# PersonaSteer V2 架构修复执行计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复HyperNetwork Cross-Attention bug、Gate集中问题、v_norm约束失效、SCL无效等4个根本问题，使Stage2/3训练真正产生query-aware的人格注入效果。

**Architecture:**
- Frozen backbone + trainable side-network（保持不变）
- HyperNetwork: 用Query-aware Gate Fusion替代buggy Cross-Attention，使v_t同时响应personality和query
- Injection: 保持layer-level，但升级为per-head gate + entropy正则化，防止gate集中单层
- Loss: 修复SCL正负样本构造，增强v_norm约束，新增gate entropy loss

**Tech Stack:** PyTorch, Transformers, Qwen2.5-3B / Qwen3-4B, ALOE数据集

---

## 背景与诊断

### 根本问题

| # | 问题 | 定位 | 症状 |
|---|------|------|------|
| 1 | Cross-Attention `dim=0` bug | `hyper_network.py:221` | v_t忽略query，只编码personality |
| 2 | Gate集中第8层 | `injection.py:84-88` | gate=[0.035,0.052,...,0.999]，其他层无效 |
| 3 | v_norm约束失效 | `trainer.py:79-80` | 实测31.9，目标5.0，weight=0.01太小 |
| 4 | SCL loss=0 | `losses.py:133` | pos_mask全零，无法构建正样本对 |

### 诊断数据

```
# Gate实测值（Qwen2.5 Stage2）:
Layer gates: [0.035, 0.052, 0.017, 0.038, 0.027, 0.047, 0.157, 0.200(clamped)]
原始第8层: 0.9999（inject_magnitude=19.53，远超hidden_states norm）

# Cross-Attention bug:
attn_weights = torch.softmax(attn_scores, dim=0)  # batch=1时永远等于1.0
→ z_fused = 1.0 * V = V（完全忽略Query）

# v_norm:
实测 norm=31.9，目标5.0，penalty=(31.9-32)²×0.01≈0.0001（几乎无约束）

# SCL:
pos_threshold=0.7太高 → personality相似度均<0.7 → pos_mask全零 → loss=0
```

---

## Task 1: 修复 HyperNetwork Cross-Attention

**文件:** `src/models/hyper_network.py`

**当前问题代码** (`hyper_network.py:215-223`):
```python
Q = self.query_proj(z_query)
K = self.key_proj(z_personality)
V = self.value_proj(z_personality)
attn_scores = (Q * K).sum(dim=-1) / (self.v_dim ** 0.5)
attn_weights = torch.softmax(attn_scores, dim=0).unsqueeze(-1)  # BUG: dim=0!
z_fused = self.attn_norm(attn_weights * V)
```

**目标设计:** Query-aware Gate Fusion，用α权重显式建模query对persona的影响

```
persona_emb + query_emb → concat → MLP → α (sigmoid, scalar)
fused = α * persona_emb + (1-α) * query_emb
```

### Step 1: 读取并理解现有forward代码

```bash
sed -n '130,260p' src/models/hyper_network.py
```

重点关注：
- `__init__` 中 `query_proj/key_proj/value_proj` 的初始化（行~59-63）
- `forward` 中 Cross-Attention 部分（行~215-223）
- `history_projector` 和 `projector` 的接口不变

### Step 2: 修改 `__init__`，用 query_gate 替代三个 proj

将 `__init__` 中（约 `hyper_network.py:59-63`）：
```python
self.query_proj = nn.Linear(v_dim, v_dim)
self.key_proj = nn.Linear(v_dim, v_dim)
self.value_proj = nn.Linear(v_dim, v_dim)
```

替换为：
```python
# Query-aware gate fusion（替代 buggy Cross-Attention）
# alpha = sigmoid(MLP([persona; query])) → 动态融合权重
self.query_gate = nn.Sequential(
    nn.Linear(v_dim * 2, v_dim // 4),
    nn.SiLU(),
    nn.Linear(v_dim // 4, 1),
    nn.Sigmoid(),
)
```

同时删除 `attn_norm` 初始化（约 `hyper_network.py:63-64`）：
```python
self.attn_norm = nn.LayerNorm(v_dim)  # 删除此行
```

### Step 3: 修改 `forward` 中的融合逻辑

将（约 `hyper_network.py:215-223`）：
```python
Q = self.query_proj(z_query)
K = self.key_proj(z_personality)
V = self.value_proj(z_personality)
attn_scores = (Q * K).sum(dim=-1) / (self.v_dim ** 0.5)
attn_weights = torch.softmax(attn_scores, dim=0).unsqueeze(-1)
z_fused = self.attn_norm(attn_weights * V)
```

替换为：
```python
# Query-aware gate fusion
# alpha: 当query与persona差异大时，alpha偏向query；相似时偏向persona
gate_input = torch.cat([z_personality, z_query], dim=-1)  # (batch, v_dim*2)
alpha = self.query_gate(gate_input)  # (batch, 1)
z_fused = alpha * z_personality + (1 - alpha) * z_query  # (batch, v_dim)
```

### Step 4: 运行单元测试验证接口不变

```bash
cd /home/kemove/Desktop/Projects/3-PersonaSteer_V2
/home/kemove/anaconda3/envs/pytorch/bin/python -m pytest tests/test_hyper_network.py -v 2>&1 | tail -20
```

预期：相关测试通过（输出形状不变）

### Step 5: 快速验证 alpha 值合理

```bash
CUDA_VISIBLE_DEVICES=0 /home/kemove/anaconda3/envs/pytorch/bin/python << 'EOF'
import torch, yaml, sys
sys.path.insert(0, '/home/kemove/Desktop/Projects/3-PersonaSteer_V2')
from src.models.hyper_network import HyperNetwork
from transformers import AutoModelForCausalLM, AutoTokenizer
config = yaml.safe_load(open('configs/train_stage2.yaml'))
backbone = AutoModelForCausalLM.from_pretrained(config['base_model'],
    trust_remote_code=True, torch_dtype=torch.float32, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(config['base_model'], trust_remote_code=True)
hn = HyperNetwork(encoder=backbone.model, v_dim=1024, hidden_dim=4096,
    num_inject_layers=8, use_layer_embedding=True)
hn._tokenizer = tokenizer
v_prev = torch.zeros(1, 1024)
v_t, v_norm = hn(["He is empathetic and creative."], ["Hey how are you?"], v_prev)
print(f"v_t shape: {v_t.shape}")  # expect (1, 8, 1024)
print(f"v_norm: {v_norm.item():.4f}")
EOF
```

预期输出：`v_t shape: torch.Size([1, 8, 1024])`

### Step 6: Commit

```bash
cd /home/kemove/Desktop/Projects/3-PersonaSteer_V2
git add src/models/hyper_network.py
git commit -m "fix: replace buggy cross-attention with query-aware gate fusion in HyperNetwork"
```

---

## Task 2: 升级 Gate 机制（Per-head + Entropy Loss）

**文件:** `src/models/injection.py`

**当前问题:**
- `DynamicGate.forward` 输出 per-layer gate（8个），第8层恒趋向饱和
- `set_intervention_vector` 中简单 clamp，训练-推理不一致
- 无entropy正则化，gate可以集中在单层

**目标:**
- 将 gate 输出改为 per-head（32 heads × 8 layers = 256维），或保持8维但加entropy
- 在 `DynamicGate.forward` 中直接返回 entropy loss 供trainer使用
- 在训练中（`DynamicGate.forward`）和推理中（`set_intervention_vector`）统一应用 clamp(max=0.3)

### Step 1: 读取 DynamicGate 的完整实现

```bash
sed -n '35,92p' src/models/injection.py
```

### Step 2: 修改 `DynamicGate.forward` 统一 clamp 并返回 entropy

将 `injection.py:71-91` 的 forward 方法改为：

```python
def forward(self, v_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算gate值和entropy loss

    Returns:
        gate_values: (batch, num_layers)，已clamp到[0, gate_max]
        entropy_loss: 标量，gate分布的负熵（最小化 = 最大化均匀性）
    """
    if v_t.dim() == 3:
        v_flat = v_t.mean(dim=1).float()
    else:
        v_flat = v_t.float()

    gate_values = self.gate_mlp(v_flat)  # (batch, num_layers)

    # 训练和推理统一限制，防止单层饱和
    gate_values = gate_values.clamp(min=0.0, max=self.gate_max)

    # Gate entropy loss: 鼓励各层均匀使用，防止集中
    # p = gate / sum(gate)，最大化熵 = 最小化负熵
    p = gate_values / (gate_values.sum(dim=-1, keepdim=True) + 1e-8)
    entropy_loss = (p * torch.log(p + 1e-8)).sum(dim=-1).mean()  # 负熵，minimize it

    return gate_values, entropy_loss
```

同时在 `__init__` 中添加 `gate_max` 参数（`injection.py:56` 附近）：
```python
def __init__(self, v_dim: int, num_layers: int, gate_hidden_dim: int = 256, gate_max: float = 0.3):
    super().__init__()
    self.gate_max = gate_max
    # 其余不变
```

### Step 3: 修改 `set_intervention_vector` 与 `forward` 保持一致

将 `injection.py:145-156` 的 `set_intervention_vector` 改为：

```python
def set_intervention_vector(self, v_t: torch.Tensor) -> None:
    gate_dtype = next(self.gate.parameters()).dtype
    if v_t.dim() == 3:
        self.current_v_t = v_t
        v_t_mean = v_t.mean(dim=1).to(gate_dtype)
        gate_values, _ = self.gate(v_t_mean)  # 推理时丢弃entropy
        self.current_gate_values = gate_values
    else:
        self.current_v_t = v_t
        gate_values, _ = self.gate(v_t.to(gate_dtype))
        self.current_gate_values = gate_values
```

### Step 4: 在 `SteeringInjection` 中暴露 entropy loss

在 `SteeringInjection` 类中添加 `compute_gate_entropy_loss` 方法（约 `injection.py:135`后）：

```python
def compute_gate_entropy_loss(self, v_t: torch.Tensor) -> torch.Tensor:
    """计算gate entropy loss，用于训练时约束gate分布均匀性"""
    gate_dtype = next(self.gate.parameters()).dtype
    if v_t.dim() == 3:
        v_mean = v_t.mean(dim=1).to(gate_dtype)
    else:
        v_mean = v_t.to(gate_dtype)
    _, entropy_loss = self.gate(v_mean)
    return entropy_loss
```

### Step 5: 运行注入相关测试

```bash
/home/kemove/anaconda3/envs/pytorch/bin/python -m pytest tests/ -k "inject or gate" -v 2>&1 | tail -20
```

### Step 6: Commit

```bash
git add src/models/injection.py
git commit -m "feat: add per-layer gate entropy loss and unify clamp in DynamicGate"
```

---

## Task 3: 修复 SCL Loss 并增强 v_norm 约束

**文件:** `src/training/losses.py`, `src/training/trainer.py`

**当前问题:**
- `losses.py:141-161` 的 `_build_pos_mask_by_personality`: `pos_threshold=0.7` 太高，导致mask全零
- `trainer.py:79-80`: `v_norm_weight=0.01`，`v_norm_target=32.0`，约束太弱

### Step 1: 修复 SCL 的 pos_threshold

在 `losses.py:60-68` 的 `__init__` 中，将默认 threshold 从 0.7 降到 0.3：

```python
def __init__(
    self,
    temperature: float = 0.07,
    pos_threshold: float = 0.3,   # 修改：0.7 → 0.3
    use_embedding_similarity: bool = False,
):
```

同时在 `_build_pos_mask_by_personality`（约 `losses.py:154-161`）中，添加降级策略：当正样本对仍为0时，使用随机正对：

```python
def _build_pos_mask_by_personality(self, personalities: list[str]) -> torch.Tensor:
    batch_size = len(personalities)
    pos_mask = torch.zeros(batch_size, batch_size)
    for i in range(batch_size):
        for j in range(batch_size):
            if i != j:
                sim = SequenceMatcher(None, personalities[i], personalities[j]).ratio()
                if sim > self.pos_threshold:
                    pos_mask[i, j] = 1.0
    # 降级策略：若没有正样本对，则跳过（返回全零，trainer会检测到并跳过）
    return pos_mask
```

### Step 2: 修复 SCL 中 pos_mask 构建的 user_id 机制

检查 trainer 调用 SCL 的方式（`trainer.py` 约 `240-245` 行）：

```bash
grep -n "scl\|personalities\|user_id\|persona_label" src/training/trainer.py | head -20
```

如果 trainer 传入的是 `personalities`（文本），修改为也传入 `user_ids`，优先使用 user_id 匹配：

在 `losses.py:69-98` 的 `forward` 中，增加 `user_ids` 可选参数：

```python
def forward(
    self,
    v_t: torch.Tensor,
    personalities: list[str],
    personality_embeddings: Optional[torch.Tensor] = None,
    user_ids: Optional[list[str]] = None,  # 新增：优先用user_id做正样本
) -> torch.Tensor:
    if user_ids is not None:
        # 最可靠的方式：同user_id为正样本
        labels = torch.tensor([hash(uid) % 10000 for uid in user_ids], device=v_t.device)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        pos_mask.fill_diagonal_(0)
    elif self.use_embedding_similarity and personality_embeddings is not None:
        pos_mask = self._build_pos_mask_by_embedding(personality_embeddings)
    else:
        pos_mask = self._build_pos_mask_by_personality(personalities).to(v_t.device)
    # 其余逻辑不变
```

### Step 3: 增强 v_norm 约束

在 `trainer.py:79-80` 修改默认值：

```python
self.v_norm_weight = config.get("v_norm_weight", 0.1)   # 修改: 0.01 → 0.1
self.v_norm_target = config.get("v_norm_target", 5.0)   # 修改: 32.0 → 5.0
```

### Step 4: 在 trainer 中集成 gate entropy loss

在 `trainer.py` 的 `train_step` 中（约 `330-334`，v_norm 之后），添加：

```python
# Gate entropy loss（防止gate集中单层）
gate_entropy_weight = self.config.get("gate_entropy_weight", 0.01)
if hasattr(self.model, 'injection') and gate_entropy_weight > 0:
    loss_gate_entropy = self.model.injection.compute_gate_entropy_loss(v_t)
    loss += gate_entropy_weight * loss_gate_entropy
```

同时在日志行（约 `trainer.py:335`）记录：

```python
epoch_sft_loss += loss_sft.item()
# 新增：记录gate entropy
if hasattr(self, '_gate_entropy_sum'):
    self._gate_entropy_sum += loss_gate_entropy.item()
```

在 epoch summary 中输出：
```python
logger.info(f"Epoch {epoch}: loss={...}, sft={...}, scl={...:.4f}, gate_entropy={...:.4f}")
```

### Step 5: 运行 loss 测试

```bash
/home/kemove/anaconda3/envs/pytorch/bin/python -m pytest tests/ -k "loss or scl" -v 2>&1 | tail -20
```

### Step 6: 快速验证 SCL 是否有正样本对

```bash
/home/kemove/anaconda3/envs/pytorch/bin/python << 'EOF'
import sys
sys.path.insert(0, '/home/kemove/Desktop/Projects/3-PersonaSteer_V2')
from src.training.losses import SupervisedContrastiveLoss
import torch

scl = SupervisedContrastiveLoss(temperature=0.07, pos_threshold=0.3)
# 模拟batch中有2个相同user的样本
personalities = [
    "He is independent and empathetic.",
    "She is creative and outgoing.",
    "He is independent and empathetic.",  # 同user
]
user_ids = ["u001", "u002", "u001"]
v_t = torch.randn(3, 1024)
loss = scl(v_t, personalities, user_ids=user_ids)
print(f"SCL loss: {loss.item():.4f}")  # 应该 > 0
EOF
```

预期：`SCL loss: X.XXXX`（非零值）

### Step 7: Commit

```bash
git add src/training/losses.py src/training/trainer.py
git commit -m "fix: lower SCL pos_threshold to 0.3, strengthen v_norm (weight=0.1, target=5.0), add gate entropy loss"
```

---

## Task 4: 更新训练配置文件

**文件:** `configs/train_stage2.yaml`, `configs/train_stage2_qwen3.yaml`, `configs/train_stage3.yaml`, `configs/train_stage3_qwen3.yaml`

### Step 1: 更新 Stage 2 Qwen2.5 配置

修改 `configs/train_stage2.yaml` 中的 `training` 块：

```yaml
training:
  stage: 2
  num_epochs: 3
  learning_rate: 0.00005
  weight_decay: 0.01
  max_grad_norm: 1.0

  sft_weight: 1.0
  scl_weight: 0.1          # 修改: 0.0 → 0.1
  temperature: 0.07

  gate_min_value: 0.0
  gate_reg_weight: 0.0
  gate_entropy_weight: 0.01  # 新增

  v_norm_weight: 0.1        # 新增（覆盖trainer默认值）
  v_norm_target: 5.0        # 新增（覆盖trainer默认值）

  use_dual_loss: false       # 修改: true → false（dual_loss设计有缺陷，改回标准SFT）

  warmup_steps: 50
  use_amp: true
  output_dir: checkpoints/stage2_v2
  save_interval: 1
  log_interval: 10
```

### Step 2: 同步更新其他3个配置文件

对 `train_stage2_qwen3.yaml`, `train_stage3.yaml`, `train_stage3_qwen3.yaml` 做相同修改：
- `scl_weight: 0.1`（stage3 已有，stage2 新增）
- `gate_entropy_weight: 0.01`
- `v_norm_weight: 0.1`
- `v_norm_target: 5.0`
- `use_dual_loss: false`

Stage3 额外配置：
```yaml
training:
  scl_weight: 0.1      # 不变（已有）
  # 其他同Stage2
```

### Step 3: Commit

```bash
git add configs/train_stage2.yaml configs/train_stage2_qwen3.yaml configs/train_stage3.yaml configs/train_stage3_qwen3.yaml
git commit -m "config: update stage2/3 configs with fixed loss weights, disable dual_loss"
```

---

## Task 5: 适配 persona_steer.py 的新接口

**文件:** `src/models/persona_steer.py`

**目标:** 确保 `forward()` 正确传递 user_ids 给 SCL，generate() 接口不变

### Step 1: 检查 forward 中 SCL 调用方式

```bash
grep -n "scl\|user_id\|personalities" src/models/persona_steer.py | head -20
```

### Step 2: 如果 forward 调用 SCL，传入 user_ids

在 `persona_steer.py` 的 `forward` 中找到调用 loss 的地方，确认 user_ids 被传递：

```python
# 在 forward() 签名中添加 user_ids 参数
def forward(self, input_ids, v_prev, personality_texts, user_query_texts,
            labels=None, user_ids=None, ...):
    ...
```

实际 SCL 调用在 `trainer.py`，不在 persona_steer.py，所以主要确认 trainer 能获取 user_ids。

### Step 3: 检查 trainer 能否获取 user_ids

```bash
grep -n "user_id\|batch\[" src/training/trainer.py | head -20
```

确认 dataloader 会提供 user_ids 字段。如果没有，在 `collator.py` 中添加：

```python
# 在 collator 返回的 batch 中包含 user_ids
return {
    ...
    "user_ids": [sample["user_id"] for sample in batch],
}
```

### Step 4: 在 trainer.train_step 中传递 user_ids 给 SCL

```python
# trainer.py 中调用 scl 的地方
user_ids = batch.get("user_ids", None)
loss_scl = self.loss_fn.scl(v_t, personalities, user_ids=user_ids)
```

### Step 5: 运行完整模型前向测试

```bash
CUDA_VISIBLE_DEVICES=0 /home/kemove/anaconda3/envs/pytorch/bin/python -m pytest tests/test_persona_steer.py -v 2>&1 | tail -30
```

### Step 6: Commit

```bash
git add src/models/persona_steer.py src/data/collator.py src/training/trainer.py
git commit -m "feat: pass user_ids to SCL for reliable positive sample construction"
```

---

## Task 6: 端到端烟雾测试

**目标:** 验证修改后整个训练流程可以正常运行 1 个 epoch，各 loss 正常

### Step 1: 小批量训练测试（只跑10步）

```bash
CUDA_VISIBLE_DEVICES=0 /home/kemove/anaconda3/envs/pytorch/bin/python scripts/train.py \
  --config configs/train_stage2.yaml \
  --debug \
  --max_steps 10 \
  2>&1 | grep -E "loss=|sft=|scl=|gate|v_norm|ERROR|Traceback"
```

预期输出（每步）：
```
loss=X.XXX, sft=X.XXX, scl=X.XXX, gate_entropy=X.XXX
```
关键验证：
- `scl=` 不为 0.0000
- 无 ERROR / Traceback
- `gate_entropy=` 有值

### Step 2: 检查 gate 分布

在10步之后停止，临时打印 gate 值：

```bash
CUDA_VISIBLE_DEVICES=0 /home/kemove/anaconda3/envs/pytorch/bin/python << 'EOF'
# 加载刚训练10步的checkpoint，检查gate激活
import torch, yaml, sys
sys.path.insert(0, '/home/kemove/Desktop/Projects/3-PersonaSteer_V2')
# 参考 Task 1 Step 5 的验证脚本，输出gate值
# 预期: 8个gate值差异明显，而非集中在第8层
EOF
```

### Step 3: Commit（如有额外修复）

```bash
git add -A
git commit -m "test: verify end-to-end training with fixed architecture"
```

---

## Task 7: 正式训练 Stage1（重训）

**注意:** 修改了 HyperNetwork 结构，Stage1 的 checkpoint 需要重训。

### Step 1: 启动 Qwen2.5 Stage1

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/kemove/anaconda3/envs/pytorch/bin/python \
  scripts/train.py --config configs/train_stage1.yaml \
  > logs/stage1_v2_qwen25_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID=$!"
```

### Step 2: 启动 Qwen3 Stage1（并行）

```bash
CUDA_VISIBLE_DEVICES=1 nohup /home/kemove/anaconda3/envs/pytorch/bin/python \
  scripts/train.py --config configs/train_stage1_qwen3.yaml \
  > logs/stage1_v2_qwen3_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID=$!"
```

### Step 3: 监控训练进度

每30分钟检查一次：
```bash
# 查看最新epoch loss
grep "Epoch [0-9]" logs/stage1_v2_*.log | tail -10
```

预期：Qwen2.5 loss 从 ~0.3 开始下降，Qwen3 从 ~0.6 开始下降

---

## Task 8: 训练 Stage2 并验证效果

### Step 1: Stage1 完成后启动 Stage2

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/kemove/anaconda3/envs/pytorch/bin/python \
  scripts/train.py --config configs/train_stage2.yaml \
  > logs/stage2_v2_qwen25_$(date +%Y%m%d_%H%M%S).log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup /home/kemove/anaconda3/envs/pytorch/bin/python \
  scripts/train.py --config configs/train_stage2_qwen3.yaml \
  > logs/stage2_v2_qwen3_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Step 2: 验证 SCL 不为零

```bash
grep "scl=" logs/stage2_v2_*.log | head -5
```

预期：`scl=0.XXXX`（非零）

### Step 3: 训练完成后快速生成测试

参考之前的生成测试脚本，测试3个不同人格的样本：

```bash
CUDA_VISIBLE_DEVICES=0 /home/kemove/anaconda3/envs/pytorch/bin/python /tmp/test_stage2_gen.py
```

验收标准：
- 无重复字符退化
- 回复体现人格特征（不只说"As an AI..."）
- Qwen3 和 Qwen2.5 均通过

---

## Task 9: 训练 Stage3 并最终评估

### Step 1: Stage2 完成后启动 Stage3

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/kemove/anaconda3/envs/pytorch/bin/python \
  scripts/train.py --config configs/train_stage3.yaml \
  > logs/stage3_v2_qwen25_$(date +%Y%m%d_%H%M%S).log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup /home/kemove/anaconda3/envs/pytorch/bin/python \
  scripts/train.py --config configs/train_stage3_qwen3.yaml \
  > logs/stage3_v2_qwen3_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Step 2: 生成所有对话

```bash
CONV_DIR="results/conversations_v2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$CONV_DIR"

# 8个模型并行生成
for spec in "baseline:baseline:configs/train_stage1.yaml:0" \
            "stage1:checkpoints/stage1_v2/best.pt:configs/train_stage1.yaml:0" \
            "stage2:checkpoints/stage2_v2/best.pt:configs/train_stage2.yaml:0" \
            "stage3:checkpoints/stage3_v2/best.pt:configs/train_stage3.yaml:0"; do
  name=$(echo $spec | cut -d: -f1)
  ckpt=$(echo $spec | cut -d: -f2)
  cfg=$(echo $spec | cut -d: -f3)
  gpu=$(echo $spec | cut -d: -f4)
  CUDA_VISIBLE_DEVICES=$gpu /home/kemove/anaconda3/envs/pytorch/bin/python \
    scripts/generate_all_conversations.py \
    --config "$cfg" --checkpoint "$ckpt" \
    --output "$CONV_DIR/${name}_qwen25.json" --gpu 0 &
done
wait
echo "生成完成: $CONV_DIR"
```

### Step 3: 启动 Judge V3 评估

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
for model in baseline_qwen25 baseline_qwen3 stage1_qwen25 stage1_qwen3 \
             stage2_qwen25 stage2_qwen3 stage3_qwen25 stage3_qwen3; do
  /home/kemove/anaconda3/envs/pytorch/bin/python scripts/judge_v3_from_conversations.py \
    --conversations "$CONV_DIR/${model}.json" \
    --output "results/judge_v3_v2_${model}_${TIMESTAMP}.json" &
done
wait
```

### Step 4: 对比新旧评估结果

```bash
/home/kemove/anaconda3/envs/pytorch/bin/python << 'EOF'
import json
ts = "TIMESTAMP"  # 替换为实际timestamp
models = ['baseline_qwen25','stage1_qwen25','stage2_qwen25','stage3_qwen25']
print("模型          | Weighted | Style | Content | Consistency")
for m in models:
    d = json.load(open(f"results/judge_v3_v2_{m}_{ts}.json"))
    w = d.get('weighted',{}).get('mean',0)
    s = d.get('style',{}).get('mean',0)
    c = d.get('content',{}).get('mean',0)
    k = d.get('consistency',{}).get('mean',0)
    print(f"{m:20s} | {w:.3f}    | {s:.2f}  | {c:.2f}    | {k:.2f}")
EOF
```

验收标准：
- Stage3 > Stage2 > Stage1 > Baseline（Qwen3 和 Qwen2.5 均有阶梯式提升）
- Weighted score > 3.5（优于当前 3.0 的全默认值）

---

## 验收标准总览

| 指标 | 修复前 | 目标 | 验证方法 |
|------|--------|------|----------|
| Cross-Attention | dim=0 bug，忽略query | alpha值0.2-0.8，随query变化 | Task 1 Step 5 |
| Gate分布 | 第8层=0.999，其他<0.15 | 各层gate均在0.05-0.25 | Task 6 Step 2 |
| SCL loss | 0.0000 | >0.1 | Task 3 Step 6 / Task 8 Step 2 |
| v_t norm | 31.9 | ~5.0 | 训练日志观察 |
| 生成效果 | Qwen2.5:"As an AI..." | 体现人格特征 | Task 8 Step 3 |
| Judge V3 | baseline=stage2=stage3=3.0 | stage3>3.5 | Task 9 Step 4 |

---

## 关键文件速查

| 文件 | 关键行 | 内容 |
|------|--------|------|
| `src/models/hyper_network.py` | 59-63, 215-223 | Cross-Attention → Gate Fusion |
| `src/models/injection.py` | 71-91, 139-156 | Gate forward + set_intervention |
| `src/training/losses.py` | 60, 133, 141-161 | SCL threshold + pos_mask |
| `src/training/trainer.py` | 79-80, 330-334 | v_norm weight/target + gate entropy |
| `configs/train_stage2.yaml` | training.scl_weight 等 | 超参数 |
