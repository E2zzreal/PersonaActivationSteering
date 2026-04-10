# PersonaSteer V2 修复方案 V4

**日期**: 2026-03-28
**版本**: V4 - 根因修复
**状态**: 待执行

---

## 1. 修复总览表

| 优先级 | 文件 | 修改内容 | 根因 |
|--------|------|---------|------|
| **P0** | `src/models/injection.py` | Gate初始化：sigmoid偏置初始化为-3（约0.05） | B |
| **P0** | `src/training/trainer.py` | 训练目标：计算clean和injected双loss | A |
| **P0** | `configs/train_stage1_qwen3.yaml` | Qwen3注入层改为[8-15] | C |
| **P1** | `src/evaluation/auto_metrics.py` | 分别计算clean/injected loss | D |
| **P1** | `src/models/hyper_network.py` | 可选：简化layer embedding | E |
| **P2** | `scripts/evaluate.py` | Baseline评估禁用注入 | B |

---

## 2. 根因分析

### 根因A: 训练目标错误

**现象**: Stage训练后生成质量崩溃，评估分数下降

**根因**: 在注入后的logits上计算SFT loss，导致模型同时学习：
- 正确生成（来自真实标签）
- 注入干扰（来自注入后的分布偏移）

**正确做法**: 对比学习范式
- `loss_clean`: 原始模型输出（无注入）的loss
- `loss_injected`: 注入后模型输出的loss
- `loss_total = loss_clean + alpha * loss_injected`

### 根因B: Gate初始值≈0.5

**现象**: Baseline（未训练）模型评估分数异常低

**根因**:
- `DynamicGate`使用Sigmoid输出，初始值≈0.5
- 未训练时50%强度注入，破坏模型原有能力

**正确做法**:
- Gate初始化偏置设为-3（约0.05，5%强度）
- Baseline评估时禁用注入（gate=0）

### 根因C: Qwen3注入位置不当

**现象**: Qwen3-4B注入后生成质量崩溃

**根因**: 注入层[0-7]位于模型前部，负责基础语义理解
- 在此注入会破坏词嵌入和早期语义提取
- 导致生成语法错误、重复内容

**正确做法**:
- Qwen3改为中部层[8-15]
- 保留前层的基础能力

### 根因D: 评估未分离注入效果

**现象**: 无法验证注入是否真正有效

**根因**: 评估时同时计算clean和injected loss，无法分离

**正确做法**:
- 分别计算两种loss
- 分析差异，验证注入效果

---

## 3. 修复代码

### 修复A: 训练目标修复（双loss）

**文件**: `src/training/trainer.py`

```python
def train_step_with_dual_loss(self, batch, turn_idx):
    """
    对比学习范式的双loss训练

    1. loss_clean: 原始模型输出（注入禁用）
    2. loss_injected: 注入后的模型输出
    3. loss_total = loss_clean + alpha * loss_injected

    Args:
        batch: 训练批次
        turn_idx: 当前轮次索引

    Returns:
        dict: 包含各项损失
    """
    # ... 前向传播准备代码 ...

    # ===== 1. 计算注入后的loss（原有逻辑） =====
    logits_injected, v_t, v_norm = self.model(
        input_ids=valid_input_ids,
        v_prev=valid_v_prev,
        user_texts=valid_user_texts,
    )

    loss_injected = compute_sft_loss(logits_injected, valid_labels)

    # ===== 2. 计算未注入的loss（禁用注入） =====
    # 保存原始gate值
    original_gate_values = self.model.injection.current_gate_values

    # 临时禁用注入（设置gate=0）
    self.model.injection.current_gate_values = torch.zeros_like(original_gate_values)

    with torch.no_grad():
        logits_clean, _, _ = self.model(
            input_ids=valid_input_ids,
            v_prev=valid_v_prev,
            user_texts=valid_user_texts,
        )

    loss_clean = compute_sft_loss(logits_clean, valid_labels)

    # 恢复gate值
    self.model.injection.current_gate_values = original_gate_values

    # ===== 3. 组合损失 =====
    # loss_clean作为基础，loss_injected作为对比
    alpha = self.config.get("injected_loss_weight", 0.5)
    loss_total = loss_clean + alpha * loss_injected

    return {
        "loss_total": loss_total,
        "loss_clean": loss_clean,
        "loss_injected": loss_injected,
    }
```

**修改训练循环**（在`train_epoch`方法中）:

```python
# 在第256行附近的forward后，替换损失计算逻辑
# 原始代码:
# loss_sft = compute_sft_loss(logits, valid_labels)

# 新代码 - 双loss训练
loss_dict = self.train_step_with_dual_loss(batch, turn_idx)
loss_sft = loss_dict["loss_clean"]  # 基础loss
loss_injected = loss_dict["loss_injected"]  # 注入loss
```

**配置文件新增参数**（在yaml中）:

```yaml
training:
  injected_loss_weight: 0.5  # 注入loss权重
  use_dual_loss: true        # 启用双loss训练
```

---

### 修复B1: Gate初始化修复

**文件**: `src/models/injection.py`

```python
class DynamicGate(nn.Module):
    """动态门控模块 - 修复初始化"""

    def __init__(
        self,
        v_dim: int = 1024,
        num_layers: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.v_dim = v_dim        self.num_layers = num_layers        self.gate_mlp = nn.Sequential(
            nn.Linear(v_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_layers),
            nn.Sigmoid()        )

        # 【修复B】初始化：使初始输出接近0（约0.05）
        # Sigmoid初始值≈0.5，我们需要偏移-3使输出≈0.05
        self._init_gate_bias()

    def _init_gate_bias(self):
        """初始化gate偏置，使初始输出接近0"""
        # 找到最后一个Linear层
        last_linear = None
        for module in self.gate_mlp.modules():
            if isinstance(module, nn.Linear) and module.out_features == self.num_layers:
                last_linear = module
                break

        if last_linear is not None:
            # 偏置初始化为-3，使Sigmoid输出≈0.05
            nn.init.constant_(last_linear.bias, -3.0)
            print(f"[DynamicGate] Initialized bias to -3.0 (output≈0.05)")

    def forward(self, v_t: torch.Tensor) -> torch.Tensor:
        """生成门控值"""
        if v_t.dim() == 3:
            batch_size, seq_len, _ = v_t.shape
            v_flat = v_t.reshape(-1, self.v_dim)
            gate_flat = self.gate_mlp(v_flat)
            gate_values = gate_flat.reshape(batch_size, seq_len, self.num_layers)
        else:
            gate_values = self.gate_mlp(v_t)
        return gate_values
```

---

### 修复B2: Baseline评估禁用注入

**文件**: `src/models/persona_steer.py`

```python
class PersonaSteerModel(nn.Module):
    """PersonaSteer 模型 - 添加baseline模式"""

    def __init__(
        self,
        config: PersonaSteerConfig,
        backbone: Optional[nn.Module] = None,
        encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        # ... 原有初始化代码 ...

        # 【修复B】添加baseline模式
        self.baseline_mode = False  # 禁用注入

    def set_baseline_mode(self, enabled: bool):
        """
        设置baseline模式

        Args:
            enabled: True=禁用注入（完全通过原始模型）,
                     False=启用注入
        """
        self.baseline_mode = enabled
        if enabled:
            # 禁用注入：将gate设为0
            if self.injection is not None:
                self.injection.current_gate_values = torch.zeros(
                    1, self.injection.num_inject_layers
                )
        print(f"[PersonaSteer] Baseline mode: {enabled}")

    def forward(
        self,
        input_ids: torch.Tensor,
        v_prev: torch.Tensor,
        user_texts: List[str],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播 - 支持baseline模式"""

        # Step 1: 生成干预向量（即使在baseline模式也生成，保持接口一致）
        if self.hyper_network is not None:
            v_t_layers, z_t, v_norm = self.hyper_network(user_texts, v_prev)
        else:
            v_t_layers = torch.zeros(
                input_ids.size(0),
                8,
                self.config.v_dim,
                device=input_ids.device,
            )
            v_norm = torch.zeros(input_ids.size(0), device=input_ids.device)

        self.current_v_t = v_t_layers

        # Step 2: 设置注入向量
        if self.baseline_mode:
            # 【修复B】Baseline模式：禁用注入
            self.injection.current_gate_values = torch.zeros(
                v_t_layers.size(0), self.injection.num_inject_layers,
                device=v_t_layers.device
            )
        else:
            self.injection.set_intervention_vector(v_t_layers)

        # Step 3: 骨干模型前向传播
        if self.backbone is not None:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
        else:
            logits = torch.randn(
                input_ids.size(0),
                input_ids.size(1),
                151936,
                device=input_ids.device,
            )

        return logits, v_t_layers, v_norm
```

**配置文件添加**:

```yaml
evaluation:
  baseline_mode: false  # 评估时是否禁用注入
```

---

### 修复C: Qwen3注入位置修复

**文件**: `configs/train_stage1_qwen3.yaml`

```yaml
# 模型配置 - Qwen3注入层改为中部层
model:
  # 【修复C】Qwen3注入层改为[8-15]（模型中部层）
  # 原因：前层[0-7]负责基础语义，注入会破坏生成能力
  inject_layers: [8, 9, 10, 11, 12, 13, 14, 15]

  v_dim: 1024
  hidden_dim: 4096
  layer_dim: 2560  # Qwen3-4B hidden_size
```

**同步修改其他Qwen3配置文件**:

```yaml
# configs/train_stage2_qwen3.yaml
model:
  inject_layers: [8, 9, 10, 11, 12, 13, 14, 15]

# configs/train_stage3_qwen3.yaml
model:
  inject_layers: [8, 9, 10, 11, 12, 13, 14, 15]
```

---

### 修复D: 评估分离clean/injected loss

**文件**: `src/evaluation/auto_metrics.py`

```python
def compute_evaluation_metrics(
    model,
    dataset,
    tokenizer,
    device: str = "cuda",
    baseline_mode: bool = False,
) -> dict:
    """
    评估模型性能，分离clean和injected loss

    Args:
        model: PersonaSteerModel
        dataset: 评估数据集
        tokenizer: 分词器
        device: 设备
        baseline_mode: True=禁用注入（baseline）, False=启用注入

    Returns:
        dict: 包含各项评估指标
    """
    model.eval()

    # 设置baseline模式
    if hasattr(model, "set_baseline_mode"):
        model.set_baseline_mode(baseline_mode)

    total_loss_clean = 0.0
    total_loss_injected = 0.0
    num_samples = 0

    with torch.no_grad():
        for sample in tqdm(dataset, desc="Evaluating"):
            # 准备输入
            inputs = tokenizer(
                sample["user_text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            # 获取标签
            labels = sample["labels"].to(device)

            # 1. 计算injected loss（启用注入）
            model.set_baseline_mode(False)
            logits_injected, _, _ = model(
                input_ids=inputs["input_ids"],
                v_prev=torch.zeros(1, model.v_dim).to(device),
                user_texts=[sample["user_text"]],
            )
            loss_injected = F.cross_entropy(
                logits_injected.view(-1, logits_injected.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

            # 2. 计算clean loss（禁用注入）
            model.set_baseline_mode(True)
            logits_clean, _, _ = model(
                input_ids=inputs["input_ids"],
                v_prev=torch.zeros(1, model.v_dim).to(device),
                user_texts=[sample["user_text"]],
            )
            loss_clean = F.cross_entropy(
                logits_clean.view(-1, logits_clean.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

            total_loss_clean += loss_clean.item()
            total_loss_injected += loss_injected.item()
            num_samples += 1

    # 恢复模型状态
    model.set_baseline_mode(False)

    avg_loss_clean = total_loss_clean / max(num_samples, 1)
    avg_loss_injected = total_loss_injected / max(num_samples, 1)

    # 计算注入增益
    injection_gain = avg_loss_clean - avg_loss_injected
    injection_gain_pct = (injection_gain / avg_loss_clean) * 100

    return {
        "loss_clean": avg_loss_clean,
        "loss_injected": avg_loss_injected,
        "injection_gain": injection_gain,
        "injection_gain_pct": injection_gain_pct,
        "num_samples": num_samples,
    }
```

---

### 修复E: 架构简化（可选）

**文件**: `src/models/hyper_network.py`

```python
class HyperNetwork(nn.Module):
    """超网络 - 可选简化layer embedding"""

    def __init__(
        self,
        encoder: nn.Module,
        v_dim: int = 1024,
        hidden_dim: int = 4096,
        num_layers: int = 3,
        encoder_dim: int = None,
        num_inject_layers: int = 8,
        use_layer_embedding: bool = False,  # 【修复E】默认禁用
    ):
        super().__init__()
        # ... 其他初始化 ...
        self.use_layer_embedding = use_layer_embedding

        if use_layer_embedding:
            self.layer_embedding = nn.Embedding(num_inject_layers, v_dim)
            nn.init.normal_(self.layer_embedding.weight, mean=0.0, std=0.02)
        else:
            self.layer_embedding = None

    def forward(self, user_texts: list[str], v_prev: torch.Tensor):
        """前向传播 - 简化版"""
        # ... 编码器处理 ...

        # 简化：所有层使用相同的干预向量
        if not self.use_layer_embedding or self.layer_embedding is None:
            v_t = self.projector(fused)  # (batch, v_dim)
            # 复制到所有注入层
            v_t_layers = v_t.unsqueeze(1).expand(
                batch_size, self.num_inject_layers, self.v_dim
            )
        else:
            # 使用层嵌入的原有逻辑
            # ...

        return v_t_layers, z_t, v_norm
```

**配置文件更新**:

```yaml
model:
  use_layer_embedding: false  # 【修复E】简化架构
```

---

## 4. 修复依赖关系

```
依赖图:
├── 修复B (Gate初始化)
│   └── 是所有其他修复的基础
│
├── 修复C (注入位置)
│   └── 依赖B：需要正确初始化的gate
│
├── 修复A (训练目标)
│   ├── 依赖B：需要正确的gate初始值
│   └── 依赖C：需要正确的注入位置
│
├── 修复D (评估)
│   └── 依赖B：需要baseline模式支持
│
└── 修复E (架构简化)
    └── 独立：可选修复
```

**执行顺序**:
1. **Phase 1** (P0 - 立即执行):
   - B1: Gate初始化
   - C: 注入位置
   - B2: Baseline模式

2. **Phase 2** (P1 - 依赖Phase 1):
   - A: 双loss训练
   - D: 评估分离

3. **Phase 3** (P2 - 可选):
   - E: 架构简化

---

## 5. 验证计划

### 5.1 单元验证

```python
# test_gate_initialization.py
def test_gate_initialization():
    """验证Gate初始化"""
    gate = DynamicGate(v_dim=1024, num_layers=8)

    # 输入随机向量
    v = torch.randn(2, 1024)
    output = gate(v)

    # 验证输出接近0.05
    expected = 0.05
    tolerance = 0.02
    assert abs(output.mean().item() - expected) < tolerance, \
        f"Gate output {output.mean().item():.3f} not close to {expected}"
```

```python
# test_baseline_mode.py
def test_baseline_mode():
    """验证baseline模式禁用注入"""
    model = create_personasteer_model(config)
    model.set_baseline_mode(True)

    # 验证gate值为0
    assert model.injection.current_gate_values.abs().sum().item() == 0
```

### 5.2 集成验证

```bash
# 1. 验证Gate初始化
python -c "
from src.models.injection import DynamicGate
import torch
gate = DynamicGate(v_dim=1024, num_layers=8)
v = torch.randn(4, 1024)
output = gate(v)
print(f'Gate output: {output.mean():.4f} (expected ~0.05)')
"

# 2. 验证注入位置
python -c "
import yaml
with open('configs/train_stage1_qwen3.yaml') as f:
    config = yaml.safe_load(f)
print(f\"Qwen3 inject_layers: {config['model']['inject_layers']} (expected [8-15])\")
"

# 3. 验证Baseline模式
python scripts/evaluate.py --baseline --model checkpoints/baseline
```

### 5.3 训练验证

```bash
# Stage 1 训练验证
python scripts/train.py --config configs/train_stage1.yaml --max-steps 10

# 检查loss变化
# 预期: loss_clean稳定，loss_injected下降
```

### 5.4 评估验证

```bash
# 完整评估流程
python scripts/evaluate.py \
    --model checkpoints/stage1 \
    --output results/eval_stage1.json

# 验证输出包含
# - loss_clean: xxx
# - loss_injected: xxx
# - injection_gain: xxx%
```

---

## 6. 预期改进

### 修复前问题

| 问题 | 表现 |
|------|------|
| Gate初始值≈0.5 | Baseline评估分数异常低 |
| 注入位置[0-7] | Qwen3生成质量崩溃 |
| 训练目标错误 | Stage训练后质量下降 |
| 无baseline对比 | 无法验证注入效果 |

### 修复后预期

| 指标 | 修复前 | 修复后预期 |
|------|--------|-----------|
| Gate初始值 | ~0.5 | ~0.05 |
| Baseline AL(K) | 1.0 | >2.0 |
| Qwen3 AL(K) | <2.0 | >3.0 |
| 注入增益 | N/A | >10% |

---

## 7. 回滚方案

如果修复后效果未改善:

1. **回滚Gate初始化**: 偏置从-3改为0
2. **回滚注入位置**: 改回[0-7]
3. **回滚训练目标**: 使用原有单loss
4. **回滚架构**: 启用layer_embedding

```bash
# 快速回滚命令
git checkout -- src/models/injection.py src/training/trainer.py configs/train_stage1_qwen3.yaml
```

---

**文档版本**: V4.0
**修复完成时间**: 待定
