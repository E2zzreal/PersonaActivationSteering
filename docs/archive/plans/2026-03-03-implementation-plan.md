# PersonaSteer 实现计划

**日期**: 2026-03-03
**状态**: 待执行
**基于**: `2026-03-02-personasteer-design.md`

---

## 执行概览

本计划将设计文档分解为可执行的实现步骤，采用渐进式开发策略：

```
Phase 0: 环境准备 (1天)
   ↓
Phase 1: Probing 实验 (1天)
   ↓
Phase 2: 核心模块实现 (2-3天)
   ↓
Phase 3: 训练流程实现 (3天)
   ↓
Phase 4: 评估与优化 (1天)
```

---

## Phase 0: 环境准备与数据预处理

### 目标
- 搭建开发环境
- 下载并预处理 ALOE 数据集
- 验证模型可访问性

### 任务清单

#### 0.1 项目初始化
- [x] 创建目录结构
- [ ] 编写 `requirements.txt`
- [ ] 编写 `pyproject.toml`
- [ ] 初始化 git 仓库
- [ ] 创建 `.gitignore`

**依赖包**:
```txt
torch>=2.1.0
transformers>=4.36.0
deepspeed>=0.12.0
accelerate>=0.25.0
datasets>=2.16.0
pyyaml>=6.0
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.8.0
seaborn>=0.13.0
openai>=1.0.0
tqdm>=4.66.0
tensorboard>=2.15.0
```

#### 0.2 数据获取与处理
```bash
# 下载 ALOE 数据集
git clone https://github.com/ShujinWu-0814/ALOE.git data/aloe_raw

# 数据预处理脚本
python scripts/preprocess_aloe.py \
    --input data/aloe_raw \
    --output data/processed \
    --min_turns 3 \
    --max_turns 10
```

**输出格式**:
```json
{
  "user_id": "u001",
  "profile": "34岁自由设计师...",
  "personality": "独立、有条理...",
  "conversations": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

#### 0.3 模型下载验证
```python
# scripts/verify_models.py
from transformers import AutoModel, AutoTokenizer

# 验证 Qwen3-4B
model = AutoModel.from_pretrained("Qwen/Qwen3-4B")
print(f"[OK] Qwen3-4B loaded: {model.config.num_hidden_layers} layers")

# 验证 Qwen3-Embedding
encoder = AutoModel.from_pretrained("Qwen/Qwen3-Embedding")
print(f"[OK] Qwen3-Embedding loaded: dim={encoder.config.hidden_size}")
```

**验收标准**:
- [ ] 所有依赖安装无错误
- [ ] ALOE 数据集下载完成 (约 5GB)
- [ ] 处理后至少 10,000 条对话样本
- [ ] Qwen3-4B 和 Qwen3-Embedding 可正常加载

---

## Phase 1: Probing 实验

### 目标
通过 Layer × Head 粒度分析，确定最优注入层配置。

### 文件创建顺序

#### 1.1 属性提取器 (`src/probing/attribute_extractor.py`)
```python
"""
从 ALOE 数据集提取人格属性标注
"""

class AttributeExtractor:
    def extract_attributes(self, sample):
        """
        输入: ALOE 样本
        输出: {
            "age": float,
            "gender": int,
            "introversion": float,
            "openness": float,
            ...
        }
        """
        pass
```

**测试**:
```bash
python -m pytest tests/test_attribute_extractor.py -v
```

#### 1.2 激活值收集器 (`src/probing/head_probing.py`)
```python
"""
收集 Qwen3-4B 各层各头的激活值
"""

class AttentionHeadProber:
    def __init__(self, model, tokenizer):
        self._register_hooks()

    def collect(self, texts: list[str]):
        """
        返回: (N, 36_layers, 32_heads, head_dim)
        """
        pass

    def compute_spearman(self, activations, attr_values):
        """
        返回: (36, 32) 相关矩阵
        """
        pass
```

**测试**:
```bash
# 小规模测试 (100 样本)
python scripts/run_probing.py --config configs/probing.yaml --debug
```

#### 1.3 可视化与层选择 (`src/probing/visualize.py`)
```python
"""
生成热力图并自动选择注入层
"""

def plot_heatmap(matrix, save_path):
    """绘制 36×32 热力图"""
    pass

def select_injection_layers(matrix, top_k=8, strategy="continuous"):
    """
    策略:
    - "top": 选择相关性最高的 k 层
    - "continuous": 选择相关性峰值附近的连续 k 层

    返回: list[int]
    """
    pass
```

#### 1.4 完整 Probing 脚本 (`scripts/run_probing.py`)
```bash
python scripts/run_probing.py \
    --model Qwen/Qwen3-4B \
    --data data/processed/train.jsonl \
    --num_samples 1000 \
    --output experiments/probing/results.json
```

**输出**:
```
experiments/probing/
├── heatmaps/
│   ├── age.png
│   ├── introversion.png
│   └── average.png
├── results.json
└── selected_layers.yaml  # 注入层配置
```

**验收标准**:
- [ ] 成功运行 1000 样本 Probing
- [ ] 生成完整热力图
- [ ] 自动选择 8 层注入配置
- [ ] 更新 `configs/model.yaml` 中的 `inject_layers`

---

## Phase 2: 核心模块实现

### 目标
实现 PersonaSteer 的三大核心组件。

### 2.1 基础组件 (`src/models/components.py`)
```python
"""
可复用的基础组件
"""

class ResidualMLP(nn.Module):
    """残差MLP块"""
    def __init__(self, dim, hidden_dim, dropout=0.1):
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.activation(self.linear1(x))
        h = self.dropout(h)
        h = self.linear2(h)
        return x + h  # 残差连接
```

**单元测试**:
```python
# tests/test_components.py
def test_residual_mlp_shape():
    mlp = ResidualMLP(dim=1024, hidden_dim=4096)
    x = torch.randn(2, 1024)
    out = mlp(x)
    assert out.shape == x.shape
```

### 2.2 超网络 (`src/models/hyper_network.py`)
```python
"""
HyperNetwork: 从用户话语生成干预向量
"""

class HyperNetwork(nn.Module):
    def __init__(self, encoder, v_dim=1024, hidden_dim=4096, num_layers=3):
        self.encoder = encoder  # 冻结的 Qwen3-Embedding
        self.history_projector = nn.Linear(v_dim, v_dim)

        layers = []
        for _ in range(num_layers):
            layers.append(ResidualMLP(v_dim, hidden_dim))
        layers.append(nn.LayerNorm(v_dim))
        self.projector = nn.Sequential(*layers)

    def forward(self, user_text: str, v_prev: torch.Tensor):
        """
        输入:
            user_text: 当前用户话语
            v_prev: 上一轮干预向量 (batch, 1024)

        输出:
            v_t: 当前干预向量 (batch, 1024)
        """
        with torch.no_grad():
            z_t = self.encoder.encode(user_text)  # (batch, 1024)

        h_prev = self.history_projector(v_prev)
        fused = z_t + h_prev
        v_t = self.projector(fused)

        return v_t
```

**测试**:
```python
def test_hyper_network_forward():
    encoder = load_frozen_encoder()
    hyper = HyperNetwork(encoder)

    v_prev = torch.zeros(2, 1024)
    v_t = hyper("你好，今天天气真好", v_prev)

    assert v_t.shape == (2, 1024)
    assert not v_t.requires_grad  # 验证 encoder 冻结
```

### 2.3 动态门控注入 (`src/models/injection.py`)
```python
"""
DynamicGate + SteeringInjection
"""

class DynamicGate(nn.Module):
    def __init__(self, v_dim=1024, num_layers=8, hidden_dim=256):
        self.gate_mlp = nn.Sequential(
            nn.Linear(v_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_layers),
            nn.Sigmoid()
        )

    def forward(self, v_t):
        return self.gate_mlp(v_t)  # (batch, num_layers)


class SteeringInjection(nn.Module):
    def __init__(self, inject_layers, v_dim=1024, layer_dim=2560):
        self.inject_layers = inject_layers
        self.gate = DynamicGate(v_dim, len(inject_layers))

        self.layer_projectors = nn.ModuleList([
            nn.Linear(v_dim, layer_dim) for _ in inject_layers
        ])

    def inject(self, hidden_states, v_t, layer_idx):
        """
        hidden_states: (batch, seq_len, 2560)
        v_t: (batch, 1024)
        layer_idx: 当前层在 inject_layers 中的索引
        """
        gate_values = self.gate(v_t)
        gate = gate_values[:, layer_idx:layer_idx+1]  # (batch, 1)

        proj = self.layer_projectors[layer_idx](v_t)  # (batch, 2560)

        return hidden_states + gate.unsqueeze(-1) * proj.unsqueeze(1)
```

**集成测试**:
```python
def test_injection_integration():
    injection = SteeringInjection([14, 15, 16], v_dim=1024, layer_dim=2560)

    h = torch.randn(2, 50, 2560)
    v = torch.randn(2, 1024)

    h_injected = injection.inject(h, v, layer_idx=0)
    assert h_injected.shape == h.shape
```

### 2.4 完整模型 (`src/models/persona_steer.py`)
```python
"""
PersonaSteer: 整合所有组件
"""

class PersonaSteerModel(nn.Module):
    def __init__(self, config):
        self.backbone = load_frozen_qwen3_4b()
        self.encoder = load_frozen_qwen3_embedding()
        self.hyper_network = HyperNetwork(self.encoder)
        self.injection = SteeringInjection(config.inject_layers)

        # 注册 hook 到 backbone
        self._register_injection_hooks()

    def _register_injection_hooks(self):
        for idx, layer_idx in enumerate(self.injection.inject_layers):
            layer = self.backbone.model.layers[layer_idx]
            layer.register_forward_hook(self._create_hook(idx))

    def forward(self, input_ids, v_prev, user_text):
        """
        input_ids: (batch, seq_len) - 当前轮次的输入 token
        v_prev: (batch, 1024) - 上一轮干预向量
        user_text: str - 用户话语文本

        返回: logits, v_t
        """
        # Step 1: 生成新干预向量
        self.current_v_t = self.hyper_network(user_text, v_prev)

        # Step 2: 前向传播 (会触发 hook 注入)
        outputs = self.backbone(input_ids)

        return outputs.logits, self.current_v_t
```

**验收标准**:
- [ ] 所有单元测试通过
- [ ] 模型可正常前向传播
- [ ] 可训练参数量约 47M
- [ ] 冻结参数不参与梯度计算

---

## Phase 3: 训练流程实现

### 3.1 数据加载器 (`src/data/aloe_dataset.py`)
```python
"""
ALOE 数据集的 PyTorch Dataset
"""

class ALOEDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_turns=10):
        self.data = self._load_jsonl(data_path)
        self.tokenizer = tokenizer
        self.max_turns = max_turns

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 展开对话轮次
        turns = []
        for i in range(0, len(sample["conversations"]), 2):
            if i // 2 >= self.max_turns:
                break

            user_msg = sample["conversations"][i]["content"]
            asst_msg = sample["conversations"][i+1]["content"]

            turns.append({
                "user_text": user_msg,
                "input_ids": self.tokenizer.encode(user_msg),
                "labels": self.tokenizer.encode(asst_msg),
            })

        return {
            "user_id": sample["user_id"],
            "profile": sample["profile"],
            "turns": turns,
        }
```

### 3.2 Collator (`src/data/collator.py`)
```python
"""
动态批处理，支持变长对话
"""

class PersonaSteerCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch):
        # 批处理每个轮次
        # 返回: {
        #   "input_ids": (batch, seq_len),
        #   "labels": (batch, seq_len),
        #   "user_texts": list[str],
        #   "user_ids": list[str],
        # }
        pass
```

### 3.3 损失函数 (`src/training/losses.py`)
```python
"""
SFT Loss + Supervised Contrastive Loss
"""

def compute_sft_loss(logits, labels):
    """标准语言模型损失"""
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, v_t, user_ids):
        """
        v_t: (batch, 1024) - 干预向量
        user_ids: list[str] - 用户 ID

        同用户为正例，不同用户为负例
        """
        # L2 归一化
        v_norm = F.normalize(v_t, dim=-1)

        # 相似度矩阵
        sim_matrix = torch.matmul(v_norm, v_norm.T) / self.temperature

        # 构建标签 (同用户=1)
        labels = torch.tensor([
            [1 if uid1 == uid2 else 0 for uid2 in user_ids]
            for uid1 in user_ids
        ], device=v_t.device)

        # InfoNCE Loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        loss = -(labels * log_prob).sum() / labels.sum()
        return loss
```

### 3.4 训练器 (`src/training/trainer.py`)
```python
"""
三阶段渐进训练器
"""

class PersonaSteerTrainer:
    def __init__(self, model, config, train_loader, eval_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.scaler = torch.cuda.amp.GradScaler()

        self.scl_loss = SupervisedContrastiveLoss()
        self.metrics = AutoMetrics()

    def train_epoch(self, epoch):
        self.model.train()

        for batch in tqdm(self.train_loader):
            # 多轮对话循环
            v_t = torch.zeros(batch_size, 1024).to(device)

            for turn_idx in range(num_turns):
                logits, v_t = self.model(
                    input_ids=batch["input_ids"][turn_idx],
                    v_prev=v_t.detach(),  # 阻断跨轮梯度
                    user_text=batch["user_texts"][turn_idx],
                )

                # 计算损失
                loss_sft = compute_sft_loss(logits, batch["labels"][turn_idx])
                loss_scl = self.scl_loss(v_t, batch["user_ids"])

                loss = (self.config.sft_weight * loss_sft +
                        self.config.scl_weight * loss_scl)

                # 反向传播
                self.scaler.scale(loss).backward()

            # 梯度裁剪与更新
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # 记录指标
            self.metrics.update_loss(loss_sft, loss_scl, loss)
```

### 3.5 训练脚本 (`scripts/train.py`)
```bash
# 阶段1
python scripts/train.py \
    --config configs/train_stage1.yaml \
    --output checkpoints/stage1

# 阶段2
python scripts/train.py \
    --config configs/train_stage2.yaml \
    --resume checkpoints/stage1/best.pt \
    --output checkpoints/stage2

# 阶段3
python scripts/train.py \
    --config configs/train_stage3.yaml \
    --resume checkpoints/stage2/best.pt \
    --output checkpoints/stage3
```

**验收标准**:
- [ ] 阶段1训练无错误，loss 收敛
- [ ] 阶段2解冻门控后 loss 继续下降
- [ ] 阶段3加入对比学习后指标提升
- [ ] 自动保存 checkpoint 和训练日志

---

## Phase 4: 评估与优化

### 4.1 自动指标评估 (`src/evaluation/auto_metrics.py`)
```python
"""
Loss, PPL, v_t 稳定性等
"""

class AutoMetricsEvaluator:
    def evaluate(self, model, eval_loader):
        results = {
            "loss": [],
            "ppl": [],
            "v_variance": [],
            "gate_distribution": [],
        }

        # 循环评估
        ...

        return results
```

### 4.2 LLM Judge 评估 (`src/evaluation/llm_judge.py`)
```python
"""
AL(K)_AVG, N-IR, N-R² 指标
"""

class LLMJudgeEvaluator:
    def __init__(self, judge_model="gpt-4o-mini"):
        self.client = OpenAI()
        self.judge_model = judge_model

    def evaluate_alignment(self, model, test_samples):
        """批量评估对齐分数"""
        scores = []
        for sample in tqdm(test_samples):
            conversation = self._generate_conversation(model, sample)
            score = self.compute_al_k_avg(
                conversation,
                sample["profile"],
                sample["personality"]
            )
            scores.append(score)

        return {
            "al_k_avg": np.mean(scores),
            "n_ir": self.compute_n_ir(scores, baseline_scores),
        }
```

### 4.3 评估脚本 (`scripts/evaluate.py`)
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/stage3/best.pt \
    --data data/processed/eval.jsonl \
    --output experiments/eval_results.json \
    --judge_model gpt-4o-mini \
    --num_samples 500
```

**输出示例**:
```json
{
  "auto_metrics": {
    "loss_sft": 2.134,
    "ppl": 8.45,
    "v_variance": 0.023
  },
  "llm_judge": {
    "al_k_avg": 4.12,
    "n_ir": 0.28,
    "n_r2": 0.87
  }
}
```

### 4.4 推理演示 (`scripts/inference.py`)
```python
"""
交互式对话演示
"""

def interactive_chat(model, tokenizer):
    print("PersonaSteer 对话演示")
    print("输入用户画像，然后开始对话")

    profile = input("用户画像: ")
    v_t = torch.zeros(1, 1024).to(device)

    while True:
        user_input = input("你: ")
        if user_input == "exit":
            break

        response, v_t = model.generate(user_input, v_t)
        print(f"AI: {response}")
```

**验收标准**:
- [ ] 自动指标评估完成
- [ ] LLM Judge 评估完成 (500 样本)
- [ ] 生成评估报告
- [ ] 推理演示可正常运行

---

## 实施时间表

| 日期 | Phase | 任务 | 预计工时 |
|------|-------|------|---------|
| Day 1 | Phase 0 | 环境搭建 + 数据处理 | 8h |
| Day 2 | Phase 1 | Probing 实验 | 8h |
| Day 3-4 | Phase 2 | 核心模块实现 | 16h |
| Day 5-7 | Phase 3 | 训练流程 | 24h |
| Day 8 | Phase 4 | 评估优化 | 8h |
| Day 9 | - | 文档整理 | 4h |
| **总计** | | | **68h (9天)** |

---

## 风险与缓解策略

| 风险 | 缓解措施 |
|------|---------|
| 显存不足 | 使用 DeepSpeed ZeRO-2 + gradient checkpointing |
| Probing 结果不明显 | 降采样层数，或直接使用论文配置 |
| 训练不收敛 | 降低学习率，增加 warmup |
| LLM Judge 成本过高 | 使用更便宜的模型 (gpt-4o-mini) |
| 对话质量不佳 | 调整注入层配置，增加训练轮次 |

---

## 检查点验证

每个 Phase 完成后执行：

```bash
# 运行全部测试
python -m pytest tests/ -v

# 检查代码规范
ruff check src/

# 验证模型可加载
python -c "from src.models import PersonaSteerModel; print('[OK] Model importable')"
```

---

## 后续优化方向

完成基础实现后，可考虑：

1. **共享记忆模块**: 解决冷启动问题
2. **更大模型**: Qwen3-14B 或 Qwen3-72B
3. **其他数据集**: CharacterChat, RolePlay
4. **多任务学习**: 同时训练多种人格属性
5. **部署优化**: 量化、剪枝、蒸馏

---

**参考设计文档**: `docs/plans/2026-03-02-personasteer-design.md`
**更新日期**: 2026-03-03
