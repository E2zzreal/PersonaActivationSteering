# PersonaSteer 复现架构设计

**日期**: 2026-03-02
**状态**: 设计完成，待实现
**目标**: 复现论文 "PersonaSteer: Eliciting Multi-turn Personalized Conversation in LLMs via Dynamic Activation Steering"

---

## 1. 项目概述

### 1.1 论文核心贡献

PersonaSteer 提出了一个动态侧网络架构，通过以下三个核心组件实现多轮个性化对话：

1. **轻量级侧网络 (Side-Network)**: 使用冻结编码器提取用户话语特征，通过可学习的 MLP 投影为动态引导向量
2. **递归状态内化机制 (RSI)**: 将对话引导建模为有状态的序列任务，解决"人格蒸发问题"
3. **共享记忆模块 (Shared Memory)**: 利用跨用户行为模式，解决冷启动问题

### 1.2 复现目标

- **优先级**: 核心验证（先实现 RSI 递归状态机制，验证其有效性）
- **骨干模型**: Qwen3-4B (冻结)
- **编码器**: Qwen3-Embedding (冻结)
- **数据集**: ALOE

### 1.3 硬件配置

- **本地**: 4×RTX 5090 (128GB 总显存)
- **云端**: 8×A100 (备用)

---

## 2. 整体架构

### 2.1 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PersonaSteer 整体架构                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    超网络 (HyperNetwork)                     │    │
│  │  ┌────────────────┐      ┌─────────────────────────────┐   │    │
│  │  │ Qwen3-Embedding│      │      Projector Layers       │   │    │
│  │  │    (冻结)       │      │        (可训练)              │   │    │
│  │  │    dim=1024    │      │  ┌─────────────────────┐    │   │    │
│  │  └───────┬────────┘      │  │ Linear: 1024→1024  │    │   │    │
│  │          │               │  │ Residual MLP ×3    │    │   │    │
│  │          ▼               │  │ LayerNorm          │    │   │    │
│  │       z_t (1024)         │  └──────────┬──────────┘    │   │    │
│  │          │               └─────────────┼───────────────┘   │    │
│  │          │     ┌───────────────┐       │                   │    │
│  │          │     │Linear Projector│       │                   │    │
│  │          │     │  1024→1024    │       │                   │    │
│  │          │     └───────┬───────┘       │                   │    │
│  │          │             │               │                   │    │
│  │          │   v_{t-1}───┘               │                   │    │
│  │          │             │               │                   │    │
│  │          └──────►(+)◄──┘               │                   │    │
│  │                  │                     │                   │    │
│  │                  └─────────────────────┘                   │    │
│  │                            │                               │    │
│  │                            ▼                               │    │
│  │                    v_t (干预向量)                           │    │
│  └────────────────────────────┼────────────────────────────────┘    │
│                               │                                     │
│                               ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    注入模块 (Injection)                      │    │
│  │         v_t ──▶ [Layer Projector + Dynamic Gate] × N       │    │
│  └────────────────────────────┼────────────────────────────────┘    │
│                               │                                     │
│                               ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                     Qwen3-4B (冻结)                          │    │
│  │                    中间层注入 steering                        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                               │                                     │
│                               ▼                                     │
│                          个性化响应                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心数据流

1. 用户话语 $u_t$ → Qwen3-Embedding 编码 → 语义向量 $z_t$
2. $z_t$ + $v_{t-1}$ (经 Linear Projector) → 相加融合
3. 融合结果 → Residual MLP ×3 → 新干预向量 $v_t$
4. $v_t$ → 层投影 + 动态门控 → 注入 Qwen3-4B 中间层
5. Qwen3-4B 生成个性化响应

### 2.3 可训练参数汇总

| 模块 | 参数量 | 说明 |
|------|--------|------|
| Qwen3-Embedding | ~600M | 冻结 |
| Qwen3-4B | ~4B | 冻结 |
| History Projector | 1M | 1024×1024 |
| Residual MLP ×3 | 25M | 3×(1024×4096×2) |
| Layer Projectors ×8 | 21M | 8×(1024×2560) |
| Dynamic Gate MLP | 265K | 1024×256 + 256×8 |
| **总可训练参数** | **~47M** | 仅占主干 ~1.2% |

---

## 3. 模块详细设计

### 3.1 超网络 (HyperNetwork)

```python
class HyperNetwork(nn.Module):
    """
    输入: 当前对话 u_t, 上一轮干预向量 v_{t-1}
    输出: 当前干预向量 v_t
    """

    # 组件1: 冻结编码器
    encoder: Qwen3Embedding  # 冻结，dim=1024

    # 组件2: 历史状态投影
    history_projector: nn.Linear(1024, 1024)

    # 组件3: 残差MLP堆叠 (3层)
    projector_layers: nn.Sequential(
        ResidualMLP(1024, 4096),
        ResidualMLP(1024, 4096),
        ResidualMLP(1024, 4096),
        nn.LayerNorm(1024)
    )

class ResidualMLP(nn.Module):
    """单个残差MLP块"""
    def __init__(self, dim, hidden_dim, dropout=0.1):
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.linear1(x)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.linear2(h)
        return x + h  # 残差连接
```

### 3.2 动态门控注入模块 (Injection)

```python
class DynamicGate(nn.Module):
    """根据 v_t 动态计算每层的门控权重"""

    def __init__(self, v_dim=1024, num_layers=8, hidden_dim=256):
        self.gate_mlp = nn.Sequential(
            nn.Linear(v_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_layers),
            nn.Sigmoid()
        )

    def forward(self, v_t):
        # v_t: (batch, 1024)
        # 返回: (batch, num_layers)
        return self.gate_mlp(v_t)


class SteeringInjection(nn.Module):
    """将 v_t 注入到 Qwen3-4B 的指定层"""

    def __init__(self, inject_layers: list[int], v_dim=1024, layer_dim=2560):
        self.inject_layers = inject_layers
        self.num_inject = len(inject_layers)

        # 动态门控
        self.gate = DynamicGate(v_dim, self.num_inject)

        # 每层投影
        self.layer_projectors = nn.ModuleList([
            nn.Linear(v_dim, layer_dim)
            for _ in range(self.num_inject)
        ])

    def inject(self, hidden_states, v_t, layer_idx):
        """
        hidden_states: (batch, seq_len, 2560)
        v_t: (batch, 1024)
        layer_idx: 当前层在 inject_layers 中的索引
        """
        gate_values = self.gate(v_t)  # (batch, num_inject)
        gate = gate_values[:, layer_idx:layer_idx+1]  # (batch, 1)

        proj = self.layer_projectors[layer_idx](v_t)  # (batch, 2560)

        # 广播并注入
        return hidden_states + gate.unsqueeze(-1) * proj.unsqueeze(1)
```

---

## 4. Probing 预实验设计

### 4.1 实验目标

在 Layer × Head 粒度分析 Qwen3-4B 对人格属性的敏感性，指导注入层选择。

### 4.2 人格属性

从 ALOE 数据集提取：

```python
attributes = {
    # 来自 profile
    "age": "continuous",
    "gender": "binary",
    "occupation_type": "ordinal",

    # 来自 personality
    "introversion": "continuous",
    "openness": "continuous",
    "enthusiasm": "continuous",
    "methodical": "continuous",
}
```

### 4.3 实验流程

```
Step 1: 数据准备
├─ 从 ALOE 采样 500-1000 个用户
├─ 提取/标注各属性值
└─ 准备输入: 用户首轮对话 u_1

Step 2: 激活值收集
├─ 前向传播 Qwen3-4B
├─ Hook 提取每层每头的 attention output
├─ 形状: (N_samples, 36_layers, 32_heads, head_dim)
└─ 取最后 token 或 mean pooling

Step 3: 相关性分析
├─ 对每个属性:
│   ├─ 对每个 (layer, head):
│   │   ├─ 训练线性探针 或 计算特征均值
│   │   └─ 计算 Spearman ρ
│   └─ 生成 36×32 热力图
└─ 汇总: 跨属性平均热力图

Step 4: 确定注入配置
├─ 分析热力图，找高相关区域
├─ 选择 Top-K 层 (如 8 层)
└─ 写入配置文件
```

### 4.4 实现代码

```python
class AttentionHeadProber:
    """Layer × Head 级别的 Probing 分析"""

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.activations = {}
        self._register_hooks()

    def _register_hooks(self):
        for idx, layer in enumerate(self.model.model.layers):
            layer.self_attn.register_forward_hook(
                self._create_hook(idx)
            )

    def _create_hook(self, layer_idx):
        def hook(module, inputs, outputs):
            hidden = outputs[0]
            batch, seq_len, hidden_dim = hidden.shape
            num_heads = self.model.config.num_attention_heads
            head_dim = hidden_dim // num_heads

            reshaped = hidden.view(batch, seq_len, num_heads, head_dim)
            self.activations[layer_idx] = reshaped[:, -1, :, :].detach().cpu()
        return hook

    @torch.no_grad()
    def collect(self, texts: list[str], batch_size=16):
        """收集所有样本的激活值"""
        all_acts = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch_texts, return_tensors="pt",
                padding=True, truncation=True, max_length=512
            ).to(self.device)

            self.model(**inputs)

            batch_acts = torch.stack([
                self.activations[l] for l in range(len(self.activations))
            ], dim=1)
            all_acts.append(batch_acts)

        return torch.cat(all_acts, dim=0)

    def compute_spearman(self, activations, attr_values):
        """计算 Spearman 相关矩阵"""
        from scipy.stats import spearmanr

        N, L, H, D = activations.shape
        result = np.zeros((L, H))

        for l in range(L):
            for h in range(H):
                features = activations[:, l, h, :].mean(dim=-1).numpy()
                rho, _ = spearmanr(features, attr_values)
                result[l, h] = rho if not np.isnan(rho) else 0

        return result


def select_injection_layers(avg_matrix, top_k=8, strategy="continuous"):
    """根据平均热力图选择注入层"""
    layer_importance = avg_matrix.mean(axis=1)

    if strategy == "top":
        top_layers = np.argsort(layer_importance)[-top_k:][::-1]
        return sorted(top_layers.tolist())

    elif strategy == "continuous":
        peak = np.argmax(layer_importance)
        half_k = top_k // 2
        start = max(0, peak - half_k)
        end = min(len(layer_importance), start + top_k)
        return list(range(start, end))
```

### 4.5 预期输出

```
Layer Probing Results (Qwen3-4B, 36 layers)
───────────────────────────────────────────
Layer 18: 0.52 ████████████████████  ← Top 1
Layer 16: 0.49 ███████████████████   ← Top 2
Layer 20: 0.47 ██████████████████    ← Top 3
...

推荐注入配置:
  策略: continuous
  注入层: [14, 15, 16, 17, 18, 19, 20, 21]
```

---

## 5. 训练流程设计

### 5.1 三阶段渐进训练

```
┌─────────────────────────────────────────────────────────────────────┐
│                          训练流程                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  阶段1: SFT 基础训练                                                 │
│  ├─ 目标: 跑通流程，验证架构正确性                                    │
│  ├─ 损失: L_sft = -log P(y|x, v_t)                                  │
│  ├─ 门控: 固定为均匀值 (α_l = 0.5)                                   │
│  ├─ 训练: HyperNetwork + Layer Projectors                           │
│  └─ 预计: 2-3 epochs                                                │
│                                                                      │
│                          ▼                                          │
│                                                                      │
│  阶段2: 门控学习                                                     │
│  ├─ 目标: 学习哪些层更重要                                           │
│  ├─ 损失: L_sft                                                     │
│  ├─ 门控: 解冻 Dynamic Gate MLP                                     │
│  ├─ 训练: 全部可训练参数                                             │
│  └─ 预计: 1-2 epochs                                                │
│                                                                      │
│                          ▼                                          │
│                                                                      │
│  阶段3: 对比学习增强                                                 │
│  ├─ 目标: 增强人格区分度                                             │
│  ├─ 损失: L = L_sft + λ·L_scl                                       │
│  ├─ L_scl: 监督对比损失，同用户正例，不同用户负例                     │
│  ├─ 训练: 全部可训练参数                                             │
│  └─ 预计: 1-2 epochs                                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 数据组织

```python
# 单条训练样本
{
    "user_id": "user_001",
    "conversation": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
    ],
    "profile": "34岁自由设计师，喜欢徒步...",
    "personality": "独立、有条理、热情..."
}

# 训练时按轮次展开
Turn 1: v_0=zeros → encode(u_1) → v_1 → generate(response_1)
Turn 2: v_1      → encode(u_2) → v_2 → generate(response_2)
Turn 3: v_2      → encode(u_3) → v_3 → generate(response_3)
```

### 5.3 训练配置

| 项目 | 设置 | 说明 |
|------|------|------|
| 优化器 | AdamW | β1=0.9, β2=0.999 |
| 学习率 | 1e-4 → 1e-5 | 余弦退火 |
| Batch Size | 4×4=16 | 4卡，每卡4 |
| 梯度累积 | 4 | 有效batch=64 |
| 梯度裁剪 | 1.0 | 防止梯度爆炸 |
| v_{t-1} 梯度 | detach | 防止跨轮梯度爆炸 |
| 精度 | bf16 | 5090 原生支持 |
| 框架 | DeepSpeed ZeRO-2 | 4卡并行 |

---

## 6. 评估指标设计

### 6.1 指标体系

```
训练时自动指标 (每 epoch):
├─ Loss: L_sft, L_scl, L_total
├─ PPL: 生成困惑度
├─ v_t 稳定性: Var(v_t) 跨轮次方差
└─ Gate 分布: 各层门控值统计

验证时核心指标 (关键节点):
├─ AL(K)_AVG: K 轮对话的平均对齐分数
├─ N-IR: 归一化改进率
└─ N-R²: 行为演化稳定性
```

### 6.2 自动指标实现

```python
class AutoMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = {"sft": [], "scl": [], "total": []}
        self.ppls = []
        self.v_vars = []
        self.gate_stats = []

    def update_loss(self, sft, scl, total):
        self.losses["sft"].append(sft)
        self.losses["scl"].append(scl)
        self.losses["total"].append(total)

    def update_ppl(self, logits, labels):
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        self.ppls.append(torch.exp(ce_loss).item())

    def update_v_stability(self, v_sequence):
        if len(v_sequence) > 1:
            v_stack = torch.stack(v_sequence)
            variance = v_stack.var(dim=0).mean().item()
            self.v_vars.append(variance)

    def summarize(self):
        return {
            "loss_sft": np.mean(self.losses["sft"]),
            "loss_scl": np.mean(self.losses["scl"]),
            "loss_total": np.mean(self.losses["total"]),
            "ppl": np.mean(self.ppls),
            "v_variance": np.mean(self.v_vars) if self.v_vars else 0,
        }
```

### 6.3 LLM-as-Judge 指标

```python
class LLMJudgeMetrics:
    def __init__(self, judge_model="gpt-4o-mini"):
        self.client = OpenAI()
        self.judge_model = judge_model

    def compute_alignment_score(self, response, profile, personality):
        """计算单轮对齐分数 (1-5)"""
        prompt = f"""请评估以下 AI 回复与用户画像的对齐程度。

用户画像: {profile}
用户性格: {personality}
AI 回复: {response}

评分标准 (1-5):
1 - 完全不符合用户特征
2 - 少量符合
3 - 部分符合
4 - 大部分符合
5 - 完全符合，自然流畅

请只输出一个数字 (1-5):"""

        result = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0
        )
        return int(result.choices[0].message.content.strip())

    def compute_al_k_avg(self, conversation, profile, personality):
        """计算 K 轮平均对齐分数"""
        scores = []
        for turn in conversation:
            if turn["role"] == "assistant":
                score = self.compute_alignment_score(
                    turn["content"], profile, personality
                )
                scores.append(score)
        return np.mean(scores) if scores else 0

    def compute_n_ir(self, scores_method, scores_baseline):
        """归一化改进率"""
        method_avg = np.mean(scores_method)
        baseline_avg = np.mean(scores_baseline)
        if baseline_avg == 0:
            return 0
        return (method_avg - baseline_avg) / baseline_avg

    def compute_n_r2(self, scores_sequence):
        """行为演化稳定性"""
        if len(scores_sequence) < 2:
            return 1.0

        x = np.arange(len(scores_sequence))
        y = np.array(scores_sequence)

        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
        n_r2 = r2 * (1 + 0.1 * np.sign(slope))

        return np.clip(n_r2, 0, 1)
```

### 6.4 评估时机

| 时机 | 指标 | 样本量 |
|------|------|--------|
| 每 100 steps | loss, PPL | 全量 |
| 每 epoch | 完整自动指标 | 全量 |
| 阶段1结束 | LLM-Judge | 100 samples |
| 阶段2结束 | LLM-Judge | 300 samples |
| 阶段3结束 | LLM-Judge | 全部 eval set |

---

## 7. 项目结构

```
PersonaSteer/
├── README.md
├── CLAUDE.md
├── requirements.txt
├── pyproject.toml
│
├── configs/
│   ├── model.yaml
│   ├── train_stage1.yaml
│   ├── train_stage2.yaml
│   ├── train_stage3.yaml
│   ├── probing.yaml
│   └── eval.yaml
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hyper_network.py
│   │   ├── injection.py
│   │   ├── persona_steer.py
│   │   └── components.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── aloe_dataset.py
│   │   ├── preprocessor.py
│   │   └── collator.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── scheduler.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── auto_metrics.py
│   │   ├── llm_judge.py
│   │   └── reporter.py
│   ├── probing/
│   │   ├── __init__.py
│   │   ├── head_probing.py
│   │   ├── attribute_extractor.py
│   │   └── visualize.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── logging.py
│       └── hooks.py
│
├── scripts/
│   ├── run_probing.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── data/
│   ├── aloe/
│   │   ├── conversations.jsonl
│   │   ├── personality.jsonl
│   │   ├── personality_eval.jsonl
│   │   ├── profile.jsonl
│   │   └── profile_eval.jsonl
│   └── processed/
│
├── experiments/
│   ├── probing/
│   └── training/
│
├── checkpoints/              # gitignore
├── outputs/                  # gitignore
└── tests/
    ├── test_hyper_network.py
    ├── test_injection.py
    ├── test_dataset.py
    └── test_metrics.py
```

---

## 8. 配置文件

### 8.1 model.yaml

```yaml
model:
  backbone:
    name: "Qwen/Qwen3-4B"
    torch_dtype: "bfloat16"
    freeze: true

  encoder:
    name: "Qwen/Qwen3-Embedding"
    dim: 1024
    freeze: true

  hyper_network:
    v_dim: 1024
    hidden_dim: 4096
    num_residual_layers: 3
    dropout: 0.1

  injection:
    inject_layers: [14, 15, 16, 17, 18, 19, 20, 21]  # Probing 后更新
    layer_dim: 2560
    gate_hidden_dim: 256
```

### 8.2 train_stage1.yaml

```yaml
stage: 1
description: "SFT 基础训练，固定门控"

training:
  epochs: 3
  batch_size: 4
  gradient_accumulation: 4
  learning_rate: 1e-4
  lr_scheduler: "cosine"
  warmup_ratio: 0.05
  weight_decay: 0.01
  max_grad_norm: 1.0

loss:
  sft_weight: 1.0
  scl_weight: 0.0

gate:
  freeze: true
  init_value: 0.5

distributed:
  strategy: "deepspeed_zero2"
  num_gpus: 4
```

### 8.3 train_stage3.yaml

```yaml
stage: 3
description: "加入对比学习"

training:
  epochs: 2
  batch_size: 4
  gradient_accumulation: 4
  learning_rate: 5e-5
  lr_scheduler: "cosine"
  warmup_ratio: 0.02

loss:
  sft_weight: 1.0
  scl_weight: 0.1
  scl_temperature: 0.07
  negative_samples: 4

gate:
  freeze: false

resume_from: "checkpoints/stage2_best/model.pt"
```

---

## 9. 运行命令

```bash
# 1. Probing 实验
python scripts/run_probing.py --config configs/probing.yaml

# 2. 阶段1训练
python scripts/train.py --config configs/train_stage1.yaml

# 3. 阶段2训练
python scripts/train.py --config configs/train_stage2.yaml

# 4. 阶段3训练
python scripts/train.py --config configs/train_stage3.yaml

# 5. 评估
python scripts/evaluate.py --config configs/eval.yaml \
    --checkpoint checkpoints/stage3_best/model.pt

# 6. 推理演示
python scripts/inference.py --checkpoint checkpoints/stage3_best/model.pt
```

---

## 10. 实施计划

| 阶段 | 任务 | 预计时间 |
|------|------|----------|
| 0 | 环境搭建 + 数据准备 | 1 天 |
| 1 | Probing 实验 | 1 天 |
| 2 | 模型代码实现 | 2-3 天 |
| 3 | 阶段1训练 + 调试 | 2 天 |
| 4 | 阶段2训练 | 1 天 |
| 5 | 阶段3训练 | 1 天 |
| 6 | 评估 + 报告 | 1 天 |
| **总计** | | **9-10 天** |

---

## 附录: 参考资料

- 论文: PersonaSteer: Eliciting Multi-turn Personalized Conversation in LLMs via Dynamic Activation Steering
- 数据集: https://github.com/ShujinWu-0814/ALOE
- Qwen3 模型: https://huggingface.co/Qwen
