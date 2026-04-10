# PersonaSteer V2

基于论文 "PersonaSteer: Eliciting Multi-turn Personalized Conversation in LLMs via Dynamic Activation Steering" 的完整实现。

## 项目概述

PersonaSteer V2 是一个基于动态激活引导（Dynamic Activation Steering）的多轮个性化对话系统。通过在大语言模型的特定层注入干预向量，实现对模型输出的精准控制，使其生成符合特定人格画像的对话内容。

### 核心特性

- 🎯 **动态激活引导** - 通过向量注入技术精准控制模型行为
- 🧠 **超网络架构** - 从用户话语动态生成干预向量
- 🔄 **递归状态记忆** - 支持多轮对话的状态保持
- 🎚️ **动态门控机制** - 自适应调整注入强度
- 📊 **完整评估体系** - 自动指标 + LLM Judge 双重评估
- 🧪 **105个单元测试** - 保证代码质量和可靠性

## 当前状态 ⚠️

**已知问题**:
1. 🔴 **Gate Init Bias 硬编码** - 配置文件中的 `gate_init_bias` 参数无效（详见 [docs/analysis/known_issues.md](docs/analysis/known_issues.md)）
2. 🟡 **部分模型生成空回复** - 5/15 模型生成失败
3. 🟡 **Activation Steering 效果不明显** - 需要增大gate值

**快速开始**:
- [训练实验记录](docs/experiments/training_log.md)
- [快速参考指南](docs/guides/quick_reference.md)
- [优化方案分析](docs/optimization_analysis.md)

## 项目结构

```
PersonaSteer_V2/
├── docs/                     # 📚 文档（新增）
│   ├── experiments/          # 实验记录
│   ├── analysis/             # 问题分析
│   └── guides/               # 使用指南
├── src/                      # 源代码
│   ├── probing/             # Probing 实验模块
│   │   ├── attribute_extractor.py    # 属性提取器
│   │   ├── head_probing.py           # 激活值收集器
│   │   └── visualize.py              # 可视化与层选择
│   ├── models/              # 核心模型
│   │   ├── components.py             # 基础组件
│   │   ├── hyper_network.py          # 超网络
│   │   ├── injection.py              # 动态门控注入
│   │   └── persona_steer.py          # 完整模型
│   ├── data/                # 数据处理
│   │   ├── aloe_dataset.py           # ALOE 数据集
│   │   └── collator.py               # 批处理器
│   ├── training/            # 训练模块
│   │   ├── losses.py                 # 损失函数
│   │   └── trainer.py                # 训练器
│   └── evaluation/          # 评估模块
│       ├── auto_metrics.py           # 自动指标评估
│       └── llm_judge.py              # LLM Judge 评估
├── scripts/                 # 工具脚本
│   ├── preprocess_aloe.py            # 数据预处理
│   ├── verify_models.py              # 模型验证
│   ├── run_probing.py                # Probing 实验
│   ├── train.py                      # 训练脚本
│   ├── evaluate.py                   # 评估脚本
│   └── inference.py                  # 推理演示
├── configs/                 # 配置文件
│   ├── model.yaml                   # 模型配置
│   ├── probing.yaml                 # Probing 实验配置
│   ├── eval.yaml                    # 评估配置
│   ├── train_stage1.yaml            # 阶段1训练配置
│   ├── train_stage2.yaml            # 阶段2训练配置
│   └── train_stage3.yaml            # 阶段3训练配置
├── tests/                   # 单元测试 (105个测试)
├── docs/                    # 文档
│   └── plans/               # 设计文档和实施计划
├── requirements.txt         # 依赖包
├── pyproject.toml          # 项目配置
└── README.md               # 本文件
```

## 快速开始

### 环境要求

- Python >= 3.9
- CUDA >= 11.8 (推荐)
- 至少 16GB GPU 显存（用于训练）

### 安装依赖

```bash
# 使用 pip
pip install -r requirements.txt

# 或使用 pip 安装项目（推荐）
pip install -e .
```

### 主要依赖

- `torch>=2.1.0` - 深度学习框架
- `transformers>=4.36.0` - Hugging Face 模型库
- `deepspeed>=0.12.0` - 分布式训练
- `accelerate>=0.25.0` - 训练加速
- `datasets>=2.16.0` - 数据集管理
- `openai>=1.0.0` - LLM Judge 评估

## 使用指南

### 1. 数据准备

```bash
# 下载 ALOE 数据集
git clone https://github.com/ShujinWu-0814/ALOE.git data/aloe_raw

# 预处理数据
python scripts/preprocess_aloe.py \
    --input data/aloe_raw \
    --output data/processed \
    --min_turns 3 \
    --max_turns 10
```

**数据集说明**：
- 每个样本包含10轮对话（20条消息）
- 训练默认使用前6轮（可在配置文件中修改 `data.max_turns`）
- 评估默认生成前4轮（可在生成脚本中修改）

### 2. Probing 实验（确定注入层）

```bash
python scripts/run_probing.py \
    --model Qwen/Qwen3-4B \
    --data data/processed/train.jsonl \
    --num_samples 1000 \
    --output experiments/probing/results.json
```

**输出**:
- `experiments/probing/heatmaps/` - 热力图
- `experiments/probing/selected_layers.yaml` - 推荐注入层配置

### 3. 三阶段训练

#### Stage 1: 仅训练 HyperNetwork

```bash
python scripts/train.py \
    --config configs/train_stage1.yaml
```

#### Stage 2: 联合训练（解冻 gate）

```bash
python scripts/train.py \
    --config configs/train_stage2.yaml \
    --resume checkpoints/stage1/best.pt
```

#### Stage 3: 加入对比学习

```bash
python scripts/train.py \
    --config configs/train_stage3.yaml \
    --resume checkpoints/stage2/best.pt
```

### 4. 模型评估

#### 方法A：优化评估流程（推荐）

先生成所有模型的对话，再批量评估（节省50%时间）：

```bash
# 1. 并行生成对话（4 GPU）
bash scripts/run_optimized_evaluation.sh
```

生成的对话保存在 `results/conversations_*/` 目录。

#### 方法B：传统评估

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/stage3/best.pt \
    --data data/processed/eval.jsonl \
    --output experiments/eval_results.json \
    --judge_model gpt-4o-mini \
    --num_samples 500
```

**评估指标**:
- **自动指标**: Loss, PPL, v_variance, gate_distribution
- **LLM Judge**:
  - V1: 严格评分（1-5分）
  - V2: A/B对比（模型间对比）
  - V3: 多维度评分（风格/内容/一致性）

**评估配置**：
- 默认评估50个样本，每个样本4轮对话
- 修改评估轮次：编辑 `scripts/generate_all_conversations.py` 第66-67行
- 修改生成长度：`--max_new_tokens` 参数（默认150）

### 5. 推理演示

```bash
python scripts/inference.py \
    --checkpoint checkpoints/stage3/best.pt
```

交互式对话示例：
```
PersonaSteer 对话演示
输入用户画像: 34岁自由设计师，独立有条理，喜欢简约风格

你: 你好，今天天气真好
AI: 是啊，这样的好天气很适合工作。我今天的设计项目进展顺利。

你: 周末有什么计划吗？
AI: 我打算在家整理工作室，顺便看看新的设计作品集。简单而充实。
```

## 模型架构

### 核心组件

1. **HyperNetwork（超网络）**
   - 输入: 用户话语文本 + 历史干预向量
   - 输出: 当前轮次的干预向量 v_t
   - 结构: Qwen3-Embedding (冻结) + 3层 ResidualMLP

2. **DynamicGate（动态门控）**
   - 输入: 干预向量 v_t
   - 输出: 各注入层的门控系数 (0-1)
   - 作用: 自适应调整注入强度

3. **SteeringInjection（向量注入）**
   - 在指定层将干预向量注入到隐藏状态
   - 公式: `h' = h + gate * proj(v_t)`
   - 注入层: [14-21] (8层，由 Probing 实验确定)

4. **PersonaSteerModel（完整模型）**
   - 骨干模型: Qwen3-4B (冻结)
   - 可训练参数: ~47M (~1.19%)
   - Hook 机制: 自动注入到指定层

### 训练策略

**三阶段渐进训练**:

| 阶段 | Gate 状态 | 对比损失 | 学习率 | Epoch |
|------|----------|---------|--------|-------|
| Stage 1 | 冻结 | 禁用 | 1e-4 | 3 |
| Stage 2 | 解冻 | 禁用 | 5e-5 | 2 |
| Stage 3 | 解冻 | 启用 | 3e-5 | 2 |

**损失函数**:
- SFT Loss: 标准语言模型交叉熵损失
- Supervised Contrastive Loss: 同用户为正例，不同用户为负例

## 测试

运行所有测试:

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定模块测试
python -m pytest tests/test_persona_steer.py -v

# 运行代码规范检查
ruff check src/ tests/ scripts/
```

**测试覆盖**:
- Phase 1: 17个测试 ✅
- Phase 2: 51个测试 ✅
- Phase 3: 24个测试 ✅
- Phase 4: 13个测试 ✅
- **总计: 105个测试全部通过**

## 性能指标

### 模型参数

- 骨干模型参数: 约 3.95B (冻结)
- 可训练参数: 约 47M (~1.19%)
- 注入层数: 8层 [14-21]
- 干预向量维度: 1024

### 训练配置

- 批大小: 4 (可根据显存调整)
- 梯度累积: 支持
- 混合精度: AMP (自动混合精度)
- 梯度裁剪: max_norm=1.0
- 优化器: AdamW

### 预期性能

根据论文，经过完整训练后应达到：
- AL(K)_AVG: > 4.0
- N-IR: > 0.25
- PPL: < 10.0

## 配置说明

### 训练配置文件

配置文件位于 `configs/` 目录：

```yaml
# 模型配置
model:
  inject_layers: [14, 15, 16, 17, 18, 19, 20, 21]  # 注入层 (8层)
  v_dim: 1024                          # 向量维度
  hidden_dim: 4096                     # MLP隐藏层维度

# 数据配置
data:
  train_path: data/processed/train.jsonl
  eval_path: data/processed/eval.jsonl
  max_turns: 6                         # 最大对话轮次
  batch_size: 4

# 训练配置
training:
  stage: 1                             # 训练阶段
  num_epochs: 3
  learning_rate: 1e-4
  sft_weight: 1.0
  scl_weight: 0.0                      # Stage 3 设为 0.1
```

## 配置说明

### 训练配置

所有训练配置文件位于 `configs/` 目录：

**关键参数**：

```yaml
# 数据配置
data:
  max_turns: 6              # 训练使用的对话轮次（数据集有10轮）
  batch_size: 2             # 批次大小

# 模型配置
model:
  inject_layers: [14,15,16,17,18]  # 注入层（通过Probing实验确定）
  v_dim: 1024               # 干预向量维度
  hidden_dim: 4096          # 超网络隐藏层维度
  gate_hidden_dim: 256      # 门控网络维度

# 训练配置
training:
  epochs: 5                 # 训练轮数
  learning_rate: 1e-4       # 学习率
  warmup_steps: 100         # 预热步数
```

**修改训练轮次**：
- 编辑 `configs/train_stage*.yaml` 中的 `data.max_turns`
- 可选值：6（默认）、8、10（完整数据集）
- 注意：更长序列需要更多显存

### 评估配置

**修改评估轮次**：
```python
# 编辑 scripts/generate_all_conversations.py
for turn in sample.get("conversations", [])[:8]:  # 改为 [:12] 或 [:20]
    if len([...]) >= 4:  # 改为 >= 6 或 >= 10
```

**修改生成长度**：
```bash
python scripts/generate_all_conversations.py --max_new_tokens 200  # 默认150
```

### GPU配置

**并行生成对话**：
```bash
# 编辑 scripts/run_optimized_evaluation.sh
CUDA_VISIBLE_DEVICES=0 python ...  # 指定GPU
```

**训练时GPU选择**：
```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py ...  # 多GPU训练
```

## 常见问题

### Q: 显存不足怎么办？

A: 尝试以下方法：
1. 减小 `batch_size`（默认2，可改为1）
2. 减少 `max_turns`（默认6，可改为4）
3. 使用梯度累积：`gradient_accumulation_steps: 4`
4. 启用混合精度训练（已默认启用 fp16）
5. 使用 DeepSpeed ZeRO-2
6. 使用 gradient checkpointing

### Q: 为什么训练只用6轮而不是完整的10轮？

A: 主要原因：
- **显存限制**：10轮序列长度约2000+ tokens，batch_size=2时容易OOM
- **训练效率**：前6轮已包含足够的persona信息
- **实验验证**：6轮在效果和效率间取得平衡

如需使用完整10轮，修改 `configs/train_stage*.yaml` 中的 `data.max_turns: 10`

### Q: 评估为什么只生成4轮对话？

A: 评估4轮是为了测试模型在短对话场景下的persona保持能力。可根据需求修改：
- 编辑 `scripts/generate_all_conversations.py`
- 修改第66-67行的轮次限制

### Q: 如何加速评估流程？

A: 使用优化评估脚本：
```bash
bash scripts/run_optimized_evaluation.sh
```
优势：
- 先生成对话，再批量评估（避免重复生成）
- 4 GPU并行生成（加速4倍）
- 节省约50%总时间

### Q: 三个Judge评估方法有什么区别？

A:
- **V1 (严格评分)**：单一1-5分评分，关注整体质量
- **V2 (A/B对比)**：两个模型对比，判断哪个更好
- **V3 (多维度)**：分别评估风格、内容、一致性三个维度

### Q: 如何选择注入层？

A: 运行Probing实验自动选择：
```bash
python scripts/run_probing.py --model Qwen/Qwen3-4B --data data/processed/train.jsonl
```
输出推荐层配置到 `experiments/probing/selected_layers.yaml`

### Q: Stage1/2/3有什么区别？

A:
- **Stage1**：仅训练HyperNetwork，backbone冻结
- **Stage2**：联合训练，解冻gate网络
- **Stage3**：加入对比学习损失，增强persona区分度

必须按顺序训练，每个stage加载上一stage的checkpoint。

### Q: Probing 实验结果不明显？

A: 可以：
- 增加采样数量 (--num_samples)
- 使用论文中的默认层配置 [14-21]
- 检查数据质量

### Q: 训练不收敛？

A: 建议：
- 降低学习率
- 增加 warmup steps
- 检查数据预处理
- 验证模型加载是否正确

### Q: 如何使用自己的数据集？

A: 数据格式需要符合：
```json
{
  "user_id": "u001",
  "profile": "用户画像描述",
  "personality": "人格特征描述",
  "conversations": [
    {"role": "user", "content": "用户消息"},
    {"role": "assistant", "content": "助手回复"}
  ]
}
```

## 贡献指南

本项目使用 Agent Team 协作开发，由以下团队成员完成：

- **env-engineer** - 环境准备
- **probing-researcher** - Probing 实验
- **model-architect** - 模型架构
- **training-engineer** - 训练流程
- **evaluation-engineer** - 评估优化

### 代码规范

- 遵循 PEP 8 规范
- 使用 ruff 进行代码检查
- 每个模块必须有文件头注释
- 公开函数必须有 docstring
- 使用 type hints

## 引用

如果本项目对您的研究有帮助，请引用原始论文：

```bibtex
@article{personasteer2024,
  title={PersonaSteer: Eliciting Multi-turn Personalized Conversation in LLMs via Dynamic Activation Steering},
  author={...},
  journal={...},
  year={2024}
}
```

## 许可证

MIT License

## 致谢

- 感谢原论文作者提供的理论基础
- 感谢 Hugging Face 提供的模型和工具
- 感谢 ALOE 数据集的贡献者
- 本项目由 Claude Opus 4.6 辅助开发

## 联系方式

- Issues: [GitHub Issues](https://github.com/personasteer/v2/issues)
- Email: team@personasteer.ai

---

**PersonaSteer V2** - 让大语言模型拥有真实的个性！🎭
