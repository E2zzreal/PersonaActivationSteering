# PersonaSteer

基于动态激活引导（Dynamic Activation Steering）的多轮个性化对话系统。通过超网络将 personality 描述实时转化为干预向量，注入 Qwen3-4B 的指定层，使模型输出与特定人格保持一致。

---

## 项目状态（2026-04-17）

### 核心发现

三源并行评估（严格 rubric，GPT-5.4 + Gemini 集成）确认：

| 来源 | 人格对齐分 | 说明 |
|------|---:|------|
| Claude-Opus-4.6 | **3.833** | 教师模型上限 |
| Qwen3-4B baseline | 2.833 | 无注入基准 |
| ALOE gold（当前训练数据） | 2.600 | **低于 baseline** |

**根本结论**：训练数据上限（2.6）低于模型自然输出水平（2.83），是注入增益有限的根因，而非注入机制本身的问题。

### P0 修复状态

| 问题 | 状态 |
|------|------|
| gate_init_bias/gate_max 硬编码，YAML 配置无效 | ✅ 已修复 |
| batch_size=1 导致 SCL 永远返回 0 | ✅ 已修复（PersonalityGroupedSampler） |
| ALOE 训练数据上限不足 | 进行中（Claude 数据重建，~661 条并行对话生成中） |

### 当前优先级

**D（数据）> E（评估）> L（损失）> I（注入）**

最高优先级是用 Claude 生成的高质量多轮对话替换 ALOE 训练数据，将数据上限从 2.6 提升至 3.5+。

---

## 架构概述

```
personality 描述
      ↓
  HyperNetwork          ← 编码器（共享 Qwen3-4B 权重，冻结）
  (Qwen3-4B encoder     + 3层 ResidualMLP
   + ResidualMLP)       → 生成干预向量 v_t（多层，每注入层独立）
      ↓
  DynamicGate           ← 根据 v_t 计算各层门控系数 g_i ∈ [0, gate_max]
      ↓
  SteeringInjection     ← h'_i = h_i + g_i * proj_i(v_t)
      ↓
  Qwen3-4B backbone     ← 冻结，通过 forward hook 注入
      ↓
    输出
```

**可训练参数**：HyperNetwork + DynamicGate + layer projectors（约 47M，~1.2%）

**注入层**：默认 `[4..11]`（前层，由 B1 归因实验确认优于 `[8..15]`）

---

## 项目结构

```
PersonaSteer/
├── src/                          # 核心源代码
│   ├── models/
│   │   ├── persona_steer.py      # PersonaSteerModel + PersonaSteerConfig
│   │   ├── injection.py          # DynamicGate + SteeringInjection
│   │   ├── hyper_network.py      # HyperNetwork
│   │   └── components.py         # 基础组件
│   ├── data/
│   │   ├── aloe_dataset.py       # ALOEDataset
│   │   ├── collator.py           # PersonaSteerCollator
│   │   └── grouped_sampler.py    # PersonalityGroupedSampler（SCL 正例保证）
│   ├── training/
│   │   ├── trainer.py            # PersonaSteerTrainer
│   │   └── losses.py             # SFT Loss + SupervisedContrastiveLoss
│   ├── evaluation/
│   │   ├── llm_judge.py          # LLM Judge 模块
│   │   ├── auto_metrics.py       # 自动指标
│   │   └── thinking_leak.py      # 思考泄露检测
│   └── probing/                  # 层位置归因实验
│
├── scripts/                      # 当前在用脚本
│   ├── train.py                  # 三阶段训练入口
│   ├── v4_eval_qwen3.py          # 模型评估（最新版）
│   ├── build_claude_sft_data.py  # 构建 Claude SFT/DPO 训练集
│   ├── detect_thinking_leak.py   # 思考泄露检测
│   ├── auto_train_pipeline.py    # 自动化训练流程
│   └── audit/                    # 评估与审计脚本
│       ├── generate_parallel_dialogues.py  # 三源并行对话生成
│       ├── score_three_sources.py          # 三源 LLM Judge 评分
│       ├── run_teacher_persona_audit.py    # 教师人格审计
│       └── judge_teacher_persona_audit.py
│
├── configs/                      # 主线配置
│   ├── api_config.yaml           # API 配置（BLSC 代理）
│   ├── train_stage1_qwen3.yaml   # Stage1：仅训练 HyperNetwork
│   ├── train_stage2_qwen3.yaml   # Stage2：联合训练 + Gate（SCL 已修复）
│   ├── train_stage3_qwen3.yaml   # Stage3：全模块 + 对比学习
│   └── archive/                  # 历史实验配置（exp_gate 系列等）
│
├── data/
│   ├── aloe_raw/                 # ALOE 原始数据集（git submodule）
│   ├── split/train.jsonl         # 训练集（2,777 条，57 个 personality）
│   └── split/val.jsonl           # 验证集（661 条）
│
├── results/
│   ├── parallel_dialogues/       # 三源并行对话数据
│   └── three_source_score_v3/    # 最新三源评分结果
│
├── docs/
│   ├── 历史实验结论速查.md        # B1/M15/三源/归因优先级结论速查
│   ├── 三源人格对齐能力综合分析报告_2026-04-17.md
│   ├── P0修复与D层重建计划_2026-04-17.md
│   ├── 阶段结论补充说明_2026-04-16.md
│   └── analysis/known_issues.md
│
└── evidence/                     # 历史决策留档（按日期）
```

---

## 快速开始

### 环境要求

- Python >= 3.9，CUDA >= 11.8
- GPU 显存：训练约 16-20GB（Qwen3-4B fp16 + HyperNetwork）
- conda 环境：`persona`

### 安装

```bash
pip install -r requirements.txt
```

### 数据准备

```bash
# ALOE 数据集（git submodule）
git submodule update --init data/aloe_raw

# 预处理（已完成，结果在 data/split/）
python data/convert_aloe.py --input data/aloe_raw --output data/split
```

### 三阶段训练

```bash
# Stage 1：训练 HyperNetwork（gate 冻结）
python scripts/train.py --config configs/train_stage1_qwen3.yaml

# Stage 2：联合训练（gate 解冻，SCL 启用，grouped sampler 自动激活）
python scripts/train.py \
  --config configs/train_stage2_qwen3.yaml \
  --resume checkpoints/stage1_qwen3/best.pt

# Stage 3：全模块 + 对比学习
python scripts/train.py \
  --config configs/train_stage3_qwen3.yaml \
  --resume checkpoints/stage2_qwen3/best.pt
```

**注意**：Stage2/3 配置已设置 `batch_size: 4` + `scl_weight: 0.1`，`PersonalityGroupedSampler` 会自动启用，保证每个 batch 内有同 personality 的正例对。

### 模型评估

```bash
# 生成对话并用 LLM Judge 评分
python scripts/v4_eval_qwen3.py \
  --checkpoint checkpoints/stage1_qwen3/best.pt \
  --data data/split/val.jsonl \
  --output results/eval_stage1/

# 三源并行评估（对比 Claude / Qwen3 baseline / 训练模型）
python scripts/audit/score_three_sources.py \
  --parallel_input results/parallel_dialogues/dialogues.json \
  --output_dir results/three_source_score_v3
```

### 构建 Claude SFT 数据（D 层重建）

```bash
# 第一步：生成三源并行对话（全量）
python scripts/audit/generate_parallel_dialogues.py \
  --aloe_input data/split/val.jsonl \
  --output results/parallel_dialogues/dialogues_full.json \
  --n 661 --max_turns 5 --device cuda:2

# 第二步：构建 SFT 训练集
python scripts/build_claude_sft_data.py \
  --input results/parallel_dialogues/dialogues_full.json \
  --output data/claude_sft/train.jsonl \
  --source claude \
  --build_dpo \
  --dpo_rejected qwen3 \
  --dpo_output data/claude_dpo/train.jsonl

# 第三步：用 Claude 数据重训 Stage1
python scripts/train.py \
  --config configs/train_stage1_qwen3.yaml \
  --data data/claude_sft/train.jsonl \
  --output checkpoints/stage1_claude_sft
```

---

## 训练配置说明

### 关键参数

```yaml
model:
  inject_layers: [4, 5, 6, 7, 8, 9, 10, 11]  # B1 归因：前层优于中层
  v_dim: 1024
  gate_hidden_dim: 256

data:
  batch_size: 4       # Stage2/3 需要 ≥2 以使 SCL 生效

training:
  scl_weight: 0.1     # >0 时自动启用 PersonalityGroupedSampler
  gate_init_bias: -2.0  # sigmoid(-2) ≈ 0.12，控制 gate 初始强度
  gate_max: 1.0
```

### gate_init_bias 效果

| gate_init_bias | sigmoid 值 | 初始注入强度 |
|---|---|---|
| -3.0 | 0.047 | 极弱 |
| -2.0 | 0.119 | 弱（默认） |
| -1.0 | 0.269 | 中等 |
| 0.0 | 0.500 | 强 |

---

## 评估体系

当前使用 **v3 严格 rubric**（2026-04-17 确立）：

- **评分维度**：persona 专属性，而非回复质量
- **关键规则**：好但通用 = 3 分；5 分仅给 persona 不可替换的回复
- **Judge 模型**：GPT-5.4（主）+ Gemini-3.1-Pro-Preview（辅）集成均值
- **评估粒度**：多轮完整对话（非单轮）

历史评估版本对比：

| 版本 | Rubric | Judge | 已知偏差 |
|------|--------|-------|---------|
| v2 | 弱，无行为锚点 | GPT-5.2 | 系统性虚高 ~0.6 分 |
| v3（当前） | 严格，含 CoT | GPT-5.4 + Gemini | 无已知系统偏差 |

---

## 数据格式

训练数据格式（`data/split/train.jsonl`）：

```json
{
  "user_id": "u1234",
  "profile": "He is a 30-year-old environmental scientist...",
  "personality": "He has an adventurous spirit...",
  "conversations": [
    {"role": "user", "content": "Hey there!"},
    {"role": "assistant", "content": "..."}
  ]
}
```

ALOE 数据集统计：57 个唯一 personality，每个出现 48-61 次，总计 2,777 条训练样本。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| `docs/历史实验结论速查.md` | 所有历史实验关键数字一页速查 |
| `docs/三源人格对齐能力综合分析报告_2026-04-17.md` | 三源评估权威报告 |
| `docs/P0修复与D层重建计划_2026-04-17.md` | 当前推进计划 |
| `docs/阶段结论补充说明_2026-04-16.md` | D>E>L>I 归因分析 |
| `docs/evaluation/known_issues.md` | 已知问题记录 |
| `docs/评估口径统一规范.md` | 评估标准说明 |

---

## 许可证

MIT License

## 致谢

- ALOE 数据集：[ShujinWu-0814/ALOE](https://github.com/ShujinWu-0814/ALOE)
- 骨干模型：[Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
- 开发辅助：Claude Opus 4.6 / Sonnet 4.6
