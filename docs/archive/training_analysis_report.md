# PersonaSteer 训练与分析报告

> 生成日期: 2026-04-11

## 1. 项目概述

### 1.1 目标
PersonaSteer 是一个基于 HyperNetwork 的人格引导模型，旨在通过注入层(injection layers)使大语言模型生成符合特定人格特质的回复。

### 1.2 技术架构
- **基础模型**: Qwen3-4B
- **核心组件**: HyperNetwork + Injection Layers
- **训练阶段**: Stage 1 → Stage 2 → Stage 3 (渐进式训练)

---

## 2. 训练配置

### 2.1 三种配置对比

| 配置名称 | 注入层数 | 特点 |
|----------|----------|------|
| **Neuroticism** | 3层 | 最少注入，专注神经质特质 |
| **Minimal** | 6层 | 中等注入，平衡配置 |
| **Baseline** | 8层 | 最多注入，全覆盖配置 |

### 2.2 训练阶段说明

| 阶段 | 描述 |
|------|------|
| Stage 1 | 初始训练，基础人格注入 |
| Stage 2 | 强化训练，人格特质增强 |
| Stage 3 | 精调阶段，人格表达优化 |

---

## 3. 评估结果

### 3.1 LLM Judge 评分 (人格一致性)

使用 GPT-5.2 作为 LLM Judge，对生成回复进行 1-5 分的人格一致性评分。

#### 原始评分热力图

| 配置 | Stage 1 | Stage 2 | Stage 3 | 平均分 |
|------|---------|---------|---------|--------|
| **Minimal** | 3.0 | 3.1 | **3.3** | **3.13** |
| Baseline | 3.2 | 2.8 | 2.89 | 2.96 |
| Neuroticism | 2.8 | 3.0 | 2.7 | 2.83 |

#### 训练 Loss 对比

| 配置 | Stage 1 | Stage 2 | Stage 3 | 平均Loss |
|------|---------|---------|---------|----------|
| Neuroticism | 3.12 | 3.05 | **3.07** | **3.08** |
| Minimal | 3.18 | 3.10 | 3.15 | 3.14 |
| Baseline | 3.25 | 3.20 | 3.22 | 3.22 |

### 3.2 Loss-Score 相关性分析

- **相关系数**: 0.345
- **结论**: 低训练 Loss 不一定带来高人格一致性评分
- **可能原因**: Neuroticism 配置存在过拟合现象

---

## 4. 关键发现

### 4.1 最佳配置

| 指标 | 最佳配置 | 数值 |
|------|----------|------|
| LLM Judge 评分 | **Minimal (6层)** | 3.13 |
| 训练 Loss | Neuroticism (3层) | 3.08 |
| 稳定性 | Baseline (8层) | std=0.80 |

### 4.2 Stage 进展趋势

- **Minimal**: Stage 1→3 评分持续上升 (3.0→3.1→3.3) ✓
- **Baseline**: Stage 1→2 下降，Stage 3 略有回升
- **Neuroticism**: 波动较大，无明确趋势

### 4.3 核心问题：思考过程泄露

模型生成的回复暴露了内部推理过程，例如：

```
❌ 问题回复示例:
"Okay, the user is asking how I'm doing... I need to respond in a friendly way.
First, I should acknowledge their message..."

✅ 期望回复示例:
"That sounds great! I've been working on some interesting projects lately.
How about you?"
```

---

## 5. 后处理优化尝试

### 5.1 过滤方法

采用正则表达式移除思考过程：
- 移除 "Okay, the user..." 开头
- 移除 "首先，我需要..." 中文思考
- 移除 "用户:" 后续生成内容

### 5.2 后处理效果

| 配置 | 原评分 | 新评分 | 变化 |
|------|--------|--------|------|
| minimal_stage2 | 3.10 | **3.40** | **+0.30** |
| minimal_stage1 | 3.00 | 3.10 | +0.10 |
| baseline_stage2 | 2.80 | 2.90 | +0.10 |
| minimal_stage3 | 3.30 | 2.90 | **-0.40** |
| baseline_stage3 | 2.89 | 2.56 | -0.33 |

### 5.3 后处理局限性

- ✅ 成功移除部分思考过程开头
- ✅ 成功移除后续对话生成
- ❌ 剩余内容仍含 "I should..." 思考句式
- ❌ 治标不治本，需从训练层面解决

---

## 6. 回复样本对比

### 样本1: 婚礼话题

**用户输入**: "I just got married, and we're planning a wedding party next month!"

| 配置 | 评分 | 回复质量 |
|------|------|----------|
| Baseline | 4.0 | 热情祝贺，询问细节 ✓ |
| Neuroticism | 3.0 | 详细但偏离人格 |
| Minimal | 2.0 | 过于简短 |

### 样本2: 啤酒酿造话题

**用户输入**: "I've been experimenting with some new beer recipes..."

| 配置 | 评分 | 回复质量 |
|------|------|----------|
| Minimal | 4.0 | 友好回应，提供帮助 ✓ |
| Neuroticism | 4.0 | 包含思考过程 |
| Baseline | 2.0 | 思考过程明显 |

---

## 7. 问题根因分析

### 7.1 思考过程污染来源

1. **训练数据问题**: 训练样本可能包含 CoT (Chain-of-Thought) 格式
2. **模型泛化**: Qwen3-4B 预训练中包含推理任务
3. **指令缺失**: 训练时未明确要求隐藏思考过程

### 7.2 Loss-Score 负相关

Neuroticism 配置 Loss 最低但评分不高，原因：
- 过拟合训练数据的格式（包括思考过程）
- 未学到人格表达的本质

---

## 8. 改进建议

### 8.1 短期方案（推理时）

| 方法 | 难度 | 效果 |
|------|------|------|
| 后处理过滤 | 低 | 中 |
| Stop Tokens | 低 | 中高 |
| Prompt 优化 | 中 | 高 |

### 8.2 长期方案（训练时）

| 方法 | 难度 | 效果 |
|------|------|------|
| 数据清洗 | 中 | 高 |
| SFT 数据重构 | 高 | 最高 |
| DPO 偏好优化 | 高 | 高 |

### 8.3 推荐配置

**当前最佳**: Minimal (6层注入)
- LLM Judge 评分最高 (3.13)
- Stage 3 达到最佳效果 (3.3)
- 平衡了注入层数与泛化能力

---

## 9. 结论

1. **Minimal 配置表现最佳**，建议后续实验以此为基础
2. **思考过程泄露是主要问题**，需从训练数据层面解决
3. **Loss 不是唯一指标**，应结合 LLM Judge 评分综合评估
4. **后处理有一定效果但有限**，根本解决需改进训练流程

---

## 附录

### A. 文件结构

```
PersonaSteer/
├── configs/
│   ├── train_stage1_qwen3_baseline_gpu3.yaml
│   ├── train_stage1_qwen3_neuroticism_gpu2.yaml
│   └── train_stage1_qwen3_probing_minimal_gpu1.yaml
├── scripts/
│   ├── train.py
│   ├── eval_new_experiments.py
│   ├── reprocess_evaluate.py
│   └── visualize_results.py
├── results/
│   ├── eval_20260411_163906/      # 原始评估
│   └── reprocess_20260411_182520/ # 后处理评估
└── docs/
    └── training_analysis_report.md
```

### B. 可视化图表

- `fig1_judge_heatmap.png` - LLM Judge 评分热力图
- `fig2_train_loss.png` - 训练 Loss 对比 (log scale)
- `fig3_loss_vs_score.png` - Loss vs Score 散点图
- `fig4_trends.png` - Stage 进展趋势
- `fig5_response_length.png` - 回复长度分布