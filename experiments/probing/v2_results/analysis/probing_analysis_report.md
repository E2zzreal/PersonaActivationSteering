# Probing 实验分析报告

**实验日期**: 2026-04-10
**模型**: Qwen/Qwen3-4B
**样本数**: 100 (快速验证)
**分析方法**: Layer × Head 粒度 Spearman 相关性分析

---

## 1. 实验概述

### 1.1 目标
通过 Probing 分析确定最优的 Activation Steering 注入层配置，解决当前干预效果不佳的问题。

### 1.2 方法
- 收集模型各层注意力头的激活值
- 计算与人格属性的 Spearman 相关性
- 应用多种层选择策略：top, continuous, early_peak, multi_scale
- 生成推荐配置

### 1.3 分析框架
采用**大五人格理论（Big Five / OCEAN模型）**分析人格特质：
- **内向性 (introversion)**: 内向 vs 外向
- **开放性 (openness)**: 保守 vs 开放/好奇
- **尽责性 (conscientiousness)**: 随意 vs 认真负责
- **宜人性 (agreeableness)**: 冷漠 vs 友好合作
- **神经质 (neuroticism)**: 稳定 vs 焦虑敏感

另外包含两个人口统计学属性：age（年龄）、gender（性别）

---

## 2. Probing 实验结果

### 2.1 各属性相关性统计

| 属性 | 最大相关性 | 最小相关性 | 平均|相关性| | 峰值层 | 峰值层相关性 | Top 3 层 |
|------|-----------|-----------|------------|--------|-------------|----------|
| age | NaN | NaN | NaN | 0 | NaN | [15, 16, 17] |
| gender | 0.377 | -0.354 | 0.140 | **3** | 0.187 | [3, 30, 31] |
| introversion | **0.393** | -0.342 | 0.117 | **28** | 0.164 | [16, 28, 31] |
| openness | 0.324 | -0.371 | 0.078 | 15 | 0.107 | [8, 10, 15] |
| conscientiousness | 0.326 | -0.291 | 0.103 | **0** | 0.149 | [0, 5, 22] |
| agreeableness | 0.242 | -0.247 | 0.074 | 11 | 0.108 | [11, 23, 29] |
| neuroticism | **0.395** | -0.378 | 0.091 | 10 | 0.144 | [2, 10, 25] |

### 2.2 峰值层分布

```
Layer 0:  conscientiousness (嵌入层)
Layer 3:  gender (浅层)
Layer 10: neuroticism (前段)
Layer 11: agreeableness (前段)
Layer 15: openness (中段)
Layer 28: introversion (深层)
```

**平均峰值层**: 9.57

### 2.3 层选择策略结果

| 策略 | 选择的层 |
|------|---------|
| top | 按平均相关性排序选Top 8 |
| continuous | 从峰值向两边扩展 |
| **early_peak** (推荐) | **[0, 1, 2, 3, 4, 5, 6, 7]** |
| multi_scale | 前中后各选若干层 |

**推荐策略**: early_peak
**推荐理由**: 平均峰值层为 9.6，说明前段层包含更多人格信息

---

## 3. 数据集人格特质分析

### 3.1 数据集统计

- **总样本数**: 3,438
- **数据来源**: ALOE 数据集 (data/processed/train.jsonl)

### 3.2 关键词频率 Top 20

| 关键词 | 出现次数 | 覆盖率 |
|--------|---------|--------|
| empathetic | 1,173 | 34.1% |
| creative | 915 | 26.6% |
| disciplined | 850 | 24.7% |
| understanding | 777 | 22.6% |
| enthusiastic | 761 | 22.1% |
| compassionate | 619 | 18.0% |
| innovative | 552 | 16.1% |
| supportive | 544 | 15.8% |
| social | 495 | 14.4% |
| adventurous | 423 | 12.3% |
| independent | 416 | 12.1% |
| spontaneous | 405 | 11.8% |
| considerate | 401 | 11.7% |
| caring | 384 | 11.2% |
| practical | 343 | 10.0% |
| curious | 312 | 9.1% |
| organized | 303 | 8.8% |
| imaginative | 298 | 8.7% |
| diligent | 290 | 8.4% |
| outgoing | 270 | 7.9% |

### 3.3 大五人格特质分布

| 特质 | 覆盖率 | 平均得分 | 高频关键词 |
|------|--------|---------|-----------|
| **宜人性** | **87.0%** | 0.917 | empathetic, compassionate, caring, understanding, supportive |
| **尽责性** | 63.8% | 0.785 | disciplined, organized, diligent, methodical |
| **开放性** | 60.8% | 0.836 | creative, innovative, curious, imaginative, adventurous |
| **内向性** | 51.4% | 0.235 | independent ↔ social, outgoing, enthusiastic |
| **神经质** | **24.2%** | 0.515 | (极少出现焦虑、敏感相关描述) |

---

## 4. 数据集 vs Probing 对比分析

### 4.1 对比表格

| 属性 | 数据集覆盖率 | Probing最大相关性 | 峰值层 | 问题标记 |
|------|------------|------------------|--------|---------|
| age | 100.0% | NaN | 0 | 数据提取可能有问题 |
| gender | 100.0% | 0.377 | 3 | 正常 |
| introversion | 51.4% | **0.393** | 28 | 深层编码 |
| openness | 60.8% | 0.324 | 15 | 中层编码 |
| conscientiousness | 63.8% | 0.326 | **0** | 嵌入层编码 |
| agreeableness | **87.0%** | 0.242 | 11 | ⚠️ 覆盖高但相关性低 |
| neuroticism | **24.2%** | **0.395** | 10 | ✓ 覆盖低但相关性高 |

### 4.2 关键发现

#### 4.2.1 宜人性悖论
- **现象**: 数据集中最常描述的特质（87%覆盖），但 Probing 相关性最低（0.24）
- **原因分析**:
  - 描述同质化严重，大量样本都是 "empathetic, understanding, compassionate"
  - 缺乏区分度，模型难以学到有区分性的表征
  - 平均得分 0.917，几乎所有样本都被标记为高宜人性

#### 4.2.2 神经质反常
- **现象**: 数据集最少描述的特质（24%覆盖），但相关性最高（0.40）
- **可能原因**:
  - 少数样本的神经质描述信号强、对比明显
  - 模型对情绪特质敏感
  - 有焦虑/敏感描述的样本与无描述的样本形成鲜明对比

#### 4.2.3 尽责性浅层编码
- **现象**: 峰值在第0层（嵌入层），相关性 0.326
- **解释**: disciplined/organized 等词可能在词嵌入空间就有区分性
- **启示**: 尽责性是词汇层面的特征，不需要深层语义理解

#### 4.2.4 内向性深层编码
- **现象**: 峰值在第28层（深层），相关性 0.393
- **解释**: 内向/外向需要理解社交行为模式，需要深层语义处理
- **启示**: 干预内向性需要作用于深层

#### 4.2.5 性别浅层编码
- **现象**: 峰值在第3层，相关性 0.378
- **解释**: 性别通过代词（he/she）、称谓（Mr./Ms.）等表层特征编码
- **启示**: 符合预期，性别是浅层特征

---

## 5. 对 Activation Steering 的启示

### 5.1 当前配置问题

| 配置项 | 当前值 | Probing推荐 | 问题 |
|--------|--------|------------|------|
| inject_layers | [8, 9, 10, 11, 12, 13, 14, 15] | [0, 1, 2, 3, 4, 5, 6, 7] | 错过浅层关键信息 |
| gate_init_bias | -2.0 (硬编码) | 应可配置 | 干预强度固定 |

### 5.2 干预效果不佳的原因分析

#### 原因1: 干预层位置错误
- 当前配置 [8-15] 跳过了浅层
- 尽责性、性别在浅层（0-3层）编码
- 推荐使用 [0-7] 的 early_peak 策略

#### 原因2: 目标特质选择问题
- 如果训练目标是宜人性，但该特质在模型中表征弱（相关性仅0.24）
- 建议：
  - 优先干预神经质（相关性最高 0.40）
  - 或内向性（相关性 0.39，有明确的深层表征）
  - 避免以宜人性为主要干预目标

#### 原因3: 数据质量问题
- 宜人性描述过于同质化（87%样本都是高宜人性）
- 神经质样本太少（仅24%覆盖）
- 建议：增加数据多样性，特别是：
  - 增加低宜人性样本
  - 增加高/低神经质对比样本

#### 原因4: 干预强度固定
- gate_init_bias = -2.0 硬编码，干预强度固定约12%
- 不同特质可能需要不同干预强度
- 建议：根据特质类型调整干预强度

### 5.3 改进建议

#### 短期改进
1. **更新 inject_layers**: 从 [8-15] 改为 [0-7]
2. **修复 gate_init_bias 硬编码**: 允许配置文件指定
3. **选择更好的干预目标**: 优先神经质或内向性

#### 中期改进
1. **扩充数据集**: 增加人格特质多样性
2. **特质特定层配置**: 不同特质使用不同的注入层
   - 尽责性: [0, 1, 2, 3, 4, 5]
   - 神经质: [8, 9, 10, 11, 12]
   - 内向性: [24, 25, 26, 27, 28, 29, 30, 31]

#### 长期改进
1. **动态层选择**: 根据输入内容动态选择干预层
2. **多特质联合干预**: 同时干预多个特质，使用不同层配置
3. **干预强度自适应**: 根据当前表征强度调整干预力度

---

## 6. 可视化资源

### 6.1 热力图
位置: `experiments/probing/v2_results/heatmaps/`

- correlation_age.png
- correlation_gender.png
- correlation_introversion.png
- correlation_openness.png
- correlation_conscientiousness.png
- correlation_agreeableness.png
- correlation_neuroticism.png

### 6.2 配置文件
位置: `experiments/probing/v2_results/layer_configs/`

- top_8_layers.yaml
- continuous_8_layers.yaml
- early_peak_8_layers.yaml (推荐)
- multi_scale_8_layers.yaml

### 6.3 统计数据
- `experiments/probing/v2_results/analysis/correlation_stats.json`
- `experiments/probing/v2_results/analysis/recommended_config.yaml`
- `experiments/probing/v2_results/v2_results.json`

---

## 7. 后续行动

- [ ] 使用推荐的 inject_layers [0-7] 重新训练
- [ ] 修复 gate_init_bias 硬编码问题
- [ ] 扩充数据集，增加人格特质多样性
- [ ] 使用 1000 样本重新运行 Probing 获得更可靠结果
- [ ] 实现特质特定的层配置策略

---

## 附录: 大五人格理论简介

**大五人格模型（Big Five / OCEAN）** 是心理学中最广泛接受的人格特质模型。

| 维度 | 英文 | 高分特征 | 低分特征 |
|------|------|---------|---------|
| 开放性 | Openness | 好奇、创意、开放 | 传统、务实、保守 |
| 尽责性 | Conscientiousness | 有条理、勤奋、负责 | 随意、马虎、拖延 |
| 外向性 | Extraversion | 活跃、健谈、外向 | 安静、内向、保守 |
| 宜人性 | Agreeableness | 友好、善良、合作 | 冷漠、挑剔、竞争 |
| 神经质 | Neuroticism | 敏感、焦虑、情绪化 | 稳定、冷静、平和 |

注：本研究使用 introversion（内向性）作为 extraversion 的反向指标。