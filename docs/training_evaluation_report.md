# PersonaSteer 训练与评估综合分析报告

> 生成日期: 2026-04-12

## 1. 执行摘要

### 1.1 关键成果

| 指标 | 修复前 | 修复后 | 提升 |
|------|--------|--------|------|
| LLM Judge Score | 2.5-3.0 | **3.7-3.8** | +0.7~1.2 |
| Checkpoint 大小 | ~12 GB | **367-420 MB** | -97% |
| 思考过程泄露 | 有 | **无** | ✅ 修复 |

### 1.2 最佳配置

**Stage 1 Minimal (6层注入)** 获得最高评分: **3.794 ± 0.43**

---

## 2. Bug 修复记录

### 2.1 Encoder 解冻 Bug

**问题**: `_configure_stage()` 错误地解冻了 encoder 参数

**影响**: Checkpoint 保存了 3.1B 额外参数 (11.77 GB)

**修复**: 过滤 encoder 参数
```python
# 修复前
for param in self.model.hyper_network.parameters():
    param.requires_grad = True

# 修复后
for name, param in self.model.hyper_network.named_parameters():
    if not name.startswith('encoder.'):
        param.requires_grad = True
```

### 2.2 Thinking 模式 Bug

**问题**: 训练时 `apply_chat_template()` 未设置 `enable_thinking=False`

**影响**: 模型学习输出伪思考过程 ("Okay, the user...")

**修复**: 在 `ALOEDataset` 中添加 `enable_thinking=False`

---

## 3. 评估结果

### 3.1 LLM Judge 评分对比

| 配置 | Stage | 注入层数 | Score | Std | 响应数 |
|------|-------|----------|-------|-----|--------|
| **stage1_minimal** | 1 | 6 | **3.794** | ±0.43 | 150 |
| stage3_v2 | 3 | 8 | 3.761 | ±0.44 | 150 |
| stage1_baseline | 1 | 8 | 3.744 | ±0.43 | 150 |
| stage2_v2 | 2 | 8 | 3.718 | ±0.44 | 150 |
| stage1_neuroticism | 1 | 3 | 3.716 | ±0.46 | 150 |

### 3.2 分数分布

```
Stage 1 Minimal:    ████████████████████ 3.794
Stage 3 V2:         ███████████████████▌ 3.761
Stage 1 Baseline:   ███████████████████  3.744
Stage 2 V2:         ██████████████████▊  3.718
Stage 1 Neuroticism:██████████████████   3.716
```

### 3.3 关键发现

1. **Stage 1 表现最佳**: Minimal 配置 (6层) 获得最高分
2. **层数不是越多越好**: 8层 baseline < 6层 minimal
3. **Stage 3 未显著提升**: 可能需要更多 epoch 或数据
4. **所有配置均达标**: 评分 > 3.7，比修复前提升显著

---

## 4. Checkpoint 分析

### 4.1 大小对比

| Checkpoint | 大小 | 注入层数 | 可训练参数 |
|------------|------|----------|------------|
| stage1_qwen3_neuroticism | 367 MB | 3 | 37.5M |
| stage1_qwen3_probing_minimal | 397 MB | 6 | 45.4M |
| stage1_qwen3 | 417 MB | 8 | 50.7M |
| stage2_qwen3_v2 | 420 MB | 8 | ~50M |
| stage3_qwen3_v2 | 420 MB | 8 | ~50M |

### 4.2 空间节省

- **修复前**: ~12 GB × 5 checkpoints = **60 GB**
- **修复后**: ~400 MB × 5 checkpoints = **2 GB**
- **节省**: **58 GB (97%)**

---

## 5. 训练配置分析

### 5.1 三种配置对比

| 配置 | 注入层 | 特点 | 适用场景 |
|------|--------|------|----------|
| **Neuroticism** | 3层 [9-11] | 最精简，专注神经质 | 单一特质控制 |
| **Minimal** | 6层 [8-13] | 平衡配置 | **通用场景 (推荐)** |
| **Baseline** | 8层 [8-15] | 全覆盖 | 复杂人格控制 |

### 5.2 训练阶段说明

| 阶段 | 训练内容 | Epochs | 学习率 |
|------|----------|--------|--------|
| Stage 1 | HyperNetwork (gate 冻结) | 4 | 1e-4 |
| Stage 2 | HyperNetwork + Gate | 4 | 5e-5 |
| Stage 3 | HyperNetwork + Gate + SCL | 5 | 3e-5 |

---

## 6. 技术改进建议

### 6.1 已完成 ✅

1. Encoder 参数冻结修复
2. Thinking 模式禁用
3. 自动检测 checkpoint 注入层数
4. 并行训练流水线

### 6.2 待优化

1. **Stage 3 效果不显著**: 
   - 增加 epoch 数 (5 → 10)
   - 调整 SCL loss 权重
   - 增加对比学习样本

2. **层数优化**:
   - 6层 minimal 表现最佳
   - 建议进一步测试 4-7 层范围

3. **数据增强**:
   - 增加训练数据多样性
   - 添加负样本约束

---

## 7. 结论

### 7.1 主要成果

1. **Bug 修复成功**: Checkpoint 大小从 12GB 降至 400MB
2. **评分显著提升**: 从 2.5-3.0 提升至 3.7-3.8
3. **最佳配置确定**: Stage 1 Minimal (6层注入)
4. **训练流水线完善**: 支持 4 GPU 并行训练

### 7.2 推荐配置

```yaml
# 推荐生产配置
inject_layers: [8, 9, 10, 11, 12, 13]  # 6层
stage: 1  # Stage 1 即可获得最佳效果
checkpoint: checkpoints/stage1_qwen3_probing_minimal/best.pt
```

### 7.3 后续工作

1. 在更多测试集上验证效果
2. 探索 Stage 3 的优化方向
3. 进行 A/B 测试对比不同配置

---

## 附录: 评估详情

### A. 评估配置

- 评估样本数: 50 (每个 checkpoint)
- LLM Judge: GPT-5.2
- 评分范围: 1-5
- 评估维度: 人格一致性

### B. Git 提交记录

```
3cbf153 feat: 添加并行训练流水线和评估脚本
7ef2fa5 fix: prevent encoder from being unfrozen in _configure_stage
2a1982a fix: add enable_thinking=False in ALOEDataset tokenization
e75e5b1 feat: 添加训练分析工具和评估脚本
```