# PersonaSteer 训练与评估综合分析报告

> 生成日期: 2026-04-12
> 更新: 添加 Baseline 对比分析

## 1. 执行摘要

### 1.1 ⚠️ 关键发现

**Baseline (无注入) 表现最佳！**

| 配置 | Score | 思考泄露率 |
|------|-------|------------|
| **baseline (无注入)** | **3.803** | **1.3%** |
| stage1_minimal | 3.794 | 22.7% |
| stage3_v2 | 3.761 | 34.0% |
| stage1_baseline | 3.744 | 24.7% |
| stage2_v2 | 3.718 | 24.0% |
| stage1_neuroticism | 3.716 | 27.3% |

**结论**: PersonaSteer 注入目前**未能提升**人格一致性，反而引入了思考过程泄露问题。

---

## 2. 完整评估对比

### 2.1 LLM Judge 评分排名

```
baseline (无注入)      ████████████████████▌ 3.803  🥇
stage1_minimal         ████████████████████  3.794  🥈
stage3_v2              ███████████████████▊  3.761  🥉
stage1_baseline        ███████████████████   3.744
stage2_v2              ██████████████████▊  3.718
stage1_neuroticism     ██████████████████   3.716
```

### 2.2 思考过程泄露统计

| 配置 | 泄露数 | 总数 | 泄露率 |
|------|--------|------|--------|
| **baseline** | 2 | 150 | **1.3%** ✅ |
| stage1_minimal | 34 | 150 | 22.7% ❌ |
| stage1_baseline | 37 | 150 | 24.7% ❌ |
| stage2_v2 | 36 | 150 | 24.0% ❌ |
| stage1_neuroticism | 41 | 150 | 27.3% ❌ |
| stage3_v2 | 51 | 150 | **34.0%** ❌❌ |

---

## 3. 问题分析

### 3.1 为什么 Baseline 表现更好？

**可能原因**:

1. **注入干扰**: HyperNetwork 生成的干预向量可能干扰了模型的正常推理
2. **训练不充分**: 模型可能未学到有效的人格注入方式
3. **评估偏差**: LLM Judge 可能更偏好自然流畅的回复，而非刻意的人格表达

### 3.2 思考过程泄露问题

**现象**: 注入后的模型倾向于输出思考过程
```
示例输出:
"Okay, the user is greeting me and saying they've been busy..."
"I need to respond in a friendly and engaging way..."
```

**原因分析**:
1. 训练数据可能隐含 CoT 模式
2. 注入向量激活了模型的推理模式
3. `enable_thinking=False` 只影响 chat template，不消除已学习的行为

---

## 4. Bug 修复回顾

### 4.1 已修复的问题

| Bug | 影响 | 状态 |
|-----|------|------|
| Encoder 解冻 | Checkpoint 12GB → 400MB | ✅ 已修复 |
| Thinking 标记 | 训练数据格式 | ✅ 已修复 |

### 4.2 待解决的问题

| 问题 | 影响 | 优先级 |
|------|------|--------|
| 思考过程泄露 | 22-34% 输出含思考 | 🔴 高 |
| 注入效果不佳 | Baseline 更优 | 🔴 高 |
| Stage 3 泄露加剧 | 34% 泄露率 | 🟡 中 |

---

## 5. 样本对比分析

### 5.1 样本 1 对比

**User**: "Hey there! How's it going? I've been pretty busy lately..."

**Baseline (Score: 3.525)**:
> Hey there! I'm doing great, thanks for asking. I've been pretty busy too—just got married last month...

**Stage1 Minimal (Score: 3.470)**:
> Okay, the user is greeting me and saying they've been busy with exciting plans. I need to respond in a friendly and engaging way...

**分析**: Baseline 回复更自然，注入版本泄露了思考过程。

### 5.2 样本 2 对比

**User**: "I just got married, and we're planning a wedding party..."

**Baseline (Score: 4.089)**:
> Oh, that's really cool! Congratulations on your marriage! It must be an exciting time...

**Stage1 Minimal (Score: 3.165)**:
> I understand how overwhelming it can be. It's a big event with so many details...

**分析**: Baseline 更热情、更符合人格，注入版本回复较机械。

---

## 6. 技术改进建议

### 6.1 紧急优先

1. **消除思考泄露**:
   - 在推理时添加更强的过滤
   - 使用 stop tokens 阻止思考输出
   - 重新训练，添加负样本约束

2. **注入机制优化**:
   - 减小注入向量强度
   - 调整注入层位置
   - 添加注入效果的监控指标

### 6.2 中期改进

1. **训练策略**:
   - 增加人格一致性奖励
   - 使用 RLHF/DPO 优化
   - 添加对比学习样本

2. **评估改进**:
   - 增加人格特质细分评估
   - 人工评估对比
   - A/B 测试验证

---

## 7. 结论

### 7.1 当前状态

| 方面 | 状态 |
|------|------|
| Bug 修复 | ✅ 完成 |
| Checkpoint 大小 | ✅ 优化 (97%↓) |
| 思考泄露 | ❌ 未解决 |
| 人格注入效果 | ❌ 未达预期 |

### 7.2 关键结论

1. **Baseline 仍是最佳选择**: 无注入时得分最高 (3.803)
2. **注入引入了问题**: 思考泄露率 22-34%
3. **需要重新设计**: 当前注入策略未能有效提升人格一致性

### 7.3 下一步行动

1. **短期**: 添加推理时过滤，减少思考泄露
2. **中期**: 重新设计注入机制和训练策略
3. **长期**: 考虑 RLHF/DPO 等更高级优化方法

---

## 附录: 详细数据

### A. 评估配置

- 样本数: 150 / checkpoint
- LLM Judge: GPT-5.2
- 评分范围: 1-5
- 评估维度: 人格一致性

### B. Checkpoint 信息

| Checkpoint | 大小 | 注入层数 |
|------------|------|----------|
| baseline | - | 0 |
| stage1_qwen3_neuroticism | 367 MB | 3 |
| stage1_qwen3_probing_minimal | 397 MB | 6 |
| stage1_qwen3 | 417 MB | 8 |
| stage2_qwen3_v2 | 420 MB | 8 |
| stage3_qwen3_v2 | 420 MB | 8 |