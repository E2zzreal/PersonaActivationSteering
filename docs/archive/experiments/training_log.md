# PersonaSteer 训练实验记录

## 实验时间线

### Stage 1: HyperNetwork 训练 (2026-04-01)

| 模型 | 基础模型 | 配置 | 最佳Loss | 状态 |
|------|---------|------|---------|------|
| stage1 | Qwen2.5-3B | train_stage1.yaml | 1.48 | ✅ 成功 |
| stage1_qwen3 | Qwen3-4B | train_stage1_qwen3.yaml | 1.48 | ✅ 成功 |

**训练参数**:
- Epochs: 4
- Learning rate: 1e-4
- Batch size: 2
- Max turns: 6

---

### Stage 2: Gate 初始化实验 (2026-04-03)

目的: 测试不同 gate_init_bias 对训练的影响

| 实验名 | gate_init_bias | 实际值 | 最佳Loss | 生成结果 | 状态 |
|--------|---------------|--------|---------|---------|------|
| exp_gate_init_0 | 0.0 | -2.0 (硬编码) | 1.72 | 空回复 | ❌ 失败 |
| exp_gate_init_neg1 | -1.0 | -2.0 (硬编码) | 1.69 | 空回复 | ❌ 失败 |
| exp_gate_init_neg2 | -2.0 | -2.0 (硬编码) | 1.81 | 空回复 | ❌ 失败 |
| exp_gate_init_neg3 | -3.0 | -2.0 (硬编码) | 1.92 | 正常 | ✅ 成功 |

**关键发现**: 配置文件中的 gate_init_bias 未生效，全部使用硬编码 -2.0

---

### Stage 2: Gate 正则化实验 (2026-04-03)

目的: 测试不同正则化强度和学习率

| 实验名 | gate_reg_weight | Learning Rate | 最佳Loss | 生成结果 | 状态 |
|--------|----------------|---------------|---------|---------|------|
| exp_gate_reg_0.001_lr5e5 | 0.001 | 5e-5 | 1.71 | 正常 | ✅ 成功 |
| exp_gate_reg_0.01_lr1e4 | 0.01 | 1e-4 | 1.68 | 正常 | ✅ 成功 |
| exp_gate_reg_0.01_lr3e5 | 0.01 | 3e-5 | 1.43 | 空回复 | ❌ 失败 |
| exp_gate_reg_0.05_lr5e5 | 0.05 | 5e-5 | 1.73 | 空回复 | ❌ 失败 |

---

### Stage 3: 对比学习实验 (2026-04-04)

目的: 加入对比学习损失，增强人格区分度

| 实验名 | 配置 | 最佳Loss | 生成结果 | 状态 |
|--------|------|---------|---------|------|
| stage3_auto | train_stage3_auto.yaml | 1.75 | 正常 | ✅ 成功 |
| stage3_gate_init_0 | train_stage3_gate_init_0.yaml | 1.75 | 正常 | ✅ 成功 |
| stage3_gate_reg_0.01_lr1e4 | train_stage3_gate_reg_0.01_lr1e4.yaml | 1.75 | 正常 | ✅ 成功 |
| stage3_gate_reg_0.05_lr5e5 | train_stage3_gate_reg_0.05_lr5e5.yaml | 1.75 | 空回复 | ❌ 失败 |

---

## 模型状态汇总

### ✅ 成功模型 (8个)

1. **baseline** - 原始基线模型
2. **stage1** - HyperNetwork训练 (Qwen2.5-3B)
3. **stage1_qwen3** - HyperNetwork训练 (Qwen3-4B)
4. **exp_gate_init_neg3** - Gate初始化-3.0
5. **exp_gate_reg_0.001_lr5e5** - 轻正则化
6. **exp_gate_reg_0.01_lr1e4** - 中等正则化
7. **stage3_auto** - 自动配置Stage3
8. **stage3_gate_init_0** - Stage3 Gate初始化0
9. **stage3_gate_reg_0.01_lr1e4** - Stage3 正则化0.01

### ❌ 失败模型 (5个)

1. **exp_gate_init_0** - 空回复
2. **exp_gate_init_neg1** - 空回复
3. **exp_gate_init_neg2** - 空回复
4. **exp_gate_reg_0.01_lr3e5** - 空回复
5. **exp_gate_reg_0.05_lr5e5** - 空回复
6. **stage3_gate_reg_0.05_lr5e5** - 空回复

---

## 评估结果 (模拟评估)

| 排名 | 模型 | 分数 | 备注 |
|------|------|------|------|
| 🥇 1 | stage1_qwen3 | 3.07 | 最佳 |
| 🥈 2 | baseline | 3.07 | 并列最佳 |
| 🥉 3 | exp_gate_init_neg3 | 3.00 | |
| 4 | stage3_auto | 3.00 | |
| 5 | stage3_gate_reg_0.01_lr1e4 | 3.00 | |
| 6 | stage1 | 2.96 | |
| 7 | exp_gate_reg_0.001_lr5e5 | 2.95 | |
| 8 | stage3_gate_init_0 | 2.94 | |
| 9 | exp_gate_reg_0.01_lr1e4 | 2.93 | |

---

## 关键问题记录

### 问题1: Gate Init Bias 硬编码
- **位置**: `src/models/injection.py:134`
- **影响**: 所有模型实际使用 -2.0，配置无效
- **状态**: 待修复

### 问题2: 部分模型生成空回复
- **数量**: 5/15 模型
- **原因**: 待调查（可能与随机初始化有关）
- **状态**: 待修复

### 问题3: Activation Steering 效果不明显
- **现象**: baseline和stage1表现相当
- **原因**: gate值太小（~0.12），干预强度弱
- **状态**: 待优化

---

## 下一步计划

1. 修复 gate_init_bias 传递问题
2. 测试更大的初始gate值（-1.0, -0.5）
3. 重新训练失败模型
4. 使用真实LLM Judge API评估

---

*记录时间: 2026-04-10*
