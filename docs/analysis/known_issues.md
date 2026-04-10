# PersonaSteer 已知问题分析

## 严重问题

### 1. Gate Init Bias 硬编码 ⚠️ CRITICAL

**问题描述**:
配置文件中的 `gate_init_bias` 参数无效，实际代码中硬编码为 -2.0。

**代码位置**:
```python
# src/models/injection.py:129-135
self.gate = DynamicGate(
    v_dim,
    self.num_inject_layers,
    gate_hidden_dim,
    gate_max=1.0,
    gate_init_bias=-2.0,  # 硬编码！
)
```

**影响**:
- 所有实验的 gate_init_bias 配置无效
- 无法测试不同初始gate值的效果
- 可能导致部分模型训练失败

**修复方案**:
```python
# 修改为从参数传入
gate_max=gate_max,
gate_init_bias=gate_init_bias,
```

**状态**: 🔴 待修复

---

### 2. 模型生成空回复 ⚠️ HIGH

**受影响模型** (5/15):
1. exp_gate_init_0
2. exp_gate_init_neg1
3. exp_gate_init_neg2
4. exp_gate_reg_0.01_lr3e5
5. exp_gate_reg_0.05_lr5e5
6. stage3_gate_reg_0.05_lr5e5

**症状**:
- 生成回复为空字符串 `""`
- 对话轮数正常（8轮）
- 训练损失正常收敛

**可能原因**:
1. **Gate值问题**: 硬编码-2.0导致某些模型gate行为异常
2. **随机初始化**: 特定随机种子导致数值不稳定
3. **模型加载**: 参数加载时部分权重未匹配
4. **生成参数**: temperature/top_p设置不当

**调查方向**:
```python
# 添加调试代码检查实际值
print(f"v_t norm: {v_t.norm().item()}")
print(f"gate values: {gate_values}")
print(f"proj output: {proj_output.norm().item()}")
```

**状态**: 🟡 调查中

---

## 中等问题

### 3. Activation Steering 效果不明显

**现象**:
- baseline: 3.07分
- stage1_qwen3: 3.07分
- stage3_auto: 3.00分

**分析**:
1. **干预强度太弱**: sigmoid(-2.0) ≈ 0.12，只有12%强度
2. **系统提示词足够**: 添加的角色扮演提示词已有效
3. **对比学习效果有限**: Stage3相比Stage1没有明显提升

**建议**:
- 增大gate_init_bias到 -1.0 或 -0.5
- 测试干预强度 25%-50% 的效果

**状态**: 🟡 待优化

---

### 4. 配置混乱

**问题**:
- 多个版本的配置文件（train_stage1.yaml, train_stage1_qwen3.yaml）
- 不同模型使用不同的基础模型（Qwen2.5 vs Qwen3）
- 注入层配置不一致

**整理建议**:
```
checkpoints/
├── stage1/
│   ├── qwen2.5/          # Qwen2.5-3B
│   └── qwen3/            # Qwen3-4B
├── stage2/
│   ├── gate_init/
│   └── gate_reg/
└── stage3/
```

**状态**: 🟢 低优先级

---

## 轻微问题

### 5. 代码警告

**警告信息**:
```
FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated.
`torch_dtype` is deprecated! Use `dtype` instead!
```

**影响**: 无功能影响，仅代码风格问题

**状态**: 🟢 可忽略

---

## 修复优先级

| 优先级 | 问题 | 影响 | 预计时间 |
|--------|------|------|---------|
| P0 | Gate Init Bias 硬编码 | 高 | 30分钟 |
| P0 | 模型生成空回复 | 高 | 2-4小时 |
| P1 | Activation Steering 效果 | 中 | 1-2天 |
| P2 | 配置混乱 | 低 | 2-3小时 |
| P3 | 代码警告 | 无 | 30分钟 |

---

## 修复后的预期效果

### 短期（修复P0问题）
- 所有模型正常生成回复
- 可以测试不同gate_init_bias的效果
- 成功率从 8/15 提升到 15/15

### 中期（优化P1问题）
- Stage3模型明显优于baseline
- 角色扮演评分提升到 3.5-4.0
- 体现Activation Steering优势

---

*记录时间: 2026-04-10*
