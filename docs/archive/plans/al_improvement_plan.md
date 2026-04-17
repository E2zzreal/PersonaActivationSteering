# AL (Average Likert) 提升计划

## 当前状态分析

### AL分数现状

| 模型 | AL分数 | 状态 | 与Baseline差距 |
|------|--------|------|---------------|
| baseline | 3.07 | ✅ | - |
| stage1_qwen3 | 3.07 | ✅ | 0.00 |
| stage3_auto | 3.00 | ✅ | -0.07 |
| exp_gate_init_neg3 | 3.00 | ✅ | -0.07 |
| stage3_gate_reg_0.01_lr1e4 | 3.00 | ✅ | -0.07 |

**关键发现**:
1. **Baseline已经很强** - 系统提示词引导的角色扮演效果良好
2. **Activation Steering无明显优势** - Stage3与Baseline相当甚至略低
3. **AL分数偏低** - 距离理想分数（4.0+）还有很大提升空间

### 核心问题诊断

#### 问题1: 干预强度太弱 🔴
```python
gate_init_bias = -2.0  # 硬编码
sigmoid(-2.0) ≈ 0.119  # 只有11.9%干预强度

# 干预公式
h' = h + gate * proj(v_t)
h' = h + 0.12 * proj(v_t)  # 干预太弱
```

#### 问题2: 人格特征提取不足 🟡
- HyperNetwork仅从用户话语提取特征
- 缺乏对完整人格描述的深度编码
- 干预向量v_t的表达能力有限

#### 问题3: 多轮一致性不足 🟡
- 当前v_t递归更新机制简单
- 缺乏长期人格记忆
- 多轮后人格特征逐渐淡化

---

## AL提升策略

### 策略1: 增强干预强度（预期提升 +0.3-0.5）

**目标**: 将gate初始值从0.12提升到0.25-0.35

**具体措施**:
1. **修复gate_init_bias传递**（必须）
2. **系统测试不同初始值**:
   ```python
   gate_init_bias_candidates = [-1.5, -1.0, -0.5, 0.0]
   # sigmoid(-1.5) ≈ 0.18
   # sigmoid(-1.0) ≈ 0.27
   # sigmoid(-0.5) ≈ 0.38
   # sigmoid(0.0)  ≈ 0.50
   ```

3. **动态gate调度**:
   - 初始高干预，逐渐降低
   - 根据对话轮数调整强度

**实验设计**:
```yaml
# configs/exp_gate_init_sweep.yaml
experiments:
  - name: gate_init_-1.5
    gate_init_bias: -1.5
  - name: gate_init_-1.0
    gate_init_bias: -1.0
  - name: gate_init_-0.5
    gate_init_bias: -0.5
```

---

### 策略2: 增强人格编码（预期提升 +0.2-0.4）

**目标**: 提升干预向量v_t的人格表达能力

**具体措施**:

#### 2.1 改进HyperNetwork结构
```python
# 当前结构
v_dim: 1024
hidden_dim: 4096
num_layers: 3

# 建议增强方案A: 增加深度
num_layers: 5  # 3→5

# 建议增强方案B: 增加宽度
hidden_dim: 6144  # 4096→6144

# 建议增强方案C: 添加Attention
self.persona_attention = nn.MultiheadAttention(v_dim, num_heads=8)
```

#### 2.2 分离Persona和Query编码
```python
# 当前: 简单拼接
fused = persona_emb + query_emb

# 建议: 独立编码后融合
persona_features = self.persona_encoder(personality_text)
query_features = self.query_encoder(user_query)
fused = self.fusion_gate(persona_features, query_features)
```

#### 2.3 添加人格记忆模块
```python
class PersonaMemory(nn.Module):
    """维护跨轮次的人格状态"""
    def __init__(self, v_dim):
        self.memory = nn.GRUCell(v_dim, v_dim)
    
    def update(self, current_v, hidden_state):
        return self.memory(current_v, hidden_state)
```

---

### 策略3: 改进训练目标（预期提升 +0.2-0.3）

#### 3.1 增加人格一致性损失
```python
# 当前损失
loss = sft_loss + scl_loss

# 建议增加
persona_consistency_loss = compute_persona_consistency(
    generated_text, 
    personality_description
)
loss = sft_loss + scl_loss + 0.3 * persona_consistency_loss
```

#### 3.2 使用更强的基线模型
```python
# 当前
base_model: Qwen3-4B

# 建议尝试
base_model: Qwen3-8B  # 更大的模型
# 或
base_model: Qwen3-4B-Instruct  # 指令微调版本
```

#### 3.3 增加训练数据
```python
# 当前
max_turns: 6
num_samples: 2777

# 建议
max_turns: 10  # 使用完整对话
num_samples: 5000+  # 数据增强
```

---

### 策略4: 优化系统提示词（预期提升 +0.1-0.2）

**当前提示词**:
```
You are role-playing as a person with the following personality: {personality}.
Respond naturally as this person would.
```

**优化方案**:
```
You are embodying the following character. Think, feel, and respond exactly as this person would.

Character Profile:
{personality}

Important:
- Use first-person perspective ("I", "my")
- Show the character's traits through actions and speech
- Maintain consistency with the character's values and habits
- React emotionally as the character would
```

---

## 执行计划

### 阶段1: 基础修复（Week 1）

**目标**: 修复硬编码问题，建立测试基准

**任务**:
1. ✅ 修复 `gate_init_bias` 传递（30分钟）
2. ✅ 测试 gate_init_bias = -1.0, -0.5, 0.0
3. ✅ 重新训练3个模型
4. ✅ 评估AL分数变化

**成功标准**:
- 至少一个模型的AL > 3.3
- 无模型生成空回复

---

### 阶段2: 结构优化（Week 2-3）

**目标**: 增强HyperNetwork和干预机制

**任务**:
1. 实现分离的Persona/Query编码器
2. 添加人格记忆模块
3. 训练并对比效果
4. 优化系统提示词

**成功标准**:
- AL > 3.5
- Stage3明显优于Baseline

---

### 阶段3: 训练优化（Week 4）

**目标**: 改进训练目标和数据

**任务**:
1. 实现人格一致性损失
2. 使用完整10轮对话训练
3. 尝试更大基线模型（如资源允许）
4. 最终评估和对比

**成功标准**:
- AL > 4.0
- 达到论文报告水平

---

## 实验追踪表

| 实验 | Gate Init | 结构改进 | 训练优化 | AL分数 | 状态 |
|------|-----------|---------|---------|--------|------|
| baseline | - | - | - | 3.07 | ✅ 基准 |
| stage1_qwen3 | -2.0 | - | - | 3.07 | ✅ 基准 |
| exp_gate_-1.0 | -1.0 | - | - | ? | 🔄 待测试 |
| exp_gate_-0.5 | -0.5 | - | - | ? | 🔄 待测试 |
| exp_gate_0.0 | 0.0 | - | - | ? | 🔄 待测试 |
| stage2_enhanced | -1.0 | ✅ | - | ? | ⏳ 计划中 |
| stage3_enhanced | -1.0 | ✅ | ✅ | ? | ⏳ 计划中 |

---

## 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 增大gate导致生成不稳定 | 中 | 高 | 渐进测试，从-1.5开始 |
| 结构改进增加训练难度 | 中 | 中 | 保持学习率，增加warmup |
| 资源不足训练大模型 | 低 | 中 | 使用梯度累积，减少batch size |
| AL提升不明显 | 低 | 高 | 多维度评估，及时调整策略 |

---

## 预期成果

### 短期（2周）
- AL从3.07提升到3.5+
- 解决所有模型生成失败问题
- 建立有效的gate初始化策略

### 中期（1个月）
- AL达到4.0+
- Stage3明显优于Baseline
- 验证Activation Steering有效性

### 长期（2个月）
- 达到或超过论文报告水平
- 形成可复现的最佳实践
- 开源优化后的模型和代码

---

*计划制定: 2026-04-10*
*目标AL: 4.0+*
*当前AL: 3.07*
*需提升: +0.93*
