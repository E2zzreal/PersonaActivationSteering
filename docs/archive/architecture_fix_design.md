# PersonaSteer V2 架构修复设计方案

**日期**: 2026-04-01
**版本**: v2.0
**状态**: 设计阶段

---

## 1. 问题诊断总结

### 1.1 当前训练结果

| 模型 | Stage2 Loss | Stage3 Loss | 生成效果 |
|------|-------------|-------------|----------|
| Qwen2.5-3B | 0.6265 | 0.6323 | **失败**：回到AI身份，人格注入无效 |
| Qwen3-3B | 0.1062 | 0.0968 | **微弱**：略有共情增强，但差异很小 |

### 1.2 根本问题定位

#### 问题1: Cross-Attention实现错误
```python
# 当前实现（错误）
attn_scores = (Q * K).sum(dim=0) / sqrt(v_dim)  # dim=0错误！
# batch=1时，sum(dim=0)消除了batch维度，导致v_t忽略query
```

**影响**: v_t只编码personality，不考虑user query，无法实现动态人格注入。

#### 问题2: Gate集中在单层
```
实际gate激活值（Qwen2.5 Stage2）:
  Layer 0-6: [0.035, 0.052, 0.017, 0.038, 0.027, 0.047, 0.157]
  Layer 7:   0.200 (被clamp限制)
```

**影响**: 只有第8层工作，其他7层几乎无效，且被clamp削弱。

#### 问题3: v_norm约束失效
```
实际v_t norm: 31.9
目标v_t norm: 5.0
v_norm_weight: 0.01（太小）
```

**影响**: v_t过大，引入noise，破坏模型原有能力。

#### 问题4: SCL loss = 0
```python
# 正负样本构造错误
positive_mask = (labels == labels[i])  # 包含了自己
negative_mask = (labels != labels[i])  # 空集
```

**影响**: 对比学习完全失效，无法学习persona区分性。

---

## 2. 架构修复方案

### 2.1 设计原则

1. **保持核心架构**: Frozen backbone + Trainable side-network
2. **Query-aware**: HyperNetwork必须同时考虑personality和query
3. **精确注入**: Head-level injection，避免residual stream污染
4. **可解释性**: Gate值可视化，便于调试

### 2.2 核心组件设计

#### 2.2.1 HyperNetwork（修复Cross-Attention）

**设计思路**: 使用门控融合替代Cross-Attention

```
输入: personality_text, query_text
  ↓
Encoder (共享backbone)
  ↓
persona_emb, query_emb
  ↓
Query-aware Gate: α = sigmoid(MLP([persona_emb; query_emb]))
  ↓
Fusion: fused = α * persona_emb + (1-α) * query_emb
  ↓
Per-layer Projectors
  ↓
输出: v_layers (B, num_layers, v_dim)
```

**关键改进**:
- ✅ 移除buggy cross-attention
- ✅ 显式建模query影响（通过α）
- ✅ 简单、稳定、易训练

#### 2.2.2 Injection Module（Head-level注入）

**设计思路**: 参考"Style Modulation Heads"论文，在head级别注入

```
输入: hidden_states (B, seq_len, num_heads, head_dim), v_t (B, v_dim)
  ↓
Per-head Gate: gates = sigmoid(MLP(v_t))  # (B, num_heads)
  ↓
Clamp: gates = clamp(gates, min=0, max=0.3)
  ↓
Per-head Projector: head_vectors = Proj(v_t)  # (B, num_heads, head_dim)
  ↓
Injection: output = hidden_states + gates * head_vectors
  ↓
输出: injected_states
```

**关键改进**:
- ✅ Per-head gate（32个gate vs 8个gate）
- ✅ 防止集中：添加gate熵正则化
- ✅ 精确注入：只影响特定heads

#### 2.2.3 Loss函数优化

**总Loss**:
```
L_total = L_sft + λ_scl * L_scl + λ_vnorm * L_vnorm + λ_entropy * L_entropy
```

**各项Loss**:

1. **SFT Loss** (不变)
   ```
   L_sft = -log P(response | query, personality)
   ```

2. **SCL Loss** (修复)
   ```python
   # 修复正负样本构造
   sim_matrix = v_norm @ v_norm.T / temperature
   pos_mask = (labels == labels.T) & ~eye_mask  # 排除自己
   L_scl = -log(exp(sim_pos) / sum(exp(sim_all)))
   ```

3. **V-norm Loss** (增强)
   ```
   L_vnorm = ||v_t|| - target||^2
   target: 32 → 5.0
   weight: 0.01 → 0.1
   ```

4. **Gate Entropy Loss** (新增)
   ```
   L_entropy = -Σ p_i * log(p_i)  # 最大化熵，防止集中
   weight: 0.01
   ```

### 2.3 训练配置

#### Stage 1: Layer Projectors (保持不变)
```yaml
trainable: layer_projectors
frozen: backbone, hyper_network, gate
epochs: 3
lr: 1e-4
```

#### Stage 2: HyperNetwork + Gate (修复后)
```yaml
trainable: hyper_network, gate, layer_projectors
frozen: backbone
epochs: 3
lr: 5e-5

loss_weights:
  sft: 1.0
  scl: 0.1      # 增强: 0.01 → 0.1
  v_norm: 0.1   # 增强: 0.01 → 0.1
  gate_entropy: 0.01  # 新增

hyperparameters:
  v_norm_target: 5.0   # 降低: 32 → 5
  gate_max: 0.3        # 提高: 0.2 → 0.3
  temperature: 0.1     # SCL温度
```

#### Stage 3: Full Fine-tuning (修复后)
```yaml
trainable: all (except backbone)
frozen: backbone
epochs: 2
lr: 1e-5
# loss_weights同Stage 2
```

---

## 3. 实现计划

### 3.1 代码修改清单

| 文件 | 修改内容 | 优先级 |
|------|----------|--------|
| `src/models/hyper_network.py` | 重写forward，移除cross-attention，添加query-aware gate | P0 |
| `src/models/injection.py` | 改为head-level injection，添加per-head gate | P0 |
| `src/training/losses.py` | 修复SCL，增强v_norm，添加gate_entropy | P0 |
| `configs/train_stage2.yaml` | 更新loss weights和hyperparameters | P0 |
| `src/models/persona_steer.py` | 适配新的injection接口 | P1 |
| `scripts/analyze_gates.py` | 新增：可视化gate分布 | P2 |

### 3.2 实施步骤

**Phase 1: 核心修复 (1天)**
1. 修改HyperNetwork (2小时)
2. 修改Injection Module (2小时)
3. 修改Loss函数 (1小时)
4. 单元测试 (1小时)

**Phase 2: 训练验证 (2天)**
1. Stage1训练 (4小时)
2. Stage2训练 (6小时)
3. Stage3训练 (4小时)
4. 生成样本分析 (2小时)

**Phase 3: 评估对比 (1天)**
1. Judge V3评估 (4小时)
2. 人工样本对比 (2小时)
3. Gate分布分析 (2小时)

---

## 4. 预期效果

### 4.1 定量指标

| 指标 | 当前 | 目标 |
|------|------|------|
| Qwen2.5 Stage3 Loss | 0.632 | < 0.3 |
| Qwen3 Stage3 Loss | 0.097 | < 0.08 |
| v_t norm | 31.9 | ~5.0 |
| Gate分布熵 | 0.5 (集中) | > 2.0 (均匀) |
| SCL loss | 0.0 | > 0.5 |

### 4.2 定性效果

**Baseline**:
```
USER: Hey! How are you?
ASST: As an AI, I don't have feelings...
```

**修复后期望**:
```
USER: Hey! How are you?
ASST (empathetic persona): I'm doing well, thanks for asking!
     How about you? I'd love to hear about your day.
```

---

## 5. 风险与备选方案

### 5.1 风险

1. **Head-level injection复杂度高**: 32 heads × 8 layers = 256个gate
   - **缓解**: 先在4层测试，验证后扩展

2. **训练不稳定**: 多个loss项可能冲突
   - **缓解**: 逐步增加loss权重，先训练SFT再加SCL

3. **效果仍然微弱**: 架构修复后仍无明显改善
   - **备选**: 考虑LoRA微调backbone（放弃frozen）

### 5.2 备选方案

**方案A: 简化版（如果head-level太复杂）**
- 保持layer-level injection
- 只修复HyperNetwork和Loss
- 添加gate正则化

**方案B: 激进版（如果效果仍差）**
- 使用LoRA微调backbone
- HyperNetwork只生成LoRA参数
- 参考PEFT方法

---

## 6. 参考文献

1. **Style Modulation Heads** (2026) - Head-level persona control
2. **Style Vectors** (2024) - Activation steering方法
3. **StyleVector** (2025, ACL) - Contrastive activation steering
4. **Concept Steering (RFM)** (2025) - Per-block concept vectors

---

## 附录A: 当前架构问题示意图

```
当前架构（有bug）:
  Personality → Encoder → persona_emb ─┐
                                        ├→ Cross-Attn (dim=0 bug!) → v_t
  Query → Encoder → query_emb ─────────┘
                                        ↓
                                   Layer Gates (集中在layer 7)
                                        ↓
                                   Residual Stream Injection
                                        ↓
                                   生成（人格微弱）

修复后架构:
  Personality → Encoder → persona_emb ─┐
                                        ├→ Query-aware Gate → Fusion → v_t
  Query → Encoder → query_emb ─────────┘
                                        ↓
                                   Head Gates (32 heads, 熵正则化)
                                        ↓
                                   Head-level Injection
                                        ↓
                                   生成（人格明显）
```

---

**设计完成，等待实施批准。**
