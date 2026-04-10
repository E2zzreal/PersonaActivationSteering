# HyperNetwork 缺少 Query 输入问题分析报告

**日期**: 2026-03-30
**问题等级**: P0 (Critical)
**影响范围**: 核心架构缺陷，导致训练后评分低于 baseline

---

## 1. 问题描述

### 1.1 核心问题

**HyperNetwork 当前设计缺少用户 query 输入，导致注入向量完全静态化。**

**当前签名**:
```python
# src/models/hyper_network.py:125-129
def forward(
    self,
    user_texts: list[str],      # 实际传入的是 personality 文本
    v_prev: torch.Tensor,        # 上一轮注入向量
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

**实际调用** (src/models/persona_steer.py:254):
```python
v_t_layers, z_t, v_norm = self.hyper_network(user_texts, v_prev)
```

**训练时传入** (src/training/trainer.py:273-277):
```python
logits, v_t, v_norm = self.model(
    input_ids=valid_input_ids,
    v_prev=valid_v_prev,
    user_texts=valid_user_texts,  # 这里传的是用户query，但变量名误导
)
```

### 1.2 致命缺陷

**问题链**:
1. 同一 personality → encoder 输出 `z_t` 固定
2. `v_prev` 在多轮对话中也趋于固定（因为 HyperNetwork 输出恒定）
3. `z_t + v_prev` → `v_t` 每一轮完全相同
4. **HyperNetwork 退化成静态查找表**：一个 personality 对应一个固定向量

**数学表达**:
```
z_t = Encoder(personality)  # 固定
v_t = MLP(z_t + v_prev)     # 由于 z_t 固定，v_prev 也固定 → v_t 恒定
v_prev_next = v_t           # 下一轮的 v_prev 也是同一个值
```

**结果**:
- ❌ 没有上下文感知能力
- ❌ 没有时序变化
- ❌ 没有动态响应能力
- ❌ 所有对话、所有轮次都用同一个注入向量

---

## 2. 实验证据

### 2.1 评估结果对比 (exp016-017)

| 模型 | AL(K)_AVG | 说明 |
|------|-----------|------|
| **Baseline** (personality as system prompt) | **4.26** | 直接传入完整语义 |
| Stage1 | 4.12 | 静态注入向量 |
| Stage2 | 4.28 | 解冻 Gate，仍然静态 |
| Stage3 | 4.14 | 加对比学习，无改善 |

**新 prompt 下** (exp017):
| 模型 | AL(K)_AVG | 下降幅度 |
|------|-----------|---------|
| Baseline | 4.12 | -0.14 |
| Stage3 | **3.02** | **-1.12** |

### 2.2 关键观察

1. **三阶段无递进效果**：Stage1/2/3 评分差距 <0.2，说明在优化一个本质上静态的东西
2. **训练后评分骤降**：新 prompt 下 baseline 几乎不受影响，但训练后模型骤降 1.12 分
3. **信息损失严重**：baseline 能看到完整 personality 文本，HyperNetwork 将其压缩为固定向量

---

## 3. 根因分析

### 3.1 架构层面

**设计缺陷**：HyperNetwork 只接收 personality，不接收当前对话上下文（user query）

**对比**：
- **Baseline**: `backbone(input_ids, system_prompt=personality)` → backbone 能根据 query 动态理解 personality
- **PersonaSteer**: `v_t = HyperNetwork(personality)` → 固定向量，无法根据 query 调整

### 3.2 训练目标不一致

- **训练目标**: SFT loss（模仿 assistant 回复）
- **评估目标**: Personality alignment（回复与人格一致）
- **实际学到**: 写好回复，而非以特定人格写回复

### 3.3 为何 baseline 更好

1. **完整语义**: Backbone 直接看到 personality 文本，能理解细微差异
2. **动态交互**: Backbone 的 attention 机制能让 personality 和 query 交互
3. **上下文感知**: 不同 query 下，backbone 对 personality 的理解不同

---

## 4. 解决方案

### 方案 A: Query-Aware HyperNetwork（推荐，P0）

#### 4.1 设计思路

**核心**: 让 HyperNetwork 同时接收 personality 和 user query，生成上下文相关的注入向量。

**新签名**:
```python
def forward(
    self,
    personality_texts: list[str],  # 人格描述
    user_query_texts: list[str],   # 当前用户输入
    v_prev: torch.Tensor,          # 历史注入向量
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

#### 4.2 Fusion 函数设计

**选项 1: Cross-Attention（推荐）**
```python
# 分离编码
z_personality = encoder(personality_texts)  # (batch, v_dim)
z_query = encoder(user_query_texts)         # (batch, v_dim)

# Cross-attention: query attends to personality
Q = linear_q(z_query)           # (batch, v_dim)
K = linear_k(z_personality)     # (batch, v_dim)
V = linear_v(z_personality)     # (batch, v_dim)

attn_scores = softmax(Q @ K.T / sqrt(v_dim))
z_fused = attn_scores @ V       # (batch, v_dim)

# 融合历史
z_t = z_fused + history_proj(v_prev)
v_t = MLP(z_t)
```

**优点**:
- 表达力强，可学习 personality 和 query 的交互关系
- 不同 query 下，personality 的不同方面被激活
- 符合 Transformer 范式

**选项 2: Gated Fusion（轻量）**
```python
z_personality = encoder(personality_texts)
z_query = encoder(user_query_texts)

# 门控融合
gate = sigmoid(linear_gate(concat(z_personality, z_query)))
z_fused = gate * z_personality + (1 - gate) * z_query

z_t = z_fused + history_proj(v_prev)
v_t = MLP(z_t)
```

**优点**:
- 参数少，训练快
- 可解释性强（gate 值表示 personality vs query 的权重）

**选项 3: Concat + MLP（最简单）**
```python
z_personality = encoder(personality_texts)
z_query = encoder(user_query_texts)

z_concat = concat(z_personality, z_query, v_prev)  # (batch, 3*v_dim)
v_t = MLP(z_concat)  # 输出 (batch, v_dim)
```

**优点**:
- 实现简单
- 让 MLP 自己学习如何融合

**缺点**:
- 参数量大（输入维度 3x）
- 可解释性差

#### 4.3 推荐方案

**优先级排序**:
1. **Cross-Attention**（表达力最强，符合 Transformer 范式）
2. **Gated Fusion**（轻量，可解释）
3. **Concat + MLP**（备选，最简单）

#### 4.4 实现步骤

**Step 1: 修改 HyperNetwork 签名**
```python
# src/models/hyper_network.py
def forward(
    self,
    personality_texts: list[str],
    user_query_texts: list[str],
    v_prev: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 分离编码
    z_personality = self.encode_text(personality_texts)
    z_query = self.encode_text(user_query_texts)

    # Cross-attention fusion
    z_fused = self.cross_attention(z_query, z_personality)

    # 融合历史
    z_t = z_fused + self.history_projector(v_prev)

    # 投影
    v_t_layers = self.projector(z_t)

    return v_t_layers, z_t, v_norm
```

**Step 2: 添加 Cross-Attention 模块**
```python
class HyperNetwork(nn.Module):
    def __init__(self, ...):
        ...
        # Cross-attention 组件
        self.query_proj = nn.Linear(v_dim, v_dim)
        self.key_proj = nn.Linear(v_dim, v_dim)
        self.value_proj = nn.Linear(v_dim, v_dim)
        self.attn_norm = nn.LayerNorm(v_dim)

    def cross_attention(self, z_query, z_personality):
        Q = self.query_proj(z_query)
        K = self.key_proj(z_personality)
        V = self.value_proj(z_personality)

        attn_scores = torch.matmul(Q, K.T) / (self.v_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, V)
        return self.attn_norm(attn_output)
```

**Step 3: 修改数据流**
```python
# src/training/trainer.py:273-277
# 当前（错误）
logits, v_t, v_norm = self.model(
    input_ids=valid_input_ids,
    v_prev=valid_v_prev,
    user_texts=valid_user_texts,  # 实际是 query，但名字误导
)

# 修改后（正确）
logits, v_t, v_norm = self.model(
    input_ids=valid_input_ids,
    v_prev=valid_v_prev,
    personality_texts=valid_personalities,  # 人格描述
    user_query_texts=valid_user_texts,      # 用户输入
)
```

**Step 4: 修改 PersonaSteerModel.forward**
```python
# src/models/persona_steer.py:230-236
def forward(
    self,
    input_ids: torch.Tensor,
    v_prev: torch.Tensor,
    personality_texts: List[str],  # 新增
    user_query_texts: List[str],   # 重命名
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if self.hyper_network is not None:
        self.injection.injection_enabled = False
        v_t_layers, z_t, v_norm = self.hyper_network(
            personality_texts,
            user_query_texts,
            v_prev
        )
        self.injection.injection_enabled = True
```

**Step 5: 修改数据 collator**
```python
# src/data/collator.py
# 确保 batch 中同时包含 personalities 和 user_texts
batch = {
    "input_ids": ...,
    "labels": ...,
    "personalities": [sample["personality"] for sample in samples],
    "user_texts": [sample["user_query"] for sample in samples],
    ...
}
```

#### 4.5 预期效果

**定量**:
- AL(K)_AVG 提升 0.5-1.0 分（目标 >4.5）
- Stage1/2/3 出现递进效果（每阶段 +0.2-0.3）
- 新 prompt 下评分下降幅度减小（<0.3）

**定性**:
- v_t 根据不同 query 动态变化
- 同一 personality 在不同话题下注入向量不同
- 多轮对话中 v_t 有时序演化

---

### 方案 B: Per-Layer Adaptive Gate（P2）

**当前**: Gate 是全局标量（所有层共享同一个 gate 值）

**改进**: Per-layer gate，不同层可以有不同注入强度

**设计**:
```python
# 当前
gate = sigmoid(gate_mlp(z_t))  # (batch, 1)

# 改进
gate = sigmoid(gate_mlp(z_t))  # (batch, num_layers)
```

**预期**:
- 浅层注入"风格"（语法、语气）
- 深层注入"内容"（观点、偏好）

**优先级**: P2（在方案 A 之后实施）

---

### 方案 C: 对比学习增强（P1）

**当前问题**: Stage3 的对比学习权重过低（scl_weight=0.01），无效果

**改进**:
1. 增大权重：scl_weight=0.1
2. Hard negative mining：同 batch 中选择最相似但不同的 personality
3. Persona-specific contrastive loss：同人格不同对话的 v_t 应相似，不同人格的 v_t 应区分

**优先级**: P1（与方案 A 并行）

---

## 5. 实施计划

### 5.1 优先级

| 优先级 | 方案 | 预期收益 | 实现难度 | 工作量 |
|--------|------|---------|---------|--------|
| **P0** | A: Query-Aware HyperNetwork | 高 | 中 | 2-3天 |
| P1 | C: 对比学习增强 | 中 | 低 | 0.5天 |
| P2 | B: Per-Layer Adaptive Gate | 中 | 中 | 1天 |

### 5.2 实施步骤

**Week 1: 方案 A（Query-Aware）**
- Day 1: 实现 Cross-Attention fusion
- Day 2: 修改数据流和训练脚本
- Day 3: 训练 Stage1 并评估

**Week 2: 对比实验**
- Day 1: A/B 对比（有 query vs 无 query）
- Day 2: 实施方案 C（对比学习增强）
- Day 3: 完整三阶段训练

**Week 3: 进阶优化**
- Day 1: 实施方案 B（Per-Layer Gate）
- Day 2-3: 最终评估和调优

### 5.3 验收标准

**必须达成**:
- [ ] AL(K)_AVG > 4.5（当前 baseline 4.26）
- [ ] Stage1/2/3 递进效果明显（每阶段 +0.2）
- [ ] v_t 在不同 query 下有显著差异（方差 >0.1）

**期望达成**:
- [ ] AL(K)_AVG > 4.8
- [ ] 新 prompt 下评分下降 <0.3
- [ ] 多轮对话中 v_t 有时序演化

---

## 6. 风险与缓解

### 6.1 风险

1. **训练不稳定**: 增加 query 输入后，输入空间变大，可能难以收敛
   - **缓解**: 使用较小学习率（1e-5），增加 warmup steps

2. **过拟合 query**: 模型可能忽略 personality，只关注 query
   - **缓解**: 在 fusion 中增加 personality 权重，或使用 gated fusion

3. **显存不足**: Cross-attention 增加参数量
   - **缓解**: 使用 gradient checkpointing，或降级到 gated fusion

### 6.2 回退方案

如果方案 A 失败：
1. 尝试方案 C（对比学习增强）
2. 简化 fusion 函数（从 cross-attention 降级到 concat+MLP）
3. 考虑架构重构（如 LoRA-based steering）

---

## 7. 总结

**核心问题**: HyperNetwork 缺少 query 输入，导致注入向量完全静态化，无法根据对话上下文动态调整。

**推荐方案**: Query-Aware HyperNetwork + Cross-Attention Fusion

**预期收益**: AL(K)_AVG 提升 0.5-1.0 分，三阶段训练出现递进效果。

**下一步**: 立即实施方案 A，2-3 天内完成实现和初步评估。
