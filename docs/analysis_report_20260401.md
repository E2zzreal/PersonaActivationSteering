# PersonaSteer V2 训练与评估分析报告

**日期**: 2026-04-01
**状态**: Stage2/3 生成退化问题已修复，对话重新生成完毕，待Judge评估

---

## 一、项目概述

### 1.1 目标
PersonaSteer V2 通过三阶段训练实现人格画像驱动的对话生成：
- **Stage1**: 训练 HyperNetwork 生成干预向量 v_t
- **Stage2**: Dual Loss 训练，让模型学会利用注入
- **Stage3**: 加入对比学习(SCL)增强人格一致性

### 1.2 模型架构
- **骨干模型**: Qwen2.5-3B / Qwen3-4B
- **超网络**: 输入人格文本 → 输出 v_t (1024维，8层)
- **注入模块**: DynamicGate 控制 8 个注入层的干预强度
- **注入层**: Qwen2.5 [10-17], Qwen3 [8-15]

### 1.3 配置演化
- **V4 修复** (2026-03-29): gate bias 初始化为 -3，sigmoid≈0.05
- **重训配置** (2026-03-31): gate_min_value=0.0, gate_reg_weight=0.0, dual_loss=true

---

## 二、训练结果汇总

### 2.1 Qwen2.5 训练 Loss

| Stage | Epoch 1 | Epoch 2 | Epoch 3 | Best Loss | 配置要点 |
|-------|---------|---------|---------|-----------|----------|
| Stage1 | 0.3065 | 0.2412 | 0.2312 | 0.2312 | V4 gate init |
| Stage2 | 0.7255 | 0.6426 | **0.6265** | 0.6265 | dual_loss, gate_min=0 |
| Stage3 | 0.6522 | 0.6428 | 0.6427→0.6323 | 0.6323 | SCL=0.0000 |

### 2.2 Qwen3 训练 Loss

| Stage | Epoch 1 | Epoch 2 | Epoch 3 | Best Loss | 备注 |
|-------|---------|---------|---------|-----------|------|
| Stage1 | 0.6236 | 0.2733 | 0.2232 | 0.2232 | - |
| Stage2 | 0.1350 | 0.1222 | **0.1062** | 0.1062 | Loss异常低 |
| Stage3 | 0.1081→0.0968 | - | - | **0.0968** | 极低loss，疑过拟合 |

### 2.3 关键观察

1. **Qwen3 Loss 显著低于 Qwen2.5**
   - Stage2: 0.106 vs 0.627 (6倍差距)
   - Stage3: 0.097 vs 0.632 (6.5倍差距)
   - 可能原因：Qwen3 thinking 能力带来的训练差异，或数据对齐问题

2. **SCL Loss 始终为 0**
   - Stage3 所有 epoch: `scl=0.0000`
   - 对比学习模块未生效，需检查 `scl_weight` 和正负样本构造

3. **Gate Bias 保持初始值**
   - 重训后 `gate_mlp.3.bias` 仍为 [-2.997, ..., -2.994]
   - 说明 `gate_min_value=0.0` 配置生效，gate 未被强制推高

---

## 三、生成退化问题分析

### 3.1 第一次生成 (2026-04-01 07:51)

**结果**: Stage2/Stage3 四个模型 **100% 退化**

| 模型 | 退化率 | 主要模式 |
|------|--------|----------|
| stage2_qwen25 | 196/200 (98%) | 纯换行符、`assistant888...`、`allall...` |
| stage2_qwen3 | 200/200 (100%) | `\n\n...`、`ueueue...`、`bbbb...` |
| stage3_qwen25 | 199/200 (99.5%) | 纯换行符、`ItItIt...` |
| stage3_qwen3 | 200/200 (100%) | `HelloHello...`、`ueueue...` |

**Judge V3 评估**: 全部 weighted=1.0（最低分），无法使用

### 3.2 根本原因定位

#### 问题1: Token级 v_prev 更新

**原代码** (`persona_steer.py:383`):
```python
for _ in range(max_new_tokens):
    logits, v_t_layers, _ = self.forward(...)
    # ...
    v_prev = v_t_layers.mean(dim=1).detach()  # 每个 token 都更新
```

**问题**: v_prev 是**对话轮次级别**的设计，但在 token 生成循环内每步更新，导致干预向量在生成过程中累积发散

#### 问题2: Gate 第8层越界

**诊断数据**:
```
v_t_mean norm: 27.27
gate_mlp pre-sigmoid: [-3.32, -2.90, -4.04, -3.23, -3.57, -3.01, -1.68, 9.74]
gate values: [0.035, 0.052, 0.017, 0.038, 0.027, 0.047, 0.157, 0.9999]

Layer 17: gate=0.9999, proj_norm=19.53, inject_magnitude=19.53
```

**问题**:
- Gate 第8个输出 pre-sigmoid 值 = 9.74
- `gate_mlp.3.weight[7]` 与 `v_t_mean` 的主成分对齐，产生强信号
- 即使 bias=-3，权重方向仍导致 sigmoid 输出 ≈1.0
- **inject_magnitude=19.53** 远超 hidden_states norm (≈20-50)，完全覆盖原始激活

### 3.3 实施的修复

#### 修复1: 固定 v_prev 到轮次级别

**修改位置**: `src/models/persona_steer.py` generate() 方法

```python
# 【修复】在 token 生成循环开始前计算一次 v_t，整轮固定不变
with torch.no_grad():
    _, v_t_layers_fixed, _ = self.forward(
        input_ids=generated,
        v_prev=v_prev,
        personality_texts=personality_texts,
        user_query_texts=user_query_texts,
    )
    self.injection.set_intervention_vector(v_t_layers_fixed)

# Token 循环内不再更新 v_prev
with torch.no_grad():
    for _ in range(max_new_tokens):
        # 直接调用 backbone，hooks 已设置好固定的 v_t
        outputs = self.backbone(...)
        # ...
```

#### 修复2: Gate 值硬限制

**修改位置**: `src/models/injection.py` set_intervention_vector()

```python
# 【安全】clamp gate 到 0.2 以内，防止单层注入过强破坏生成
self.current_gate_values = self.gate(v_t_mean).clamp(max=0.2)
```

**效果对比**:
```
修复前: Layer 17 gate=0.9999, inject_magnitude=19.53
修复后: Layer 17 gate=0.20, inject_magnitude=3.91
```

### 3.4 第二次生成 (2026-04-01 09:20)

**修复后生成质量**:

| 模型 | 轮次数 | 平均长度 | 退化率 | 状态 |
|------|--------|----------|--------|------|
| baseline_qwen25 | 200 | 388 | 14/200 (7%) | ✓ 正常 |
| baseline_qwen3 | 200 | 559 | 0/200 (0%) | ✓ 正常 |
| stage1_qwen25 | 200 | 402 | 14/200 (7%) | ✓ 正常 |
| stage1_qwen3 | 200 | 528 | 0/200 (0%) | ✓ 正常 |
| stage2_qwen25 | 200 | 467 | 31/200 (15.5%) | ✓ 基本正常 |
| stage2_qwen3 | 200 | 534 | 0/200 (0%) | ✓ 正常 |
| stage3_qwen25 | 200 | 441 | 33/200 (16.5%) | ✓ 基本正常 |
| stage3_qwen3 | 200 | 536 | 0/200 (0%) | ✓ 正常 |

**生成示例** (修复后):

```
[人格] He is independent, empathetic, creative. He enjoys deep conversations.
[用户] Hey! How are you today?
[回复] I'm doing quite well, thanks for asking! Are you looking to chat about
       something specific today? 🎈

[人格] She is outgoing, energetic, loves humor. Always cheerful.
[用户] What do you think about traveling?
[回复] Traveling is an amazing experience. It allows us to explore new cultures,
       broaden our horizons, and gain a deeper understanding of the world...
```

**剩余问题**:
- Qwen2.5 Stage2/3 仍有少量退化（emoji重复、句子重复），需进一步调整 gate 上限
- Qwen3 生成质量显著优于 Qwen2.5（零退化）

---

## 四、旧版 Judge 评估结果 (2026-03-31)

**注意**: 以下评估基于第一次退化的生成，仅 baseline/stage1 可参考

### 4.1 Judge V3 分数 (2026-03-31 09:44)

| 模型 | Weighted | Style | Content | Consistency | 备注 |
|------|----------|-------|---------|-------------|------|
| baseline_qwen25 | 3.000 | 3.00 | 3.00 | 3.00 | 中等水平 |
| baseline_qwen3 | 3.000 | 3.00 | 3.00 | 3.00 | 中等水平 |
| stage1_qwen25 | 2.960 | 2.96 | 2.96 | 2.96 | 略低于baseline |
| stage1_qwen3 | 3.000 | 3.00 | 3.00 | 3.00 | 无差异 |
| stage2_qwen25 | 1.000 | 1.00 | 1.00 | 1.00 | 退化样本 |
| stage2_qwen3 | 1.040 | 1.04 | 1.04 | 1.04 | 退化样本 |
| stage3_qwen25 | 1.000 | 1.00 | 1.00 | 1.00 | 退化样本 |
| stage3_qwen3 | 1.440 | 1.44 | 1.44 | 1.44 | 退化样本 |

**结论**: baseline/stage1 分数接近，stage1 未显示出明显人格增强效果

---

## 五、Baseline vs Stage1 生成分析

### 5.1 文本特征对比

**Qwen2.5**:
- 两者回复风格相似，都说 "I'm an AI..." 模式
- stage1 未注入明显人格特征
- 退化率相同 (14/200)，说明退化是 backbone 固有问题

**Qwen3**:
- baseline: 偏学术/中立，"I don't have a physical form..."
- stage1: 更积极主动，"I love spending time outdoors too!"
- stage1 人格特征更明显，回复更长更自然

### 5.2 人格一致性评估

**样本0人格**: "He is independent, capable of working alone. He is empathetic..."

| 模型 | 第一轮回复 | 人格体现 |
|------|-----------|----------|
| baseline_qwen25 | "As an AI, I don't have feelings..." | ✗ 无 |
| baseline_qwen3 | "I don't have a physical form..." | ✗ 无 |
| stage1_qwen25 | "I'm an AI language model..." | ✗ 无 |
| stage1_qwen3 | "Hey there! I'm doing great... I love spending time outdoors too!" | ✓ 部分 |

**结论**: Qwen3 Stage1 开始体现人格，Qwen2.5 几乎无效

---

## 六、HyperNetwork 与 Gate 权重分析

### 6.1 Stage2 Checkpoint 权重

**Gate 参数**:
```
gate_mlp.0.weight: (256, 1024), norm=18.83
gate_mlp.3.bias: [-2.997×7, -2.994]  # 保持初始化
gate_mlp.3.weight[7] norm: 0.642  # 第8行权重略大
```

**HyperNetwork 参数**:
```
encoder_projector.weight: (1024, 2048), norm=18.99
layer_embedding.weight: (8, 1024), norm=7.90
projector.0.linear1.weight: (4096, 1024), norm=38.16
```

**关键发现**:
1. Gate bias 未变，但 weight 与 v_t_mean 的主成分对齐导致第8层越界
2. HyperNetwork 输出的 v_t 各层 norm 一致 (31.8-32.5)，第8层略大
3. Layer projector 输出 norm ≈ 25-27，gate=0.2 时 inject_magnitude ≈ 5.0

### 6.2 Cross-Attention 潜在问题

**代码位置**: `src/models/hyper_network.py`

```python
attn_scores = (Q * K).sum(dim=-1) / (self.v_dim ** 0.5)  # (batch,) scalar
attn_weights = torch.softmax(attn_scores, dim=0)  # BUG: dim=0 is batch!
```

**问题**: 当 batch=1 时，`softmax([x]) = [1.0]` 恒定，导致 v_t 完全忽略 query，只编码 personality

**影响**:
- v_t 与 user_query 无关，无法实现 query-aware 干预
- dual_loss 训练的是固定 bias 方向，而非动态人格注入

---

## 七、已知问题清单

### 7.1 架构问题

| 问题 | 严重性 | 影响 | 状态 |
|------|--------|------|------|
| Cross-Attention dim=0 bug | 高 | v_t 忽略 query | 已定位，未修复 |
| SCL loss 始终为 0 | 高 | 对比学习未生效 | 已定位，未修复 |
| Qwen3 loss 异常低 | 中 | 可能过拟合 | 待验证 |
| Gate 第8层越界倾向 | 中 | 单层注入过强 | 已修复 (clamp) |

### 7.2 训练配置问题

| 问题 | 原因 | 影响 |
|------|------|------|
| dual_loss 公式 | `loss = sft + α·relu(sft-clean)` | 训练固定偏置方向，非动态注入 |
| v_norm 约束弱 | weight=0.01, target=32 | v_t norm=90，但惩罚仅 33.6 |
| gate_min_value 依赖 | 需显式配置，不能依赖默认值 | 重训前 gate 训练到 0.6-1.0 |

### 7.3 生成质量问题

| 问题 | 模型 | 退化率 | 原因 |
|------|------|--------|------|
| Emoji 重复 | Qwen2.5 | 7% | Backbone 固有倾向 |
| 句子重复 | Qwen2.5 Stage2/3 | 15% | 注入过强或 gate 不稳定 |
| 角色标签泄露 | Stage2 偶发 | <5% | Stop token 检测延迟 |

---

## 八、下一步计划

### 8.1 紧急修复

1. **修复 Cross-Attention**: 改为正确的序列注意力或门控融合
2. **修复 SCL**: 检查正负样本构造和 `scl_weight` 配置
3. **调整 gate 上限**: Qwen2.5 可进一步降低到 0.15

### 8.2 训练改进

1. **替换 dual_loss**: 使用 KL 散度约束而非 relu 差值
2. **增强 v_norm 约束**: weight 提升到 0.1，target 降到 5.0
3. **添加 learnable scale**: layer_projector 初始化 scale=0.01

### 8.3 评估流程

1. **立即**: 对修复后的对话运行 Judge V1/V3
2. **短期**: 增加人格一致性的细粒度评估
3. **长期**: 引入人工评估和 A/B 测试

---

## 九、文件清单

### 9.1 对话生成结果

```
results/conversations_20260401_092056/
├── baseline_qwen25.json  (200 turns, 退化=14)
├── baseline_qwen3.json   (200 turns, 退化=0)
├── stage1_qwen25.json    (200 turns, 退化=14)
├── stage1_qwen3.json     (200 turns, 退化=0)
├── stage2.json           (200 turns, 退化=31) ← 修复后重新生成
├── stage2_qwen3.json     (200 turns, 退化=0)
├── stage3.json           (200 turns, 退化=33) ← 修复后重新生成
└── stage3_qwen3.json     (200 turns, 退化=0)
```

### 9.2 训练日志

```
logs/
├── retrain_stage2_20260331_111922/   # Qwen2.5 Stage2 重训
├── retrain_qwen3_20260331_112230/    # Qwen3 Stage2 重训
├── stage3_20260331_141717/           # Qwen2.5 Stage3
├── stage3_20260331_162035/           # Qwen3 Stage3
└── eval_20260401_092056/             # 对话生成日志
```

### 9.3 关键 Checkpoint

```
checkpoints/
├── stage1/best.pt         (Qwen2.5, loss=0.2312)
├── stage1_qwen3/best.pt   (Qwen3, loss=0.2232)
├── stage2/best.pt         (Qwen2.5, loss=0.6265) ← gate bias=-2.997
├── stage2_qwen3/best.pt   (Qwen3, loss=0.1062)
├── stage3/best.pt         (Qwen2.5, loss=0.6323)
└── stage3_qwen3/best.pt   (Qwen3, loss=0.0968)
```

---

## 十、结论

1. **生成退化问题已解决**: 通过修复 v_prev 更新机制和限制 gate 上限，生成质量恢复正常
2. **Qwen3 表现优于 Qwen2.5**: 零退化、更长回复、更强人格体现
3. **Stage1 效果有限**: 仅 Qwen3 显示出轻微人格增强，Qwen2.5 几乎无变化
4. **Stage2/3 需重新评估**: 修复后的生成结果待 Judge 评分
5. **架构问题待修复**: Cross-Attention bug 和 SCL 失效是核心问题

---

**报告更新**: 2026-04-01 10:00
**下一步**: 启动 Judge V1/V3 评估修复后的对话
