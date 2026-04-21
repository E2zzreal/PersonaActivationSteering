# PersonaSteer 项目实验报告

日期：2026-04-21
状态：阶段性结论

---

## 1. 项目概述

PersonaSteer 旨在通过动态激活引导（Dynamic Activation Steering）实现多人格对话生成。核心架构：超网络（HyperNetwork）将人格描述转化为干预向量（v_t），通过门控注入机制注入到冻结的 Qwen3-4B backbone 的指定层，使模型输出符合特定人格。

**核心问题**：注入机制能否比简单的 system prompt 提供更好的人格对齐？

---

## 2. 实验路径与关键发现

### 2.1 数据层诊断（D 层）

| 实验 | 数据 | 均分 | 结论 |
|------|------|---:|------|
| ALOE 原始 | 2777 条, 57 personality | 2.600 | 训练数据低于 baseline |
| Qwen3-4B baseline | — | 2.833 | 模型自然输出 |
| Claude SFT Stage1 | 661 条, 14 personality | 3.000 | 突破 baseline |
| Claude SFT Stage2 | 同上, +gate | 3.033 | 当前 ALOE 最佳 |
| Big Five Stage1 | 199 条, 16 personality | 3.503 | 结构化 personality 有效 |

**关键转折**：ALOE 的 57 个 personality 在 encoder embedding 空间的余弦相似度为 0.961（几乎不可区分），而 Big Five 16 个结构化 personality 的 5D 向量相似度仅 0.037。

### 2.2 注入机制诊断（I 层）

| 实验 | 改动 | 均分 | 结论 |
|------|------|---:|------|
| A1 (Embedding Table) | 14 个正交可学习向量 | 3.000 | encoder 不是唯一瓶颈 |
| B (gate=0.0) | 注入强度 12%→50% | 3.000 | gate 强度不是瓶颈 |
| Prompt 升级试验 | 行为约束 prompt | 4.00 | Claude 已够好 |

### 2.3 损失函数诊断（L 层）

**SCL（对比损失）三层 bug 修复历程**：

| Bug | 原因 | 修复 |
|-----|------|------|
| batch_size=1 → 返回 0 | 单样本无对比 | batch_size=4 + grouped sampler |
| user_ids 唯一 → pos_mask=0 | 优先用 user_ids 匹配 | 改用 personality 字符串匹配 |
| 全同 personality → 无负例 | sampler 设计缺陷 | 混合 batch（2 personality × 2 samples）|

修复后 SCL 仍无效（ALOE 数据 Δ=0.003/6 epochs），但在 Big Five 数据上首次出现有意义下降（1.069→0.952, -11%）。

### 2.4 公平对比（关键结论）

| 模型 | 严格 v3 rubric | 分布 |
|------|---:|------|
| Big Five PersonaSteer | 3.503 | 4×2, 111×3, 61×4, 21×5 |
| Qwen3-4B Baseline（同 prompt） | 3.500 | 6×2, 15×3, 27×4, 2×5 |
| 差异 | +0.003 | 不显著 |

**结论：PersonaSteer 注入机制在 personality 已在 prompt 中时不提供额外增益。3.033→3.503 的提升完全来自更好的 personality 描述。**

---

## 3. Probing 层分析

对冻结 Qwen3-4B backbone 做 36 层线性探针（16 Big Five personality × 5 contexts = 80 样本）：

| Big Five 维度 | 最强层 | 峰值 R² | 分布模式 |
|--------------|---:|---:|------|
| C (尽责性) | 19 | **0.815** | 中深层，最强维度 |
| A (宜人性) | 7-21 | 0.711 | 中间层稳定带 |
| E (外向性) | 35 | 0.685 | 逐层递增，深层最强 |
| O (开放性) | 8-10 | 0.386 | 较浅层，最弱维度 |
| N (神经质) | 29 | 0.396 | 全程较弱 |

**推荐注入窗口 [16..23]**（overall R²=0.559），与之前经验选择的 [8..15] 和 B1 归因的 [4..11] 均不同。

---

## 4. 泛化能力验证

用 4 个训练中未见过的 Big Five personality 测试：

| 新 Persona | 与最近训练 persona 距离 | 均分 |
|------------|---:|---:|
| Cold Visionary | cos=0.734 | **3.80** |
| Anxious Creator | cos=0.852 | **3.75**（含 1 个 5 分） |
| Chaotic Empath | cos=0.941 | 3.40 |
| Blunt Pragmatist | cos=0.726 | 3.20 |
| **平均** | — | **3.526** |

泛化均分 3.526 高于训练内 3.348，说明 5D Big Five 分支具有良好的泛化能力。但此增益同样可能来自 prompt 而非注入。

---

## 5. 整体结论链

```
ALOE 数据太弱（2.6）→ 换 Claude 数据（4.0）→ PersonaSteer 3.0（SFT 磨平差异）
  → 不是注入机制问题（A1/B 诊断排除）
  → 不是 Claude 质量问题（prompt 试验排除）
  → 是 personality 描述问题（encoder cos=0.961）
    → Big Five 结构化 personality → 3.503
      → 但 baseline 不注入也 3.500
      → 最终结论：提升来自 personality 描述质量，非注入机制
```

**PersonaSteer 在 personality 已在 prompt 中的标准设置下，注入机制的增益为零。**

---

## 6. 技术贡献（独立于注入机制增益）

尽管注入机制未超过 baseline，项目在以下方面有技术贡献：

### 6.1 Big Five 结构化人格配置
- 16 个覆盖 OCEAN 全空间的 personality 配置（5D 余弦均值 0.037）
- 每个配置含 behavioral_markers 和 anti_patterns
- 可复用于任何人格对话研究

### 6.2 层级人格信息 Probing
- 首次对 Qwen3-4B 做 Big Five 维度的逐层探针分析
- 发现 C（尽责性）在中深层最强（R²=0.815），E（外向性）逐层递增
- 为注入层选择提供了理论依据

### 6.3 三源评估管道
- ALOE / Claude / Qwen3 并行多轮对话生成 + 统一 LLM Judge 评分
- 严格 v3 rubric（"好但通用=3"保护规则）
- GPT-5.4 + Gemini 集成评估

### 6.4 Bug 修复与工程改进
- 5 个 P0 bug 修复（gate 传递、SCL 三层 bug）
- PersonalityGroupedSampler（混合 personality 批次）
- KV cache 生成加速（~50-100x）
- HyperNetwork Big Five 5D 分支（向后兼容）

---

## 7. 未来研究方向

### 7.1 无 Prompt 注入实验（验证注入的独立价值）

**假设**：当 personality 描述不在 system prompt 中时，v_t 注入是唯一的人格信号来源，此时注入机制应体现出不可替代的价值。

**实验设计**：
- 控制组：system prompt 不含 personality，仅普通对话指令
- 实验组 A：system prompt 不含 personality + PersonaSteer v_t 注入
- 实验组 B：system prompt 含 personality（当前设置，作为上限参考）

**预期结果**：
- 控制组：~2.0（无人格信号）
- 实验组 A：如果注入有效，应显著高于控制组
- 实验组 B：~3.5（已验证）

**意义**：如果 A 显著高于控制组，证明 v_t 能独立携带人格信息。这意味着 PersonaSteer 在以下场景有独特价值：
- 不想占用 context window 的长对话
- 需要通过 API 参数（而非文本）控制人格的部署场景
- 多智能体系统中每个 agent 的人格由向量定义

### 7.2 多轮对话一致性测试

**假设**：在 10-20 轮的长对话中，system prompt 中的人格描述被 attention 机制逐渐稀释，而 v_t 作为每层的固定注入不受对话长度影响。

**实验设计**：
- 生成 20 轮对话
- 每 5 轮评估一次人格一致性（第 1-5 轮、6-10 轮、11-15 轮、16-20 轮）
- 对比：prompt-only vs prompt+injection
- 关键指标：后半段（11-20 轮）的人格对齐分是否下降

**预期结果**：
- prompt-only：前半段正常，后半段人格逐渐退化
- prompt+injection：后半段人格仍然稳定

**意义**：如果注入在长对话中表现出优势，PersonaSteer 在多轮对话场景（客服、心理咨询、长期陪伴）有实用价值。

### 7.3 连续人格插值实验

**假设**：5D Big Five 向量支持连续插值，可以在两个极端人格之间平滑过渡。System prompt 无法做到这种连续控制。

**实验设计**：
- 选择两个极端 persona（如 Explorer E=+0.9 和 Hermit E=-0.9）
- 从 E=+0.9 以 0.2 的步长过渡到 E=-0.9（共 10 个插值点）
- 对每个插值点生成回复，观察语言风格的渐变

**预期结果**：
- E=+0.9：高能量、社交、兴奋
- E=+0.5：活跃但克制
- E=0.0：中性
- E=-0.5：安静、简短
- E=-0.9：极度内向、惜字如金

**意义**：如果插值平滑，PersonaSteer 可以用于：
- 游戏 NPC 的动态人格调节
- 心理治疗中根据患者状态调整 AI 的互动风格
- 研究人格维度对语言的因果影响（机制可解释性）

### 7.4 注意力头级别 Probing（机制可解释性）

对 Probing 推荐的 top-5 层（Layer 16-24）做注意力头分析：
- 32 heads × 5 层 = 160 个探针
- 目标：找到「人格专用注意力头」
- 与 Anthropic 的 SAE 特征研究类似，但聚焦于人格维度

---

## 8. 文件索引

### Main 分支
| 文件 | 说明 |
|------|------|
| configs/big5_personalities.json | 16 个 Big Five personality 配置 |
| docs/历史实验结论速查.md | A-L 节完整实验结论 |
| docs/Big5人格分支设计文档_2026-04-20.md | Big Five 架构设计 |

### feature/big5-personality 分支
| 文件 | 说明 |
|------|------|
| src/models/hyper_network.py | Big Five 5D 分支（big5_projector + big5_gate）|
| src/models/persona_steer.py | KV cache 生成 + big5_scores 透传 |
| src/data/aloe_dataset.py | 支持 big5_scores 字段 |
| src/data/collator.py | 收集 big5_scores 为 tensor |
| src/training/trainer.py | 传递 big5_scores + SCL 用 personality 匹配 |
| scripts/generate_big5_data.py | 跨 persona 数据生成 |
| scripts/probe_big5_layers.py | 层级 Probing |
| scripts/eval_big5_model.py | Big Five 模型评估 |
| results/big5_probing/layer_probe.json | 36 层探针结果 |
| results/big5_eval/ | 评估结果（完整 + 严格重评 + baseline 对比）|
