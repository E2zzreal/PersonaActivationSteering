# PersonaSteer 完整实验报告

日期：2026-04-21（持续更新）
分支：feature/film-injection（基于 feature/big5-personality）
状态：FiLM v2 架构验证通过，等待扩展数据训练

---

## 1. 项目目标

通过动态激活引导（Dynamic Activation Steering）实现多人格对话生成。核心架构：HyperNetwork 将人格描述转化为干预向量 v_t，通过注入机制注入冻结的 Qwen3-4B backbone 指定层，使输出符合目标人格。

**核心问题**：注入机制能否比 system prompt 提供更好的人格对齐？

---

## 2. 实验路径总览

```
ALOE 原始 (2.6) → Claude SFT (3.0) → 诊断 3 分墙 → Big Five (3.503)
  → 但 baseline 也 3.500 → 注入无增益
    → FiLM 架构重设计 → 过拟合/崩溃 → v_t 区分度诊断 → 突破 (cos 0.997→0.640)
      → 首次实现 prompt+inject 模式不崩溃 + 人格有区分
```

---

## 3. 各阶段详细结论

### 3.1 ALOE 原始数据（失败）

| 指标 | 值 |
|------|---:|
| 数据 | 2777 条，57 personality |
| 均分 | 2.600 |
| Qwen3-4B baseline | 2.833 |

**结论**：ALOE 训练数据回复质量低于模型自然输出。模型在"学习模仿差的回复"。

### 3.2 Claude 教师数据（突破 baseline）

| 阶段 | 数据 | 均分 |
|------|------|---:|
| Claude SFT Stage 1 | 661 条，14 personality | 3.000 |
| Claude SFT Stage 2（+gate） | 同上 | 3.033 |

**结论**：Claude 数据质量好（教师 4.0），但 SFT 蒸馏将人格特征"磨平"到 3.0。

### 3.3 诊断"3 分墙"

| 实验 | 改动 | 均分 | 结论 |
|------|------|---:|------|
| A1（Embedding Table） | 14 个正交可学习向量替代 encoder | 3.000 | encoder 不是唯一瓶颈 |
| B（gate 强度） | 注入从 12%→50% | 3.000 | gate 强度不是瓶颈 |
| Prompt 升级试验 | 更好的行为约束 prompt | 4.00 | Claude 生成端够好 |
| DPO | chosen/rejected 偏好训练 | 失败 | 80% step margin=0，注入太弱 |

**结论**：问题不在注入机制的强度或编码器架构，在于 personality 描述本身不可区分。

### 3.4 Big Five 结构化人格（突破 3 分）

**核心发现**：ALOE 57 个 personality 在 encoder embedding 空间余弦相似度 0.961（几乎不可区分）。

方案：设计 16 个覆盖 OCEAN 全空间的 Big Five personality，5D 余弦均值 0.037。

| 模型 | 严格 v3 rubric |
|------|---:|
| Big Five PersonaSteer | 3.503 |
| Qwen3-4B Baseline（同 prompt） | 3.500 |
| 差异 | +0.003（无显著增益） |

**结论**：3.033→3.503 的提升完全来自更好的 personality 描述质量，非注入机制。

### 3.5 Probing 层分析

对冻结 Qwen3-4B 做 36 层线性探针（16 personality × 5 contexts = 80 样本）：

| Big Five 维度 | 最强层 | 峰值 R² |
|--------------|---:|---:|
| C（尽责性） | 19 | 0.815 |
| A（宜人性） | 7-21 | 0.711 |
| E（外向性） | 35 | 0.685 |
| O（开放性） | 8-10 | 0.386 |
| N（神经质） | 29 | 0.396 |

推荐注入窗口 **[16..23]**（overall R²=0.559）。

**结论**：人格信息在 hidden state 中线性可解码，FiLM 调制在理论上可操作这些方向。

---

## 4. FiLM v2 架构重设计

### 4.1 架构

```
旧：h' = h + gate_i · Linear(v_t_i)           （加法，1 标量 gate/层）
新：h' = h + Δγ(v_t) ⊙ h + β(v_t)           （Residual FiLM，2560 维/层）
```

- Δγ: `tanh(Linear(v_t)) * gamma_scale`，控制特征放大/压制
- β: `tanh(Linear(v_t)) * beta_scale`，加法偏移
- 自定义 autograd `FiLMInjectFunction`：梯度流向 Δγ/β，h 梯度保留（连接多层）
- 移除 DynamicGate（Δγ 本身是 2560 维细粒度门控）

### 4.2 训练策略：Prompt Dropout + Curriculum

```
Phase 1: prompt_include_rate=0.0（injection-only，逼 HyperNetwork 学人格编码）
Phase 2: prompt_include_rate=0.5（学会与 prompt 协作）
Phase 3: prompt_include_rate=0.7（面向部署微调）
```

### 4.3 Bug 修复历程

| Bug | 现象 | 根因 | 修复 |
|-----|------|------|------|
| 多层梯度断裂 | 只有 layer 7 有非零 Δγ/β | FiLMInjectFunction 返回 None 给 h | 返回 `grad_output * (1+Δγ)` |
| NaN 爆炸 | 训练全 NaN | LayerNorm 将零初始化输出放大到 std=1 → Δγ≈1 → (1+Δγ)^8 爆炸 | 移除 LN，加 tanh 软限制 |
| 生成崩溃（空字符串） | beta_norm=19 破坏 logit 分布 | β 无范围限制，过拟合学到大偏移 | β 加 tanh，降低 scale |
| 重复 token 退化 | "ThankThank..."、"Hey!Hey!" | 199 条数据 FiLM 过拟合 | 扩展数据（进行中） |
| 无人格区分 | 3 个 personality 生成完全相同回复 | ALOE 回复本身通用，FiLM 学到统一调制 | 数据质量 > 数据量 |

### 4.4 v_t 区分度诊断（关键发现）

直接测量 16 个 Big Five personality 的 v_t 余弦相似度：

```
v_t  cos = 0.9971  ← 几乎完全相同！
Δγ   cos = 1.0000  ← 调制参数完全相同
β    cos = 1.0000

根因（三层叠加）：
  ① encoder 输出:  norm=65, cos=0.961（personality 文本太相似）
  ② big5 分支:     norm=4.5（被 encoder 信号淹没，65:4.5 = 14:1）
  ③ big5_gate β:   ≈ 0.15-0.28（Big Five 权重仅 15-28%）
  
  → v_t ≈ 80% × encoder（全部相同） + 20% × big5（有差异但太弱）
  → FiLM 对所有 personality 做完全相同的调制 → 无区分
```

### 4.5 v_t 区分度修复（突破）

三管齐下：

| 修复 | 作用 |
|------|------|
| F.normalize(z_personality) 和 F.normalize(z_big5) | 消除 65:4.5 尺度失衡 |
| big5_gate 初始 bias=+1.0 (sigmoid=0.73) | 初始时 73% 信号来自 Big5 |
| SCL 从 Phase 1 启用 | 对比损失直接推开不同 personality 的 v_t |

**结果**：

```
v_t cos: 0.9971 → 0.6401  (min=0.12, max=0.99)
```

**生成测试（prompt+inject, gamma=0.01, beta=0.02）**：

| Personality | 回复 | 状态 |
|------------|------|:----:|
| Scholar | "I find myself more inclined to engage in thoughtful conversation..." | ✓ 符合人格 |
| Explorer | "Oh wow, that sounds *so* cool! I'm all in!" | ✓ 符合人格 |
| Hermit | "Parties can be so loud and overwhelming. I'd rather stay home..." | ✓ 符合人格 |

**首次实现：prompt+inject 模式不崩溃 + 人格有区分。**

但 inject-only 模式仍无区分（三者生成相同回复）→ FiLM scale 过小（0.01/0.02），注入信号不足以独立驱动人格。

---

## 5. 数据集分析

### 5.1 全部数据集

| 数据集 | 样本数 | 生成方式 | Big5 | 回复质量 | OCEAN 多样性 |
|--------|------:|--------|:---:|:---:|------|
| ALOE 原始 | 2777 | ALOE 自带 | ❌ | ★★☆ (2.6) | cos=0.961 |
| Claude SFT | 661 | Claude 教师 | ❌ | ★★★★☆ (3.0+) | 14 persona |
| Claude DPO | 3305 | Claude 教师 | ❌ | ★★★★☆ | DPO 格式 |
| Big Five 原始 | 471→生成中 | Claude 教师 | ✓ | ★★★★☆ | **cos=0.037** |
| ALOE→Big5 | 2777 | LLM 映射 | ✓(噪声) | ★★☆ | cos=0.798 |
| ALOE 并行 Claude | 生成中 | Claude 教师 | ✓(映射) | ★★★★☆ | cos=0.798 |

### 5.2 ALOE 人格的根本缺陷

```
ALOE 57 个人格全部偏「好人」：
  A(宜人性): 全部 ≥ +0.4（没有冷漠/对抗性人格）
  C(尽责性): mean=+0.67（几乎没有散漫人格）
  N(神经质): mean=-0.36（很少焦虑/情绪化人格）
  
设计 16 个 Big Five 人格覆盖全空间：
  A: [-0.8, +0.9]  |  E: [-0.9, +0.9]  |  N: [-0.9, +0.8]
```

### 5.3 高分 vs 低分回复特征

| 高分 (4-5) | 低分 (3) |
|-----------|---------|
| 具体行为描述（"Bali 冲浪"、"读哲学文本"） | 泛泛而谈（"taking charge"） |
| 角色扮演动作（`*leans back*`） | 跳出角色（"I'm a language model"） |
| 情感与隐喻丰富 | 回复短、功能性 |
| 易高分：Artist, Dreamer, Explorer | 易低分：Scholar, Commander |

---

## 6. 当前状态

### 已解决

- ✅ FiLM 架构实现（Residual FiLM + 自定义 autograd + tanh 限制）
- ✅ 多层梯度链正确传播
- ✅ 数值稳定性（无 NaN）
- ✅ v_t 区分度（0.997 → 0.640）
- ✅ prompt+inject 模式不崩溃 + 人格有区分
- ✅ ALOE 57 personality → Big Five 5D 映射
- ✅ 归一化融合 + gate 偏置 + SCL 组合方案

### 未解决

- ❌ inject-only 模式无人格区分（FiLM scale 过小）
- ❌ 尚未量化 prompt+inject vs prompt-only 的分数差（需 LLM Judge）
- ❌ 训练数据不足且质量参差

### 进行中

- 🔄 Big Five 扩展数据生成（150 inputs × 16 personas → 2400 条，Claude 教师）
- 🔄 ALOE 并行对话 Claude 回复生成（2590/2777）

---

## 7. 下一步计划

### 7.1 近期（数据完成后立即执行）

1. Big Five 扩展数据（~2400 条）训练 FiLM → 评估 prompt+inject vs prompt-only
2. 逐步提高 FiLM scale（0.01→0.02→0.05），找到"有区分 + 不崩溃"的最优区间
3. inject-only 评估：测试注入的独立价值

### 7.2 中期

4. 用 ALOE 并行 Claude 回复数据进一步扩展训练集
5. 多轮对话一致性测试（10-20 轮）
6. 连续人格插值实验（E=+0.9 → E=-0.9）

### 7.3 远期

7. 注意力头级别 Probing（找"人格专用头"）
8. 无 prompt 注入场景的实用价值验证
9. 论文撰写

---

## 8. 核心教训

1. **注入是否有效 = v_t 是否真正不同**，v_t 是否不同 = 输入信号区分度
2. **数据质量 > 数据量**：3248 条 ALOE 回复不如 471 条 Claude 回复有用
3. **encoder 信号会淹没结构化信号**：需要归一化 + gate 偏置 + 对比损失组合修复
4. **FiLM scale 有安全上限**：gamma>0.05 或 beta>0.1 导致 8 层连乘爆炸
5. **SFT 会磨平人格差异**：需要 Prompt Dropout + SCL 创造训练压力

---

## 9. 文件索引

### feature/film-injection 分支

| 文件 | 说明 |
|------|------|
| src/models/injection.py | FiLMInjectFunction + FiLMSteeringInjection |
| src/models/hyper_network.py | 归一化融合 + big5_gate 偏置初始化 |
| src/models/persona_steer.py | injection_type="film" 支持 |
| src/data/aloe_dataset.py | prompt_include_rate (Prompt Dropout) |
| src/training/trainer.py | FiLM 模式优化器/损失简化 |
| scripts/train_film.py | 三阶段课程训练脚本 |
| scripts/eval_film_v2.py | 三线对比评估 |
| scripts/convert_aloe_to_big5.py | ALOE→Big5 数据转换 |
| scripts/build_claude_big5_dataset.py | Claude 回复 + Big5 数据构建 |
| configs/train_film_v2.yaml | FiLM 训练配置 |
| docs/plans/2026-04-21-film-injection-redesign.md | FiLM 设计文档 |
