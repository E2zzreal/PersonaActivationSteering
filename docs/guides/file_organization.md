# PersonaSteer 文件组织规范

## 目录结构

```
PersonaSteer/
├── checkpoints/          # 模型检查点（保留策略见下）
├── configs/             # 配置文件（全部保留）
├── data/                # 数据文件（全部保留）
├── docs/                # 文档（全部保留）
├── logs/                # 训练日志（精简保留）
├── results/             # 生成结果（精选保留）
├── scripts/             # 脚本（全部保留）
├── src/                 # 源代码（全部保留）
└── Qwen/                # 模型文件（全部保留）
```

---

## Checkpoints 保留策略

### 保留规则

| 模型类型 | 保留文件 | 删除文件 |
|---------|---------|---------|
| **成功模型** | best.pt, epoch_*.pt | - |
| **失败模型** | best.pt（用于分析） | epoch_*.pt |
| **Stage1** | best.pt, epoch_*.pt | - |

### 当前保留的Checkpoints

#### Stage 1 (基础训练)
- ✅ `stage1/best.pt` + epoch_*.pt
- ✅ `stage1_qwen3/best.pt` + epoch_*.pt

#### Stage 2 (Gate实验)
- ✅ `exp_gate_init_neg3/best.pt` + epoch_*.pt （成功）
- ⚠️ `exp_gate_init_0/best.pt` （失败，仅保留best.pt）
- ⚠️ `exp_gate_init_neg1/best.pt` （失败，仅保留best.pt）
- ⚠️ `exp_gate_init_neg2/best.pt` （失败，仅保留best.pt）
- ✅ `exp_gate_reg_0.001_lr5e5/best.pt` + epoch_*.pt （成功）
- ✅ `exp_gate_reg_0.01_lr1e4/best.pt` + epoch_*.pt （成功）
- ⚠️ `exp_gate_reg_0.01_lr3e5/best.pt` （失败，仅保留best.pt）
- ⚠️ `exp_gate_reg_0.05_lr5e5/best.pt` （失败，仅保留best.pt）

#### Stage 3 (对比学习)
- ✅ `stage3_auto/best.pt` + epoch_*.pt
- ✅ `stage3_gate_init_0/best.pt` + epoch_*.pt
- ✅ `stage3_gate_reg_0.01_lr1e4/best.pt` + epoch_*.pt
- ⚠️ `stage3_gate_reg_0.05_lr5e5/best.pt` （失败，仅保留best.pt）

**总大小**: ~331GB（已清理182GB）

---

## Results 保留策略

### 保留规则

| 文件类型 | 保留策略 |
|---------|---------|
| **对话文件** | 只保留最新批次（20260410_103948） |
| **评估文件** | 保留最新评估结果 |
| **失败模型** | 删除生成结果（空回复无价值） |

### 当前保留的生成结果

#### ✅ 成功模型（8个）
1. `conversations_baseline_20260410_103948.json`
2. `conversations_stage1_20260410_103948.json`
3. `conversations_stage1_qwen3_20260410_103948.json`
4. `conversations_exp_gate_init_neg3_20260410_103948.json`
5. `conversations_exp_gate_reg_0.001_lr5e5_20260410_103948.json`
6. `conversations_exp_gate_reg_0.01_lr1e4_20260410_103948.json`
7. `conversations_stage3_auto_20260410_103948.json`
8. `conversations_stage3_gate_init_0_20260410_103948.json`
9. `conversations_stage3_gate_reg_0.01_lr1e4_20260410_103948.json`

#### ❌ 已删除（失败模型）
- ~~conversations_exp_gate_init_0_*.json~~ （空回复）
- ~~conversations_exp_gate_init_neg1_*.json~~ （空回复）
- ~~conversations_exp_gate_init_neg2_*.json~~ （空回复）
- ~~conversations_exp_gate_reg_0.01_lr3e5_*.json~~ （空回复）
- ~~conversations_exp_gate_reg_0.05_lr5e5_*.json~~ （空回复）
- ~~conversations_stage3_gate_reg_0.05_lr5e5_*.json~~ （空回复）

#### ❌ 已删除（旧批次）
- ~~conversations_*_20260410_074204.json~~ （旧批次）
- ~~conversations_*_20260405_*.json~~ （旧批次）
- ~~conversations_*_20260404_*.json~~ （旧批次）

---

## Logs 保留策略

### 保留规则

| 日志类型 | 保留策略 |
|---------|---------|
| **训练日志** | 保留每个模型的最新日志（184xxx批次） |
| **生成日志** | 保留pipeline日志 |
| **评估日志** | 保留最新评估日志 |
| **中间日志** | 删除 |

### 当前保留的日志

#### 训练日志
- `exp_gate_init_0_20260403_184208.log`
- `exp_gate_init_0_20260403_184715.log`
- `exp_gate_init_neg1_20260403_184202.log`
- `exp_gate_init_neg1_20260403_184706.log`
- `exp_gate_init_neg2_20260403_184204.log`
- `exp_gate_init_neg2_20260403_184709.log`
- `exp_gate_init_neg3_20260403_184206.log`
- `exp_gate_init_neg3_20260403_184712.log`
- `stage3_auto_20260404_081448.log`
- `stage3_gate_init_0_20260404_081923.log`
- `stage3_gate_reg_0.01_lr1e4_20260404_081926.log`
- `stage3_gate_reg_0.05_lr5e5_20260404_081929.log`

#### 生成日志
- `generation_pipeline_20260410_103948.log`
- `generate_fixed_20260410_102415.log`

#### 评估日志
- `eval_all_models_mock_20260410_153400.log`
- `eval_all_models_20260410_153117.log`
- `eval_remaining.log`

#### 其他
- `auto_test_20260404_084045.log`
- `pipeline_nohup.log`

#### ❌ 已删除的日志类型
- ~~exp_*_20260403_175*.log~~ （旧批次训练）
- ~~eval_stage3_20260405_*.log~~ （旧评估）
- ~~eval_baselines_20260405_*.log~~ （旧评估）
- ~~batch_generate_*.log~~ （中间日志）
- ~~auto_pipeline_*.log~~ （中间日志）
- ~~pipeline.log~~ （中间日志）

---

## 存储统计

| 目录 | 清理前 | 清理后 | 节省 |
|------|--------|--------|------|
| checkpoints/ | 513GB | 331GB | 182GB |
| results/ | ~2GB | ~500MB | ~1.5GB |
| logs/ | ~5GB | ~2GB | ~3GB |
| **总计** | **~520GB** | **~333GB** | **~187GB** |

---

## 备份建议

### 重要文件（建议备份）
1. `checkpoints/stage1_qwen3/best.pt` - 最佳模型
2. `checkpoints/stage3_auto/best.pt` - 完整训练模型
3. `results/conversations_stage1_qwen3_20260410_103948.json` - 最佳生成结果
4. `docs/` - 所有文档

### 可删除文件（如空间不足）
1. 失败模型的best.pt（保留1-2个用于分析）
2. 成功模型的epoch_*.pt（只保留best.pt）
3. 旧日志文件

---

*整理时间: 2026-04-10*
