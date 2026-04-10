# PersonaSteer V2 实习生操作手册

**版本**: 2026-03-30 v2
**适用**: Query-Aware HyperNetwork (方案A) 训练监控与评估

**⚠️ 重要提示**:
- 训练完成后**不会自动评估**，需要手动启动
- Stage1 完成后**不会自动启动 Stage2**，需要手动启动
- 每个阶段完成后都需要人工确认和启动下一步

---

## 1. 训练监控

### 1.1 检查训练进程

```bash
# 查看训练进程
ps aux | grep "train.py" | grep -v grep

# 查看 GPU 使用情况
nvidia-smi

# 预期输出：
# - Qwen2.5: PID 2751849, GPU 0, ~14GB
# - Qwen3: PID 2756250, GPU 1, ~16GB
```

### 1.2 查看训练进度

```bash
# Qwen2.5 进度
tail -50 logs/train_stage1_qwen25_20260330_112754.log | grep "Epoch"

# Qwen3 进度
tail -50 logs/train_stage1_qwen3_20260330_113212.log | grep "Epoch"

# 查找最新 loss
grep "Epoch [0-9]/3:" logs/train_stage1_qwen25_*.log | tail -1
```

### 1.3 训练完成标志

```bash
# 检查是否保存 checkpoint
ls -lh checkpoints/stage1/best.pt
ls -lh checkpoints/stage1_qwen3/best.pt

# 查看训练日志最后几行
tail -20 logs/train_stage1_qwen25_20260330_112754.log
```

**完成标志**:
- 日志显示 `Epoch 3/3: 100%`
- 出现 `Checkpoint saved to checkpoints/*/best.pt`
- 进程自动退出

---

## 2. 训练完成后操作

### 2.1 确认 Checkpoint

```bash
# 检查文件大小和时间
ls -lh checkpoints/stage1/
ls -lh checkpoints/stage1_qwen3/

# 预期：每个目录有 4 个文件
# - best.pt
# - epoch_1.pt
# - epoch_2.pt
# - epoch_3.pt
```

### 2.2 启动评估

#### Qwen2.5 评估

```bash
cd /home/kemove/Desktop/Projects/3-PersonaSteer_V2

# 激活环境
source /home/kemove/anaconda3/bin/activate pytorch

# 启动评估
nohup python scripts/evaluate.py \
  --checkpoint checkpoints/stage1/best.pt \
  --config configs/eval.yaml \
  --output results/stage1_qwen25_eval_$(date +%Y%m%d).json \
  > logs/eval_stage1_qwen25_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "Qwen2.5 评估 PID: $!"
```

#### Qwen3 评估

```bash
# 启动评估
nohup python scripts/evaluate.py \
  --checkpoint checkpoints/stage1_qwen3/best.pt \
  --config configs/eval.yaml \
  --output results/stage1_qwen3_eval_$(date +%Y%m%d).json \
  > logs/eval_stage1_qwen3_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "Qwen3 评估 PID: $!"
```

### 2.3 监控评估进度

```bash
# 查看评估日志
tail -f logs/eval_stage1_qwen25_*.log

# 查看评估结果
ls -lh results/stage1_*_eval_*.json
```

---

## 3. Stage2/3 训练启动

**⚠️ 重要**: Stage1 评估完成后，需要手动启动 Stage2 训练。

### 3.1 启动 Stage2 训练

```bash
cd /home/kemove/Desktop/Projects/3-PersonaSteer_V2
source /home/kemove/anaconda3/bin/activate pytorch

# Qwen2.5 Stage2
nohup python scripts/train.py \
  --config configs/train_stage2.yaml \
  > logs/train_stage2_qwen25_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "Qwen2.5 Stage2 PID: $!"

# Qwen3 Stage2 (使用 GPU1)
CUDA_VISIBLE_DEVICES=1 nohup python scripts/train.py \
  --config configs/train_stage2_qwen3.yaml \
  > logs/train_stage2_qwen3_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "Qwen3 Stage2 PID: $!"
```

### 3.2 启动 Stage3 训练

**Stage2 训练和评估完成后**:

```bash
# Qwen2.5 Stage3
nohup python scripts/train.py \
  --config configs/train_stage3.yaml \
  > logs/train_stage3_qwen25_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Qwen3 Stage3 (使用 GPU1)
CUDA_VISIBLE_DEVICES=1 nohup python scripts/train.py \
  --config configs/train_stage3_qwen3.yaml \
  > logs/train_stage3_qwen3_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## 4. 结果记录

### 3.1 提取关键指标

```bash
# 查看评估结果
cat results/stage1_qwen25_eval_*.json | grep -E "AL\(K\)|loss|accuracy"

# 或使用 jq (如果安装)
jq '.metrics' results/stage1_qwen25_eval_*.json
```

### 3.2 更新 Taskboard

编辑 `taskboard_update.md`，添加：

```markdown
## 训练完成 - 2026-03-30

### Qwen2.5-3B
- 训练时长: XX 小时
- 最终 Loss: X.XXXX
- Checkpoint: checkpoints/stage1/best.pt

### Qwen3-4B
- 训练时长: XX 小时
- 最终 Loss: X.XXXX
- Checkpoint: checkpoints/stage1_qwen3/best.pt

### 评估结果
- AL(K)_AVG: X.XX
- 对比基线: +X.XX
```

---

## 5. 常见问题

### 4.1 训练进程意外退出

```bash
# 查看日志最后 50 行
tail -50 logs/train_stage1_qwen25_*.log

# 常见错误：
# - OOM: 显存不足
# - CUDA error: GPU 故障
# - 需要联系导师
```

### 4.2 评估失败

```bash
# 检查 checkpoint 是否存在
ls -lh checkpoints/stage1/best.pt

# 检查配置文件
cat configs/eval.yaml

# 重新运行评估（见 2.2）
```

### 4.3 磁盘空间不足

```bash
# 检查磁盘空间
df -h /home/kemove

# 如果空间不足，删除旧归档
rm -rf checkpoints/archive_20260330_pre_queryaware/
```

---

## 6. 联系方式

**遇到以下情况请立即联系导师**:
- 训练进程异常退出
- GPU 错误或硬件故障
- 评估结果异常（loss > 10 或 < 0.1）
- 不确定如何操作

**正常情况无需打扰**:
- 训练正常进行中
- 评估正常运行中

---

## 7. 快速检查清单

**每 2 小时检查一次**:
- [ ] 训练进程是否运行
- [ ] GPU 使用率是否正常 (30-50%)
- [ ] 日志是否有错误信息

**Stage1 训练完成后**:
- [ ] 确认 checkpoint 文件存在 (4个文件)
- [ ] 启动 Stage1 评估脚本
- [ ] 记录训练时长和最终 loss
- [ ] 更新 taskboard_update.md

**Stage1 评估完成后**:
- [ ] 提取 AL(K)_AVG 等关键指标
- [ ] 更新文档记录评估结果
- [ ] **启动 Stage2 训练** (见第3节)

**Stage2/3 完成后**:
- [ ] 重复上述流程
- [ ] 每个阶段都需要手动启动下一步
- [ ] 通知导师阶段性结果

---

## 附录: 文件路径速查

```
训练日志:
  logs/train_stage1_qwen25_20260330_112754.log
  logs/train_stage1_qwen3_20260330_113212.log

Checkpoint:
  checkpoints/stage1/best.pt
  checkpoints/stage1_qwen3/best.pt

配置文件:
  configs/train_stage1.yaml
  configs/train_stage1_qwen3.yaml
  configs/eval.yaml

结果输出:
  results/stage1_qwen25_eval_*.json
  results/stage1_qwen3_eval_*.json
```
