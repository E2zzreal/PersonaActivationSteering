#!/bin/bash
# 重新训练Stage2和Stage3（修复gate_max问题）

set -e

PROJECT_ROOT="/home/kemove/Desktop/PersonaSteer"
PYTHON="python"
LOG_DIR="$PROJECT_ROOT/logs"

cd "$PROJECT_ROOT"

echo "=== 重新训练Stage2和Stage3（修复gate_max=0.3问题） ==="
echo "开始时间: $(date)"

# 备份旧checkpoint
echo "[$(date)] 备份旧checkpoint..."
mv checkpoints/stage2_qwen3_v2 checkpoints/stage2_qwen3_v2_backup_gatemax03 || true
mv checkpoints/stage3_qwen3_v2 checkpoints/stage3_qwen3_v2_backup_gatemax03 || true

# Stage2训练
echo "[$(date)] Stage2 训练开始..."
CUDA_VISIBLE_DEVICES=2,3 $PYTHON scripts/train.py \
    --config configs/train_stage2_qwen3.yaml \
    2>&1 | tee "$LOG_DIR/stage2_qwen3_v3_$(date +%Y%m%d_%H%M%S).log"

# Stage3训练
echo "[$(date)] Stage3 训练开始..."
CUDA_VISIBLE_DEVICES=2,3 $PYTHON scripts/train.py \
    --config configs/train_stage3_qwen3.yaml \
    2>&1 | tee "$LOG_DIR/stage3_qwen3_v3_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date)] 训练完成"
echo "结束时间: $(date)"

# 测试生成质量
echo "[$(date)] 测试生成质量..."
$PYTHON scripts/test_generation_quality_v2.py 2>&1 | tee results/generation_quality_v3_$(date +%Y%m%d_%H%M%S).log
