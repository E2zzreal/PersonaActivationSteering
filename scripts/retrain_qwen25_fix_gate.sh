#!/bin/bash
set -e

PROJECT_ROOT="/home/kemove/Desktop/PersonaSteer"
PYTHON="python"

cd "$PROJECT_ROOT"

echo "=== 重新训练Qwen2.5 Stage2和Stage3 ==="

# 备份
mv checkpoints/stage2_v2 checkpoints/stage2_v2_backup_gatemax03 || true
mv checkpoints/stage3_v2 checkpoints/stage3_v2_backup_gatemax03 || true

# Stage2
CUDA_VISIBLE_DEVICES=0,1 $PYTHON scripts/train.py \
    --config configs/train_stage2.yaml \
    2>&1 | tee logs/stage2_qwen25_v3_$(date +%Y%m%d_%H%M%S).log

# Stage3
CUDA_VISIBLE_DEVICES=0,1 $PYTHON scripts/train.py \
    --config configs/train_stage3.yaml \
    2>&1 | tee logs/stage3_qwen25_v3_$(date +%Y%m%d_%H%M%S).log

echo "Qwen2.5训练完成"
