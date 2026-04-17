#!/bin/bash
# 重新训练Stage2/3（修复gate约束问题后）
# 修复内容:
#   1. 所有Stage2/3配置文件添加 gate_min_value=0.0, gate_reg_weight=0.0
#   2. trainer.py dual_loss公式修复为惩罚质量下降而非累加
#   3. persona_steer.py generate()添加im_end停止token
set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/retrain_stage2_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "重新训练Stage2/3 (修复gate约束问题)"
echo "开始时间: $(date)"
echo "=========================================="

# 备份旧checkpoints
mkdir -p checkpoints/backup_${TIMESTAMP}
cp checkpoints/stage2/best.pt checkpoints/backup_${TIMESTAMP}/stage2_qwen25_best.pt 2>/dev/null || true
cp checkpoints/stage3/best.pt checkpoints/backup_${TIMESTAMP}/stage3_qwen25_best.pt 2>/dev/null || true
cp checkpoints/stage2_qwen3/best.pt checkpoints/backup_${TIMESTAMP}/stage2_qwen3_best.pt 2>/dev/null || true
cp checkpoints/stage3_qwen3/best.pt checkpoints/backup_${TIMESTAMP}/stage3_qwen3_best.pt 2>/dev/null || true
echo "旧checkpoints已备份到 checkpoints/backup_${TIMESTAMP}/"

echo ""
echo "[1/4] 训练 Qwen2.5 Stage2 (GPU0)..."
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --config configs/train_stage2.yaml \
    --device cuda:0 \
    > "$LOG_DIR/stage2_qwen25.log" 2>&1 &
PID_S2_25=$!

echo "[3/4] 训练 Qwen3 Stage2 (GPU1)..."
CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
    --config configs/train_stage2_qwen3.yaml \
    --device cuda:0 \
    > "$LOG_DIR/stage2_qwen3.log" 2>&1 &
PID_S2_Q3=$!

echo "等待Stage2训练完成..."
wait $PID_S2_25
echo "Qwen2.5 Stage2 完成"
wait $PID_S2_Q3
echo "Qwen3 Stage2 完成"

echo ""
echo "[2/4] 训练 Qwen2.5 Stage3 (GPU0)..."
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --config configs/train_stage3.yaml \
    --device cuda:0 \
    > "$LOG_DIR/stage3_qwen25.log" 2>&1 &
PID_S3_25=$!

echo "[4/4] 训练 Qwen3 Stage3 (GPU1)..."
CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
    --config configs/train_stage3_qwen3.yaml \
    --device cuda:0 \
    > "$LOG_DIR/stage3_qwen3.log" 2>&1 &
PID_S3_Q3=$!

echo "等待Stage3训练完成..."
wait $PID_S3_25
echo "Qwen2.5 Stage3 完成"
wait $PID_S3_Q3
echo "Qwen3 Stage3 完成"

echo ""
echo "=========================================="
echo "所有Stage2/3重训完成: $(date)"
echo "日志位于: $LOG_DIR"
echo "=========================================="
