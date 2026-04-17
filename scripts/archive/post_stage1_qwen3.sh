#!/bin/bash
# Qwen3 Stage1 完成后的自动操作：
#   1. 将 checkpoint 归位到 checkpoints/stage1_qwen3/
#   2. 用修复后的配置重新启动 Stage1 训练（从 scratch，使用 query-aware 新配置）
#   3. 同时在 GPU2 启动 Qwen3 baseline 评估

set -e
cd /home/kemove/Desktop/PersonaSteer
source /home/kemove/anaconda3/bin/activate pytorch

LOG_TAG=$(date +%Y%m%d_%H%M%S)

echo "[post_stage1_qwen3] Started at $(date)"

# Step 1: 确认 Epoch 3 checkpoint 是否已保存到根目录（旧 output_dir 问题）
echo "[post_stage1_qwen3] Checking checkpoints..."
if [ -f checkpoints/best.pt ] && [ -f checkpoints/epoch_3.pt ]; then
    echo "[post_stage1_qwen3] Moving checkpoints to stage1_qwen3/"
    mkdir -p checkpoints/stage1_qwen3
    cp checkpoints/best.pt    checkpoints/stage1_qwen3/best.pt
    cp checkpoints/epoch_1.pt checkpoints/stage1_qwen3/epoch_1.pt
    cp checkpoints/epoch_2.pt checkpoints/stage1_qwen3/epoch_2.pt
    cp checkpoints/epoch_3.pt checkpoints/stage1_qwen3/epoch_3.pt
    echo "[post_stage1_qwen3] Checkpoints copied to stage1_qwen3/"
else
    echo "[post_stage1_qwen3] Checkpoints already in place or not found in root. Skipping copy."
fi

# Step 2: 启动 Qwen3 Stage2 训练（使用修复后的 yaml）
echo "[post_stage1_qwen3] Launching Qwen3 Stage2 training on GPU1..."
CUDA_VISIBLE_DEVICES=1 nohup python scripts/train.py \
    --config configs/train_stage2_qwen3.yaml \
    > logs/train_stage2_qwen3_${LOG_TAG}.log 2>&1 &
STAGE2_PID=$!
echo "[post_stage1_qwen3] Qwen3 Stage2 PID: ${STAGE2_PID}"

# Step 3: Qwen3 baseline eval 已提前手动启动，此处仅记录
echo "[post_stage1_qwen3] Note: Qwen3 baseline eval was already launched manually (PID 2765406)"

echo "[post_stage1_qwen3] Done. Stage2 PID=${STAGE2_PID}, Eval PID=${EVAL_PID}"
