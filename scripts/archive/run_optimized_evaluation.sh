#!/bin/bash
# 优化评估流程：先生成对话，再批量评估
set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/eval_${TIMESTAMP}"
CONV_DIR="results/conversations_${TIMESTAMP}"
mkdir -p "$LOG_DIR" "$CONV_DIR"

echo "=========================================="
echo "阶段1: 生成所有对话 (4 GPU并行) - $(date)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python scripts/generate_all_conversations.py --config configs/train_stage1.yaml --checkpoint baseline --output "$CONV_DIR/baseline_qwen25.json" --gpu 0 > "$LOG_DIR/gen_baseline_qwen25.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/generate_all_conversations.py --config configs/train_stage1_qwen3.yaml --checkpoint baseline --output "$CONV_DIR/baseline_qwen3.json" --gpu 0 > "$LOG_DIR/gen_baseline_qwen3.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/generate_all_conversations.py --config configs/train_stage1.yaml --checkpoint checkpoints/stage1/best.pt --output "$CONV_DIR/stage1_qwen25.json" --gpu 0 > "$LOG_DIR/gen_stage1_qwen25.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 python scripts/generate_all_conversations.py --config configs/train_stage1_qwen3.yaml --checkpoint checkpoints/stage1_qwen3/best.pt --output "$CONV_DIR/stage1_qwen3.json" --gpu 0 > "$LOG_DIR/gen_stage1_qwen3.log" 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 python scripts/generate_all_conversations.py --config configs/train_stage2.yaml --checkpoint checkpoints/stage2/best.pt --output "$CONV_DIR/stage2_qwen25.json" --gpu 0 > "$LOG_DIR/gen_stage2_qwen25.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/generate_all_conversations.py --config configs/train_stage2_qwen3.yaml --checkpoint checkpoints/stage2_qwen3/best.pt --output "$CONV_DIR/stage2_qwen3.json" --gpu 0 > "$LOG_DIR/gen_stage2_qwen3.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/generate_all_conversations.py --config configs/train_stage3.yaml --checkpoint checkpoints/stage3/best.pt --output "$CONV_DIR/stage3_qwen25.json" --gpu 0 > "$LOG_DIR/gen_stage3_qwen25.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 python scripts/generate_all_conversations.py --config configs/train_stage3_qwen3.yaml --checkpoint checkpoints/stage3_qwen3/best.pt --output "$CONV_DIR/stage3_qwen3.json" --gpu 0 > "$LOG_DIR/gen_stage3_qwen3.log" 2>&1 &
wait

echo "对话生成完成 - $(date)"
echo ""
echo "=========================================="
echo "阶段2: 批量评估 (V1/V3并行)"
echo "=========================================="

# TODO: 修改judge脚本支持读取预生成对话
echo "评估脚本需要修改以支持读取预生成对话"
echo "完成时间: $(date)"
