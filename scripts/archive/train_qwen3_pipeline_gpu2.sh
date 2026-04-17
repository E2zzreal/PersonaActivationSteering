#!/bin/bash
# Qwen3-4B完整训练流水线 - 强制使用GPU 2

set -e

# 激活虚拟环境
source /home/kemove/Desktop/PersonaSteer/venv/bin/activate

# 强制使用GPU 2
export CUDA_VISIBLE_DEVICES=2

echo "========================================="
echo "PersonaSteer V2 - Qwen3-4B 训练流水线"
echo "强制使用物理GPU 2"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__}')"
echo "========================================="

# Stage 1
echo ""
echo "=== Stage 1 / 3 ==="
python scripts/train.py \
    --config configs/train_stage1_qwen3.yaml \
    --device cuda \
    --output checkpoints/stage1_qwen3_fixed

if [ $? -ne 0 ]; then
    echo "❌ Stage 1 失败"
    exit 1
fi

# Stage 2
echo ""
echo "=== Stage 2 / 3 ==="
python scripts/train.py \
    --config configs/train_stage2_qwen3.yaml \
    --resume checkpoints/stage1_qwen3_fixed/best.pt \
    --device cuda \
    --output checkpoints/stage2_qwen3_fixed

if [ $? -ne 0 ]; then
    echo "❌ Stage 2 失败"
    exit 1
fi

# Stage 3
echo ""
echo "=== Stage 3 / 3 ==="
python scripts/train.py \
    --config configs/train_stage3_qwen3.yaml \
    --resume checkpoints/stage2_qwen3_fixed/best.pt \
    --device cuda \
    --output checkpoints/stage3_qwen3_fixed

if [ $? -ne 0 ]; then
    echo "❌ Stage 3 失败"
    exit 1
fi

echo ""
echo "========================================="
echo "✅ 全部训练完成！"
echo "Checkpoints:"
echo "  - checkpoints/stage1_qwen3_fixed"
echo "  - checkpoints/stage2_qwen3_fixed"
echo "  - checkpoints/stage3_qwen3_fixed"
echo "========================================="
