#!/bin/bash
# Qwen3-4B训练脚本 - 强制使用GPU 2

set -e

# 激活虚拟环境
source /home/kemove/Desktop/PersonaSteer/venv/bin/activate

# 强制使用GPU 2
export CUDA_VISIBLE_DEVICES=2

# 运行训练
python scripts/train.py \
    --config configs/train_stage1_qwen3.yaml \
    --device cuda \
    --output checkpoints/stage1_qwen3_fixed
