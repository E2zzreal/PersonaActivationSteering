#!/bin/bash
# PersonaSteer 自动推进脚本
# 监控Stage 1完成后自动启动Stage 2和Stage 3

set -e

PROJECT_DIR="/home/kemove/Desktop/PersonaSteer"
cd $PROJECT_DIR
source venv/bin/activate

MODEL=$1  # qwen25 或 qwen3
GPU=$2    # cuda:0 或 cuda:2

if [ -z "$MODEL" ] || [ -z "$GPU" ]; then
    echo "Usage: $0 <model> <gpu>"
    echo "  model: qwen25 或 qwen3"
    echo "  gpu: cuda:0 或 cuda:2"
    exit 1
fi

# 设置配置文件
if [ "$MODEL" == "qwen25" ]; then
    STAGE1_CONFIG="configs/train_stage1.yaml"
    STAGE2_CONFIG="configs/train_stage2.yaml"
    STAGE3_CONFIG="configs/train_stage3.yaml"
    STAGE1_CKPT="checkpoints/stage1/best.pt"
    STAGE2_CKPT="checkpoints/stage2/best.pt"
else
    STAGE1_CONFIG="configs/train_stage1_qwen3.yaml"
    STAGE2_CONFIG="configs/train_stage2_qwen3.yaml"
    STAGE3_CONFIG="configs/train_stage3_qwen3.yaml"
    STAGE1_CKPT="checkpoints/stage1_qwen3/best.pt"
    STAGE2_CKPT="checkpoints/stage2_qwen3/best.pt"
fi

echo "=== PersonaSteer 自动推进脚本 ==="
echo "模型: $MODEL"
echo "GPU: $GPU"
echo ""

# 函数：等待checkpoint
wait_for_checkpoint() {
    local ckpt_path=$1
    local stage=$2
    echo "[$(date '+%H:%M:%S')] 等待 $stage checkpoint: $ckpt_path"
    
    while [ ! -f "$ckpt_path" ]; do
        sleep 30
    done
    
    echo "[$(date '+%H:%M:%S')] ✅ $stage checkpoint 已保存!"
}

# 函数：运行训练
run_training() {
    local config=$1
    local stage=$2
    
    echo ""
    echo "=== 开始 $stage 训练 ==="
    echo "[$(date '+%H:%M:%S')] 配置: $config"
    
    python scripts/train.py --config "$config" 2>&1 | tee "logs/${MODEL}_${stage}.log"
    
    echo "[$(date '+%H:%M:%S')] ✅ $stage 训练完成"
}

# 创建日志目录
mkdir -p logs

# 等待Stage 1完成
wait_for_checkpoint "$STAGE1_CKPT" "Stage 1"

# 启动Stage 2
sleep 10
run_training "$STAGE2_CONFIG" "Stage 2"

# 启动Stage 3
sleep 10
run_training "$STAGE3_CONFIG" "Stage 3"

echo ""
echo "=== 所有Stage训练完成! ==="
echo "[$(date '+%H:%M:%S')] 模型 $MODEL 训练完成"
