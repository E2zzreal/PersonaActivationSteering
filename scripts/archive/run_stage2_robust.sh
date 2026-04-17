#!/bin/bash
# PersonaSteer Stage 2 稳健训练脚本
# 使用 nohup + disown 确保进程不被中断

cd /home/kemove/Desktop/PersonaSteer

# 激活虚拟环境
source venv/bin/activate

# 日志文件
LOG_DIR="logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M)

echo "========================================"
echo "PersonaSteer Stage 2 Training"
echo "Start Time: $(date)"
echo "========================================"

# Qwen2.5-3B Stage 2 (GPU 0)
echo "Starting Qwen2.5-3B Stage 2 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nohup python scripts/train.py \
    --config configs/train_stage2.yaml \
    --resume checkpoints/stage1/best.pt \
    > ${LOG_DIR}/train_stage2_qwen25_${TIMESTAMP}.log 2>&1 &
PID_QWEN25=$!
echo "Qwen2.5-3B PID: $PID_QWEN25"
disown $PID_QWEN25

# 等待5秒确保第一个进程稳定
sleep 5

# Qwen3-4B Stage 2 (GPU 2)
echo "Starting Qwen3-4B Stage 2 on GPU 2..."
CUDA_VISIBLE_DEVICES=2 nohup python scripts/train.py \
    --config configs/train_stage2_qwen3.yaml \
    > ${LOG_DIR}/train_stage2_qwen3_${TIMESTAMP}.log 2>&1 &
PID_QWEN3=$!
echo "Qwen3-4B PID: $PID_QWEN3"
disown $PID_QWEN3

echo "========================================"
echo "Both training processes started!"
echo "Qwen2.5-3B PID: $PID_QWEN25"
echo "Qwen3-4B PID: $PID_QWEN3"
echo "========================================"
echo "Logs:"
echo "  - ${LOG_DIR}/train_stage2_qwen25_${TIMESTAMP}.log"
echo "  - ${LOG_DIR}/train_stage2_qwen3_${TIMESTAMP}.log"
echo "========================================"

# 保存PID到文件
echo $PID_QWEN25 > ${LOG_DIR}/stage2_qwen25.pid
echo $PID_QWEN3 > ${LOG_DIR}/stage2_qwen3.pid

echo "PIDs saved to ${LOG_DIR}/stage2_*.pid"
