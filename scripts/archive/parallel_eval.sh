#!/bin/bash
# 并行评估脚本 - 使用旧checkpoint进行评估

set -e

# 激活虚拟环境
source /home/kemove/Desktop/PersonaSteer/venv/bin/activate

echo "========================================="
echo "PersonaSteer V2 - 并行评估"
echo "GPU 1: Qwen2.5-3B Baseline对比"
echo "GPU 3: Qwen3-4B Baseline对比"
echo "========================================="

# GPU 1: Qwen2.5-3B评估
(
    export CUDA_VISIBLE_DEVICES=1
    echo "[GPU 1] 开始评估Qwen2.5-3B..."
    
    python scripts/complete_comparison_eval.py \
        --model qwen25 \
        --num_samples 50 \
        --device cuda:0 \
        > logs/eval_qwen25_gpu1.log 2>&1
    
    echo "[GPU 1] Qwen2.5-3B评估完成"
) &

PID1=$!

# GPU 3: Qwen3-4B评估
(
    export CUDA_VISIBLE_DEVICES=3
    echo "[GPU 3] 开始评估Qwen3-4B..."
    
    python scripts/complete_comparison_eval.py \
        --model qwen3 \
        --num_samples 50 \
        --device cuda:0 \
        > logs/eval_qwen3_gpu3.log 2>&1
    
    echo "[GPU 3] Qwen3-4B评估完成"
) &

PID2=$!

echo ""
echo "并行评估已启动:"
echo "  GPU 1 (Qwen2.5-3B): PID $PID1"
echo "  GPU 3 (Qwen3-4B): PID $PID2"
echo ""
echo "日志文件:"
echo "  logs/eval_qwen25_gpu1.log"
echo "  logs/eval_qwen3_gpu3.log"
echo ""

# 等待两个进程完成
wait $PID1 $PID2

echo ""
echo "========================================="
echo "✅ 全部评估完成！"
echo "查看结果:"
echo "  tail -100 logs/eval_qwen25_gpu1.log"
echo "  tail -100 logs/eval_qwen3_gpu3.log"
echo "========================================="
