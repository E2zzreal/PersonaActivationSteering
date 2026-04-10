#!/bin/bash
# PersonaSteer V2 训练监控脚本
# 实时显示各 Stage 的训练进度和 GPU 状态

LOG_DIR=$(ls -td "$HOME/Desktop/Projects/3-PersonaSteer_V2/logs/pipeline_"* 2>/dev/null | head -1)

if [ -z "$LOG_DIR" ]; then
    echo "未找到训练日志目录"
    echo "请先启动训练: bash scripts/auto_train_pipeline.sh"
    exit 1
fi

echo "=============================================="
echo "PersonaSteer V2 训练监控"
echo "日志目录: $LOG_DIR"
echo "=============================================="
echo "按 Ctrl+C 退出监控"
echo ""

# 显示各 Stage 最新日志
show_stage_log() {
    local name=$1
    local log=$2

    echo "=== $name ==="
    if [ -f "$log" ]; then
        tail -5 "$log"
    else
        echo "日志文件不存在"
    fi
    echo ""
}

# 循环监控
while true; do
    echo "=============================================="
    echo "更新时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================="

    echo ""
    show_stage_log "Stage 1 Qwen2.5 GPU0" "$LOG_DIR/stage1_qwen25_gpu0.log"
    show_stage_log "Stage 1 Qwen2.5 GPU1" "$LOG_DIR/stage1_qwen25_gpu1.log"
    show_stage_log "Stage 1 Qwen2.5 GPU2" "$LOG_DIR/stage1_qwen25_gpu2.log"
    show_stage_log "Stage 1 Qwen2.5 GPU3" "$LOG_DIR/stage1_qwen25_gpu3.log"

    show_stage_log "Stage 2 Qwen2.5" "$LOG_DIR/stage2_qwen25.log"
    show_stage_log "Stage 2 Qwen3" "$LOG_DIR/stage2_qwen3.log"

    show_stage_log "Stage 3 Qwen2.5" "$LOG_DIR/stage3_qwen25.log"
    show_stage_log "Stage 3 Qwen3" "$LOG_DIR/stage3_qwen3.log"

    echo "=== GPU 状态 ==="
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F', ' '{printf "GPU %s (%s): %s%%, %s/%s MB\n", $1, $2, $3, $4, $5}'

    echo ""
    echo "----------------------------------------------"

    sleep 10
done