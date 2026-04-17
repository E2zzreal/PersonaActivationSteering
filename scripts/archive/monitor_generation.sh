#!/bin/bash
# 监控生成进度

echo "========================================"
echo "生成进度监控"
echo "========================================"
echo ""

# 检查进程
echo "进程状态:"
ps aux | grep run_generation_pipeline | grep -v grep | head -2
echo ""

# 检查GPU
echo "GPU状态:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
echo ""

# 检查最新日志
echo "最新日志:"
LATEST_LOG=$(ls -t logs/generation_pipeline_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "日志文件: $LATEST_LOG"
    echo "---"
    tail -20 "$LATEST_LOG"
else
    echo "暂无日志文件"
fi
echo ""

# 统计已生成的文件
echo "已生成文件:"
ls -lt results/conversations_*_20260410_*.json 2>/dev/null | wc -l
echo "个对话文件"
echo ""

# 检查是否有错误
echo "错误检查:"
if [ -n "$LATEST_LOG" ]; then
    grep -c "✗" "$LATEST_LOG" 2>/dev/null || echo "0个错误"
fi
echo ""

echo "========================================"
echo "监控时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
