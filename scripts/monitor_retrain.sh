#!/bin/bash
# 监控训练进度

LOG_FILE=$(ls -t logs/retrain_*.log | head -1)

echo "监控训练日志: $LOG_FILE"
echo "按Ctrl+C退出监控"
echo ""

tail -f "$LOG_FILE" | grep --line-buffered -E "Epoch|loss=|Best|Saved|训练|完成"
