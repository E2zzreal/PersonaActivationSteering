#!/bin/bash
# 训练监控脚本

echo "=========================================="
echo "PersonaSteer Training Monitor"
echo "Time: $(date)"
echo "=========================================="
echo ""

# GPU 状态
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv
echo ""

# 训练进程
echo "=== Training Processes ==="
ps aux | grep "train.py" | grep -v grep | wc -l
echo "processes running"
echo ""

# 各任务进度
echo "=== Training Progress ==="
for job in qwen3_neuroticism qwen3_minimal qwen3_baseline qwen25_baseline; do
  for stage in 1 2 3; do
    log_file="logs/training/$job/stage$stage.log"
    if [ -f "$log_file" ]; then
      # 获取epoch进度
      epoch=$(grep -oP "Epoch \d+/\d+" "$log_file" 2>/dev/null | tail -1)
      pct=$(grep -oP "\d+%" "$log_file" 2>/dev/null | tail -1)
      loss=$(grep -oP "loss=[\d.]+" "$log_file" 2>/dev/null | tail -1)
      echo "[$job] Stage $stage: $epoch $pct $loss"
    fi
  done
done
echo ""

echo "=========================================="
