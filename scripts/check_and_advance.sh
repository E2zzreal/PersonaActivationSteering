#!/bin/bash
# 检查实验状态并在完成后启动下一批

PROJECT_ROOT="/home/kemove/Desktop/PersonaSteer"
cd "$PROJECT_ROOT"

LOG_FILE="logs/pipeline.log"
echo "[$(date)] === 检查实验状态 ===" >> "$LOG_FILE"

# 检查训练进程数量
running=$(pgrep -f "train.py" 2>/dev/null | wc -l)
echo "运行中的训练进程: $running" >> "$LOG_FILE"

# 检查第一批实验完成标志
if [ -f ".batch1_done" ]; then
    batch1_status="done"
elif [ $running -eq 0 ] && [ ! -f ".batch1_done" ]; then
    # 没有运行中的进程，检查第一批是否完成
    batch1_logs=$(ls logs/exp_gate_init_*.log 2>/dev/null | wc -l)
    if [ $batch1_logs -gt 0 ]; then
        # 检查是否有完成的日志
        for log in logs/exp_gate_init_*.log; do
            if grep -q "Training completed" "$log" 2>/dev/null || \
               grep -q "best_loss" "$log" && tail -100 "$log" | grep -q "Epoch 4/4"; then
                echo "第一批实验已完成" >> "$LOG_FILE"
                touch .batch1_done
                batch1_status="done"
                break
            fi
        done
    fi
fi

# 启动第二批实验
if [ -f ".batch1_done" ] && [ ! -f ".batch2_started" ] && [ $running -eq 0 ]; then
    echo "[$(date)] === 启动第二批实验 ===" >> "$LOG_FILE"
    
    for exp in "0:gate_reg_0.001_lr5e5" "1:gate_reg_0.01_lr1e4" "2:gate_reg_0.01_lr3e5" "3:gate_reg_0.05_lr5e5"; do
        IFS=':' read -r gpu_id exp_name <<< "$exp"
        echo "[GPU $gpu_id] 启动: $exp_name" >> "$LOG_FILE"
        
        CUDA_VISIBLE_DEVICES=$gpu_id nohup python scripts/train.py \
            --config "configs/exp_${exp_name}.yaml" \
            > "logs/exp_${exp_name}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
        sleep 2
    done
    
    touch .batch2_started
fi

# 第二批完成后分析并启动Stage3
if [ -f ".batch2_started" ] && [ ! -f ".stage3_started" ] && [ $running -eq 0 ]; then
    echo "[$(date)] === 分析结果并启动Stage3 ===" >> "$LOG_FILE"
    
    # 分析结果
    python3 << 'PYTHON'
import os
import json
import re
from pathlib import Path

results = {}
all_exps = [
    "gate_init_neg1", "gate_init_neg2", "gate_init_neg3", "gate_init_0",
    "gate_reg_0.001_lr5e5", "gate_reg_0.01_lr1e4", "gate_reg_0.01_lr3e5", "gate_reg_0.05_lr5e5"
]

for name in all_exps:
    log_files = list(Path('logs').glob(f'exp_{name}_*.log'))
    if log_files:
        latest_log = max(log_files, key=os.path.getmtime)
        with open(latest_log) as f:
            content = f.read()
        best_match = re.search(r'best_loss=([0-9.]+)(?!.*best_loss)', content)
        if best_match:
            results[name] = float(best_match.group(1))

if results:
    best = min(results.items(), key=lambda x: x[1])
    print(f"最佳实验: {best[0]}, loss={best[1]:.4f}")
    
    with open('logs/best_experiment.json', 'w') as f:
        json.dump({'name': best[0], 'loss': best[1]}, f)
PYTHON

    # 启动Stage3
    if [ -f "logs/best_experiment.json" ]; then
        best_name=$(python3 -c "import json; print(json.load(open('logs/best_experiment.json'))['name'])")
        checkpoint="checkpoints/exp_${best_name}/best.pt"
        
        if [ -f "$checkpoint" ]; then
            echo "使用checkpoint: $checkpoint" >> "$LOG_FILE"
            
            # 创建Stage3配置
            cat > configs/train_stage3_auto.yaml << YAML
# 自动生成的Stage3配置
model:
  inject_layers: [8, 9, 10, 11, 12, 13, 14, 15]
  v_dim: 1024
  hidden_dim: 4096
  layer_dim: 2560

base_model: /home/kemove/Desktop/PersonaSteer/Qwen/Qwen3-4B
stage2_checkpoint: $checkpoint

data:
  tokenizer: /home/kemove/Desktop/PersonaSteer/Qwen/Qwen3-4B
  train_path: data/split/train.jsonl
  eval_path: data/split/val.jsonl
  max_turns: 6
  batch_size: 1

training:
  stage: 3
  num_epochs: 3
  learning_rate: 0.00003
  sft_weight: 1.0
  scl_weight: 0.1
  use_amp: true
  output_dir: checkpoints/stage3_auto
  gate_init_bias: -2.0

device: cuda:0
YAML
            
            CUDA_VISIBLE_DEVICES=0 nohup python scripts/train.py \
                --config configs/train_stage3_auto.yaml \
                > "logs/stage3_auto_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
            
            touch .stage3_started
            echo "Stage3已启动" >> "$LOG_FILE"
        fi
    fi
fi
