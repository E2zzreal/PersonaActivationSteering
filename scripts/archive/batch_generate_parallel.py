#!/usr/bin/env python3
"""
4卡并行批量生成对话
"""
import subprocess
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/generate_parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 模型分组（4组，每组分配到一个GPU）
GPU_ASSIGNMENTS = {
    0: ["baseline", "stage1", "stage1_qwen3", "exp_gate_init_0"],
    1: ["exp_gate_init_neg1", "exp_gate_init_neg2", "exp_gate_init_neg3", "exp_gate_reg_0.001_lr5e5"],
    2: ["exp_gate_reg_0.01_lr1e4", "exp_gate_reg_0.01_lr3e5", "exp_gate_reg_0.05_lr5e5", "stage3_auto"],
    3: ["stage3_gate_init_0", "stage3_gate_reg_0.01_lr1e4", "stage3_gate_reg_0.05_lr5e5"]
}

MODEL_CONFIGS = {
    "baseline": {"checkpoint": "baseline", "config": "configs/train_stage1_qwen3.yaml"},
    "stage1": {"checkpoint": "checkpoints/stage1/best.pt", "config": "configs/train_stage1.yaml"},
    "stage1_qwen3": {"checkpoint": "checkpoints/stage1_qwen3/best.pt", "config": "configs/train_stage1_qwen3.yaml"},
    "exp_gate_init_0": {"checkpoint": "checkpoints/exp_gate_init_0/best.pt", "config": "configs/exp_gate_init_0.yaml"},
    "exp_gate_init_neg1": {"checkpoint": "checkpoints/exp_gate_init_neg1/best.pt", "config": "configs/exp_gate_init_neg1.yaml"},
    "exp_gate_init_neg2": {"checkpoint": "checkpoints/exp_gate_init_neg2/best.pt", "config": "configs/exp_gate_init_neg2.yaml"},
    "exp_gate_init_neg3": {"checkpoint": "checkpoints/exp_gate_init_neg3/best.pt", "config": "configs/exp_gate_init_neg3.yaml"},
    "exp_gate_reg_0.001_lr5e5": {"checkpoint": "checkpoints/exp_gate_reg_0.001_lr5e5/best.pt", "config": "configs/exp_gate_reg_0.001_lr5e5.yaml"},
    "exp_gate_reg_0.01_lr1e4": {"checkpoint": "checkpoints/exp_gate_reg_0.01_lr1e4/best.pt", "config": "configs/exp_gate_reg_0.01_lr1e4.yaml"},
    "exp_gate_reg_0.01_lr3e5": {"checkpoint": "checkpoints/exp_gate_reg_0.01_lr3e5/best.pt", "config": "configs/exp_gate_reg_0.01_lr3e5.yaml"},
    "exp_gate_reg_0.05_lr5e5": {"checkpoint": "checkpoints/exp_gate_reg_0.05_lr5e5/best.pt", "config": "configs/exp_gate_reg_0.05_lr5e5.yaml"},
    "stage3_auto": {"checkpoint": "checkpoints/stage3_auto/best.pt", "config": "configs/train_stage3_auto.yaml"},
    "stage3_gate_init_0": {"checkpoint": "checkpoints/stage3_gate_init_0/best.pt", "config": "configs/train_stage3_gate_init_0.yaml"},
    "stage3_gate_reg_0.01_lr1e4": {"checkpoint": "checkpoints/stage3_gate_reg_0.01_lr1e4/best.pt", "config": "configs/train_stage3_gate_reg_0.01_lr1e4.yaml"},
    "stage3_gate_reg_0.05_lr5e5": {"checkpoint": "checkpoints/stage3_gate_reg_0.05_lr5e5/best.pt", "config": "configs/train_stage3_gate_reg_0.05_lr5e5.yaml"},
}


def generate_single(args):
    """生成单个模型"""
    name, gpu_id, timestamp = args
    config = MODEL_CONFIGS[name]
    output_file = f"results/conversations_{name}_{timestamp}.json"

    cmd = [
        "python", "scripts/generate_all_conversations_fixed.py",
        "--config", config["config"],
        "--checkpoint", config["checkpoint"],
        "--output", output_file,
        "--num_samples", "50",
        "--max_new_tokens", "150",
        "--gpu", str(gpu_id)
    ]

    logger.info(f"[GPU{gpu_id}] 开始: {name}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        if result.returncode == 0:
            with open(output_file) as f:
                data = json.load(f)
            logger.info(f"[GPU{gpu_id}] ✓ 完成: {name} ({len(data)}样本)")
            return {"name": name, "status": "success", "samples": len(data), "gpu": gpu_id}
        else:
            logger.error(f"[GPU{gpu_id}] ✗ 失败: {name}")
            return {"name": name, "status": "error", "error": result.stderr[-300:], "gpu": gpu_id}

    except Exception as e:
        logger.error(f"[GPU{gpu_id}] ✗ 异常: {name} - {str(e)}")
        return {"name": name, "status": "error", "error": str(e), "gpu": gpu_id}


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 准备任务列表
    tasks = []
    for gpu_id, models in GPU_ASSIGNMENTS.items():
        for name in models:
            tasks.append((name, gpu_id, timestamp))

    logger.info(f"=" * 60)
    logger.info(f"4卡并行生成 - {timestamp}")
    logger.info(f"总任务数: {len(tasks)}")
    logger.info(f"GPU分配: { {k: len(v) for k, v in GPU_ASSIGNMENTS.items()} }")
    logger.info(f"=" * 60)

    # 4卡并行执行
    results = {}
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(generate_single, task): task[0] for task in tasks}

        for future in as_completed(futures):
            result = future.result()
            results[result["name"]] = result

    # 汇总
    success = sum(1 for r in results.values() if r["status"] == "success")
    logger.info(f"\n完成: {success}/{len(tasks)}")

    # 保存汇总
    with open(f"results/generation_parallel_{timestamp}.json", "w") as f:
        json.dump({"timestamp": timestamp, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
