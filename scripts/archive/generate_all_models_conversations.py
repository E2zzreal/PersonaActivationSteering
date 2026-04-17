#!/usr/bin/env python3
"""
批量生成所有模型的对话并记录
"""
import subprocess
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/generate_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 模型配置
MODELS = {
    # Baseline
    "baseline": {
        "checkpoint": "baseline",
        "config": "configs/model.yaml",
        "gpu": 0
    },
    # Stage 1
    "stage1": {
        "checkpoint": "checkpoints/stage1/best.pt",
        "config": "configs/train_stage1.yaml",
        "gpu": 0
    },
    "stage1_qwen3": {
        "checkpoint": "checkpoints/stage1_qwen3/best.pt",
        "config": "configs/train_stage1.yaml",
        "gpu": 0
    },
    # Stage 2 实验
    "exp_gate_init_0": {
        "checkpoint": "checkpoints/exp_gate_init_0/best.pt",
        "config": "configs/train_stage2.yaml",
        "gpu": 0
    },
    "exp_gate_init_neg1": {
        "checkpoint": "checkpoints/exp_gate_init_neg1/best.pt",
        "config": "configs/train_stage2.yaml",
        "gpu": 0
    },
    "exp_gate_init_neg2": {
        "checkpoint": "checkpoints/exp_gate_init_neg2/best.pt",
        "config": "configs/train_stage2.yaml",
        "gpu": 0
    },
    "exp_gate_init_neg3": {
        "checkpoint": "checkpoints/exp_gate_init_neg3/best.pt",
        "config": "configs/train_stage2.yaml",
        "gpu": 0
    },
    "exp_gate_reg_0.001_lr5e5": {
        "checkpoint": "checkpoints/exp_gate_reg_0.001_lr5e5/best.pt",
        "config": "configs/train_stage2.yaml",
        "gpu": 0
    },
    "exp_gate_reg_0.01_lr1e4": {
        "checkpoint": "checkpoints/exp_gate_reg_0.01_lr1e4/best.pt",
        "config": "configs/train_stage2.yaml",
        "gpu": 0
    },
    "exp_gate_reg_0.01_lr3e5": {
        "checkpoint": "checkpoints/exp_gate_reg_0.01_lr3e5/best.pt",
        "config": "configs/train_stage2.yaml",
        "gpu": 0
    },
    "exp_gate_reg_0.05_lr5e5": {
        "checkpoint": "checkpoints/exp_gate_reg_0.05_lr5e5/best.pt",
        "config": "configs/train_stage2.yaml",
        "gpu": 0
    },
    # Stage 3 实验
    "stage3_auto": {
        "checkpoint": "checkpoints/stage3_auto/best.pt",
        "config": "configs/train_stage3_auto.yaml",
        "gpu": 0
    },
    "stage3_gate_init_0": {
        "checkpoint": "checkpoints/stage3_gate_init_0/best.pt",
        "config": "configs/train_stage3.yaml",
        "gpu": 0
    },
    "stage3_gate_reg_0.01_lr1e4": {
        "checkpoint": "checkpoints/stage3_gate_reg_0.01_lr1e4/best.pt",
        "config": "configs/train_stage3.yaml",
        "gpu": 0
    },
    "stage3_gate_reg_0.05_lr5e5": {
        "checkpoint": "checkpoints/stage3_gate_reg_0.05_lr5e5/best.pt",
        "config": "configs/train_stage3.yaml",
        "gpu": 0
    },
}


def generate_for_model(name, config, timestamp):
    """为单个模型生成对话"""
    output_file = f"results/conversations_{name}_{timestamp}.json"

    cmd = [
        "python", "scripts/generate_all_conversations.py",
        "--config", config["config"],
        "--checkpoint", config["checkpoint"],
        "--output", output_file,
        "--num_samples", "50",
        "--max_new_tokens", "150",
        "--gpu", str(config["gpu"])
    ]

    logger.info(f"=" * 60)
    logger.info(f"开始生成: {name}")
    logger.info(f"Checkpoint: {config['checkpoint']}")
    logger.info(f"Config: {config['config']}")
    logger.info(f"Output: {output_file}")
    logger.info(f"=" * 60)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30分钟超时
        )

        if result.returncode == 0:
            logger.info(f"✓ 成功: {name}")
            # 验证输出文件
            output_path = Path(output_file)
            if output_path.exists():
                with open(output_path) as f:
                    data = json.load(f)
                logger.info(f"  样本数: {len(data)}")
                if data and len(data) > 0:
                    conv_len = len(data[0].get("conversation", []))
                    logger.info(f"  首样本对话轮数: {conv_len}")
                return {"status": "success", "output": output_file, "samples": len(data)}
            else:
                logger.error(f"✗ 输出文件不存在: {output_file}")
                return {"status": "error", "error": "Output file not found"}
        else:
            logger.error(f"✗ 失败: {name}")
            logger.error(f"  错误: {result.stderr}")
            return {"status": "error", "error": result.stderr}

    except subprocess.TimeoutExpired:
        logger.error(f"✗ 超时: {name}")
        return {"status": "timeout", "error": "Timeout after 30 minutes"}
    except Exception as e:
        logger.error(f"✗ 异常: {name} - {str(e)}")
        return {"status": "error", "error": str(e)}


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}

    logger.info("=" * 60)
    logger.info(f"开始批量生成对话 - {timestamp}")
    logger.info(f"总计模型数: {len(MODELS)}")
    logger.info("=" * 60)

    for i, (name, config) in enumerate(MODELS.items(), 1):
        logger.info(f"\n[{i}/{len(MODELS)}] 处理: {name}")
        results[name] = generate_for_model(name, config, timestamp)

    # 汇总报告
    logger.info("\n" + "=" * 60)
    logger.info("生成完成汇总")
    logger.info("=" * 60)

    success_count = sum(1 for r in results.values() if r["status"] == "success")
    error_count = len(results) - success_count

    for name, result in results.items():
        status_icon = "✓" if result["status"] == "success" else "✗"
        logger.info(f"{status_icon} {name}: {result['status']}")

    logger.info("-" * 60)
    logger.info(f"成功: {success_count}/{len(MODELS)}")
    logger.info(f"失败: {error_count}/{len(MODELS)}")

    # 保存汇总结果
    summary_file = f"results/generation_summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "total_models": len(MODELS),
            "success_count": success_count,
            "error_count": error_count,
            "results": results
        }, f, indent=2)
    logger.info(f"汇总已保存: {summary_file}")


if __name__ == "__main__":
    main()
