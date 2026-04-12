#!/usr/bin/env python
"""
仅运行 Stage 3 训练（Stage 1 & 2 已完成）
"""
import subprocess
import sys
import yaml
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def modify_config_gpu(config_path: str, gpu: int, output_name: str) -> str:
    """修改配置文件的 GPU 设备"""
    config_path = PROJECT_ROOT / config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config['device'] = f"cuda:{gpu}"
    output_path = PROJECT_ROOT / "configs" / "auto" / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return str(output_path.relative_to(PROJECT_ROOT))


# Stage 3 任务配置
STAGE3_JOBS = [
    {"name": "qwen3_neuroticism", "gpu": 0, "config": "configs/train_stage3_qwen3.yaml"},
    {"name": "qwen3_minimal", "gpu": 1, "config": "configs/train_stage3_qwen3.yaml"},
    {"name": "qwen3_baseline", "gpu": 2, "config": "configs/train_stage3_qwen3.yaml"},
]


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "logs" / "training"
    log_dir.mkdir(parents=True, exist_ok=True)

    processes = []

    for job in STAGE3_JOBS:
        name = job["name"]
        gpu = job["gpu"]
        config = job["config"]

        # 修改 GPU 配置
        auto_config = modify_config_gpu(
            config, gpu, f"{name}_stage3_{timestamp}.yaml"
        )

        log_file = log_dir / f"{name}_stage3.log"

        cmd = [sys.executable, "scripts/train.py", "--config", auto_config]

        print(f"[{name}] Starting Stage 3 on GPU {gpu}")
        print(f"  Config: {auto_config}")
        print(f"  Log: {log_file}")

        with open(log_file, 'w') as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT,
            )
        processes.append((name, proc))

    # 等待所有进程完成
    print("\nWaiting for all Stage 3 training to complete...")
    results = {}
    for name, proc in processes:
        exit_code = proc.wait()
        results[name] = "success" if exit_code == 0 else "failed"
        print(f"[{name}] Stage 3: {results[name]}")

    print("\n=== Results ===")
    for name, status in results.items():
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()