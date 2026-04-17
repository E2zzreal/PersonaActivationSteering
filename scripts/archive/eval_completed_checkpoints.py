#!/usr/bin/env python
"""
评估已完成的 checkpoints (在 GPU 3 上运行)
- Stage 1: neuroticism, minimal, baseline
- Stage 2: v2
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent

# 评估配置
EVAL_JOBS = [
    {
        "name": "stage1_neuroticism",
        "checkpoint": "checkpoints/stage1_qwen3_neuroticism/best.pt",
        "stages": ["stage1"],
    },
    {
        "name": "stage1_minimal",
        "checkpoint": "checkpoints/stage1_qwen3_probing_minimal/best.pt",
        "stages": ["stage1"],
    },
    {
        "name": "stage1_baseline",
        "checkpoint": "checkpoints/stage1_qwen3/best.pt",
        "stages": ["stage1"],
    },
    {
        "name": "stage2_v2",
        "checkpoint": "checkpoints/stage2_qwen3_v2/best.pt",
        "stages": ["stage2"],
    },
]


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "logs" / "eval"
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir = PROJECT_ROOT / "results" / f"eval_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for job in EVAL_JOBS:
        name = job["name"]
        checkpoint = job["checkpoint"]
        stages = job["stages"]

        checkpoint_path = PROJECT_ROOT / checkpoint
        if not checkpoint_path.exists():
            print(f"[{name}] Checkpoint not found: {checkpoint}")
            results[name] = {"status": "skipped", "reason": "checkpoint not found"}
            continue

        log_file = log_dir / f"{name}.log"
        output_dir = results_dir / name

        cmd = [
            sys.executable,
            "scripts/v4_eval_qwen3.py",
            "--device", "cuda:3",
            "--num_samples", "50",
            "--stages", *stages,
            "--output_dir", str(output_dir),
        ]

        # 根据 checkpoint 类型添加参数
        if "stage1" in stages:
            cmd.extend(["--stage1_checkpoint", str(checkpoint_path)])
        elif "stage2" in stages:
            cmd.extend(["--stage2_checkpoint", str(checkpoint_path)])

        print(f"\n[{name}] Starting evaluation...")
        print(f"  Checkpoint: {checkpoint}")
        print(f"  Log: {log_file}")

        with open(log_file, 'w') as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT,
            )
            exit_code = proc.wait()

        results[name] = {
            "status": "success" if exit_code == 0 else "failed",
            "exit_code": exit_code,
            "checkpoint": checkpoint,
        }
        print(f"[{name}] Status: {results[name]['status']}")

    # 保存结果摘要
    import json
    summary_file = results_dir / "eval_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "results": results,
        }, f, indent=2)

    print(f"\n=== Evaluation Summary ===")
    print(f"Results saved to: {results_dir}")
    for name, res in results.items():
        print(f"  {name}: {res['status']}")


if __name__ == "__main__":
    main()