#!/usr/bin/env python
"""
PersonaSteer 并行自动化训练流水线
- 4 GPU 并行训练不同配置
- Stage 自动衔接
- 训练完成后自动评估
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TrainJob:
    """单个训练任务"""
    def __init__(self, name: str, gpu: int, base_model: str,
                 stage1_config: str, stage2_config: str, stage3_config: str,
                 checkpoint_prefix: str):
        self.name = name
        self.gpu = gpu
        self.base_model = base_model
        self.stage1_config = stage1_config
        self.stage2_config = stage2_config
        self.stage3_config = stage3_config
        self.checkpoint_prefix = checkpoint_prefix
        self.status = "pending"
        self.current_stage = 0
        self.log_dir = PROJECT_ROOT / "logs" / "training" / name
        self.log_dir.mkdir(parents=True, exist_ok=True)


# 4 GPU 并行配置
TRAIN_JOBS = {
    "qwen3_neuroticism": TrainJob(
        name="qwen3_neuroticism",
        gpu=0,
        base_model="Qwen/Qwen3-4B",
        stage1_config="configs/train_stage1_qwen3_neuroticism_gpu2.yaml",
        stage2_config="configs/train_stage2_qwen3.yaml",
        stage3_config="configs/train_stage3_qwen3.yaml",
        checkpoint_prefix="stage1_qwen3_neuroticism",
    ),
    "qwen3_minimal": TrainJob(
        name="qwen3_minimal",
        gpu=1,
        base_model="Qwen/Qwen3-4B",
        stage1_config="configs/train_stage1_qwen3_probing_minimal_gpu1.yaml",
        stage2_config="configs/train_stage2_qwen3.yaml",
        stage3_config="configs/train_stage3_qwen3.yaml",
        checkpoint_prefix="stage1_qwen3_probing_minimal",
    ),
    "qwen3_baseline": TrainJob(
        name="qwen3_baseline",
        gpu=2,
        base_model="Qwen/Qwen3-4B",
        stage1_config="configs/train_stage1_qwen3_baseline_gpu3.yaml",
        stage2_config="configs/train_stage2_qwen3.yaml",
        stage3_config="configs/train_stage3_qwen3.yaml",
        checkpoint_prefix="stage1_qwen3_baseline",
    ),
    "qwen25_baseline": TrainJob(
        name="qwen25_baseline",
        gpu=3,
        base_model="Qwen/Qwen2.5-3B",
        stage1_config="configs/train_stage1.yaml",
        stage2_config="configs/train_stage2_qwen25_nodual.yaml",
        stage3_config="configs/train_stage3_qwen25_nodual.yaml",
        checkpoint_prefix="stage1",
    ),
}


def modify_config_gpu(config_path: str, gpu: int, output_name: str) -> str:
    """修改配置文件的 GPU 设备"""
    config_path = PROJECT_ROOT / config_path

    with open(config_path) as f:
        config = yaml.safe_load(f)

    config['device'] = f"cuda:{gpu}"

    # 保存到 auto 目录
    output_path = PROJECT_ROOT / "configs" / "auto" / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return str(output_path.relative_to(PROJECT_ROOT))


def run_command(cmd: list, log_file: Path, name: str) -> int:
    """运行命令并记录日志"""
    logger.info(f"[{name}] Running: {' '.join(cmd)}")

    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
        )

    exit_code = process.wait()
    return exit_code


def run_stage(job: TrainJob, stage: int) -> bool:
    """运行单个训练阶段"""
    stage_configs = {
        1: job.stage1_config,
        2: job.stage2_config,
        3: job.stage3_config,
    }

    config_path = stage_configs[stage]
    if not config_path or not (PROJECT_ROOT / config_path).exists():
        logger.error(f"[{job.name}] Stage {stage} config not found: {config_path}")
        return False

    # 修改 GPU 配置
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    auto_config = modify_config_gpu(
        config_path, job.gpu,
        f"{job.name}_stage{stage}_{timestamp}.yaml"
    )

    # 日志文件
    log_file = job.log_dir / f"stage{stage}.log"

    # 训练命令
    cmd = [sys.executable, "scripts/train.py", "--config", auto_config]

    job.current_stage = stage
    job.status = f"stage{stage}_running"

    logger.info(f"[{job.name}] Starting Stage {stage} on GPU {job.gpu}")
    exit_code = run_command(cmd, log_file, f"{job.name}_stage{stage}")

    if exit_code == 0:
        logger.info(f"[{job.name}] Stage {stage} completed")
        return True
    else:
        logger.error(f"[{job.name}] Stage {stage} failed (exit={exit_code})")
        job.status = f"stage{stage}_failed"
        return False


def run_training_job(job: TrainJob) -> dict:
    """运行完整训练任务 (Stage 1 → 2 → 3)"""
    result = {
        "name": job.name,
        "gpu": job.gpu,
        "stages": {},
        "success": False,
        "checkpoints": [],
    }

    logger.info(f"[{job.name}] Starting training on GPU {job.gpu}")

    for stage in [1, 2, 3]:
        success = run_stage(job, stage)
        result["stages"][stage] = "success" if success else "failed"

        if not success:
            logger.error(f"[{job.name}] Training stopped at Stage {stage}")
            return result

    job.status = "completed"
    result["success"] = True

    # 查找 checkpoints
    result["checkpoints"] = find_checkpoints(job)

    logger.info(f"[{job.name}] All stages completed successfully")
    return result


def find_checkpoints(job: TrainJob) -> list[str]:
    """查找训练产生的 checkpoints"""
    checkpoint_dirs = [
        PROJECT_ROOT / "checkpoints" / job.checkpoint_prefix,
        PROJECT_ROOT / "checkpoints" / "stage2_qwen3_v2",
        PROJECT_ROOT / "checkpoints" / "stage3_qwen3_v2",
        PROJECT_ROOT / "checkpoints" / "stage2",
        PROJECT_ROOT / "checkpoints" / "stage3",
    ]

    checkpoints = []
    for d in checkpoint_dirs:
        if d.exists():
            for f in sorted(d.glob("*.pt")):
                checkpoints.append(str(f))

    return checkpoints


def run_evaluation(checkpoint: str, gpu: int = 0) -> dict:
    """运行单个 checkpoint 的评估"""
    checkpoint_path = Path(checkpoint)
    log_dir = PROJECT_ROOT / "logs" / "eval"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{checkpoint_path.stem}.log"

    # 判断模型类型
    if "qwen25" in checkpoint.lower() or "stage1" == checkpoint_path.parent.name:
        eval_script = "scripts/v4_eval_qwen25.py"
    else:
        eval_script = "scripts/v4_eval_qwen3.py"

    cmd = [
        sys.executable, eval_script,
        "--checkpoint", checkpoint,
    ]

    logger.info(f"[EVAL] Evaluating {checkpoint_path.name}")
    exit_code = run_command(cmd, log_file, f"eval_{checkpoint_path.stem}")

    return {
        "checkpoint": checkpoint,
        "success": exit_code == 0,
        "exit_code": exit_code,
    }


def main():
    parser = argparse.ArgumentParser(description="PersonaSteer Parallel Training Pipeline")
    parser.add_argument(
        "--jobs", nargs="+",
        default=list(TRAIN_JOBS.keys()),
        choices=list(TRAIN_JOBS.keys()),
        help="Training jobs to run"
    )
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation on existing checkpoints")

    args = parser.parse_args()

    start_time = datetime.now()
    results = {"jobs": {}, "evaluations": [], "start_time": str(start_time)}

    # 创建日志目录
    (PROJECT_ROOT / "logs" / "training").mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        # 并行训练
        logger.info("=" * 60)
        logger.info("STARTING PARALLEL TRAINING")
        logger.info(f"Jobs: {args.jobs}")
        logger.info("=" * 60)

        jobs_to_run = [TRAIN_JOBS[name] for name in args.jobs]

        with ThreadPoolExecutor(max_workers=len(jobs_to_run)) as executor:
            futures = {
                executor.submit(run_training_job, job): job.name
                for job in jobs_to_run
            }

            for future in as_completed(futures):
                job_name = futures[future]
                try:
                    result = future.result()
                    results["jobs"][job_name] = result
                except Exception as e:
                    logger.error(f"[{job_name}] Exception: {e}")
                    results["jobs"][job_name] = {"success": False, "error": str(e)}

    # 评估阶段
    if not args.skip_eval:
        logger.info("=" * 60)
        logger.info("STARTING EVALUATION")
        logger.info("=" * 60)

        all_checkpoints = []
        for job_name, job_result in results["jobs"].items():
            all_checkpoints.extend(job_result.get("checkpoints", []))

        # 如果 eval-only 模式，查找现有 checkpoints
        if args.eval_only:
            for job_name in args.jobs:
                job = TRAIN_JOBS[job_name]
                all_checkpoints.extend(find_checkpoints(job))

        logger.info(f"Found {len(all_checkpoints)} checkpoints to evaluate")

        for checkpoint in all_checkpoints:
            eval_result = run_evaluation(checkpoint)
            results["evaluations"].append(eval_result)
            status = "✓" if eval_result["success"] else "✗"
            logger.info(f"[EVAL] {status} {Path(checkpoint).name}")

    # 保存结果
    end_time = datetime.now()
    results["end_time"] = str(end_time)
    results["duration"] = str(end_time - start_time)

    results_file = PROJECT_ROOT / "results" / f"pipeline_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # 打印摘要
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Duration: {end_time - start_time}")
    print()

    for job_name, job_result in results["jobs"].items():
        status = "✓" if job_result.get("success") else "✗"
        print(f"[{status}] {job_name}: {job_result.get('stages', {})}")

    if results["evaluations"]:
        passed = sum(1 for e in results["evaluations"] if e["success"])
        print(f"\nEvaluations: {passed}/{len(results['evaluations'])} passed")

    print(f"\nResults: {results_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()