#!/usr/bin/env python3
"""
PersonaSteer 自动化训练流水线

功能:
- 多GPU并行训练不同配置
- 自动接力: Stage1 -> Stage2 -> Stage3 -> Evaluation
- 进度监控和日志记录

Usage:
    python scripts/auto_pipeline.py --config configs/pipeline_config.yaml
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import yaml

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pipeline.log", mode='a'),
    ]
)
logger = logging.getLogger(__name__)


class TrainingJob:
    """单个训练任务"""

    def __init__(
        self,
        name: str,
        config_path: str,
        gpu_id: int,
        stage: int = 1,
        checkpoint_dir: str = None,
    ):
        self.name = name
        self.config_path = config_path
        self.gpu_id = gpu_id
        self.stage = stage
        self.checkpoint_dir = checkpoint_dir or f"checkpoints/{name}"
        self.process: Optional[subprocess.Popen] = None
        self.log_file = f"logs/{name}_stage{stage}.log"
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.status = "pending"  # pending, running, completed, failed

    def start(self) -> bool:
        """启动训练任务"""
        if self.status != "pending":
            logger.warning(f"Job {self.name} already started")
            return False

        # 确保日志目录存在
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

        # 启动训练进程
        cmd = [
            sys.executable,
            "scripts/train.py",
            "--config", self.config_path,
        ]

        logger.info(f"Starting job {self.name} on GPU {self.gpu_id}")
        logger.info(f"Command: {' '.join(cmd)}")
        logger.info(f"Log file: {self.log_file}")

        try:
            with open(self.log_file, 'w') as log_f:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,
                )
            self.start_time = time.time()
            self.status = "running"
            return True
        except Exception as e:
            logger.error(f"Failed to start job {self.name}: {e}")
            self.status = "failed"
            return False

    def check_status(self) -> str:
        """检查任务状态"""
        if self.status in ["completed", "failed"]:
            return self.status

        if self.process is None:
            return self.status

        # 检查进程是否结束
        ret = self.process.poll()
        if ret is not None:
            self.end_time = time.time()
            if ret == 0:
                self.status = "completed"
                logger.info(f"Job {self.name} completed in {self.elapsed_time():.1f}s")
            else:
                self.status = "failed"
                logger.error(f"Job {self.name} failed with return code {ret}")
        return self.status

    def elapsed_time(self) -> float:
        """已运行时间"""
        if self.start_time is None:
            return 0
        end = self.end_time or time.time()
        return end - self.start_time

    def get_checkpoint_path(self) -> str:
        """获取最佳checkpoint路径"""
        return f"{self.checkpoint_dir}/best.pt"


class Pipeline:
    """训练流水线"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.jobs: Dict[str, TrainingJob] = {}
        self.pipeline_status = {}

    def _load_config(self, config_path: str) -> dict:
        """加载流水线配置"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _create_stage_config(self, base_config: dict, stage: int, gpu_id: int,
                             prev_checkpoint: str = None) -> str:
        """创建阶段配置文件"""
        config = base_config.copy()

        # 更新阶段
        config["training"]["stage"] = stage
        config["training"]["output_dir"] = f"checkpoints/{config['name']}_stage{stage}"

        # 设置GPU
        config["device"] = f"cuda:{gpu_id}"

        # 设置前一阶段checkpoint
        if prev_checkpoint and stage > 1:
            config[f"stage{stage-1}_checkpoint"] = prev_checkpoint

        # 保存配置
        config_path = f"configs/auto/{config['name']}_stage{stage}_gpu{gpu_id}.yaml"
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        return config_path

    def setup_jobs(self):
        """设置所有训练任务"""
        experiments = self.config.get("experiments", [])

        for exp in experiments:
            name = exp["name"]
            gpu_id = exp["gpu"]
            base_config = exp.get("config", {})

            logger.info(f"Setting up experiment: {name} on GPU {gpu_id}")

            # Stage 1
            stage1_config = self._create_stage_config(
                {"name": name, **base_config},
                stage=1,
                gpu_id=gpu_id,
            )
            job1 = TrainingJob(
                name=f"{name}_stage1",
                config_path=stage1_config,
                gpu_id=gpu_id,
                stage=1,
            )
            self.jobs[f"{name}_stage1"] = job1

            # Stage 2 (depends on Stage 1)
            job2 = TrainingJob(
                name=f"{name}_stage2",
                config_path=None,  # Will be created when Stage 1 completes
                gpu_id=gpu_id,
                stage=2,
            )
            self.jobs[f"{name}_stage2"] = job2

            # Stage 3 (depends on Stage 2)
            job3 = TrainingJob(
                name=f"{name}_stage3",
                config_path=None,  # Will be created when Stage 2 completes
                gpu_id=gpu_id,
                stage=3,
            )
            self.jobs[f"{name}_stage3"] = job3

    def run(self):
        """运行流水线"""
        logger.info("=" * 60)
        logger.info("PersonaSteer Auto Pipeline")
        logger.info("=" * 60)

        # 确保目录存在
        Path("logs").mkdir(exist_ok=True)
        Path("checkpoints").mkdir(exist_ok=True)

        # 设置任务
        self.setup_jobs()

        # 启动所有 Stage 1 任务
        stage1_jobs = [j for j in self.jobs.values() if j.stage == 1]
        for job in stage1_jobs:
            job.start()
            time.sleep(5)  # 避免同时加载模型

        # 监控循环
        while True:
            all_completed = True

            for name, job in self.jobs.items():
                status = job.check_status()

                if status == "running":
                    all_completed = False

                elif status == "completed":
                    # 检查是否需要启动下一阶段
                    self._try_start_next_stage(name)

                elif status == "failed":
                    logger.error(f"Job {name} failed, stopping pipeline")
                    return

            if all_completed:
                logger.info("All jobs completed!")
                break

            # 打印状态
            self._print_status()

            time.sleep(30)  # 每30秒检查一次

        # 运行评估
        self._run_evaluation()

    def _try_start_next_stage(self, completed_job_name: str):
        """尝试启动下一阶段"""
        # 解析任务名称
        parts = completed_job_name.rsplit("_stage", 1)
        if len(parts) != 2:
            return

        base_name = parts[0]
        completed_stage = int(parts[1])

        if completed_stage >= 3:
            return  # 已经是最后一阶段

        next_stage = completed_stage + 1
        next_job_name = f"{base_name}_stage{next_stage}"
        next_job = self.jobs.get(next_job_name)

        if next_job is None or next_job.status != "pending":
            return

        # 获取前一阶段checkpoint
        prev_job = self.jobs[completed_job_name]
        prev_checkpoint = prev_job.get_checkpoint_path()

        if not Path(prev_checkpoint).exists():
            logger.error(f"Checkpoint not found: {prev_checkpoint}")
            return

        # 创建下一阶段配置
        exp_config = next(
            (e for e in self.config["experiments"] if e["name"] == base_name),
            None
        )
        if exp_config is None:
            return

        next_config_path = self._create_stage_config(
            {"name": base_name, **exp_config.get("config", {})},
            stage=next_stage,
            gpu_id=next_job.gpu_id,
            prev_checkpoint=prev_checkpoint,
        )
        next_job.config_path = next_config_path
        next_job.checkpoint_dir = f"checkpoints/{base_name}_stage{next_stage}"

        # 启动下一阶段
        logger.info(f"Starting {next_job_name} after {completed_job_name} completed")
        next_job.start()

    def _print_status(self):
        """打印状态"""
        logger.info("-" * 40)
        for name, job in self.jobs.items():
            elapsed = job.elapsed_time()
            status_str = f"{job.status} ({elapsed:.0f}s)" if job.status == "running" else job.status
            logger.info(f"  {name}: {status_str}")
        logger.info("-" * 40)

    def _run_evaluation(self):
        """运行评估"""
        logger.info("=" * 60)
        logger.info("Running evaluation...")
        logger.info("=" * 60)

        # 找到所有 Stage 3 checkpoints
        stage3_checkpoints = []
        for name, job in self.jobs.items():
            if job.stage == 3 and job.status == "completed":
                ckpt = job.get_checkpoint_path()
                if Path(ckpt).exists():
                    stage3_checkpoints.append((name, ckpt))

        for name, ckpt in stage3_checkpoints:
            logger.info(f"Evaluating {name}...")
            eval_cmd = [
                sys.executable,
                "scripts/evaluate.py",
                "--checkpoint", ckpt,
                "--output", f"results/{name}_eval.json",
            ]
            try:
                subprocess.run(eval_cmd, check=True)
            except Exception as e:
                logger.error(f"Evaluation failed for {name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="PersonaSteer Auto Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Pipeline configuration file",
    )
    args = parser.parse_args()

    pipeline = Pipeline(args.config)
    pipeline.run()


if __name__ == "__main__":
    main()