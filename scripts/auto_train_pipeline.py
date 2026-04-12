#!/usr/bin/env python
"""
PersonaSteer 自动化训练流水线
- 支持 Qwen2.5 和 Qwen3 并行训练
- Stage 1 → Stage 2 → Stage 3 自动衔接
- 训练完成后自动评估所有 checkpoints
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/auto_train.log", mode='a')
    ]
)
logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TrainingConfig:
    """训练配置"""
    def __init__(self, name: str, model_type: str, gpu: int,
                 stage1_config: str, stage2_config: str, stage3_config: str):
        self.name = name
        self.model_type = model_type  # "qwen25" or "qwen3"
        self.gpu = gpu
        self.stage1_config = stage1_config
        self.stage2_config = stage2_config
        self.stage3_config = stage3_config


# 默认训练配置
DEFAULT_CONFIGS = {
    "qwen3_neuroticism": TrainingConfig(
        name="qwen3_neuroticism",
        model_type="qwen3",
        gpu=0,
        stage1_config="configs/train_stage1_qwen3_neuroticism_gpu2.yaml",
        stage2_config="configs/train_stage2_qwen3.yaml",
        stage3_config="configs/train_stage3_qwen3.yaml",
    ),
    "qwen3_minimal": TrainingConfig(
        name="qwen3_minimal",
        model_type="qwen3",
        gpu=1,
        stage1_config="configs/train_stage1_qwen3_probing_minimal_gpu1.yaml",
        stage2_config="configs/train_stage2_qwen3.yaml",
        stage3_config="configs/train_stage3_qwen3.yaml",
    ),
    "qwen3_baseline": TrainingConfig(
        name="qwen3_baseline",
        model_type="qwen3",
        gpu=2,
        stage1_config="configs/train_stage1_qwen3_baseline_gpu3.yaml",
        stage2_config="configs/train_stage2_qwen3.yaml",
        stage3_config="configs/train_stage3_qwen3.yaml",
    ),
    "qwen25_baseline": TrainingConfig(
        name="qwen25_baseline",
        model_type="qwen25",
        gpu=3,
        stage1_config="configs/train_stage1.yaml",
        stage2_config="configs/train_stage2_qwen25_nodual.yaml",
        stage3_config="configs/train_stage3_qwen25_nodual.yaml",
    ),
}


class ProcessManager:
    """进程管理器"""
    def __init__(self):
        self.processes: dict[str, subprocess.Popen] = {}
        self.logs: dict[str, Path] = {}

    def start(self, name: str, cmd: list, log_file: Path):
        """启动进程"""
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT,
            )
        self.processes[name] = process
        self.logs[name] = log_file
        logger.info(f"Started {name}: PID={process.pid}, Log={log_file}")

    def wait(self, name: str) -> int:
        """等待进程完成，返回退出码"""
        if name not in self.processes:
            return -1
        return self.processes[name].wait()

    def is_running(self, name: str) -> bool:
        """检查进程是否运行中"""
        if name not in self.processes:
            return False
        return self.processes[name].poll() is None

    def get_exit_code(self, name: str) -> Optional[int]:
        """获取进程退出码"""
        if name not in self.processes:
            return None
        return self.processes[name].poll()

    def tail_log(self, name: str, lines: int = 20) -> str:
        """获取日志最后几行"""
        if name not in self.logs:
            return ""
        log_file = self.logs[name]
        if not log_file.exists():
            return ""
        result = subprocess.run(
            ["tail", "-n", str(lines), str(log_file)],
            capture_output=True, text=True
        )
        return result.stdout


class AutoTrainPipeline:
    """自动化训练流水线"""

    def __init__(self, configs: list[str], skip_eval: bool = False):
        self.config_names = configs
        self.skip_eval = skip_eval
        self.manager = ProcessManager()
        self.results: dict[str, dict] = {}
        self.start_time = datetime.now()

        # 创建日志目录
        (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
        (PROJECT_ROOT / "logs" / "training").mkdir(exist_ok=True)

    def get_config(self, name: str) -> TrainingConfig:
        """获取训练配置"""
        if name not in DEFAULT_CONFIGS:
            raise ValueError(f"Unknown config: {name}")
        return DEFAULT_CONFIGS[name]

    def modify_config_for_gpu(self, config_path: str, gpu: int, stage: int) -> str:
        """修改配置文件中的 GPU 设备"""
        import yaml

        config_path = PROJECT_ROOT / config_path
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # 修改设备
        config['device'] = f"cuda:{gpu}"

        # 生成新配置文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_config_name = f"auto_{config_path.stem}_{timestamp}.yaml"
        new_config_path = PROJECT_ROOT / "configs" / "auto" / new_config_name
        new_config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(new_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        return str(new_config_path.relative_to(PROJECT_ROOT))

    def run_stage(self, config_name: str, stage: int, config: TrainingConfig) -> bool:
        """运行单个训练阶段"""
        stage_config_map = {
            1: config.stage1_config,
            2: config.stage2_config,
            3: config.stage3_config,
        }

        stage_config = stage_config_map[stage]
        if not stage_config or not (PROJECT_ROOT / stage_config).exists():
            logger.warning(f"[{config_name}] Stage {stage} config not found: {stage_config}")
            return False

        # 修改 GPU 配置
        actual_config = self.modify_config_for_gpu(stage_config, config.gpu, stage)

        # 日志文件
        log_file = PROJECT_ROOT / "logs" / "training" / f"{config_name}_stage{stage}.log"

        # 训练命令
        cmd = [
            sys.executable,
            "scripts/train.py",
            "--config", actual_config,
        ]

        process_name = f"{config_name}_stage{stage}"
        logger.info(f"[{config_name}] Starting Stage {stage} on GPU {config.gpu}")

        self.manager.start(process_name, cmd, log_file)

        # 等待完成
        exit_code = self.manager.wait(process_name)

        if exit_code == 0:
            logger.info(f"[{config_name}] Stage {stage} completed successfully")
            return True
        else:
            logger.error(f"[{config_name}] Stage {stage} failed with exit code {exit_code}")
            # 打印日志尾部
            logger.error(f"[{config_name}] Log tail:\n{self.manager.tail_log(process_name)}")
            return False

    def run_training(self, config_name: str) -> bool:
        """运行完整训练流程 (Stage 1 → 2 → 3)"""
        config = self.get_config(config_name)
        self.results[config_name] = {"stages": {}, "checkpoints": []}

        for stage in [1, 2, 3]:
            success = self.run_stage(config_name, stage, config)
            self.results[config_name]["stages"][stage] = "success" if success else "failed"

            if not success:
                logger.error(f"[{config_name}] Training stopped at Stage {stage}")
                return False

        logger.info(f"[{config_name}] All stages completed successfully")
        return True

    def find_checkpoints(self, config_name: str) -> list[str]:
        """查找训练产生的 checkpoints"""
        config = self.get_config(config_name)

        # 根据配置名确定 checkpoint 目录
        checkpoint_dirs = []
        if "neuroticism" in config_name:
            checkpoint_dirs = [
                PROJECT_ROOT / "checkpoints" / "stage1_qwen3_neuroticism",
                PROJECT_ROOT / "checkpoints" / "stage2_qwen3_v2",
                PROJECT_ROOT / "checkpoints" / "stage3_qwen3_v2",
            ]
        elif "minimal" in config_name:
            checkpoint_dirs = [
                PROJECT_ROOT / "checkpoints" / "stage1_qwen3_probing_minimal",
                PROJECT_ROOT / "checkpoints" / "stage2_qwen3_v2",
                PROJECT_ROOT / "checkpoints" / "stage3_qwen3_v2",
            ]
        elif "qwen3_baseline" in config_name:
            checkpoint_dirs = [
                PROJECT_ROOT / "checkpoints" / "stage1_qwen3_baseline",
                PROJECT_ROOT / "checkpoints" / "stage2_qwen3_v2",
                PROJECT_ROOT / "checkpoints" / "stage3_qwen3_v2",
            ]
        elif "qwen25" in config_name:
            checkpoint_dirs = [
                PROJECT_ROOT / "checkpoints" / "stage1",
                PROJECT_ROOT / "checkpoints" / "stage2",
                PROJECT_ROOT / "checkpoints" / "stage3",
            ]

        checkpoints = []
        for d in checkpoint_dirs:
            if d.exists():
                for f in d.glob("*.pt"):
                    checkpoints.append(str(f))

        return checkpoints

    def run_evaluation(self, checkpoint: str) -> dict:
        """运行单个 checkpoint 的评估"""
        checkpoint_path = Path(checkpoint)
        log_file = PROJECT_ROOT / "logs" / "training" / f"eval_{checkpoint_path.stem}.log"

        # 评估命令
        cmd = [
            sys.executable,
            "scripts/v4_eval_qwen3.py",
            "--checkpoint", checkpoint,
        ]

        process_name = f"eval_{checkpoint_path.stem}"
        logger.info(f"Starting evaluation for {checkpoint}")

        self.manager.start(process_name, cmd, log_file)
        exit_code = self.manager.wait(process_name)

        return {
            "checkpoint": checkpoint,
            "exit_code": exit_code,
            "success": exit_code == 0,
        }

    def run_all_evaluations(self):
        """运行所有 checkpoint 的评估"""
        logger.info("=" * 60)
        logger.info("Starting evaluation phase")
        logger.info("=" * 60)

        all_checkpoints = []
        for config_name in self.config_names:
            checkpoints = self.find_checkpoints(config_name)
            all_checkpoints.extend(checkpoints)
            self.results[config_name]["checkpoints"] = checkpoints

        logger.info(f"Found {len(all_checkpoints)} checkpoints to evaluate")

        eval_results = []
        for checkpoint in all_checkpoints:
            result = self.run_evaluation(checkpoint)
            eval_results.append(result)
            if result["success"]:
                logger.info(f"Evaluation passed: {checkpoint}")
            else:
                logger.error(f"Evaluation failed: {checkpoint}")

        self.results["evaluations"] = eval_results

    def save_results(self):
        """保存结果报告"""
        results_file = PROJECT_ROOT / "results" / f"auto_train_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results saved to {results_file}")

    def print_summary(self):
        """打印摘要"""
        elapsed = datetime.now() - self.start_time

        print("\n" + "=" * 60)
        print("TRAINING PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Duration: {elapsed}")
        print()

        for config_name in self.config_names:
            if config_name in self.results:
                stages = self.results[config_name].get("stages", {})
                checkpoints = self.results[config_name].get("checkpoints", [])

                print(f"[{config_name}]")
                print(f"  Stages: {stages}")
                print(f"  Checkpoints: {len(checkpoints)}")
                print()

        if "evaluations" in self.results:
            evals = self.results["evaluations"]
            passed = sum(1 for e in evals if e["success"])
            print(f"Evaluations: {passed}/{len(evals)} passed")

        print("=" * 60)

    def run(self):
        """运行完整流水线"""
        logger.info("=" * 60)
        logger.info("STARTING AUTO TRAINING PIPELINE")
        logger.info(f"Configs: {self.config_names}")
        logger.info("=" * 60)

        # 训练阶段 - 并行运行
        training_success = {}
        for config_name in self.config_names:
            success = self.run_training(config_name)
            training_success[config_name] = success

        # 评估阶段
        if not self.skip_eval:
            self.run_all_evaluations()

        # 保存结果
        self.save_results()

        # 打印摘要
        self.print_summary()

        return all(training_success.values())


def main():
    parser = argparse.ArgumentParser(description="PersonaSteer Auto Training Pipeline")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["qwen3_neuroticism", "qwen3_minimal", "qwen3_baseline", "qwen25_baseline"],
        choices=list(DEFAULT_CONFIGS.keys()),
        help="Training configurations to run"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation phase"
    )

    args = parser.parse_args()

    pipeline = AutoTrainPipeline(args.configs, args.skip_eval)
    success = pipeline.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()