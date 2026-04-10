#!/usr/bin/env python
"""
PersonaSteer 评估脚本
运行自动指标评估和 LLM Judge 评估
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.aloe_dataset import ALOEDataset
from src.data.collator import PersonaSteerCollator
from src.evaluation.auto_metrics import AutoMetricsEvaluator
from src.evaluation.llm_judge import LLMJudgeEvaluator, load_baseline_scores
from src.models.persona_steer import PersonaSteerModel, PersonaSteerConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Evaluate PersonaSteer model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/eval.jsonl",
        help="Path to evaluation data",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="experiments/eval_results.json",
        help="Output path for evaluation results",
    )

    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4o-mini",
        help="LLM Judge model name",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )

    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to baseline scores JSON for N-IR calculation",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    parser.add_argument(
        "--skip_llm_judge",
        action="store_true",
        help="Skip LLM Judge evaluation",
    )

    parser.add_argument(
        "--baseline_mode",
        action="store_true",
        help="Run in baseline mode: disable all injection, evaluate raw backbone",
    )

    parser.add_argument(
        "--judge_models",
        type=str,
        default=None,
        help="Comma-separated list of judge models (e.g. 'Claude-Sonnet-4.6,GPT-5.2'). Overrides --judge_model",
    )

    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, config: dict) -> PersonaSteerModel:
    """从 checkpoint 加载模型（含 backbone）"""
    from transformers import AutoConfig, AutoModelForCausalLM
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = config.get("model", {})
    base_model_path = config.get("base_model")
    if base_model_path is None:
        raise ValueError("config must contain 'base_model' path")

    # 从 backbone config 读取实际 hidden_size
    backbone_cfg = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    actual_layer_dim = backbone_cfg.hidden_size
    logger.info(f"Backbone hidden_size: {actual_layer_dim}")

    persona_config = PersonaSteerConfig(
        inject_layers=model_config.get("inject_layers", [14, 15, 16, 17, 18, 19, 20, 21]),
        v_dim=model_config.get("v_dim", 1024),
        hidden_dim=model_config.get("hidden_dim", 4096),
        layer_dim=actual_layer_dim,
        gate_hidden_dim=model_config.get("gate_hidden_dim", 256),
    )

    # 加载 backbone
    logger.info(f"Loading backbone from {base_model_path}")
    backbone = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        use_cache=False,
    )
    encoder = backbone.model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    model = PersonaSteerModel(config=persona_config, encoder=encoder)
    if hasattr(model, "hyper_network") and model.hyper_network is not None:
        model.hyper_network._tokenizer = tokenizer
    model.set_backbone(backbone)

    # 加载 HyperNetwork / injection 权重
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()
    loaded, skipped = 0, 0
    for key, val in state_dict.items():
        if key in model_state and model_state[key].shape == val.shape:
            model_state[key] = val
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(model_state)
    logger.info(f"Checkpoint loaded: {loaded} tensors loaded, {skipped} skipped (shape mismatch/not found)")

    model.to(device)
    model.eval()
    return model, tokenizer


def create_eval_loader(data_path: str, config: dict, tokenizer=None) -> DataLoader:
    """创建评估数据加载器"""
    # 如果没有传入 tokenizer，从 config 加载（避免与 load_model_from_checkpoint 重复加载）
    if tokenizer is None:
        tokenizer_name = config.get("data", {}).get("tokenizer", "Qwen/Qwen3-4B")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    # 数据参数
    data_config = config.get("data", {})
    max_turns = data_config.get("max_turns", 6)
    batch_size = data_config.get("batch_size", 4)

    # 创建数据集
    dataset = ALOEDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_turns=max_turns,
    )

    logger.info(f"Evaluation dataset: {len(dataset)} samples")

    # Collator
    collator = PersonaSteerCollator(tokenizer)

    # DataLoader
    eval_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=data_config.get("num_workers", 2),
        pin_memory=True,
    )

    return eval_loader, tokenizer


def run_auto_evaluation(
    model: PersonaSteerModel,
    eval_loader: DataLoader,
    device: str,
) -> dict:
    """运行自动指标评估"""
    logger.info("Running auto metrics evaluation...")

    evaluator = AutoMetricsEvaluator(device=device)
    results = evaluator.evaluate(model, eval_loader)

    return results


def run_llm_judge_evaluation(
    model: PersonaSteerModel,
    test_samples: list,
    tokenizer,
    judge_model: str,
    baseline_scores: list | None = None,
) -> dict:
    """运行 LLM Judge 评估"""
    logger.info(f"Running LLM Judge evaluation with {judge_model}...")

    evaluator = LLMJudgeEvaluator(judge_model=judge_model)
    results = evaluator.evaluate_alignment(
        model=model,
        test_samples=test_samples,
        tokenizer=tokenizer,
        baseline_scores=baseline_scores,
    )

    return results


def load_config(config_path: str | None) -> dict:
    """加载配置文件"""
    if config_path is None:
        # 无默认配置，必须传入 --config
        raise ValueError("--config is required. No default config available.")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_test_samples(data_path: str, num_samples: int | None = None) -> list:
    """加载测试样本"""
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    if num_samples is not None:
        samples = samples[:num_samples]

    return samples


def main():
    """主函数"""
    args = parse_args()

    # 加载配置
    config = load_config(args.config)

    # 加载模型（同时返回 tokenizer，避免重复加载）
    model, tokenizer = load_model_from_checkpoint(args.checkpoint, config)

    # baseline_mode: 禁用注入，评估原始骨干模型
    if args.baseline_mode:
        model.baseline_mode = True
        logger.info("Baseline mode enabled: injection disabled")

    # 创建评估数据加载器（复用已加载的 tokenizer）
    eval_loader, tokenizer = create_eval_loader(args.data, config, tokenizer=tokenizer)

    # 加载测试样本 (用于 LLM Judge)
    test_samples = load_test_samples(args.data, args.num_samples)

    # 评估结果
    results = {
        "checkpoint": args.checkpoint,
        "baseline_mode": args.baseline_mode,
        "num_eval_samples": len(test_samples),
    }

    # 自动指标评估
    auto_metrics = run_auto_evaluation(model, eval_loader, args.device)
    results["auto_metrics"] = auto_metrics

    # LLM Judge 评估
    if not args.skip_llm_judge:
        # 加载 baseline 分数
        baseline_scores = None
        if args.baseline:
            baseline_scores = load_baseline_scores(args.baseline)

        # 支持多个 judge 模型
        judge_models = []
        if args.judge_models:
            judge_models = [m.strip() for m in args.judge_models.split(",")]
        else:
            judge_models = [args.judge_model]

        for judge_model in judge_models:
            logger.info(f"Running LLM Judge with model: {judge_model}")
            llm_judge_metrics = run_llm_judge_evaluation(
                model=model,
                test_samples=test_samples,
                tokenizer=tokenizer,
                judge_model=judge_model,
                baseline_scores=baseline_scores,
            )
            key = f"llm_judge_{judge_model.replace('.', '_').replace('-', '_')}"
            results[key] = llm_judge_metrics

        # 兼容旧格式：单模型时同时保存到 llm_judge
        if len(judge_models) == 1:
            results["llm_judge"] = results[f"llm_judge_{judge_models[0].replace('.', '_').replace('-', '_')}"]

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation results saved to {output_path}")
    logger.info(f"Results: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
