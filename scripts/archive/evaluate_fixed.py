#!/usr/bin/env python
"""
PersonaSteer 评估脚本 - 修复版
正确加载backbone和encoder，支持LLM Judge评估
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForCausalLM

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.aloe_dataset import ALOEDataset
from src.data.collator import PersonaSteerCollator
from src.evaluation.auto_metrics import AutoMetricsEvaluator
from src.evaluation.llm_judge import LLMJudgeEvaluator
from src.models.persona_steer import PersonaSteerModel, PersonaSteerConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PersonaSteer model (fixed)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (required if not baseline)")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--base_model", type=str, default=None, help="Base model path")
    parser.add_argument("--data", type=str, default="data/split/val.jsonl", help="Eval data path")
    parser.add_argument("--output", type=str, default="experiments/eval_results.json", help="Output path")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--baseline", action="store_true", help="Run baseline evaluation (no injection)")
    parser.add_argument("--judge_model", type=str, default="Claude-Sonnet-4.6", help="LLM Judge model")
    parser.add_argument("--skip_llm_judge", action="store_true", help="Skip LLM Judge evaluation")
    return parser.parse_args()


def load_config(config_path):
    if config_path is None:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_model_with_backbone(config, base_model_path, device="cuda"):
    """创建模型并加载backbone和encoder"""
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    
    model_config = config.get("model", {})
    
    # 加载backbone配置
    logger.info(f"Loading backbone config from {base_model_path}")
    backbone_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    actual_layer_dim = backbone_config.hidden_size
    logger.info(f"Backbone hidden_size: {actual_layer_dim}")
    
    # 创建PersonaSteer配置
    persona_config = PersonaSteerConfig(
        inject_layers=model_config.get("inject_layers", [10, 11, 12, 13, 14, 15, 16, 17]),
        v_dim=model_config.get("v_dim", 1024),
        hidden_dim=model_config.get("hidden_dim", 4096),
        layer_dim=actual_layer_dim,
        gate_hidden_dim=model_config.get("gate_hidden_dim", 256),
    )
    
    # 加载encoder
    logger.info(f"Loading encoder from {base_model_path}")
    # 将encoder放在CPU上以节省GPU内存，使用float16
    encoder = AutoModel.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16  # 使用float16减少内存
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # 创建模型
    model = PersonaSteerModel(config=persona_config, encoder=encoder)
    
    # 设置v_norm_clip
    v_norm_clip = model_config.get("v_norm_clip", 10.0)
    if hasattr(model, 'hyper_network') and model.hyper_network is not None:
        model.hyper_network.v_norm_clip = v_norm_clip
        logger.info(f"Set v_norm_clip={v_norm_clip}")
    
    # 设置tokenizer
    if hasattr(model, 'hyper_network') and model.hyper_network is not None:
        model.hyper_network._tokenizer = tokenizer
    
    # 加载backbone
    logger.info(f"Loading backbone from {base_model_path}")
    backbone = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )
    
    # 设置backbone
    model.set_backbone(backbone)
    
    return model, tokenizer


def load_checkpoint(model, checkpoint_path, device="cuda"):
    """加载checkpoint到模型"""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    # 加载参数
    model_state = model.state_dict()
    loaded_keys = []
    for key in state_dict:
        if key in model_state:
            model_state[key] = state_dict[key]
            loaded_keys.append(key)
    
    model.load_state_dict(model_state)
    logger.info(f"Loaded {len(loaded_keys)} parameters from checkpoint")
    
    if "best_loss" in checkpoint:
        logger.info(f"Checkpoint best_loss: {checkpoint['best_loss']}")
    
    return model


def create_eval_loader(data_path, tokenizer, config):
    """创建评估数据加载器"""
    data_config = config.get("data", {})
    max_turns = data_config.get("max_turns", 6)
    batch_size = data_config.get("batch_size", 1)  # 减小batch size以节省内存
    
    dataset = ALOEDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_turns=max_turns,
    )
    
    logger.info(f"Evaluation dataset: {len(dataset)} samples")
    
    collator = PersonaSteerCollator(tokenizer)
    
    eval_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )
    
    return eval_loader


def load_test_samples(data_path, num_samples=None):
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
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 确定base_model路径
    base_model_path = args.base_model
    if base_model_path is None:
        base_model_path = config.get("base_model", "/home/kemove/.cache/modelscope/Qwen/Qwen2___5-3B")
    
    # 创建模型
    model, tokenizer = create_model_with_backbone(config, base_model_path, args.device)
    
    # 【V4修复】Baseline评估时禁用注入
    if args.baseline:
        model.baseline_mode = True
        logger.info("Baseline mode enabled - injection disabled")
    
    # 如果不是baseline评估，加载checkpoint
    if not args.baseline:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when not running baseline evaluation")
        model = load_checkpoint(model, args.checkpoint, args.device)
    
    model.to(args.device)
    model.eval()
    
    # 将encoder移回CPU以节省GPU内存
    if hasattr(model, 'hyper_network') and model.hyper_network is not None:
        model.hyper_network.encoder = model.hyper_network.encoder.cpu()
        logger.info("Moved encoder to CPU to save GPU memory")
    
    # 清理内存
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    logger.info("Memory cleaned up")
    
    # 创建数据加载器
    eval_loader = create_eval_loader(args.data, tokenizer, config)
    
    # 加载测试样本（用于LLM Judge）
    test_samples = load_test_samples(args.data, args.num_samples)
    
    # 运行自动指标评估
    logger.info("Running auto metrics evaluation...")
    evaluator = AutoMetricsEvaluator(device=args.device)
    auto_results = evaluator.evaluate(model, eval_loader)
    
    # 释放evaluator以节省GPU内存
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Evaluator released, GPU memory freed")
    
    # 保存结果
    output = {
        "checkpoint": args.checkpoint if not args.baseline else "baseline",
        "num_eval_samples": len(test_samples),
        "auto_metrics": auto_results,
    }
    
    # LLM Judge评估
    if not args.skip_llm_judge:
        # 释放模型以释放GPU内存，然后重新加载（避免OOM）
        logger.info("Unloading model to free GPU memory for LLM Judge phase...")
        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        import time
        time.sleep(3)  # 等待GPU驱动完全释放显存
        gc.collect()
        torch.cuda.empty_cache()
        
        # 重新加载模型（在CPU上先加载，再移到GPU）
        logger.info("Reloading model for LLM Judge...")
        model, _ = create_model_with_backbone(config, base_model_path, "cpu")
        if not args.baseline:
            model = load_checkpoint(model, args.checkpoint, "cpu")
        gc.collect()
        torch.cuda.empty_cache()
        model.to(args.device)
        model.eval()
        if hasattr(model, 'hyper_network') and model.hyper_network is not None:
            model.hyper_network.encoder = model.hyper_network.encoder.cpu()
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"Running LLM Judge evaluation with {args.judge_model}...")
        base_url = "https://llmapi.blsc.cn"
        judge = LLMJudgeEvaluator(
            judge_model=args.judge_model,
            base_url=base_url
        )
        llm_results = judge.evaluate_alignment(
            model=model,
            test_samples=test_samples,
            tokenizer=tokenizer,
        )
        output["llm_judge"] = llm_results
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Auto metrics: {json.dumps(auto_results, indent=2)}")
    if not args.skip_llm_judge and "llm_judge" in output:
        logger.info(f"LLM Judge: {json.dumps(output['llm_judge'], indent=2)}")


if __name__ == "__main__":
    main()
