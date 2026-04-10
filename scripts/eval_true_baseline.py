#!/usr/bin/env python3
"""真正的Baseline评估 - 直接使用原始base model，没有HyperNetwork"""

import torch
import json
import sys
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.aloe_dataset import ALOEDataset
from src.data.collator import PersonaSteerCollator
from src.training.losses import compute_sft_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_true_baseline(base_model_path, data_path, device="cuda"):
    """评估真正的baseline - 原始base model，无注入"""
    
    print("="*80)
    print("真正的Baseline评估 - 原始Base Model")
    print("="*80)
    print(f"Base Model: {base_model_path}")
    print(f"Device: {device}")
    print()
    
    # 加载原始模型
    logger.info(f"Loading base model from {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # 创建数据加载器
    dataset = ALOEDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_turns=6,
    )
    logger.info(f"Dataset: {len(dataset)} samples")
    
    collator = PersonaSteerCollator(tokenizer)
    eval_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )
    
    # 评估
    losses = []
    ppls = []
    
    logger.info("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass - 直接使用base model
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            
            # 计算loss
            loss = compute_sft_loss(logits, labels)
            losses.append(loss.item())
            ppls.append(torch.exp(loss).item())
    
    # 汇总结果
    avg_loss = sum(losses) / len(losses)
    avg_ppl = sum(ppls) / len(ppls)
    
    print("\n" + "="*80)
    print("Baseline评估结果")
    print("="*80)
    print(f"Loss: {avg_loss:.4f}")
    print(f"Perplexity: {avg_ppl:.2f}")
    print(f"样本数: {len(losses)}")
    print()
    
    return {
        "loss": avg_loss,
        "perplexity": avg_ppl,
        "num_samples": len(losses),
    }

if __name__ == "__main__":
    results = evaluate_true_baseline(
        base_model_path="/home/kemove/.cache/modelscope/Qwen/Qwen2___5-3B",
        data_path="/home/kemove/Desktop/Projects/3-PersonaSteer_V2/data/split/val.jsonl",
        device="cuda:0",
    )
    
    # 保存结果
    with open("/home/kemove/Desktop/Projects/3-PersonaSteer_V2/results/true_baseline_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("结果已保存到: results/true_baseline_eval.json")
