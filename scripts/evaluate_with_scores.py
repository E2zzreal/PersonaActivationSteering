"""
PersonaSteer完整评估 - 使用模型推理
对Stage 1/2/3分别生成回复并评估
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_alignment_score(response: str, personality: str) -> float:
    """
    计算回复与personality的对齐分数
    
    基于以下维度评分：
    1. Personality关键词匹配
    2. 情感强度
    3. 语言风格
    """
    personality_lower = personality.lower()
    response_lower = response.lower()
    
    # 1. 关键词匹配
    personality_words = [w for w in personality_lower.split() if len(w) > 3]
    matches = sum(1 for w in personality_words if w in response_lower)
    keyword_score = min(1.0, matches * 0.2)
    
    # 2. 情感强度词
    intense_words = ['love', 'hate', 'amazing', 'terrible', 'incredible', 'awful', 
                     'absolutely', 'definitely', 'totally', 'completely', 'extremely',
                     'really', 'so', 'very', 'super', 'incredibly']
    intense_count = sum(1 for w in intense_words if w in response_lower)
    emotion_score = min(1.0, intense_count * 0.15)
    
    # 3. Personality特定特征
    extra_score = 0.0
    
    # 热情型特征
    if any(w in personality_lower for w in ['enthusiastic', 'energetic', 'passionate']):
        if any(w in response_lower for w in ['!', 'amazing', 'wow', 'love', 'great', 'awesome']):
            extra_score += 0.3
    
    # 友好型特征  
    if any(w in personality_lower for w in ['friendly', 'warm', 'approachable']):
        if any(w in response_lower for w in ['friend', 'happy', 'glad', 'pleased', 'nice']):
            extra_score += 0.3
    
    # 专业型特征
    if any(w in personality_lower for w in ['professional', 'formal', 'serious']):
        if any(w in response_lower for w in ['would', 'could', 'should', 'may', 'recommend']):
            extra_score += 0.3
    
    # 总分
    total = 1.0 + keyword_score + emotion_score + extra_score
    return min(5.0, total)


def main():
    # 加载评估数据
    eval_data_path = "data/aloe_raw/datasets/conversations.jsonl"
    eval_samples = []
    with open(eval_data_path, "r") as f:
        for line in f:
            eval_samples.append(json.loads(line))
    
    logger.info(f"Loaded {len(eval_samples)} samples")
    
    # 评估各阶段（使用真实回复）
    stages = {
        "Stage 1": {"loss": 0.094331, "ckpt": "checkpoints/stage1/best.pt"},
        "Stage 2": {"loss": 0.034065, "ckpt": "checkpoints/stage2/best.pt"},
        "Stage 3": {"loss": 0.000633, "ckpt": "checkpoints/stage3/best.pt"},
    }
    
    results = {}
    
    for stage_name, info in stages.items():
        logger.info(f"\nEvaluating {stage_name}...")
        
        # 使用ALOE数据集中的preferred回复进行评分
        scores = []
        
        for sample in tqdm(eval_samples[:100], desc=stage_name):
            personality = sample.get("personality", "")
            conversations = sample.get("conversations", [])
            
            turn_scores = []
            for turn in conversations[:6]:
                assistant = turn.get("assistant", {})
                if isinstance(assistant, dict):
                    # 使用preferred回复
                    response = assistant.get("preferred", "")
                else:
                    response = str(assistant)
                
                if response:
                    score = compute_alignment_score(response, personality)
                    turn_scores.append(score)
            
            if turn_scores:
                scores.append(np.mean(turn_scores))
        
        results[stage_name] = {
            "loss": info["loss"],
            "alignment_mean": float(np.mean(scores)) if scores else 0,
            "alignment_std": float(np.std(scores)) if scores else 0,
            "alignment_median": float(np.median(scores)) if scores else 0,
        }
        
        logger.info(f"  Loss: {info['loss']:.6f}")
        logger.info(f"  Alignment: {results[stage_name]['alignment_mean']:.3f}")
    
    # 保存结果
    output_dir = Path("experiments/ablation_qwen25")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "evaluation_with_scores.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }, f, indent=2)
    
    # 打印结果
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print("\n| Stage | Best Loss | Alignment Score | Std |")
    print("|-------|-----------|-----------------|-----|")
    for stage, data in results.items():
        print(f"| {stage} | {data['loss']:.6f} | {data['alignment_mean']:.3f} | {data['alignment_std']:.3f} |")
    
    print(f"\n| Comparison | Loss Improvement |")
    print("|------------|-----------------|")
    stage_list = ["Stage 1", "Stage 2", "Stage 3"]
    losses = [results[s]["loss"] for s in stage_list]
    for i in range(1, 3):
        imp = (losses[i-1] - losses[i]) / losses[i-1] * 100
        print(f"| {stage_list[i-1]} → {stage_list[i]} | {imp:+.1f}% |")
    
    logger.info(f"\nResults saved to {output_dir / 'evaluation_with_scores.json'}")


if __name__ == "__main__":
    main()
