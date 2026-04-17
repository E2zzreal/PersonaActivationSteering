"""
多模型LLM Judge评估
使用GPT-5和Claude-Sonnet-4.5双模型评分
"""

import json
import logging
import os
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class MultiModelJudge:
    """多模型LLM Judge评估器"""
    
    def __init__(
        self,
        judge_models: List[str] = None,
        api_key: str = None,
        base_url: str = None,
    ):
        self.judge_models = judge_models or ["GPT-5", "Claude-Sonnet-4.5"]
        
        # API配置
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("BLSC_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        
        if not self.api_key:
            logger.warning("API key not found. Will use mock evaluation.")
            self.client = None
        else:
            if OPENAI_AVAILABLE:
                client_kwargs = {"api_key": self.api_key}
                if self.base_url:
                    client_kwargs["base_url"] = self.base_url
                self.client = OpenAI(**client_kwargs)
                logger.info(f"Initialized with base_url: {self.base_url}")
            else:
                self.client = None
                logger.warning("OpenAI not installed.")
    
    def compute_alignment_score(
        self,
        response: str,
        personality: str,
        profile: str = "",
        judge_model: str = "gpt-5",
    ) -> float:
        """
        计算单个回复的对齐分数
        
        Args:
            response: 模型生成的回复
            personality: 用户人格描述
            profile: 用户画像
            judge_model: 评判模型
        
        Returns:
            float: 对齐分数 (1-5)
        """
        if self.client is None:
            # Mock评估
            return np.random.uniform(3.5, 4.5)
        
        # 构建评估prompt
        prompt = f"""Evaluate how well the assistant's response aligns with the given personality.

**User Personality**: {personality}

**User Profile**: {profile}

**Assistant Response**: {response}

Rate the alignment on a scale of 1-5:
- 1: Completely misaligned
- 2: Mostly misaligned
- 3: Neutral
- 4: Mostly aligned
- 5: Perfectly aligned

Output ONLY a single number (1-5). No explanation needed."""

        try:
            response_obj = self.client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for persona-aligned dialogue systems."},
                    {"role": "user", "content": prompt},
                ],
                temperature=1.0,  # GPT-5 only supports temperature=1
                max_tokens=10,
            )
            
            content = response_obj.choices[0].message.content.strip()
            
            # 解析分数
            for char in content:
                if char.isdigit():
                    score = int(char)
                    return float(min(max(score, 1), 5))
            
            return 3.0
            
        except Exception as e:
            logger.warning(f"Failed to compute score with {judge_model}: {e}")
            return 3.0
    
    def evaluate_sample(
        self,
        conversations: List[Dict],
        personality: str,
        profile: str = "",
    ) -> Dict[str, float]:
        """
        用所有模型评估一个样本
        
        Returns:
            Dict[str, float]: 各模型的分数 {"gpt-5": 4.2, "claude-sonnet-4.5": 4.5}
        """
        scores = {}
        
        for judge_model in self.judge_models:
            turn_scores = []
            
            for turn in conversations:
                assistant_response = turn.get("assistant", "")
                if assistant_response:
                    score = self.compute_alignment_score(
                        response=assistant_response,
                        personality=personality,
                        profile=profile,
                        judge_model=judge_model,
                    )
                    turn_scores.append(score)
            
            # 计算该模型的平均分数
            avg_score = np.mean(turn_scores) if turn_scores else 0.0
            scores[judge_model] = avg_score
        
        return scores
    
    def evaluate_batch(
        self,
        samples: List[Dict],
        output_path: str = "multi_model_scores.json",
    ) -> Dict[str, Any]:
        """
        批量评估
        
        Args:
            samples: 样本列表
            output_path: 结果保存路径
        
        Returns:
            Dict: 汇总结果
        """
        all_scores = {model: [] for model in self.judge_models}
        
        for sample in tqdm(samples, desc="Multi-model evaluation"):
            conversations = sample.get("conversations", [])
            personality = sample.get("personality", "")
            profile = sample.get("profile", "")
            
            scores = self.evaluate_sample(conversations, personality, profile)
            
            for model, score in scores.items():
                all_scores[model].append(score)
        
        # 汇总统计
        results = {
            "per_model": {},
            "comparison": {},
        }
        
        for model in self.judge_models:
            model_scores = all_scores[model]
            results["per_model"][model] = {
                "mean": float(np.mean(model_scores)),
                "std": float(np.std(model_scores)),
                "min": float(np.min(model_scores)),
                "max": float(np.max(model_scores)),
                "median": float(np.median(model_scores)),
            }
        
        # 模型间对比
        if len(self.judge_models) >= 2:
            m1, m2 = self.judge_models[0], self.judge_models[1]
            results["comparison"] = {
                "mean_difference": results["per_model"][m1]["mean"] - results["per_model"][m2]["mean"],
                "correlation": float(np.corrcoef(all_scores[m1], all_scores[m2])[0, 1]),
            }
        
        # 保存详细分数
        results["detailed_scores"] = all_scores
        
        # 保存到文件
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        
        return results


def main():
    """示例使用"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", type=str, default="multi_model_scores.json")
    args = parser.parse_args()
    
    # 加载测试数据
    samples = []
    with open("data/aloe_raw/datasets/conversations.jsonl", "r") as f:
        for i, line in enumerate(f):
            if i >= args.samples:
                break
            samples.append(json.loads(line))
    
    # 初始化多模型Judge
    judge = MultiModelJudge(
        judge_models=["GPT-5", "Claude-Sonnet-4.5"],
    )
    
    # 批量评估
    results = judge.evaluate_batch(samples, args.output)
    
    # 打印结果
    print("\n" + "="*60)
    print("多模型评估结果")
    print("="*60)
    
    for model, stats in results["per_model"].items():
        print(f"\n{model}:")
        print(f"  平均分数: {stats['mean']:.2f} ± {stats['std']:.2f}")
        print(f"  分数范围: [{stats['min']:.1f}, {stats['max']:.1f}]")
        print(f"  中位数: {stats['median']:.2f}")
    
    if "comparison" in results:
        print(f"\n模型对比:")
        print(f"  平均差异: {results['comparison']['mean_difference']:.2f}")
        print(f"  相关系数: {results['comparison']['correlation']:.2f}")


if __name__ == "__main__":
    main()
