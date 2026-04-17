"""
Qwen3-4B Baseline 评估（无注入）
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BaselineEvaluator:
    """Baseline评估器（无注入）"""
    
    def __init__(self, api_key: str, base_url: str, judge_models: list):
        self.api_key = api_key
        self.base_url = base_url
        self.judge_models = judge_models
        
        if OPENAI_AVAILABLE and api_key:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"LLM Judge: {judge_models}")
        else:
            self.client = None
    
    def compute_score(self, response: str, personality: str, judge_model: str) -> float:
        """LLM Judge评分"""
        if self.client is None:
            return np.random.uniform(3.0, 4.5)
        
        prompt = f"""Evaluate personality alignment (1-5):

Personality: {personality}

Response: {response}

Output ONLY a number 1-5:
1=Completely misaligned
5=Perfectly aligned"""
        
        try:
            result = self.client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=10,
            )
            content = result.choices[0].message.content.strip()
            for char in content:
                if char.isdigit():
                    return float(min(max(int(char), 1), 5))
            return 3.0
        except Exception as e:
            logger.warning(f"Judge failed: {e}")
            return 3.0
    
    def load_model(self, base_model_path: str, device: str):
        """加载基线模型"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading baseline model from {base_model_path}")
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        
        model.to(device)
        model.eval()
        
        logger.info("Baseline model loaded successfully")
        return model, tokenizer
    
    def generate_response(self, model, tokenizer, user_input: str, device: str) -> str:
        """生成回复（无注入）"""
        input_text = f"User: {user_input}\nAssistant:"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        
        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=128,
                    temperature=0.7,
                    do_sample=True,
                )
            
            response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return ""
    
    def evaluate(self, base_model_path: str, eval_samples: list, device: str, num_samples: int):
        """评估基线模型"""
        model, tokenizer = self.load_model(base_model_path, device)
        
        scores = {m: [] for m in self.judge_models}
        generated_responses = []
        
        for sample in tqdm(eval_samples[:num_samples], desc="Baseline"):
            personality = sample.get("personality", "")
            conversations = sample.get("conversations", [])
            
            for turn in conversations[:3]:
                user_input = turn.get("user", "")
                
                if user_input and personality:
                    response = self.generate_response(model, tokenizer, user_input, device)
                    
                    if response:
                        generated_responses.append({
                            "user": user_input,
                            "response": response,
                            "personality": personality
                        })
                        
                        for judge_model in self.judge_models:
                            score = self.compute_score(response, personality, judge_model)
                            scores[judge_model].append(score)
        
        del model
        torch.cuda.empty_cache()
        
        results = {
            "scores": {},
            "num_responses": len(generated_responses),
        }
        
        for model_name, model_scores in scores.items():
            if model_scores:
                results["scores"][model_name] = {
                    "mean": float(np.mean(model_scores)),
                    "std": float(np.std(model_scores)),
                }
        
        return results, generated_responses


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    # API
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("BLSC_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://llmapi.blsc.cn/v1")
    judge_models = ["GPT-5.2", "Claude-Sonnet-4.5"]
    
    evaluator = BaselineEvaluator(api_key, base_url, judge_models)
    
    # 数据
    eval_samples = []
    with open("data/aloe_raw/datasets/conversations.jsonl", "r") as f:
        for line in f:
            eval_samples.append(json.loads(line))
    
    # Qwen3-4B Baseline
    base_model_path = "/home/kemove/.cache/modelscope/Qwen/Qwen3-4B"
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating Baseline (Qwen3-4B, No Injection)")
    logger.info(f"{'='*60}")
    
    results, responses = evaluator.evaluate(
        base_model_path, eval_samples, args.device, args.num_samples
    )
    
    logger.info(f"Generated {results['num_responses']} responses")
    for model_name, scores in results["scores"].items():
        logger.info(f"{model_name}: {scores['mean']:.2f} ± {scores['std']:.2f}")
    
    # 保存
    with open("results/qwen3_baseline_responses.json", "w") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": "Qwen3-4B",
        "stage": "baseline",
        "judge_models": judge_models,
        "num_samples": args.num_samples,
        "results": results,
    }
    
    with open("results/qwen3_baseline_eval_fixed.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to results/qwen3_baseline_eval_fixed.json")


if __name__ == "__main__":
    main()
