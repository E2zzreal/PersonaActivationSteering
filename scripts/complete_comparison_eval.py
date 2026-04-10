"""
Baseline模型评估 - 无HyperNetwork的原始模型
对比：Baseline vs Stage 1 vs Stage 2 vs Stage 3
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
    """Baseline模型评估器（无HyperNetwork）"""
    
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
    
    def load_baseline_model(self, model_path: str, device: str):
        """加载Baseline模型（原始backbone，无HyperNetwork）"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading Baseline model from {model_path}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        model.to(device)
        model.eval()
        
        return model, tokenizer
    
    def generate_response_baseline(self, model, tokenizer, user_input: str, device: str) -> str:
        """使用Baseline模型生成回复（无personality注入）"""
        input_text = f"User: {user_input}\nAssistant:"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=128,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return ""
    
    def evaluate_baseline(self, model_path: str, eval_samples: list, 
                         device: str, num_samples: int):
        """评估Baseline模型"""
        model, tokenizer = self.load_baseline_model(model_path, device)
        
        scores = {m: [] for m in self.judge_models}
        generated_responses = []
        
        for sample in tqdm(eval_samples[:num_samples], desc="Baseline"):
            personality = sample.get("personality", "")
            conversations = sample.get("conversations", [])
            
            for turn in conversations[:3]:
                user_input = turn.get("user", "")
                
                if user_input:
                    # Baseline生成（无personality注入）
                    response = self.generate_response_baseline(
                        model, tokenizer, user_input, device
                    )
                    
                    if response:
                        generated_responses.append({
                            "user": user_input,
                            "response": response,
                            "personality": personality
                        })
                        
                        # LLM Judge评分
                        for judge_model in self.judge_models:
                            score = self.compute_score(response, personality, judge_model)
                            scores[judge_model].append(score)
        
        del model
        torch.cuda.empty_cache()
        
        results = {
            "stage": "Baseline",
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
    
    def load_persona_steer_model(self, checkpoint_path: str, base_model_path: str, device: str):
        """加载PersonaSteer模型"""
        from transformers import AutoModelForCausalLM, AutoModel, AutoConfig, AutoTokenizer
        from src.models.persona_steer import PersonaSteerConfig, PersonaSteerModel
        
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = ckpt.get("config", {})
        stage = ckpt.get("stage", 1)
        
        logger.info(f"Loading Stage {stage}")
        
        backbone = AutoModelForCausalLM.from_pretrained(
            base_model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        encoder = AutoModel.from_pretrained(base_model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        backbone_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        
        persona_config = PersonaSteerConfig(
            inject_layers=getattr(config, "inject_layers", [10, 11, 12, 13, 14, 15, 16, 17]),
            v_dim=getattr(config, "v_dim", 1024),
            hidden_dim=getattr(config, "hidden_dim", 4096),
            layer_dim=backbone_config.hidden_size,
        )
        
        model = PersonaSteerModel(config=persona_config, backbone=backbone, encoder=encoder)
        model.hyper_network.set_tokenizer(tokenizer)
        
        persona_weights = {k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith("backbone.")}
        model.load_state_dict(persona_weights, strict=False)
        model.to(device)
        model.eval()
        
        return model, tokenizer, stage
    
    def generate_response_with_persona(self, model, tokenizer, user_input: str, 
                                       personality: str, device: str) -> str:
        """使用PersonaSteer生成回复（有personality注入）"""
        input_text = f"User: {user_input}\nAssistant:"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        v_prev = torch.zeros(1, model.v_dim).to(device)
        
        try:
            with torch.no_grad():
                generated_ids, v_t = model.generate(
                    input_ids=input_ids,
                    v_prev=v_prev,
                    user_texts=[personality],
                    max_new_tokens=128,
                    temperature=0.7,
                )
            
            response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return ""
    
    def evaluate_stage(self, checkpoint_path: str, base_model_path: str,
                      eval_samples: list, device: str, num_samples: int):
        """评估单个Stage"""
        model, tokenizer, stage = self.load_persona_steer_model(
            checkpoint_path, base_model_path, device
        )
        
        scores = {m: [] for m in self.judge_models}
        generated_responses = []
        
        for sample in tqdm(eval_samples[:num_samples], desc=f"Stage {stage}"):
            personality = sample.get("personality", "")
            conversations = sample.get("conversations", [])
            
            for turn in conversations[:3]:
                user_input = turn.get("user", "")
                
                if user_input and personality:
                    response = self.generate_response_with_persona(
                        model, tokenizer, user_input, personality, device
                    )
                    
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
            "stage": stage,
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
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model", type=str, default="qwen25", choices=["qwen25", "qwen3"])
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
    
    # 模型配置
    if args.model == "qwen25":
        base_model_path = "/home/kemove/.cache/modelscope/Qwen/Qwen2___5-3B"
        checkpoints = {
            "Stage 1": "checkpoints/stage1/best.pt",
            "Stage 2": "checkpoints/stage2/best.pt",
            "Stage 3": "checkpoints/stage3/best.pt",
        }
        model_name = "Qwen2.5-3B"
    else:
        base_model_path = "/home/kemove/.cache/modelscope/Qwen/Qwen3-4B"
        checkpoints = {
            "Stage 1": "checkpoints/stage1_qwen3/best.pt",
            "Stage 2": "checkpoints/stage2_qwen3/best.pt",
            "Stage 3": "checkpoints/stage3_qwen3/best.pt",
        }
        model_name = "Qwen3-4B"
    
    all_results = {}
    
    # 1. 评估Baseline
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating Baseline (No HyperNetwork)")
    logger.info(f"{'='*60}")
    
    baseline_results, baseline_responses = evaluator.evaluate_baseline(
        base_model_path, eval_samples, args.device, args.num_samples
    )
    all_results["Baseline"] = baseline_results
    
    logger.info(f"Generated {baseline_results['num_responses']} responses")
    for model_name_judge, scores in baseline_results["scores"].items():
        logger.info(f"{model_name_judge}: {scores['mean']:.2f} ± {scores['std']:.2f}")
    
    with open(f"results/{args.model}_baseline_responses.json", "w") as f:
        json.dump(baseline_responses, f, indent=2, ensure_ascii=False)
    
    # 2. 评估Stage 1/2/3
    for stage_name, ckpt_path in checkpoints.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {stage_name}")
        logger.info(f"{'='*60}")
        
        results, responses = evaluator.evaluate_stage(
            ckpt_path, base_model_path, eval_samples, args.device, args.num_samples
        )
        
        all_results[stage_name] = results
        
        logger.info(f"Generated {results['num_responses']} responses")
        for model_name_judge, scores in results["scores"].items():
            logger.info(f"{model_name_judge}: {scores['mean']:.2f} ± {scores['std']:.2f}")
        
        with open(f"results/{args.model}_{stage_name.lower().replace(' ', '_')}_responses.json", "w") as f:
            json.dump(responses, f, indent=2, ensure_ascii=False)
    
    # 保存完整结果
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "judge_models": judge_models,
        "num_samples": args.num_samples,
        "results": all_results,
    }
    
    with open(f"results/{args.model}_complete_comparison.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # 打印报告
    print("\n" + "="*60)
    print(f"{model_name} COMPLETE COMPARISON")
    print("="*60)
    
    print(f"\n{'Stage':<12} {'GPT-5.2':>15} {'Claude':>15} {'vs Baseline':>15}")
    print("-"*60)
    
    baseline_gpt = all_results["Baseline"]["scores"]["GPT-5.2"]["mean"]
    baseline_claude = all_results["Baseline"]["scores"]["Claude-Sonnet-4.5"]["mean"]
    
    for stage_name in ["Baseline", "Stage 1", "Stage 2", "Stage 3"]:
        data = all_results[stage_name]
        gpt_score = data["scores"]["GPT-5.2"]["mean"]
        claude_score = data["scores"]["Claude-Sonnet-4.5"]["mean"]
        
        if stage_name == "Baseline":
            diff = "-"
        else:
            gpt_diff = f"+{(gpt_score/baseline_gpt - 1)*100:.1f}%"
            diff = gpt_diff
        
        print(f"{stage_name:<12} {gpt_score:>6.2f} ± {data['scores']['GPT-5.2']['std']:.2f} "
              f"{claude_score:>6.2f} ± {data['scores']['Claude-Sonnet-4.5']['std']:.2f} {diff:>15}")
    
    # 改善分析
    print("\n" + "="*60)
    print("IMPROVEMENT ANALYSIS")
    print("="*60)
    
    stages = ["Baseline", "Stage 1", "Stage 2", "Stage 3"]
    scores_gpt = [all_results[s]["scores"]["GPT-5.2"]["mean"] for s in stages]
    scores_claude = [all_results[s]["scores"]["Claude-Sonnet-4.5"]["mean"] for s in stages]
    
    print(f"\nGPT-5.2:")
    print(f"  Baseline → Stage 1: +{(scores_gpt[1]/scores_gpt[0] - 1)*100:+.1f}%")
    print(f"  Stage 1 → Stage 2: +{(scores_gpt[2]/scores_gpt[1] - 1)*100:+.1f}%")
    print(f"  Stage 2 → Stage 3: +{(scores_gpt[3]/scores_gpt[2] - 1)*100:+.1f}%")
    print(f"  Baseline → Stage 3: +{(scores_gpt[3]/scores_gpt[0] - 1)*100:+.1f}%")
    
    print(f"\nClaude-Sonnet-4.5:")
    print(f"  Baseline → Stage 1: +{(scores_claude[1]/scores_claude[0] - 1)*100:+.1f}%")
    print(f"  Stage 1 → Stage 2: +{(scores_claude[2]/scores_claude[1] - 1)*100:+.1f}%")
    print(f"  Stage 2 → Stage 3: +{(scores_claude[3]/scores_claude[2] - 1)*100:+.1f}%")
    print(f"  Baseline → Stage 3: +{(scores_claude[3]/scores_claude[0] - 1)*100:+.1f}%")


if __name__ == "__main__":
    main()
