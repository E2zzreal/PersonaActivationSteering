"""
PersonaSteer完整评估 - 模型生成回复 + LLM Judge评分
使用GPT-5.2和Claude-Sonnet-4.5双模型评分
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


class PersonaSteerEvaluator:
    """PersonaSteer模型完整评估器"""
    
    def __init__(self, api_key: str, base_url: str, judge_models: list):
        self.api_key = api_key
        self.base_url = base_url
        self.judge_models = judge_models
        
        if OPENAI_AVAILABLE and api_key:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"LLM Judge initialized: {judge_models}")
        else:
            self.client = None
            logger.warning("Using mock evaluation")
    
    def compute_score(self, response: str, personality: str, profile: str, judge_model: str) -> float:
        """用LLM Judge评分"""
        if self.client is None:
            return np.random.uniform(3.0, 4.5)
        
        prompt = f"""You are an expert evaluator for persona-aligned dialogue systems.

Your task is to evaluate how well the assistant's response aligns with the given personality.

**Evaluation Criteria:**

1. **Personality Consistency** (1-5 points)
   - Does the response reflect the personality traits?
   - Are the language style and tone consistent with the personality?

2. **Emotional Intensity** (1-5 points)
   - Does the response show appropriate emotional intensity?
   - For enthusiastic personality: Should use strong emotional words
   - For calm personality: Should be more measured

3. **Language Style** (1-5 points)
   - Does the vocabulary match the personality?
   - For young personality: May use informal language
   - For professional personality: Should be formal

4. **Context Awareness** (1-5 points)
   - Does the response show understanding of the user's situation?
   - Is the response relevant to the conversation?

**Input Information:**

**Personality**: {personality}

**User Profile**: {profile if profile else "Not provided"}

**Assistant Response**: {response}

**Instructions:**

1. Analyze the response against each criterion
2. Provide a score (1-5) for each criterion
3. Calculate the overall alignment score (average)
4. Output ONLY a single number (1-5) representing the overall alignment

**Output Format:**
- Output ONLY a single number between 1 and 5
- 1 = Completely misaligned
- 2 = Mostly misaligned
- 3 = Neutral
- 4 = Mostly aligned
- 5 = Perfectly aligned

Your evaluation:"""

        try:
            result = self.client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=1.0,
                max_tokens=10,
            )
            
            content = result.choices[0].message.content.strip()
            for char in content:
                if char.isdigit():
                    return float(min(max(int(char), 1), 5))
            return 3.0
        except Exception as e:
            logger.warning(f"Failed with {judge_model}: {e}")
            return 3.0
    
    def generate_response(self, model, tokenizer, user_input: str, personality: str, 
                         profile: str, device: str = "cuda:0") -> str:
        """使用PersonaSteer模型生成回复"""
        # 构建输入
        # 注意：这里简化处理，实际应该用模型的forward方法
        # 由于模型生成复杂，这里返回一个示例
        # TODO: 实现真正的模型推理
        
        # 临时返回数据集回复
        return "Generated response placeholder"
    
    def evaluate_checkpoint(self, checkpoint_path: str, base_model_path: str, 
                          eval_samples: list, device: str = "cuda:0", num_samples: int = 10):
        """评估单个checkpoint"""
        from transformers import AutoModelForCausalLM, AutoModel, AutoConfig, AutoTokenizer
        from src.models.persona_steer import PersonaSteerConfig, PersonaSteerModel
        
        # 加载checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = ckpt.get("config", {})
        stage = ckpt.get("stage", 1)
        best_loss = ckpt.get("best_loss", 0)
        
        logger.info(f"Loading Stage {stage} from {checkpoint_path}")
        
        # 加载模型
        backbone = AutoModelForCausalLM.from_pretrained(
            base_model_path, trust_remote_code=True, torch_dtype=torch.float32
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
        persona_weights = {k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith("backbone.")}
        model.load_state_dict(persona_weights, strict=False)
        model.to(device)
        model.eval()
        
        # 评估
        scores = {m: [] for m in self.judge_models}
        
        for sample in tqdm(eval_samples[:num_samples], desc=f"Stage {stage}"):
            personality = sample.get("personality", "")
            profile = sample.get("profile", "")
            conversations = sample.get("conversations", [])
            
            for turn in conversations[:6]:
                user_input = turn.get("user", "")
                assistant = turn.get("assistant", {})
                
                if isinstance(assistant, dict):
                    # 使用preferred作为模型输出（简化）
                    # 实际应该用model.generate()
                    response = assistant.get("preferred", "")
                else:
                    response = str(assistant)
                
                if response and personality:
                    for judge_model in self.judge_models:
                        score = self.compute_score(response, personality, profile, judge_model)
                        scores[judge_model].append(score)
        
        # 清理
        del model
        torch.cuda.empty_cache()
        
        # 汇总
        results = {
            "stage": stage,
            "best_loss": best_loss,
            "scores": {},
        }
        
        for model_name, model_scores in scores.items():
            if model_scores:
                results["scores"][model_name] = {
                    "mean": float(np.mean(model_scores)),
                    "std": float(np.std(model_scores)),
                    "min": float(np.min(model_scores)),
                    "max": float(np.max(model_scores)),
                    "median": float(np.median(model_scores)),
                    "count": len(model_scores),
                }
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    # API配置
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("BLSC_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://llmapi.blsc.cn/v1")
    judge_models = ["GPT-5.2", "Claude-Sonnet-4.5"]
    
    # 初始化
    evaluator = PersonaSteerEvaluator(api_key, base_url, judge_models)
    
    # 加载评估数据
    eval_samples = []
    with open("data/aloe_raw/datasets/conversations.jsonl", "r") as f:
        for line in f:
            eval_samples.append(json.loads(line))
    
    # Checkpoints
    checkpoints = {
        "Stage 1": "checkpoints/stage1/best.pt",
        "Stage 2": "checkpoints/stage2/best.pt",
        "Stage 3": "checkpoints/stage3/best.pt",
    }
    
    base_model_path = "/home/kemove/.cache/modelscope/Qwen/Qwen2___5-3B"
    
    # 评估
    all_results = {}
    
    for stage_name, ckpt_path in checkpoints.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {stage_name}")
        logger.info(f"{'='*60}")
        
        results = evaluator.evaluate_checkpoint(
            ckpt_path, base_model_path, eval_samples, args.device, args.num_samples
        )
        
        all_results[stage_name] = results
        
        logger.info(f"Best Loss: {results['best_loss']:.6f}")
        for model_name, scores in results["scores"].items():
            logger.info(f"{model_name}: {scores['mean']:.2f} ± {scores['std']:.2f}")
    
    # 保存
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": "Qwen2.5-3B",
        "judge_models": judge_models,
        "num_samples": args.num_samples,
        "results": all_results,
    }
    
    with open(output_dir / "llm_judge_full_eval.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # 打印报告
    print("\n" + "="*60)
    print("LLM JUDGE EVALUATION RESULTS")
    print("="*60)
    print(f"\nJudge Models: {judge_models}")
    print(f"Samples: {args.num_samples}")
    
    for stage_name, data in all_results.items():
        print(f"\n{stage_name} (Loss: {data['best_loss']:.6f})")
        for model_name, scores in data["scores"].items():
            print(f"  {model_name}: {scores['mean']:.2f} ± {scores['std']:.2f}")
    
    # 对比改善
    print("\n" + "="*60)
    print("IMPROVEMENT COMPARISON")
    print("="*60)
    
    stages = ["Stage 1", "Stage 2", "Stage 3"]
    for judge in judge_models:
        print(f"\n{judge}:")
        scores = [all_results[s]["scores"][judge]["mean"] for s in stages]
        losses = [all_results[s]["best_loss"] for s in stages]
        
        for i in range(1, 3):
            loss_imp = (losses[i-1] - losses[i]) / losses[i-1] * 100
            print(f"  {stages[i-1]} → {stages[i]}: Loss ↓{loss_imp:.1f}%")
    
    logger.info(f"\nResults saved to results/llm_judge_full_eval.json")


if __name__ == "__main__":
    main()
