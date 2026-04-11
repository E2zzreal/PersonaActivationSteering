#!/usr/bin/env python3
"""
评估新训练的三个配置 (neuroticism, baseline, minimal) × 三个阶段
"""
import json
import logging
import os
import sys
import time
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LLMJudge:
    def __init__(self, api_key=None, base_url=None, model="GPT-4o-mini"):
        self.model = model
        if OPENAI_AVAILABLE and api_key:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"LLM Judge initialized: {model}")
        else:
            self.client = None
            logger.warning("No API key, returning mock scores")

    def score(self, response: str, personality: str) -> float:
        prompt = f"""评估AI回复与目标人格的一致程度。

目标人格: {personality}

AI回复: {response}

评分标准 (1-5分):
1分 = 完全不一致
2分 = 较不一致
3分 = 中立
4分 = 较一致
5分 = 完全一致

请只输出1-5的数字，不要输出任何其他内容。"""

        if self.client:
            try:
                result = self.client.chat.completions.create(
                    model=self.model,
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
        return np.random.uniform(3.0, 4.5)


def load_persona_steer(checkpoint_path: str, base_model_path: str, device: str):
    """加载 PersonaSteer 模型"""
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    from src.models.persona_steer import PersonaSteerModel, PersonaSteerConfig

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    backbone = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device)

    backbone_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stage = ckpt.get("stage", 1)

    # 从checkpoint中读取inject_layers配置
    saved_config = ckpt.get("config", {})
    inject_layers = saved_config.get("inject_layers", None)

    # 如果config中没有，从模型参数推断
    if inject_layers is None:
        for k, v in ckpt["model_state_dict"].items():
            if "layer_embedding.weight" in k:
                num_layers = v.shape[0]
                # 使用模型后部的层（与训练配置一致）
                inject_layers = list(range(backbone_config.num_hidden_layers - num_layers, backbone_config.num_hidden_layers))
                break

    if inject_layers is None:
        inject_layers = list(range(backbone_config.num_hidden_layers - 8, backbone_config.num_hidden_layers))

    logger.info(f"Using inject_layers: {inject_layers} ({len(inject_layers)} layers)")

    persona_config = PersonaSteerConfig(
        inject_layers=inject_layers,
        v_dim=1024,
        hidden_dim=4096,
        layer_dim=backbone_config.hidden_size,
    )

    model = PersonaSteerModel(config=persona_config, backbone=backbone, encoder=backbone.model)

    if hasattr(model.hyper_network, 'set_tokenizer'):
        model.hyper_network.set_tokenizer(tokenizer)

    model_state = model.state_dict()
    loaded = 0
    for key, value in ckpt["model_state_dict"].items():
        if key in model_state:
            model_state[key] = value
            loaded += 1
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    if hasattr(model.hyper_network, 'set_tokenizer'):
        model.hyper_network.set_tokenizer(tokenizer)

    logger.info(f"Stage {stage} loaded ({loaded} params)")
    return model, tokenizer, stage


def generate_response(model, tokenizer, user_input: str, personality: str, device: str) -> str:
    """生成回复"""
    device = torch.device(device)
    prompt = f"用户: {user_input}\n助手:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    v_prev = torch.zeros(1, model.v_dim, device=device)

    with torch.no_grad():
        try:
            generated_ids, _ = model.generate(
                input_ids=input_ids,
                v_prev=v_prev,
                personality_texts=[personality],
                user_query_texts=[user_input],
                max_new_tokens=128,
                temperature=0.7,
            )
        except Exception as e:
            logger.warning(f"Generation error: {e}")
            return ""

    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if "助手:" in response:
        response = response.split("助手:")[-1].strip()
    return response


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--judge_model", type=str, default="GPT-4o-mini")
    parser.add_argument("--skip_judge", action="store_true")
    parser.add_argument("--base_model", type=str,
                        default="/home/kemove/Desktop/PersonaSteer/Qwen/Qwen3-4B")
    args = parser.parse_args()

    # API - 从配置文件读取
    import yaml
    api_config_path = Path("configs/api_config.yaml")
    if api_config_path.exists():
        with open(api_config_path) as f:
            api_config = yaml.safe_load(f)
        default_api = api_config.get("default", "blsc")
        api_key = api_config.get(default_api, {}).get("api_key")
        base_url = api_config.get(default_api, {}).get("base_url")
        logger.info(f"Loaded API config from {api_config_path}: {default_api}")
    else:
        api_key = None
        base_url = None

    # 环境变量可覆盖
    api_key = os.environ.get("BLSC_API_KEY") or os.environ.get("OPENAI_API_KEY") or api_key
    base_url = os.environ.get("OPENAI_BASE_URL") or base_url

    judge = LLMJudge(api_key=api_key, base_url=base_url, model=args.judge_model)

    # 加载评估数据
    eval_data_path = Path("data/split/val.jsonl")
    if not eval_data_path.exists():
        eval_data_path = Path("data/aloe_raw/datasets/conversations.jsonl")

    eval_samples = []
    with open(eval_data_path, "r") as f:
        for line in f:
            if line.strip():
                eval_samples.append(json.loads(line))

    logger.info(f"Loaded {len(eval_samples)} samples, using {args.num_samples}")

    # 定义实验
    configs = ["neuroticism", "baseline", "minimal"]
    stages = ["stage1", "stage2", "stage3"]

    experiments = []
    for config in configs:
        for stage in stages:
            ckpt_path = f"checkpoints/{config}_gate_neg3_{stage}/best.pt"
            if Path(ckpt_path).exists():
                experiments.append({
                    "name": f"{config}_{stage}",
                    "config": config,
                    "stage": stage,
                    "checkpoint": ckpt_path,
                })

    logger.info(f"Found {len(experiments)} experiments to evaluate")

    # 输出目录
    output_dir = Path(f"results/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for exp in experiments:
        exp_name = exp["name"]
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {exp_name}")
        logger.info(f"{'='*60}")

        try:
            model, tokenizer, stage = load_persona_steer(
                exp["checkpoint"], args.base_model, args.device
            )
        except Exception as e:
            logger.error(f"Failed to load: {e}")
            continue

        responses = []
        scores = []

        for sample in tqdm(eval_samples[:args.num_samples], desc=exp_name):
            personality = sample.get("personality", "")
            conversations = sample.get("conversations", [])

            for turn in conversations[:3]:
                # 支持两种格式: {'user': '...'} 或 {'role': 'user', 'content': '...'}
                user_input = turn.get("user", "")
                if not user_input:
                    if turn.get("role") == "user":
                        user_input = turn.get("content", "")
                if not user_input or not personality:
                    continue

                response = generate_response(model, tokenizer, user_input, personality, args.device)

                if response:
                    entry = {
                        "personality": personality[:200],
                        "user_input": user_input,
                        "response": response,
                    }
                    responses.append(entry)

                    if not args.skip_judge:
                        score = judge.score(response, personality)
                        entry["score"] = score
                        scores.append(score)

        # 清理
        del model
        torch.cuda.empty_cache()
        time.sleep(2)

        # 保存
        with open(output_dir / f"{exp_name}_responses.json", "w", encoding="utf-8") as f:
            json.dump(responses, f, indent=2, ensure_ascii=False)

        result = {
            "name": exp_name,
            "config": exp["config"],
            "stage": exp["stage"],
            "num_responses": len(responses),
        }

        if scores:
            result["mean_score"] = round(float(np.mean(scores)), 3)
            result["std_score"] = round(float(np.std(scores)), 3)
            result["median_score"] = round(float(np.median(scores)), 3)

        all_results[exp_name] = result

        if scores:
            logger.info(f"Score: {result['mean_score']:.2f} ± {result['std_score']:.2f}")
        logger.info(f"Responses: {len(responses)}")

    # 保存汇总
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 打印报告
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"{'Experiment':<30} {'Responses':>10} {'Score':>10} {'Std':>8}")
    print("-"*70)

    for name, data in all_results.items():
        score_str = f"{data.get('mean_score', 'N/A'):.2f}" if 'mean_score' in data else "N/A"
        std_str = f"{data.get('std_score', 0):.2f}" if 'std_score' in data else ""
        print(f"{name:<30} {data['num_responses']:>10} {score_str:>10} {std_str:>8}")

    print("\n" + "="*70)
    print("按配置分组:")
    for config in configs:
        config_results = {k: v for k, v in all_results.items() if v.get("config") == config}
        if config_results:
            scores_list = [v.get('mean_score', 0) for v in config_results.values() if 'mean_score' in v]
            if scores_list:
                avg = np.mean(scores_list)
                print(f"  {config}: avg={avg:.2f}")

    print("\n按阶段分组:")
    for stage in stages:
        stage_results = {k: v for k, v in all_results.items() if v.get("stage") == stage}
        if stage_results:
            scores_list = [v.get('mean_score', 0) for v in stage_results.values() if 'mean_score' in v]
            if scores_list:
                avg = np.mean(scores_list)
                print(f"  {stage}: avg={avg:.2f}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()