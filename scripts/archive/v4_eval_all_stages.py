#!/usr/bin/env python
"""
V4 完整评估脚本
评估 baseline + Stage 1 + Stage 2，生成示例文本 + LLM Judge 评分
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


# ============================================================
# LLM Judge
# ============================================================

class LLMJudge:
    def __init__(self, api_key=None, base_url=None, model="GPT-5.2"):
        self.model = model
        if OPENAI_AVAILABLE and api_key:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"LLM Judge initialized: {model}")
        else:
            self.client = None
            logger.warning("No API key, returning random scores")

    def score(self, response: str, personality: str) -> float:
        prompt = f"""你是一个人格一致性评估专家。请评估以下AI回复与目标人格的一致程度。

目标人格: {personality}

AI回复: {response}

评分标准:
1 - 完全不一致
2 - 较不一致
3 - 中立
4 - 较一致
5 - 完全一致

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


# ============================================================
# Model Loaders
# ============================================================

def load_baseline(base_model_path: str, device: str):
    """加载原始模型（无 persona injection）"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"[Baseline] Loading raw model from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    return model, tokenizer


def load_persona_steer(checkpoint_path: str, base_model_path: str, device: str):
    """加载 PersonaSteer 模型（Stage 1/2/3）"""
    from transformers import (
        AutoModelForCausalLM,
        AutoConfig,
        AutoTokenizer,
    )
    from src.models.persona_steer import PersonaSteerModel, PersonaSteerConfig

    logger.info(f"[PersonaSteer] Loading checkpoint {checkpoint_path}")
    device = torch.device(device)

    # 加载 backbone
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    backbone = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device)

    backbone_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)

    # 加载 checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stage = ckpt.get("stage", 1)

    # 模型配置
    persona_config = PersonaSteerConfig(
        inject_layers=backbone_config.num_hidden_layers - 8,  # [10-17] for Qwen2.5
        v_dim=1024,
        hidden_dim=4096,
        layer_dim=backbone_config.hidden_size,
    )
    # 更正 inject_layers
    persona_config.inject_layers = list(range(
        backbone_config.num_hidden_layers - 8,
        backbone_config.num_hidden_layers
    ))

    # 创建模型（共享 encoder）
    model = PersonaSteerModel(config=persona_config, backbone=backbone, encoder=backbone.model)

    # 设置 encoder tokenizer（HyperNetwork 编码 user_texts 需要）
    if hasattr(model.hyper_network, 'encoder') and hasattr(model.hyper_network, 'set_tokenizer'):
        model.hyper_network.set_tokenizer(tokenizer)

    # 加载 checkpoint（只加载可训练参数）
    model_state = model.state_dict()
    loaded = 0
    for key, value in ckpt["model_state_dict"].items():
        if key in model_state:
            model_state[key] = value
            loaded += 1
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    # 设置 tokenizer（HyperNetwork 需要）
    model.hyper_network.set_tokenizer(tokenizer)

    logger.info(f"[PersonaSteer] Stage {stage} loaded ({loaded} params)")
    return model, tokenizer, stage


# ============================================================
# Generation
# ============================================================

def generate_baseline(model, tokenizer, user_input: str, personality: str,
                      profile: str, device: str) -> str:
    """Baseline 生成（人格作为 system prompt）"""
    prompt = f"你的人格特征是：{personality}\n\n你的个人简介：{profile}\n\n用户: {user_input}\n助手:"

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def generate_persona_steer(model, tokenizer, user_input: str, personality: str,
                           profile: str, device: str) -> str:
    """PersonaSteer 生成（人格通过 hypernetwork 注入）"""
    device = torch.device(device)

    prompt = f"用户: {user_input}\n助手:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    v_prev = torch.zeros(1, model.v_dim, device=device)

    with torch.no_grad():
        generated_ids, _ = model.generate(
            input_ids=input_ids,
            v_prev=v_prev,
            user_texts=[personality],
            max_new_tokens=150,
            temperature=0.7,
        )

    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if "助手:" in response:
        response = response.split("助手:")[-1].strip()
    return response


# ============================================================
# Main Evaluation
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=30,
                        help="每个 stage 评估的样本数")
    parser.add_argument("--max_turns_per_sample", type=int, default=3,
                        help="每个样本最多评估的对话轮次")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--judge_model", type=str, default="GPT-5.2")
    parser.add_argument("--output_dir", type=str, default="results/v4_qwen25_eval")
    parser.add_argument("--base_model", type=str,
                        default="/home/kemove/.cache/modelscope/Qwen/Qwen2___5-3B")
    parser.add_argument("--skip_judge", action="store_true",
                        help="跳过 LLM Judge，只生成示例文本")
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # API 配置
    api_key = os.environ.get("BLSC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://llmapi.blsc.cn/v1")

    # 初始化 Judge
    judge = LLMJudge(api_key=api_key, base_url=base_url, model=args.judge_model)

    # 加载评估数据
    eval_samples = []
    with open("data/split/val.jsonl", "r") as f:
        for line in f:
            if line.strip():
                eval_samples.append(json.loads(line))

    logger.info(f"Loaded {len(eval_samples)} eval samples, will use {args.num_samples}")

    # 定义评估目标
    targets = [
        {
            "name": "baseline",
            "type": "baseline",
            "checkpoint": None,
        },
        {
            "name": "stage1",
            "type": "persona_steer",
            "checkpoint": "checkpoints/best.pt",
        },
        {
            "name": "stage2",
            "type": "persona_steer",
            "checkpoint": "checkpoints/v4_qwen25_stage2/best.pt",
        },
    ]

    all_results = {}

    for target in targets:
        stage_name = target["name"]
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {stage_name}")
        logger.info(f"{'='*60}")

        # 加载模型
        try:
            if target["type"] == "baseline":
                model, tokenizer = load_baseline(args.base_model, args.device)
            else:
                model, tokenizer, stage = load_persona_steer(
                    target["checkpoint"], args.base_model, args.device
                )
        except Exception as e:
            logger.error(f"Failed to load {stage_name}: {e}")
            continue

        responses = []
        scores = []

        for sample in tqdm(
            eval_samples[:args.num_samples],
            desc=f"[{stage_name}] Generating",
        ):
            personality = sample.get("personality", "")
            profile = sample.get("profile", "")
            conversations = sample.get("conversations", [])

            turn_count = 0
            for turn in conversations:
                if turn_count >= args.max_turns_per_sample:
                    break

                # 数据格式: role/content 或 user/assistant
                user_input = turn.get("user", "") or (
                    turn.get("content", "") if turn.get("role") == "user" else ""
                )
                if not user_input or not personality:
                    continue

                try:
                    if target["type"] == "baseline":
                        response = generate_baseline(
                            model, tokenizer, user_input, personality, profile, args.device
                        )
                    else:
                        response = generate_persona_steer(
                            model, tokenizer, user_input, personality, profile, args.device
                        )

                    if response:
                        entry = {
                            "personality": personality[:200],
                            "profile": profile[:200],
                            "user_input": user_input,
                            "response": response,
                            "stage": stage_name,
                        }
                        responses.append(entry)

                        # LLM Judge
                        if not args.skip_judge:
                            score = judge.score(response, personality)
                            entry["judge_score"] = score
                            scores.append(score)

                        turn_count += 1

                except Exception as e:
                    logger.warning(f"Generation failed: {e}")
                    continue

        # 释放显存
        del model
        torch.cuda.empty_cache()
        time.sleep(2)

        # 保存生成文本
        responses_path = output_dir / f"{stage_name}_responses.json"
        with open(responses_path, "w", encoding="utf-8") as f:
            json.dump(responses, f, indent=2, ensure_ascii=False)
        logger.info(f"  Saved {len(responses)} responses to {responses_path}")

        # 汇总
        result = {
            "stage": stage_name,
            "checkpoint": target["checkpoint"],
            "num_responses": len(responses),
            "judge_model": args.judge_model,
            "num_scored": len(scores),
        }
        if scores:
            result["mean_score"] = round(float(np.mean(scores)), 3)
            result["std_score"] = round(float(np.std(scores)), 3)
            result["median_score"] = round(float(np.median(scores)), 3)
            result["min_score"] = round(float(np.min(scores)), 3)
            result["max_score"] = round(float(np.max(scores)), 3)

        all_results[stage_name] = result
        logger.info(f"  Score: {result.get('mean_score', 'N/A')} ± {result.get('std_score', 'N/A')}")

    # 保存汇总结果
    summary_path = output_dir / "eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 打印对比表
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Stage':<12} {'Responses':>10} {'Score':>8} {'Std':>6}")
    logger.info(f"{'-'*40}")
    for name, data in all_results.items():
        score_str = f"{data.get('mean_score', 'N/A'):.2f}" if 'mean_score' in data else "N/A"
        std_str = f"{data.get('std_score', 0):.2f}" if 'std_score' in data else ""
        logger.info(f"{name:<12} {data['num_responses']:>10} {score_str:>8} {std_str:>6}")

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"Summary: {summary_path}")
    logger.info("Responses: {stage}_responses.json in output dir")


if __name__ == "__main__":
    main()
