#!/usr/bin/env python
"""
V4 完整评估脚本 - Qwen3-4B 版本
评估 baseline + Stage1 (+ Stage2/3 待训练完成)
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

from src.evaluation.thinking_leak import conservative_clean_response

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


def load_baseline(base_model_path: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"[Baseline] Loading raw model from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    return model, tokenizer


def clean_qwen3_response(text: str) -> str:
    """清理 Qwen3 的 think token，只保留最终回复。"""
    import re
    text = re.sub(r'<think.*?</think\s*>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think.*', '', text, flags=re.DOTALL)
    return text.strip()


def load_persona_steer(checkpoint_path: str, base_model_path: str, device: str,
                       inject_layers: list = None):
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    from src.models.persona_steer import PersonaSteerModel, PersonaSteerConfig

    logger.info(f"[PersonaSteer] Loading checkpoint {checkpoint_path}")
    device = torch.device(device)

    # 先加载 checkpoint 检测注入层数
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # 自动检测 num_inject_layers
    layer_emb_key = 'hyper_network.layer_embedding.weight'
    if layer_emb_key in ckpt["model_state_dict"]:
        num_inject_layers = ckpt["model_state_dict"][layer_emb_key].shape[0]
        logger.info(f"[PersonaSteer] Detected num_inject_layers={num_inject_layers} from checkpoint")
        if inject_layers is None:
            # 根据检测到的层数推断 inject_layers
            # Qwen3-4B 有 40 层，默认从中间开始注入
            start_layer = 8  # 默认起始层
            inject_layers = list(range(start_layer, start_layer + num_inject_layers))
            logger.info(f"[PersonaSteer] Auto-detected inject_layers={inject_layers}")
    else:
        num_inject_layers = 8
        if inject_layers is None:
            inject_layers = list(range(8, 16))

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    backbone = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device)

    backbone_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)

    persona_config = PersonaSteerConfig(
        inject_layers=inject_layers,
        v_dim=1024,
        hidden_dim=4096,
        layer_dim=backbone_config.hidden_size,  # 2560 for Qwen3-4B
    )

    # 共享 encoder = backbone.model
    model = PersonaSteerModel(config=persona_config, backbone=backbone, encoder=backbone.model)

    # 设置 encoder tokenizer（HyperNetwork 编码 user_texts 需要）
    if hasattr(model.hyper_network, 'encoder') and hasattr(model.hyper_network, 'set_tokenizer'):
        model.hyper_network.set_tokenizer(tokenizer)

    # 加载 checkpoint
    stage = ckpt.get("stage", 1)

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

    logger.info(f"[PersonaSteer] Stage {stage} loaded ({loaded} params, inject_layers={inject_layers})")
    return model, tokenizer, stage


def generate_baseline(model, tokenizer, user_input: str, personality: str,
                      profile: str, device: str) -> str:
    # 使用 Qwen3 chat template + /no_think 禁用思考模式
    messages = [
        {"role": "system", "content": f"/no_think\n你的人格特征是：{personality}\n\n你的个人简介：{profile}"},
        {"role": "user", "content": user_input},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
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
    response = clean_qwen3_response(response)
    return response.strip()


def generate_persona_steer(model, tokenizer, user_input: str, personality: str,
                           profile: str, device: str) -> str:
    device = torch.device(device)
    messages = [
        {"role": "system", "content": f"/no_think\n你的人格特征是：{personality}\n\n你的个人简介：{profile}"},
        {"role": "user", "content": user_input},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    v_prev = torch.zeros(1, model.v_dim, device=device)

    with torch.no_grad():
        generated_ids, _ = model.generate(
            input_ids=input_ids,
            v_prev=v_prev,
            personality_texts=[personality],
            user_query_texts=[user_input],
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
        )

    response = tokenizer.decode(generated_ids[0][prompt_len:], skip_special_tokens=True)
    response = clean_qwen3_response(response)
    return response.strip()


def main():
    import argparse

    def parse_layer_list(value: str):
        value = (value or '').strip()
        if not value:
            return None
        return [int(item.strip()) for item in value.split(',') if item.strip()]

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--max_turns_per_sample", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--judge_model", type=str, default="GPT-5.2")
    parser.add_argument("--output_dir", type=str, default="results/v4_qwen3_eval")
    parser.add_argument("--base_model", type=str,
                        default="/home/kemove/Desktop/PersonaSteer/Qwen/Qwen3-4B")
    parser.add_argument("--skip_judge", action="store_true")
    # 动态指定要评估的 stages
    parser.add_argument("--stages", type=str, nargs="+",
                        default=["baseline", "stage1"],
                        help="Which stages to evaluate (baseline/stage1/stage2/stage3)")
    parser.add_argument("--stage2_checkpoint", type=str,
                        default="checkpoints/v4_qwen3_stage2/best.pt")
    parser.add_argument("--stage3_checkpoint", type=str,
                        default="checkpoints/stage3_qwen3_v2/best.pt")
    parser.add_argument("--stage1_checkpoint", type=str,
                        default="checkpoints/stage1_qwen3/best.pt",
                        help="Custom stage1 checkpoint path")
    parser.add_argument("--stage1_inject_layers", type=str, default=None,
                        help="逗号分隔的 stage1 注入层列表，例如 8,9,10,11,12,13,14,15")
    parser.add_argument("--stage2_inject_layers", type=str, default=None,
                        help="逗号分隔的 stage2 注入层列表")
    parser.add_argument("--stage3_inject_layers", type=str, default=None,
                        help="逗号分隔的 stage3 注入层列表")
    args = parser.parse_args()

    stage1_inject_layers = parse_layer_list(args.stage1_inject_layers)
    stage2_inject_layers = parse_layer_list(args.stage2_inject_layers)
    stage3_inject_layers = parse_layer_list(args.stage3_inject_layers)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("BLSC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://llmapi.blsc.cn/v1")
    judge = LLMJudge(api_key=api_key, base_url=base_url, model=args.judge_model)

    # 加载数据
    eval_samples = []
    with open("data/split/val.jsonl", "r") as f:
        for line in f:
            if line.strip():
                eval_samples.append(json.loads(line))

    # 如果存在之前的评估结果，先加载
    summary_path = output_dir / "eval_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            all_results = json.load(f)
        logger.info(f"Loaded previous results: {list(all_results.keys())}")
    else:
        all_results = {}

    # 定义评估目标
    stage_defs = {
        "baseline": {"type": "baseline", "checkpoint": None},
        "stage1": {
            "type": "persona_steer",
            "checkpoint": args.stage1_checkpoint,
            "inject_layers": stage1_inject_layers,
        },
        "stage2": {
            "type": "persona_steer",
            "checkpoint": args.stage2_checkpoint,
            "inject_layers": stage2_inject_layers,
        },
        "stage3": {
            "type": "persona_steer",
            "checkpoint": args.stage3_checkpoint,
            "inject_layers": stage3_inject_layers,
        },
    }

    for stage_name in args.stages:
        if stage_name in all_results and all_results[stage_name].get("num_responses", 0) > 0:
            logger.info(f"Skipping {stage_name} (already evaluated)")
            continue

        target = stage_defs[stage_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {stage_name}")
        logger.info(f"{'='*60}")

        try:
            if target["type"] == "baseline":
                model, tokenizer = load_baseline(args.base_model, args.device)
            else:
                model, tokenizer, stage = load_persona_steer(
                    target["checkpoint"], args.base_model, args.device,
                    inject_layers=target.get("inject_layers"),
                )
        except Exception as e:
            logger.error(f"Failed to load {stage_name}: {e}")
            all_results[stage_name] = {
                "stage": stage_name,
                "checkpoint": target.get("checkpoint"),
                "num_responses": 0,
                "error": str(e),
            }
            # 保存中间结果
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
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
                        leak_result = conservative_clean_response(response)
                        gate_values = None
                        gate_mean = None
                        gate_std = None
                        gate_max = None
                        gate_min = None
                        if target["type"] != "baseline" and hasattr(model, 'injection') and getattr(model.injection, 'current_gate_values', None) is not None:
                            gate_tensor = model.injection.current_gate_values.detach().float().cpu()
                            if gate_tensor.dim() >= 2 and gate_tensor.size(0) > 0:
                                gate_row = gate_tensor[0]
                                gate_values = [round(float(v), 6) for v in gate_row.tolist()]
                                gate_mean = round(float(gate_row.mean().item()), 6)
                                gate_std = round(float(gate_row.std().item()), 6)
                                gate_max = round(float(gate_row.max().item()), 6)
                                gate_min = round(float(gate_row.min().item()), 6)
                        entry = {
                            "personality": personality[:200],
                            "profile": profile[:200],
                            "user_input": user_input,
                            "response": leak_result.clean_response,
                            "raw_response": leak_result.raw_response,
                            "clean_response": leak_result.clean_response,
                            "leak_detected": leak_result.leak_detected,
                            "leak_patterns": leak_result.leak_patterns,
                            "cleaning_applied": leak_result.cleaning_applied,
                            "cleaning_skipped": leak_result.cleaning_skipped,
                            "gate_values": gate_values,
                            "gate_mean": gate_mean,
                            "gate_std": gate_std,
                            "gate_max": gate_max,
                            "gate_min": gate_min,
                            "stage": stage_name,
                        }
                        responses.append(entry)

                        if not args.skip_judge:
                            score = judge.score(leak_result.clean_response, personality)
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

        result = {
            "stage": stage_name,
            "checkpoint": target.get("checkpoint"),
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

        # 每完成一个 stage 就保存
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 打印对比表
    logger.info(f"\n{'='*60}")
    logger.info("Qwen3-4B EVALUATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Stage':<12} {'Responses':>10} {'Score':>8} {'Std':>6}")
    logger.info(f"{'-'*40}")
    for name, data in all_results.items():
        score_str = f"{data.get('mean_score', 'N/A'):.2f}" if 'mean_score' in data else "N/A"
        std_str = f"{data.get('std_score', 0):.2f}" if 'std_score' in data else ""
        logger.info(f"{name:<12} {data.get('num_responses', 0):>10} {score_str:>8} {std_str:>6}")

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
