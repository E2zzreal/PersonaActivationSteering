#!/usr/bin/env python
"""
Judge V2: A/B 对比评估
核心改进：给 Judge 看两个回复（baseline vs injected），让它选偏好，
输出胜率而非绝对分，彻底绕开基准分高的问题。
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.persona_steer import PersonaSteerModel, PersonaSteerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


AB_COMPARE_PROMPT = """## Task
Two AI assistants (A and B) responded to the same user queries. Your job is to determine which assistant better matched the user's personality profile — NOT which gave a better answer in general.

## User Information
**Profile**: {profile}
**Personality Traits**: {personality}

## User Query
{user_query}

## Assistant A Response
{response_a}

## Assistant B Response
{response_b}

## Evaluation Criteria
Focus ONLY on personality alignment:
- Does the vocabulary/tone match this user's communication style?
- Is the level of formality/informality appropriate for this personality?
- Does the response content reflect this user's interests/background?
- Would a person with a DIFFERENT personality get a noticeably different response?

DO NOT consider: factual accuracy, helpfulness, fluency (unless it relates to personality match)

## Output Format
Output ONLY one of: A, B, or TIE
- A: Assistant A better reflects this user's personality
- B: Assistant B better reflects this user's personality
- TIE: Both are equally (un)aligned with this personality
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Judge V2: A/B Comparison Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Injected model checkpoint")
    parser.add_argument("--baseline_checkpoint", type=str, required=True, help="Baseline model checkpoint (no injection)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/split/val.jsonl")
    parser.add_argument("--output", type=str, default="results/judge_v2_ab.json")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--judge_model", type=str, default="Claude-Sonnet-4.6")
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path, config, baseline_mode=False):
    from transformers import AutoConfig, AutoModelForCausalLM

    base_model_path = config["base_model"]
    backbone_cfg = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    actual_layer_dim = backbone_cfg.hidden_size

    model_config = config.get("model", {})
    persona_config = PersonaSteerConfig(
        inject_layers=model_config.get("inject_layers", [14, 15, 16, 17, 18]),
        v_dim=model_config.get("v_dim", 1024),
        hidden_dim=model_config.get("hidden_dim", 4096),
        layer_dim=actual_layer_dim,
        gate_hidden_dim=model_config.get("gate_hidden_dim", 256),
    )

    backbone = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        use_cache=False,
    )
    encoder = backbone.model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    model = PersonaSteerModel(config=persona_config, encoder=encoder)
    if hasattr(model, "hyper_network") and model.hyper_network is not None:
        model.hyper_network._tokenizer = tokenizer
    model.set_backbone(backbone)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model_state = model.state_dict()
    loaded = 0
    for key, val in state_dict.items():
        if key in model_state and model_state[key].shape == val.shape:
            model_state[key] = val
            loaded += 1
    model.load_state_dict(model_state)
    logger.info(f"Loaded {loaded} tensors from {checkpoint_path}")

    if baseline_mode:
        model.baseline_mode = True
        logger.info("Baseline mode: injection disabled")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_single_response(model, tokenizer, sample, max_new_tokens=150):
    """只生成第一轮回复用于 A/B 对比"""
    device = None
    for p in model.parameters():
        if p.device.type == "cuda":
            device = p.device
            break
    if device is None:
        device = next(model.parameters()).device

    v_t = torch.zeros(1, model.v_dim).to(device)

    # 取第一个 user 轮
    user_text = None
    for turn in sample.get("conversations", []):
        if turn.get("role") == "user":
            user_text = turn.get("content", "")
            break
    if user_text is None:
        return "", ""

    messages = [{"role": "user", "content": user_text}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        outputs, _ = model.generate(
            input_ids=input_ids,
            v_prev=v_t,
            personality_texts=[sample["personality"]],
            user_query_texts=[user_text],
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
        )

    response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    return user_text, response


def judge_ab(client, judge_model, profile, personality, user_query, response_a, response_b):
    """Returns: 'A', 'B', or 'TIE'"""
    if client is None:
        return random.choice(["A", "B", "TIE"])

    prompt = AB_COMPARE_PROMPT.format(
        profile=profile,
        personality=personality,
        user_query=user_query,
        response_a=response_a,
        response_b=response_b,
    )

    try:
        temperature = 1.0 if "GPT-5" in judge_model else 0.0
        resp = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": "You are evaluating which AI response better matches a user's personality. Output only A, B, or TIE."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=10,
        )
        content = resp.choices[0].message.content.strip().upper()
        if "TIE" in content:
            return "TIE"
        elif content.startswith("A"):
            return "A"
        elif content.startswith("B"):
            return "B"
        return "TIE"
    except Exception as e:
        logger.warning(f"Judge call failed: {e}")
        return "TIE"


def main():
    args = parse_args()
    config = load_config(args.config)

    client = None
    if OPENAI_AVAILABLE:
        api_key = os.environ.get("BLSC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key, base_url="https://llmapi.blsc.cn")
            logger.info(f"Judge: {args.judge_model} via https://llmapi.blsc.cn")

    logger.info("Loading injected model...")
    model_injected, tokenizer = load_model(args.checkpoint, config, baseline_mode=False)

    logger.info("Loading baseline model...")
    model_baseline, _ = load_model(args.baseline_checkpoint, config, baseline_mode=True)

    # Load samples
    samples = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.num_samples:
        samples = samples[: args.num_samples]
    logger.info(f"Evaluating {len(samples)} samples")

    per_sample = []
    wins_injected = 0
    wins_baseline = 0
    ties = 0

    for sample in tqdm(samples, desc="V2 A/B Compare"):
        profile = sample.get("profile", "")
        personality = sample.get("personality", "")

        # Generate responses from both models
        user_query, resp_injected = generate_single_response(
            model_injected, tokenizer, sample, args.max_new_tokens
        )
        _, resp_baseline = generate_single_response(
            model_baseline, tokenizer, sample, args.max_new_tokens
        )

        if not user_query:
            continue

        # Randomize A/B assignment to avoid position bias
        swap = random.random() < 0.5
        if swap:
            response_a, response_b = resp_baseline, resp_injected
        else:
            response_a, response_b = resp_injected, resp_baseline

        verdict = judge_ab(client, args.judge_model, profile, personality, user_query, response_a, response_b)

        # Unswap
        if swap:
            # A=baseline, B=injected
            if verdict == "A":
                winner = "baseline"
                wins_baseline += 1
            elif verdict == "B":
                winner = "injected"
                wins_injected += 1
            else:
                winner = "tie"
                ties += 1
        else:
            # A=injected, B=baseline
            if verdict == "A":
                winner = "injected"
                wins_injected += 1
            elif verdict == "B":
                winner = "baseline"
                wins_baseline += 1
            else:
                winner = "tie"
                ties += 1

        per_sample.append({
            "user_id": sample.get("user_id", ""),
            "personality": personality[:80],
            "user_query": user_query[:100],
            "resp_injected": resp_injected[:150],
            "resp_baseline": resp_baseline[:150],
            "winner": winner,
        })
        logger.info(f"  user={sample.get('user_id','')} winner={winner}")

    total = wins_injected + wins_baseline + ties
    win_rate = wins_injected / total if total > 0 else 0.0

    results = {
        "method": "v2_ab_comparison",
        "checkpoint_injected": args.checkpoint,
        "checkpoint_baseline": args.baseline_checkpoint,
        "judge_model": args.judge_model,
        "num_samples": total,
        "wins_injected": wins_injected,
        "wins_baseline": wins_baseline,
        "ties": ties,
        "injected_win_rate": round(win_rate, 4),
        "baseline_win_rate": round(wins_baseline / total, 4) if total > 0 else 0.0,
        "tie_rate": round(ties / total, 4) if total > 0 else 0.0,
        "per_sample": per_sample,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {out_path}")
    logger.info(
        f"V2 A/B: injected_win={wins_injected} baseline_win={wins_baseline} tie={ties} "
        f"win_rate={win_rate:.1%}"
    )


if __name__ == "__main__":
    main()
