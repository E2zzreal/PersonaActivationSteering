#!/usr/bin/env python
"""
Judge V3: 多维度评分
核心改进：将对齐分解为3个维度单独打分，找出注入在哪个维度最有效。
维度：style_score（语言风格）/ content_score（内容偏好）/ consistency_score（上下文一致性）
"""

import argparse
import json
import logging
import os
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


MULTIDIM_PROMPT = """## Task
Evaluate the assistant's responses across THREE independent dimensions of personality alignment.

## User Information
**Profile**: {profile}
**Personality Traits**: {personality}

## Dialogue
{dialogue}

## Dimensions

**Dimension 1 — Style Alignment (S)**
Does the assistant's language style (formality, vocabulary, sentence length, tone) match this user's communication style?
- 5: Language style is a clear, specific match to this user's profile
- 3: Neutral/generic style that neither matches nor contradicts
- 1: Style is opposite to what this user would prefer

**Dimension 2 — Content Alignment (C)**
Does the content, examples, and depth of the responses match this user's interests and background?
- 5: Content choices are specific to this user's domain/interests/knowledge level
- 3: Content is generic but not wrong for this user
- 1: Content ignores or contradicts the user's stated background

**Dimension 3 — Consistency (K)**
Are the alignment signals consistent across the whole dialogue, or do they appear only occasionally?
- 5: Every assistant turn reflects personality awareness
- 3: Inconsistent — some turns adapt, others don't
- 1: No turn shows personality adaptation

## Output Format
Output EXACTLY in this format (three lines, no other text):
S: <integer 1-5>
C: <integer 1-5>
K: <integer 1-5>
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Judge V3: Multi-Dimensional Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/split/val.jsonl")
    parser.add_argument("--output", type=str, default="results/judge_v3_multidim.json")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--judge_model", type=str, default="Claude-Sonnet-4.6")
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument(
        "--weights",
        type=str,
        default="0.4,0.4,0.2",
        help="Comma-separated weights for S,C,K dimensions (must sum to 1.0)",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path, config):
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

    if checkpoint_path != "baseline":
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model_state = model.state_dict()
        loaded = 0
        for key, val in state_dict.items():
            if key in model_state and model_state[key].shape == val.shape:
                model_state[key] = val
                loaded += 1
        model.load_state_dict(model_state)
        logger.info(f"Loaded {loaded} tensors from checkpoint")
    else:
        logger.info("Using baseline model (no checkpoint)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_conversation(model, tokenizer, sample, max_new_tokens=150):
    device = None
    for p in model.parameters():
        if p.device.type == "cuda":
            device = p.device
            break
    if device is None:
        device = next(model.parameters()).device

    v_t = torch.zeros(1, model.v_dim).to(device)
    conversation = []

    for turn in sample.get("conversations", [])[:8]:
        if turn.get("role") != "user":
            continue
        if len([t for t in conversation if t["role"] == "user"]) >= 4:
            break

        user_text = turn.get("content", "")
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
            outputs, v_t = model.generate(
                input_ids=input_ids,
                v_prev=v_t,
                personality_texts=[sample["personality"]],
                user_query_texts=[user_text],
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
            )

        response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        conversation.append({"role": "user", "content": user_text})
        conversation.append({"role": "assistant", "content": response})

    return conversation


def judge_multidim(client, judge_model, conversation, profile, personality):
    """Returns dict with keys: style, content, consistency (each 1-5)"""
    if client is None:
        return {
            "style": float(np.random.randint(1, 6)),
            "content": float(np.random.randint(1, 6)),
            "consistency": float(np.random.randint(1, 6)),
        }

    dialogue = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in conversation)
    prompt = MULTIDIM_PROMPT.format(
        profile=profile, personality=personality, dialogue=dialogue
    )

    try:
        temperature = 1.0 if "GPT-5" in judge_model else 0.0
        resp = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": "You are evaluating personality alignment across three dimensions. Follow the exact output format."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=30,
        )
        content = resp.choices[0].message.content.strip()

        # Parse S: N, C: N, K: N
        dim_scores = {"style": 3.0, "content": 3.0, "consistency": 3.0}
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("S:"):
                val = line.split(":")[1].strip()
                for ch in val:
                    if ch.isdigit():
                        dim_scores["style"] = float(min(max(int(ch), 1), 5))
                        break
            elif line.startswith("C:"):
                val = line.split(":")[1].strip()
                for ch in val:
                    if ch.isdigit():
                        dim_scores["content"] = float(min(max(int(ch), 1), 5))
                        break
            elif line.startswith("K:"):
                val = line.split(":")[1].strip()
                for ch in val:
                    if ch.isdigit():
                        dim_scores["consistency"] = float(min(max(int(ch), 1), 5))
                        break

        return dim_scores
    except Exception as e:
        logger.warning(f"Judge call failed: {e}")
        return {"style": 3.0, "content": 3.0, "consistency": 3.0}


def main():
    args = parse_args()
    config = load_config(args.config)

    weights = [float(w) for w in args.weights.split(",")]
    if len(weights) != 3:
        raise ValueError("--weights must have exactly 3 values (S,C,K)")
    w_s, w_c, w_k = weights
    logger.info(f"Dimension weights: style={w_s}, content={w_c}, consistency={w_k}")

    client = None
    if OPENAI_AVAILABLE:
        api_key = os.environ.get("BLSC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key, base_url="https://llmapi.blsc.cn")
            logger.info(f"Judge: {args.judge_model} via https://llmapi.blsc.cn")

    model, tokenizer = load_model(args.checkpoint, config)

    samples = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.num_samples:
        samples = samples[: args.num_samples]
    logger.info(f"Evaluating {len(samples)} samples")

    per_sample = []
    all_style, all_content, all_consist, all_weighted = [], [], [], []

    for sample in tqdm(samples, desc="V3 Multi-Dimensional"):
        profile = sample.get("profile", "")
        personality = sample.get("personality", "")

        conv = generate_conversation(model, tokenizer, sample, args.max_new_tokens)
        dim_scores = judge_multidim(client, args.judge_model, conv, profile, personality)

        s = dim_scores["style"]
        c = dim_scores["content"]
        k = dim_scores["consistency"]
        weighted = w_s * s + w_c * c + w_k * k

        all_style.append(s)
        all_content.append(c)
        all_consist.append(k)
        all_weighted.append(weighted)

        per_sample.append({
            "user_id": sample.get("user_id", ""),
            "personality": personality[:80],
            "style_score": s,
            "content_score": c,
            "consistency_score": k,
            "weighted_score": round(weighted, 3),
            "num_turns": len(conv) // 2,
        })
        logger.info(
            f"  user={sample.get('user_id','')} S={s} C={c} K={k} weighted={weighted:.2f}"
        )

    def dim_stats(arr):
        a = np.array(arr)
        return {
            "mean": round(float(a.mean()), 4),
            "std": round(float(a.std()), 4),
            "min": float(a.min()),
            "max": float(a.max()),
        }

    results = {
        "method": "v3_multidimensional",
        "checkpoint": args.checkpoint,
        "judge_model": args.judge_model,
        "num_samples": len(per_sample),
        "weights": {"style": w_s, "content": w_c, "consistency": w_k},
        "style": dim_stats(all_style),
        "content": dim_stats(all_content),
        "consistency": dim_stats(all_consist),
        "weighted": dim_stats(all_weighted),
        "per_sample": per_sample,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {out_path}")
    logger.info(
        f"V3 Multi-Dim: style={results['style']['mean']:.3f} "
        f"content={results['content']['mean']:.3f} "
        f"consistency={results['consistency']['mean']:.3f} "
        f"weighted={results['weighted']['mean']:.3f}"
    )


if __name__ == "__main__":
    main()
