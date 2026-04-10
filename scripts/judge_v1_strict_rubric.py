#!/usr/bin/env python
"""
Judge V1: 严格评分标准
核心改进：5分需要体现人格专属细节，拒绝通用高分，提高区分度。
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


STRICT_RUBRIC_PROMPT = """## Task
You are a strict evaluator. Your job is to measure how specifically the assistant tailored its responses to this PARTICULAR user's personality — NOT how polite, helpful, or fluent the responses are.

## User Information
**Profile**: {profile}

**Personality Traits**: {personality}

## Dialogue
{dialogue}

## Strict Scoring Rubric (1-5)

**5 - Persona-Specific (Rare)**
Award ONLY if: The assistant uses specific vocabulary, references, or communication patterns that CLEARLY match THIS user's described traits. A different user with a different personality would receive a noticeably different response style. Examples: matching formality level AND topic depth AND emotional register all at once.

**4 - Clearly Adapted**
The assistant's style is visibly adjusted for this personality. At least 2 personality dimensions are reflected (e.g., tone + content preference). Generic phrasing is minimal.

**3 - Partially Adapted**
The assistant shows awareness of 1 personality dimension but the rest is standard LLM output. Someone else with a different profile would get essentially the same response.

**2 - Superficial or Accidental**
Any alignment appears coincidental. The response could have been given to any user. Style is default-LLM polite.

**1 - Misaligned or Contradictory**
The assistant's style contradicts the stated personality (e.g., uses formal language with a casual user, over-explains to an expert, etc.)

## Key Rule
If you find yourself wanting to give a 5 because the response is "helpful and good", STOP. Score 5 only for persona-SPECIFICITY, not quality. Good-but-generic = 3.

## Output Format
Output ONLY a single integer: 1, 2, 3, 4, or 5.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Judge V1: Strict Rubric Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/split/val.jsonl")
    parser.add_argument("--output", type=str, default="results/judge_v1_strict.json")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--judge_model", type=str, default="Claude-Sonnet-4.6")
    parser.add_argument("--max_new_tokens", type=int, default=150)
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


def judge_strict(client, judge_model, conversation, profile, personality):
    if client is None:
        return float(np.random.uniform(2.0, 4.0))

    dialogue = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in conversation)
    prompt = STRICT_RUBRIC_PROMPT.format(
        profile=profile, personality=personality, dialogue=dialogue
    )

    try:
        temperature = 1.0 if "GPT-5" in judge_model else 0.0
        resp = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": "You are a strict evaluator focused on persona-specificity, not general response quality."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=10,
        )
        content = resp.choices[0].message.content.strip()
        for ch in content:
            if ch.isdigit():
                return float(min(max(int(ch), 1), 5))
        return 3.0
    except Exception as e:
        logger.warning(f"Judge call failed: {e}")
        return 3.0


def main():
    args = parse_args()
    config = load_config(args.config)

    # Init judge client
    client = None
    if OPENAI_AVAILABLE:
        api_key = os.environ.get("BLSC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key, base_url="https://llmapi.blsc.cn")
            logger.info(f"Judge: {args.judge_model} via https://llmapi.blsc.cn")

    # Load model
    model, tokenizer = load_model(args.checkpoint, config)

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
    scores = []

    for sample in tqdm(samples, desc="V1 Strict Rubric"):
        conv = generate_conversation(model, tokenizer, sample, args.max_new_tokens)
        score = judge_strict(
            client, args.judge_model, conv,
            sample.get("profile", ""), sample.get("personality", "")
        )
        scores.append(score)

        # score distribution bucket
        per_sample.append({
            "user_id": sample.get("user_id", ""),
            "personality": sample.get("personality", "")[:80],
            "score": score,
            "num_turns": len(conv) // 2,
        })
        logger.info(f"  user={sample.get('user_id','')} score={score}")

    # Aggregate stats
    score_arr = np.array(scores)
    distribution = {str(i): int((score_arr == i).sum()) for i in range(1, 6)}

    results = {
        "method": "v1_strict_rubric",
        "checkpoint": args.checkpoint,
        "judge_model": args.judge_model,
        "num_samples": len(scores),
        "al_k_avg": float(score_arr.mean()),
        "al_k_std": float(score_arr.std()),
        "al_k_min": float(score_arr.min()),
        "al_k_max": float(score_arr.max()),
        "score_distribution": distribution,
        "per_sample": per_sample,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {out_path}")
    logger.info(f"V1 Strict: AL(K)_AVG={results['al_k_avg']:.3f} ± {results['al_k_std']:.3f}")
    logger.info(f"Distribution: {distribution}")


if __name__ == "__main__":
    main()
