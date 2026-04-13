#!/usr/bin/env python
"""准备训练集监督质量审计样本：gold vs baseline。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def generate_baseline(model, tokenizer, user_input: str, personality: str, profile: str, device: str) -> str:
    messages = [
        {"role": "system", "content": f"/no_think\n你的人格特征是：{personality}\n\n你的个人简介：{profile}"},
        {"role": "user", "content": user_input},
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    return response


def collect_turns(data_path: Path):
    items = []
    with data_path.open('r', encoding='utf-8') as f:
        for sample_idx, line in enumerate(f):
            if not line.strip():
                continue
            sample = json.loads(line)
            conv = sample.get('conversations', [])
            for i in range(0, len(conv) - 1, 2):
                user_msg = conv[i]
                asst_msg = conv[i + 1]
                if user_msg.get('role') != 'user' or asst_msg.get('role') != 'assistant':
                    continue
                items.append({
                    'sample_idx': sample_idx,
                    'turn_idx': i // 2,
                    'user_id': sample.get('user_id', f'u{sample_idx}'),
                    'personality': sample.get('personality', ''),
                    'profile': sample.get('profile', ''),
                    'user_input': user_msg.get('content', ''),
                    'gold_response': asst_msg.get('content', ''),
                })
    return items


def main():
    parser = argparse.ArgumentParser(description='准备训练集监督质量审计样本')
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--base_model', type=str, default='/home/kemove/Desktop/PersonaSteer/Qwen/Qwen3-4B')
    args = parser.parse_args()

    random.seed(args.seed)
    items = collect_turns(Path(args.data))
    random.shuffle(items)
    selected = items[:args.num_samples]

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    prepared = []
    for item in selected:
        baseline_response = generate_baseline(
            model, tokenizer,
            item['user_input'], item['personality'], item['profile'], args.device
        )
        prepared.append({
            **item,
            'baseline_response': baseline_response,
        })

    Path(args.output).write_text(json.dumps(prepared, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'prepared {len(prepared)} samples -> {args.output}')


if __name__ == '__main__':
    main()
