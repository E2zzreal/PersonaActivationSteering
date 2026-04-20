#!/usr/bin/env python
"""用 16 个 Big Five personality 生成跨 persona 对比训练数据

对每条用户输入，从 16 个 personality 中抽 K 个，用 Claude 生成不同回复。
输出格式兼容 ALOEDataset，额外带 big5 分数。

用法：
  python scripts/generate_big5_data.py --n 50 --k 4 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import yaml
from openai import OpenAI

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

SYSTEM_PROMPT = """You are roleplaying a character. Your every response MUST be immediately recognizable as this personality type.

【Personality type: {name}】
{description}

【Required behavioral markers — you MUST include at least 2 per response】
{markers}

【Anti-patterns — you must NEVER do these】
{anti_patterns}

Respond in the same language as the user. Output only the response text."""


def load_big5_config(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text())
    return data['personalities']


def build_prompt(persona: dict) -> str:
    markers = '\n'.join(f'- {m}' for m in persona['behavioral_markers'])
    anti = '\n'.join(f'- {a}' for a in persona['anti_patterns'])
    return SYSTEM_PROMPT.format(
        name=persona['name'],
        description=persona['description'],
        markers=markers,
        anti_patterns=anti,
    )


def call_api(client, model, messages, timeout=90, max_retries=3) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.8, max_tokens=250, timeout=timeout)
            return (resp.choices[0].message.content or '').strip()
        except Exception as e:
            if attempt == max_retries:
                return ''
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--big5_config', default='configs/big5_personalities.json')
    parser.add_argument('--user_inputs', default='results/parallel_dialogues/dialogues.json',
                        help='用户输入来源（取第一轮 user_input）')
    parser.add_argument('--n', type=int, default=50, help='取多少条用户输入')
    parser.add_argument('--k', type=int, default=4, help='每条输入用几个 personality')
    parser.add_argument('--api_config', default='configs/api_config.yaml')
    parser.add_argument('--output', default='data/big5_cross_persona/train.jsonl')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    personas = load_big5_config(args.big5_config)
    print(f'{len(personas)} Big Five personalities loaded')

    # API
    cfg = yaml.safe_load(Path(args.api_config).read_text())
    api_cfg = cfg.get(cfg.get('default', 'blsc'), {})
    client = OpenAI(api_key=api_cfg['api_key'], base_url=api_cfg['base_url'])
    model = api_cfg.get('claude_generator_model', 'Claude-Opus-4.6')

    # 用户输入
    dialogues = json.loads(Path(args.user_inputs).read_text())
    user_inputs = []
    for conv in dialogues[:args.n]:
        if conv.get('turns'):
            user_inputs.append(conv['turns'][0]['user_input'])
    # 补充更多（如果不够从 train.jsonl 取）
    if len(user_inputs) < args.n:
        train_path = project_root / 'data' / 'split' / 'train.jsonl'
        if not train_path.exists():
            # worktree 中可能没有 data/，尝试主目录
            train_path = Path('/home/kemove/Desktop/PersonaSteer/data/split/train.jsonl')
        with open(train_path) as f:
            for line in f:
                if len(user_inputs) >= args.n:
                    break
                item = json.loads(line)
                convs = item.get('conversations', [])
                if convs and convs[0].get('role') == 'user':
                    user_inputs.append(convs[0]['content'])

    print(f'{len(user_inputs)} 条用户输入, 每条 × {args.k} personas = {len(user_inputs)*args.k} 条')

    # 断点续传
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    done_keys.add((d.get('user_input', '')[:50], d.get('persona_id', '')))
        print(f'断点续传: 已有 {len(done_keys)} 条')

    total = 0
    with open(out_path, 'a', encoding='utf-8') as fout:
        for i, user_input in enumerate(user_inputs):
            # 随机选 K 个 personality
            selected = rng.sample(personas, min(args.k, len(personas)))

            for persona in selected:
                key = (user_input[:50], persona['id'])
                if key in done_keys:
                    continue

                sys_prompt = build_prompt(persona)
                response = call_api(client, model, [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_input},
                ])
                if not response:
                    continue

                big5 = persona['big5']
                item = {
                    'persona_id': persona['id'],
                    'persona_name': persona['name'],
                    'big5_scores': [big5['O'], big5['C'], big5['E'], big5['A'], big5['N']],
                    'personality': persona['description'],
                    'profile': f"Big Five: O={big5['O']:+.1f} C={big5['C']:+.1f} E={big5['E']:+.1f} A={big5['A']:+.1f} N={big5['N']:+.1f}",
                    'user_input': user_input,
                    'user_id': f"big5_{persona['id']}_{i}",
                    'conversations': [
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": response},
                    ],
                }
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                fout.flush()
                total += 1

                if total % 10 == 0 or total <= 5:
                    print(f'  [{total}] input_{i} × {persona["id"]}: {response[:60]}...')

    print(f'\n完成: {total} 条新增, 总计 {len(done_keys)+total} 条')
    print(f'输出: {args.output}')


if __name__ == '__main__':
    main()
