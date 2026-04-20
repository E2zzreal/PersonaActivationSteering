#!/usr/bin/env python
"""生成跨 persona 对比数据

对同一批用户输入，用不同 persona 生成不同 Claude 回复。
训练时，模型必须根据 persona（而非 user_input）区分输出。

输出格式：每条 = {user_input, persona, profile, response}
同一 user_input 出现 N 次（N 个不同 persona），每次 response 不同。

用法：
  python scripts/audit/generate_cross_persona.py --n 30 --personas_per_input 3
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

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


SYSTEM_PROMPT = """You are roleplaying a character with a unique personality. Every response you give MUST be immediately recognizable as THIS specific personality.

【Your personality traits】
{personality}

【Your personal background】
{profile}

【Rules】
1. Include at least 2 IRREPLACEABLE personality markers: unique speech habits, specific experiences, distinctive emotional reactions.
2. SHOW personality through actions, not adjectives. Don't say "I'm adventurous" — share an adventure.
3. Focus on the 2-3 most distinctive traits. Stay consistent.
4. Respond in the same language as the user. Output only the response text."""


def call_api(client, model, messages, timeout=90, max_retries=3) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.8, max_tokens=250, timeout=timeout)
            return (resp.choices[0].message.content or '').strip()
        except Exception as e:
            if attempt == max_retries:
                print(f'  [warn] API failed: {e}')
                return ''
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dialogues', default='results/parallel_dialogues/dialogues.json')
    parser.add_argument('--n', type=int, default=30, help='取多少条用户输入')
    parser.add_argument('--personas_per_input', type=int, default=3,
                        help='每条输入用几个不同 persona 生成')
    parser.add_argument('--api_config', default='configs/api_config.yaml')
    parser.add_argument('--output', default='data/cross_persona/train.jsonl')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # API
    cfg = yaml.safe_load(Path(args.api_config).read_text())
    api_cfg = cfg.get(cfg.get('default', 'blsc'), {})
    client = OpenAI(api_key=api_cfg['api_key'], base_url=api_cfg['base_url'])
    model = api_cfg.get('claude_generator_model', 'Claude-Opus-4.6')

    # 加载对话（取用户输入）
    dialogues = json.loads(Path(args.dialogues).read_text())

    # 收集所有 personality + profile 组合
    persona_pool = []
    seen = set()
    for conv in dialogues:
        p = conv['personality']
        if p not in seen:
            persona_pool.append({
                'personality': p,
                'profile': conv['profile'],
            })
            seen.add(p)
    print(f'Persona pool: {len(persona_pool)} 个')

    # 收集用户输入（第一轮）
    user_inputs = []
    for conv in dialogues[:args.n]:
        if conv['turns']:
            user_inputs.append({
                'conv_id': conv['conv_id'],
                'user_input': conv['turns'][0]['user_input'],
                'original_personality': conv['personality'],
            })
    print(f'用户输入: {len(user_inputs)} 条')
    print(f'每条 × {args.personas_per_input} personas = {len(user_inputs) * args.personas_per_input} 条训练数据')

    # 断点续传
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    done_keys = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing.append(item)
                    done_keys.add((item['conv_id'], item['personality'][:50]))
        print(f'断点续传: 已有 {len(existing)} 条')

    total = 0
    with open(out_path, 'a', encoding='utf-8') as fout:
        for ui in user_inputs:
            # 选择 personas_per_input 个不同 persona（包含原始 + 随机选的）
            available = [p for p in persona_pool
                         if p['personality'] != ui['original_personality']]
            rng.shuffle(available)
            selected = available[:args.personas_per_input - 1]

            # 加上原始 persona
            orig_persona = next(p for p in persona_pool
                                if p['personality'] == ui['original_personality'])
            all_personas = [orig_persona] + selected

            for persona in all_personas:
                key = (ui['conv_id'], persona['personality'][:50])
                if key in done_keys:
                    continue

                sys_prompt = SYSTEM_PROMPT.format(
                    personality=persona['personality'],
                    profile=persona['profile'],
                )
                response = call_api(client, model, [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": ui['user_input']},
                ])

                if not response:
                    continue

                item = {
                    'conv_id': ui['conv_id'],
                    'user_input': ui['user_input'],
                    'personality': persona['personality'],
                    'profile': persona['profile'],
                    'response': response,
                    'conversations': [
                        {"role": "user", "content": ui['user_input']},
                        {"role": "assistant", "content": response},
                    ],
                    'user_id': f"cross_{ui['conv_id']}_{hash(persona['personality']) % 10000}",
                }
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                fout.flush()
                total += 1
                print(f'  [{total}] {ui["conv_id"]} × {persona["personality"][:40]}... → {response[:60]}')

    print(f'\n完成: {total} 条新增，总计 {len(existing) + total} 条')
    print(f'输出: {args.output}')


if __name__ == '__main__':
    main()
