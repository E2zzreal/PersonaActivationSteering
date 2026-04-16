#!/usr/bin/env python
"""教师人格区分能力审计：生成候选样本。"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from openai import OpenAI

GENERATOR_PROMPT = """你要扮演指定人格，输出一条面向用户的最终回复。

要求：
1. 必须自然、像真实聊天
2. 必须明显符合给定人格
3. 必须是面向用户的完成态回答
4. 不允许输出思考过程、草稿、自我分析
5. 不要出现类似“我需要…”“让我想想…”“用户在说…”这类 planning 句式
6. 长度控制在 80 到 220 个英文词附近

目标人格：
{persona_text}

用户输入：
{user_input}

请直接输出回复正文，不要输出任何解释。
"""


def load_user_inputs(path: Path, limit: int, seed: int) -> list[str]:
    records = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            convs = item.get('conversations', [])
            for turn in convs:
                if turn.get('role') == 'user' and turn.get('content'):
                    records.append(turn['content'])
                    break
    rng = random.Random(seed)
    rng.shuffle(records)
    return records[:limit]


def call_with_retry(client: OpenAI, model: str, messages: list, timeout: int = 60, max_retries: int = 3) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1.0,
                max_tokens=350,
                timeout=timeout,
            )
            return (resp.choices[0].message.content or '').strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f'  [warn] attempt {attempt}/{max_retries} failed: {e}. retrying in {wait}s...')
            if attempt == max_retries:
                raise
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description='运行教师人格区分能力审计（生成阶段）')
    parser.add_argument('--config', default='configs/audit/teacher_persona_audit_v1.json')
    parser.add_argument('--input_data', default='data/split/val.jsonl')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--base_url', default=None)
    parser.add_argument('--api_env', default='BLSC_API_KEY')
    parser.add_argument('--timeout', type=int, default=60, help='单次 API 调用超时秒数')
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    prompt_template = config.get('generator_prompt', GENERATOR_PROMPT)
    user_inputs = load_user_inputs(Path(args.input_data), config['num_user_inputs'], args.seed)

    api_key = os.environ.get(args.api_env) or os.environ.get('OPENAI_API_KEY')
    base_url = args.base_url or os.environ.get('OPENAI_BASE_URL')
    # fallback: 从 configs/api_config.yaml 读取
    if not api_key or not base_url:
        _cfg_path = Path('configs/api_config.yaml')
        if _cfg_path.exists():
            import yaml
            _cfg = yaml.safe_load(_cfg_path.read_text())
            _default = _cfg.get('default', 'blsc')
            _api_cfg = _cfg.get(_default, {})
            api_key = api_key or _api_cfg.get('api_key')
            base_url = base_url or _api_cfg.get('base_url')
    client = OpenAI(api_key=api_key, base_url=base_url)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 断点续传：加载已有结果
    if output_path.exists():
        results = json.loads(output_path.read_text(encoding='utf-8'))
        done_keys = {(r['sample_id'], r['persona_id'], r['candidate_id']) for r in results}
        print(f'断点续传：已有 {len(results)} 条，跳过已完成项')
    else:
        results = []
        done_keys = set()

    total = len(user_inputs) * len(config['personas']) * config['candidates_per_persona']
    idx = 0
    for sample_idx, user_input in enumerate(user_inputs, start=1):
        for persona in config['personas']:
            for candidate_idx in range(1, config['candidates_per_persona'] + 1):
                idx += 1
                key = (f's{sample_idx:03d}', persona['persona_id'], f'c{candidate_idx}')
                if key in done_keys:
                    print(f'[{idx}/{total}] skip {persona["persona_id"]} c{candidate_idx} (already done)')
                    continue

                prompt = prompt_template.format(
                    persona_text=persona['persona_text'],
                    user_input=user_input,
                )
                content = call_with_retry(
                    client, config['generator_model'],
                    [{'role': 'user', 'content': prompt}],
                    timeout=args.timeout,
                )
                results.append({
                    'sample_id': f's{sample_idx:03d}',
                    'user_input': user_input,
                    'persona_id': persona['persona_id'],
                    'persona_text': persona['persona_text'],
                    'candidate_id': f'c{candidate_idx}',
                    'generator_model': config['generator_model'],
                    'response': content,
                })
                output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
                print(f'[{idx}/{total}] generated {persona["persona_id"]} c{candidate_idx}')

    print(f'已写出：{output_path}')


if __name__ == '__main__':
    main()
