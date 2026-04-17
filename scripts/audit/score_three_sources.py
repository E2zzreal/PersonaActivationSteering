#!/usr/bin/env python
"""三源统一打分 v2：Claude teacher / ALOE gold / Qwen3 baseline。

特性：
- 严格 rubric：persona-specificity 而非 quality
- CoT 推理：先分析再打分，结果可审计
- 多模型集成：GPT-5.4 + GLM-5.1，取均值
- ALOE 多轮：使用完整对话评估跨轮一致性
- Qwen3 / Claude：单轮评估
- 断点续传
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import yaml
from openai import OpenAI

# ── Prompts ────────────────────────────────────────────────────────────────

SINGLE_TURN_PROMPT = """## Task
You are a strict evaluator. Measure how well the assistant's response is tailored to THIS PARTICULAR user's personality — NOT how polite, helpful, or fluent the response is.

## User Information
Profile: {profile}
Personality Traits: {personality}

## User Input
{user_input}

## Assistant Response
{response}

## Scoring Rubric (1-5)

5 - Persona-Specific (Rare)
The assistant uses specific vocabulary, tone, or communication patterns that CLEARLY match this user's traits. A different user with a different personality would receive a noticeably different response style.

4 - Clearly Adapted
The assistant's style is visibly adjusted for this personality. At least 2 personality dimensions are reflected (e.g., tone + content preference). Generic phrasing is minimal.

3 - Partially Adapted
Shows awareness of 1 personality dimension but the rest is standard LLM output. Someone else with a different profile would get essentially the same response.

2 - Superficial or Accidental
Any alignment appears coincidental. The response could have been given to any user. Style is default-LLM polite.

1 - Misaligned or Contradictory
The assistant's style contradicts the stated personality (e.g., uses formal language with a casual user, over-explains to an expert).

## Key Rule
If you find yourself wanting to give a 5 because the response is "helpful and good", STOP.
Score 5 only for persona-SPECIFICITY, not quality. Good-but-generic = 3.

## Output Format
First write 1-2 sentences of reasoning. Then on a new line write exactly:
Score: <integer>"""

MULTITURN_PROMPT = """## Task
You are a strict evaluator. Measure how consistently the assistant tailored its responses to THIS PARTICULAR user's personality across the whole dialogue — NOT how helpful or fluent the responses are.

## User Information
Profile: {profile}
Personality Traits: {personality}

## Dialogue
{dialogue}

## Scoring Rubric (1-5)

5 - Consistently Persona-Specific (Rare)
Every assistant turn reflects specific vocabulary, tone, or communication patterns that CLEARLY match this user's traits. A different user with a different personality would receive a noticeably different conversation.

4 - Mostly Adapted
Most assistant turns show clear personality adaptation. At least 2 personality dimensions are reflected consistently. Minor lapses only.

3 - Partially Adapted
Some turns adapt to the personality, others are generic. The overall impression is of an LLM that is occasionally aware of the user's traits but defaults to neutral style.

2 - Superficial or Accidental
Any alignment appears coincidental or limited to 1-2 turns out of many. Most responses could have been given to any user.

1 - Misaligned or Contradictory
The assistant's style consistently contradicts or ignores the stated personality.

## Key Rule
If you find yourself wanting to give a 5 because the responses are "helpful and good", STOP.
Score 5 only for persona-SPECIFICITY and CONSISTENCY, not quality. Good-but-generic = 3.

## Output Format
First write 1-2 sentences of reasoning. Then on a new line write exactly:
Score: <integer>"""


# ── Helpers ────────────────────────────────────────────────────────────────

def load_api_clients(cfg_path: Path) -> list[tuple[str, OpenAI, dict]]:
    """从 api_config.yaml 加载 judge_models，返回 [(model_name, client, model_cfg), ...]。"""
    cfg = yaml.safe_load(cfg_path.read_text())
    blsc_cfg = cfg.get('blsc', {})
    clients = []
    for entry in cfg.get('judge_models', []):
        model = entry['model']
        api_name = entry.get('api', 'blsc')
        api_cfg = cfg.get(api_name, blsc_cfg)
        client = OpenAI(api_key=api_cfg['api_key'], base_url=api_cfg['base_url'])
        model_cfg = {
            'max_tokens': entry.get('max_tokens', 300),
            'disable_thinking': entry.get('disable_thinking', False),
        }
        clients.append((model, client, model_cfg))
    if not clients:
        default = cfg.get('default', 'blsc')
        api_cfg = cfg.get(default, blsc_cfg)
        model = api_cfg.get('model', 'GPT-5.2')
        clients.append((model, OpenAI(api_key=api_cfg['api_key'], base_url=api_cfg['base_url']), {'max_tokens': 300, 'disable_thinking': False}))
    return clients


def parse_score(text: str) -> float | None:
    # 取最后一个 "Score: X"，避免思维链中间值干扰
    matches = re.findall(r'Score:\s*([1-5])', text or '', re.IGNORECASE)
    if matches:
        return float(matches[-1])
    # fallback：取文本最后出现的孤立数字 1-5
    for ch in reversed(text or ''):
        if ch in '12345':
            return float(ch)
    return None


def call_judge(client: OpenAI, model: str, prompt: str,
               timeout: int = 90, max_retries: int = 3,
               max_tokens: int = 300, disable_thinking: bool = False) -> str:
    extra = {}
    if disable_thinking:
        extra['extra_body'] = {'thinking': {'type': 'disabled'}}
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.2,
                max_tokens=max_tokens,
                timeout=timeout,
                **extra,
            )
            return (resp.choices[0].message.content or '').strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f'    [warn] {model} attempt {attempt}/{max_retries}: {e}. retry in {wait}s...')
            if attempt == max_retries:
                return ''
            time.sleep(wait)


def score_item(clients: list[tuple[str, OpenAI, dict]], prompt: str,
               timeout: int) -> dict:
    """用所有模型打分，返回 {per_model: {model: {raw, score}}, ensemble_score}。"""
    per_model = {}
    valid_scores = []
    for model, client, model_cfg in clients:
        raw = call_judge(client, model, prompt, timeout=timeout,
                         max_tokens=model_cfg.get('max_tokens', 300),
                         disable_thinking=model_cfg.get('disable_thinking', False))
        s = parse_score(raw)
        per_model[model] = {'raw': raw, 'score': s}
        if s is not None:
            valid_scores.append(s)
    ensemble = round(sum(valid_scores) / len(valid_scores), 3) if valid_scores else None
    return {'per_model': per_model, 'ensemble_score': ensemble}


def load_existing(path: Path) -> list:
    if path.exists():
        data = json.loads(path.read_text(encoding='utf-8'))
        return data if isinstance(data, list) else data.get('items', [])
    return []


def format_dialogue(conversations: list, max_turns: int = 10) -> str:
    lines = []
    for turn in conversations[:max_turns * 2]:
        role = 'User' if turn['role'] == 'user' else 'Assistant'
        lines.append(f'{role}: {turn["content"]}')
    return '\n'.join(lines)


def summarize(items: list, source_label: str) -> dict:
    scores = [x['ensemble_score'] for x in items if x.get('ensemble_score') is not None]
    if not scores:
        return {'source': source_label, 'n': 0}
    by_persona = defaultdict(list)
    for x in items:
        if x.get('ensemble_score') is not None:
            by_persona[x.get('persona_id', 'unknown')].append(x['ensemble_score'])
    return {
        'source': source_label,
        'n': len(scores),
        'mean': round(sum(scores) / len(scores), 3),
        'min': round(min(scores), 1),
        'max': round(max(scores), 1),
        'by_persona': {
            pid: {'n': len(ss), 'mean': round(sum(ss) / len(ss), 3)}
            for pid, ss in sorted(by_persona.items())
        },
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel_input',
                        default='results/parallel_dialogues/dialogues.json',
                        help='generate_parallel_dialogues.py 的输出文件')
    parser.add_argument('--output_dir', default='results/three_source_score_v3')
    parser.add_argument('--api_config', default='configs/api_config.yaml')
    parser.add_argument('--timeout', type=int, default=90)
    args = parser.parse_args()

    clients = load_api_clients(Path(args.api_config))
    model_names = [m for m, _, _ in clients]
    print(f'Judge 模型: {model_names}')

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    parallel = json.loads(Path(args.parallel_input).read_text(encoding='utf-8'))
    print(f'加载 {len(parallel)} 条并行对话（三源同批次，多轮）')

    source_response_key = {
        'aloe':   'aloe_response',
        'claude': 'claude_response',
        'qwen3':  'qwen3_response',
    }

    all_results: dict[str, list] = {}
    for src, resp_key in source_response_key.items():
        src_out = out / f'{src}_scores.json'
        results = load_existing(src_out)
        results_by_id = {x['item_id']: x for x in results}
        print(f'\n[{src}] 已完成 {len(results)} 条')

        for conv in parallel:
            conv_id = conv['conv_id']

            dialogue_lines = []
            for turn in conv['turns']:
                resp = turn.get(resp_key, '')
                if not resp:
                    continue
                dialogue_lines.append(f"User: {turn['user_input']}")
                dialogue_lines.append(f"Assistant: {resp}")
            if not dialogue_lines:
                continue

            prompt = MULTITURN_PROMPT.format(
                profile=conv.get('profile', 'N/A'),
                personality=conv['personality'],
                dialogue='\n'.join(dialogue_lines),
            )

            if conv_id in results_by_id:
                # 按模型粒度续传：只补分数为 None 的模型
                existing = results_by_id[conv_id]
                missing = [(m, c, cfg) for m, c, cfg in clients
                           if existing.get('per_model', {}).get(m, {}).get('score') is None]
                if not missing:
                    print(f'  skip {conv_id} (all models done)')
                    continue
                print(f'  retry {conv_id} missing={[m for m,_,_ in missing]}')
                partial = score_item(missing, prompt, args.timeout)
                existing.setdefault('per_model', {}).update(partial['per_model'])
                valid = [v['score'] for v in existing['per_model'].values()
                         if v.get('score') is not None]
                existing['ensemble_score'] = round(sum(valid) / len(valid), 3) if valid else None
            else:
                result = score_item(clients, prompt, args.timeout)
                entry = {
                    'item_id': conv_id,
                    'source': src,
                    'num_turns': conv['num_turns'],
                    **result,
                }
                results.append(entry)
                results_by_id[conv_id] = entry
            src_out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            print(f'  [{src}] {conv_id} → {results_by_id[conv_id].get("ensemble_score")}')

        all_results[src] = results

    summary = {
        'judge_models': model_names,
        'eval_mode': 'multi_turn_parallel',
        'n_conversations': len(parallel),
        'claude': summarize(all_results.get('claude', []), 'claude_opus_4.6_v2'),
        'aloe':   summarize(all_results.get('aloe', []),   'aloe_gold'),
        'qwen3':  summarize(all_results.get('qwen3', []),  'qwen3_4b_baseline'),
    }
    (out / 'comparison_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print('\n\n====== 三源多轮并行对比结果 ======')
    print(f'Judge 模型: {model_names}  |  对话数: {len(parallel)}  |  评估方式: 多轮')
    for src in ['claude', 'aloe', 'qwen3']:
        s = summary[src]
        print(f"  [{s['source']}]  n={s.get('n', 0)}  mean={s.get('mean', 'N/A')}")
    print(f'\n结果保存至 {out}')


if __name__ == '__main__':
    main()
