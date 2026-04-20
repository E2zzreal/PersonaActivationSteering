#!/usr/bin/env python
"""Prompt 升级小批量对比试验

对比旧 prompt（简单指令）vs 新 prompt（行为约束）生成的 Claude 回复，
用 LLM Judge 评分，验证人格签名密度是否提升。

用法：
  python scripts/audit/test_prompt_upgrade.py --n 10 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import yaml
from openai import OpenAI

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ── 旧 prompt ─────────────────────────────────────────────────────────
OLD_SYSTEM_PROMPT = """{personality}

你的个人简介：{profile}

请以该人格说话，只输出回复正文，不要输出思考过程。"""

# ── 新 prompt（行为约束版）────────────────────────────────────────────
NEW_SYSTEM_PROMPT = """You are roleplaying a character with a unique personality. Every response you give MUST be immediately recognizable as THIS specific personality — not something any friendly person would say.

【Your personality traits】
{personality}

【Your personal background】
{profile}

【MANDATORY rules for every response】
1. Include at least 2 IRREPLACEABLE personality markers per response:
   - Unique speech habits (catchphrases, tone words, metaphor style specific to this personality)
   - Specific personal experiences, memories or preferences consistent with the personality
   - Emotional reaction patterns unique to this personality (not generic "that's great!")

2. SHOW personality through actions, don't TELL it:
   ✗ "I'm a very adventurous person" (telling)
   ✓ "Last month I climbed an unmarked mountain trail and almost got lost — but that adrenaline rush was unreal" (showing)

3. Pick the 2-3 most distinctive traits from the personality description as your thread.
   Stay consistent across turns. Don't spread thin across all traits.

4. Respond in the same language as the user's message. Output only the response text — no thinking process or character analysis."""


# ── Judge prompt（与 score_three_sources 相同）──────────────────────
JUDGE_PROMPT = """你是一位严格的人格一致性评审专家。请评估以下回复是否展现了目标人格的**专属行为特征**。

【目标人格】
{personality}

【用户消息】
{user_input}

【AI回复】
{response}

【评分标准（1-5 分）】
5 - 有该人格不可替换的具体行为：独特的语言习惯、价值取向或情感模式
4 - 有人格印记，但偶尔略显通用
3 - 基本礼貌得体，但换成任何其他人格也说得通（好但通用 = 3）
2 - 与目标人格有冲突或几乎看不到人格特征
1 - 完全忽视人格

请输出：
分析：<简短分析>
Score: <1-5 的整数>"""


def parse_score(text: str) -> float | None:
    matches = re.findall(r'Score:\s*([1-5])', text or '', re.IGNORECASE)
    return float(matches[-1]) if matches else None


def call_api(client, model, messages, timeout=90, max_retries=3) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.8, max_tokens=300, timeout=timeout)
            return (resp.choices[0].message.content or '').strip()
        except Exception as e:
            if attempt == max_retries:
                return ''
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10, help='测试条数')
    parser.add_argument('--dialogues', default='results/parallel_dialogues/dialogues.json')
    parser.add_argument('--api_config', default='configs/api_config.yaml')
    parser.add_argument('--output', default='results/prompt_upgrade_test.json')
    args = parser.parse_args()

    # API client
    cfg = yaml.safe_load(Path(args.api_config).read_text())
    api_cfg = cfg.get(cfg.get('default', 'blsc'), {})
    client = OpenAI(api_key=api_cfg['api_key'], base_url=api_cfg['base_url'])
    gen_model = api_cfg.get('claude_generator_model', 'Claude-Opus-4.6')
    judge_model = "GPT-5.4"

    # 加载对话
    dialogues = json.loads(Path(args.dialogues).read_text())[:args.n]
    print(f'测试 {len(dialogues)} 条对话，生成模型: {gen_model}，Judge: {judge_model}')

    results = []
    for conv in dialogues:
        personality = conv['personality']
        profile = conv['profile']
        turn = conv['turns'][0]
        user_input = turn['user_input']

        print(f'\n[{conv["conv_id"]}]')

        # 生成旧 prompt 回复
        old_sys = OLD_SYSTEM_PROMPT.format(personality=personality, profile=profile)
        old_resp = call_api(client, gen_model, [
            {"role": "system", "content": old_sys},
            {"role": "user", "content": user_input},
        ])

        # 生成新 prompt 回复
        new_sys = NEW_SYSTEM_PROMPT.format(personality=personality, profile=profile)
        new_resp = call_api(client, gen_model, [
            {"role": "system", "content": new_sys},
            {"role": "user", "content": user_input},
        ])

        # Judge 评分
        old_judge = call_api(client, judge_model, [{"role": "user", "content":
            JUDGE_PROMPT.format(personality=personality, user_input=user_input, response=old_resp)}])
        new_judge = call_api(client, judge_model, [{"role": "user", "content":
            JUDGE_PROMPT.format(personality=personality, user_input=user_input, response=new_resp)}])

        old_score = parse_score(old_judge)
        new_score = parse_score(new_judge)

        print(f'  旧 prompt: {old_score} | {old_resp[:80]}...')
        print(f'  新 prompt: {new_score} | {new_resp[:80]}...')

        results.append({
            'conv_id': conv['conv_id'],
            'personality': personality[:80],
            'user_input': user_input[:80],
            'old_response': old_resp,
            'new_response': new_resp,
            'old_score': old_score,
            'new_score': new_score,
            'old_judge': old_judge,
            'new_judge': new_judge,
        })

    # 汇总
    old_scores = [r['old_score'] for r in results if r['old_score']]
    new_scores = [r['new_score'] for r in results if r['new_score']]

    print('\n' + '=' * 50)
    print(f'旧 prompt 均分: {sum(old_scores)/len(old_scores):.2f} ({len(old_scores)} 条)')
    print(f'新 prompt 均分: {sum(new_scores)/len(new_scores):.2f} ({len(new_scores)} 条)')
    print(f'提升: {sum(new_scores)/len(new_scores) - sum(old_scores)/len(old_scores):+.2f}')

    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f'输出: {args.output}')


if __name__ == '__main__':
    main()
