#!/usr/bin/env python
"""并行多轮对话生成：对同一批 ALOE 对话，让 Claude 和 Qwen3 各自独立生成多轮回复。

输出格式（每条对话）：
{
  "conv_id": "aloe_u1234",
  "user_id": "u1234",
  "personality": "...",
  "profile": "...",
  "num_turns": 5,
  "turns": [
    {"turn_idx": 0, "user_input": "...",
     "aloe_response": "...", "claude_response": "...", "qwen3_response": "..."},
    ...
  ]
}
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import yaml
from openai import OpenAI

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ── Claude API 多轮生成 ────────────────────────────────────────────────────

def build_claude_client(cfg_path: Path) -> tuple[OpenAI, str]:
    cfg = yaml.safe_load(cfg_path.read_text())
    default = cfg.get('default', 'blsc')
    api_cfg = cfg.get(default, {})
    client = OpenAI(api_key=api_cfg['api_key'], base_url=api_cfg['base_url'])
    model = api_cfg.get('claude_generator_model', 'Claude-Opus-4.6')
    return client, model


def claude_generate_turn(client: OpenAI, model: str, personality: str, profile: str,
                         history: list[dict], user_input: str,
                         timeout: int = 90, max_retries: int = 3) -> str:
    """给定历史，生成 Claude 当前轮回复。history 格式：[{role, content}, ...]"""
    system_msg = (
        f"你的人格特征是：{personality}\n\n"
        f"你的个人简介：{profile}\n\n"
        "请以该人格说话，只输出回复正文，不要输出思考过程。"
    )
    messages = [{"role": "system", "content": system_msg}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.8,
                max_tokens=200,
                timeout=timeout,
            )
            return (resp.choices[0].message.content or '').strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f'    [claude warn] attempt {attempt}/{max_retries}: {e}. retry in {wait}s...')
            if attempt == max_retries:
                return ''
            time.sleep(wait)


# ── Qwen3 本地多轮生成 ─────────────────────────────────────────────────────

_qwen3_model = None
_qwen3_tokenizer = None


def ensure_qwen3_loaded(base_model_path: str, device: str):
    global _qwen3_model, _qwen3_tokenizer
    if _qwen3_model is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f'[Qwen3] 加载模型 {base_model_path} -> {device}')
    _qwen3_tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    _qwen3_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype='auto',
    ).to(device)
    _qwen3_model.eval()
    print('[Qwen3] 模型加载完成')


def qwen3_generate_turn(personality: str, profile: str,
                        history: list[dict], user_input: str,
                        device: str) -> str:
    """给定历史，生成 Qwen3 当前轮回复。"""
    import re
    import torch

    system_content = f"/no_think\n你的人格特征是：{personality}\n\n你的个人简介：{profile}"
    messages = [{"role": "system", "content": system_content}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    prompt = _qwen3_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    input_ids = _qwen3_tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = _qwen3_model.generate(
            input_ids=input_ids,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=_qwen3_tokenizer.eos_token_id,
        )
    response = _qwen3_tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    # 清除 think token
    response = re.sub(r'<think.*?</think\s*>', '', response, flags=re.DOTALL)
    response = re.sub(r'<think.*', '', response, flags=re.DOTALL)
    return response.strip()


# ── ALOE 数据加载 ──────────────────────────────────────────────────────────

def load_aloe_conversations(path: Path, n: int, max_turns: int, seed: int) -> list[dict]:
    items = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get('personality') and item.get('conversations'):
                items.append(item)
    random.Random(seed).shuffle(items)
    result = []
    for item in items[:n]:
        convs = item['conversations']
        user_turns, aloe_responses = [], []
        for i in range(0, len(convs) - 1, 2):
            if convs[i]['role'] == 'user' and convs[i + 1]['role'] == 'assistant':
                user_turns.append(convs[i]['content'])
                aloe_responses.append(convs[i + 1]['content'])
                if len(user_turns) >= max_turns:
                    break
        if not user_turns:
            continue
        result.append({
            'conv_id': f"aloe_{item.get('user_id', '')}",
            'user_id': item.get('user_id', ''),
            'personality': item['personality'],
            'profile': item.get('profile', 'N/A'),
            'user_turns': user_turns,
            'aloe_responses': aloe_responses,
        })
    return result


# ── 主流程 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aloe_input', default='data/split/val.jsonl')
    parser.add_argument('--output', default='results/parallel_dialogues/dialogues.json')
    parser.add_argument('--api_config', default='configs/api_config.yaml')
    parser.add_argument('--base_model_path',
                        default='/home/kemove/Desktop/PersonaSteer/Qwen/Qwen3-4B')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--n', type=int, default=30, help='采样对话数')
    parser.add_argument('--max_turns', type=int, default=5, help='每条对话最多处理轮数')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_qwen3', action='store_true', help='跳过 Qwen3 生成（仅跑 Claude）')
    parser.add_argument('--timeout', type=int, default=90)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 断点续传
    if output_path.exists():
        results = json.loads(output_path.read_text(encoding='utf-8'))
        done_ids = {r['conv_id'] for r in results}
        print(f'断点续传：已完成 {len(results)} 条对话')
    else:
        results = []
        done_ids = set()

    # 加载 Claude 客户端
    claude_client, claude_model = build_claude_client(Path(args.api_config))
    print(f'Claude 生成模型: {claude_model}')

    # 加载 ALOE 对话
    conversations = load_aloe_conversations(
        Path(args.aloe_input), args.n, args.max_turns, args.seed
    )
    print(f'采样 {len(conversations)} 条 ALOE 对话，每条最多 {args.max_turns} 轮')

    # 加载 Qwen3（如需要）
    if not args.skip_qwen3:
        ensure_qwen3_loaded(args.base_model_path, args.device)

    for conv in conversations:
        conv_id = conv['conv_id']
        if conv_id in done_ids:
            print(f'skip {conv_id}')
            continue

        personality = conv['personality']
        profile = conv['profile']
        user_turns = conv['user_turns']
        aloe_responses = conv['aloe_responses']

        claude_history: list[dict] = []
        qwen3_history: list[dict] = []
        claude_responses = []
        qwen3_responses = []

        print(f'\n[{conv_id}] {len(user_turns)} 轮')
        for t_idx, user_input in enumerate(user_turns):
            # Claude
            cr = claude_generate_turn(
                claude_client, claude_model, personality, profile,
                claude_history, user_input, timeout=args.timeout
            )
            claude_responses.append(cr)
            claude_history.append({"role": "user", "content": user_input})
            claude_history.append({"role": "assistant", "content": cr})
            print(f'  turn {t_idx} claude: {cr[:60]}...')

            # Qwen3
            if not args.skip_qwen3:
                qr = qwen3_generate_turn(
                    personality, profile, qwen3_history, user_input, args.device
                )
                qwen3_responses.append(qr)
                qwen3_history.append({"role": "user", "content": user_input})
                qwen3_history.append({"role": "assistant", "content": qr})
                print(f'  turn {t_idx} qwen3:  {qr[:60]}...')
            else:
                qwen3_responses.append('')

        turns = []
        for t_idx in range(len(user_turns)):
            turns.append({
                'turn_idx': t_idx,
                'user_input': user_turns[t_idx],
                'aloe_response': aloe_responses[t_idx] if t_idx < len(aloe_responses) else '',
                'claude_response': claude_responses[t_idx],
                'qwen3_response': qwen3_responses[t_idx],
            })

        results.append({
            'conv_id': conv_id,
            'user_id': conv['user_id'],
            'personality': personality,
            'profile': profile,
            'num_turns': len(turns),
            'turns': turns,
        })
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f'  已保存 {conv_id}（累计 {len(results)} 条）')

    print(f'\n全部完成，输出: {output_path}')


if __name__ == '__main__':
    main()
