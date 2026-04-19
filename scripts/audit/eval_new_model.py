#!/usr/bin/env python
"""D4 评估：对比新训练模型与 Qwen3 baseline 的人格对齐分。

使用与 score_three_sources.py 相同的严格 v3 rubric 和 GPT-5.4 judge，
在相同的 30 条并行对话上生成新模型的多轮回复并评分。

用法：
  python scripts/audit/eval_new_model.py \
    --checkpoint checkpoints/stage1_claude_sft/best.pt \
    --dialogues results/parallel_dialogues/dialogues.json \
    --output results/eval_stage1_claude_sft \
    --device cuda:2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
import yaml
from openai import OpenAI

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ── Strict v3 judge prompt（与 score_three_sources.py 相同）─────────────
MULTITURN_PROMPT = """你是一位严格的人格一致性评审专家。请评估以下多轮对话是否展现了目标人格的**专属行为特征**。

【目标人格】
{personality}

【多轮对话】
{dialogue}

【评分标准（1-5 分）】
5 - 每轮回复都有该人格不可替换的具体行为：独特的语言习惯、价值取向或情感模式，换成其他人格就会明显不对
4 - 多数回复有人格印记，偶尔 1-2 轮略显通用
3 - 基本礼貌得体，但换成任何其他人格也说得通（好但通用 = 3）
2 - 与目标人格有冲突，或几乎看不到人格特征
1 - 完全忽视人格，像通用 AI 的标准回复

【关键规则】
- "好但通用"= 3 分，不能给 4 分
- 必须有人格专属细节才能给 4-5 分
- 先写 1-2 句分析，再给分

请输出：
分析：<简短分析>
Score: <1-5 的整数>"""


def parse_score(text: str) -> float | None:
    matches = re.findall(r'Score:\s*([1-5])', text or '', re.IGNORECASE)
    if matches:
        return float(matches[-1])
    for ch in reversed(text or ''):
        if ch in '12345':
            return float(ch)
    return None


def build_judge_client(cfg_path: Path):
    cfg = yaml.safe_load(cfg_path.read_text())
    default = cfg.get('default', 'blsc')
    api_cfg = cfg.get(default, {})
    client = OpenAI(api_key=api_cfg['api_key'], base_url=api_cfg['base_url'])
    judge_model = "GPT-5.4"
    return client, judge_model


def call_judge(client, model, prompt, timeout=60, max_retries=3) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                timeout=timeout,
            )
            return (resp.choices[0].message.content or '').strip()
        except Exception as e:
            if attempt == max_retries:
                return ''
            time.sleep(2 ** attempt)


# ── 加载 PersonaSteer checkpoint ──────────────────────────────────────
def load_model(checkpoint_path: str, base_model_path: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from src.models.persona_steer import PersonaSteerModel, PersonaSteerConfig

    print(f'[Model] 加载 checkpoint: {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 自动检测注入层
    num_inject = None
    for k, v in ckpt.items():
        if 'layer_projectors' in k:
            try:
                idx = int(k.split('layer_projectors.')[1].split('.')[0])
                num_inject = max(num_inject or 0, idx + 1)
            except Exception:
                pass
    inject_layers = list(range(4, 4 + (num_inject or 8)))
    print(f'[Model] inject_layers={inject_layers}')

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    backbone_cfg = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    backbone = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": int(device.split(':')[1]) if ':' in device else 0},
        use_cache=False,
    )

    config = PersonaSteerConfig(
        inject_layers=inject_layers,
        v_dim=1024,
        hidden_dim=4096,
        layer_dim=backbone_cfg.hidden_size,
        gate_hidden_dim=256,
        gate_init_bias=-2.0,
        gate_max=1.0,
    )
    model = PersonaSteerModel(config=config, backbone=backbone, encoder=backbone.model)
    model.hyper_network._tokenizer = tokenizer

    # 加载权重
    loaded = 0
    sd = model.state_dict()
    for k, v in ckpt.items():
        if k in sd and sd[k].shape == v.shape:
            sd[k] = v.to(sd[k].dtype)
            loaded += 1
    model.load_state_dict(sd, strict=False)

    # 确保 hyper_network 和 injection 与 backbone 在同一设备
    dev_obj = torch.device(device)
    if model.hyper_network is not None:
        for param in model.hyper_network.parameters():
            if param.device != dev_obj:
                param.data = param.data.to(dev_obj)
    if model.injection is not None:
        model.injection.to(dev_obj)

    model.eval()
    print(f'[Model] 已加载 {loaded} 个参数张量，所有组件在 {device}')
    return model, tokenizer


# ── 多轮生成 ──────────────────────────────────────────────────────────
def generate_multiturn(model, tokenizer, personality: str, profile: str,
                       user_turns: list[str], device: str) -> list[str]:
    import re as re_mod
    dev = torch.device(device)
    history = []
    responses = []
    v_prev = torch.zeros(1, model.v_dim, device=dev)

    for user_input in user_turns:
        messages = [
            {"role": "system",
             "content": f"/no_think\n你的人格特征是：{personality}\n\n你的个人简介：{profile}"}
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(dev)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            generated_ids, v_prev = model.generate(
                input_ids=input_ids,
                v_prev=v_prev,
                personality_texts=[personality],
                user_query_texts=[user_input],
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
            )

        response = tokenizer.decode(
            generated_ids[0][prompt_len:], skip_special_tokens=True)
        # 清除残余 think tokens
        response = re_mod.sub(r'<think.*?</think\s*>', '', response, flags=re_mod.DOTALL)
        response = re_mod.sub(r'<think.*', '', response, flags=re_mod.DOTALL)
        response = response.strip()

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        responses.append(response)

    return responses


def format_dialogue(turns: list[dict], response_key: str) -> str:
    lines = []
    for t in turns:
        lines.append(f"用户：{t['user_input']}")
        lines.append(f"AI：{t.get(response_key, '')}")
    return '\n'.join(lines)


# ── 主流程 ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/stage1_claude_sft/best.pt')
    parser.add_argument('--dialogues', default='results/parallel_dialogues/dialogues.json')
    parser.add_argument('--output', default='results/eval_stage1_claude_sft')
    parser.add_argument('--base_model', default='/home/kemove/Desktop/PersonaSteer/Qwen/Qwen3-4B')
    parser.add_argument('--api_config', default='configs/api_config.yaml')
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--timeout', type=int, default=60)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    responses_path = out_dir / 'responses.json'
    scores_path = out_dir / 'scores.json'

    # 加载并行对话数据
    dialogues = json.loads(Path(args.dialogues).read_text())
    print(f'加载 {len(dialogues)} 条对话')

    # ── Step 1: 生成新模型回复（支持断点续传）────────────────────────
    if responses_path.exists():
        results = json.loads(responses_path.read_text())
        done_ids = {r['conv_id'] for r in results}
        print(f'断点续传：已生成 {len(results)} 条')
    else:
        results = []
        done_ids = set()

    if len(done_ids) < len(dialogues):
        model, tokenizer = load_model(args.checkpoint, args.base_model, args.device)

        for conv in dialogues:
            conv_id = conv['conv_id']
            if conv_id in done_ids:
                continue

            user_turns = [t['user_input'] for t in conv['turns']]
            responses = generate_multiturn(
                model, tokenizer,
                conv['personality'], conv['profile'],
                user_turns, args.device
            )
            entry = {
                'conv_id': conv_id,
                'personality': conv['personality'],
                'profile': conv['profile'],
                'turns': [
                    {**t, 'new_model_response': r}
                    for t, r in zip(conv['turns'], responses)
                ],
            }
            results.append(entry)
            done_ids.add(conv_id)
            responses_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            print(f'  [{len(results)}/{len(dialogues)}] {conv_id}: {responses[0][:60]}...')

        # 释放显存
        del model
        torch.cuda.empty_cache()
        print(f'\n生成完成，保存至 {responses_path}')
    else:
        print('所有回复已存在，跳过生成')
        results = json.loads(responses_path.read_text())

    # ── Step 2: Judge 评分（支持断点续传）────────────────────────────
    if scores_path.exists():
        scored = json.loads(scores_path.read_text())
        done_score_ids = {s['conv_id'] for s in scored}
        print(f'断点续传：已评分 {len(scored)} 条')
    else:
        scored = []
        done_score_ids = set()

    judge_client, judge_model = build_judge_client(Path(args.api_config))
    print(f'Judge: {judge_model}')

    for entry in results:
        conv_id = entry['conv_id']
        if conv_id in done_score_ids:
            continue

        dialogue_text = '\n'.join(
            f"用户：{t['user_input']}\nAI：{t.get('new_model_response', '')}"
            for t in entry['turns']
        )
        prompt = MULTITURN_PROMPT.format(
            personality=entry['personality'],
            dialogue=dialogue_text,
        )
        raw = call_judge(judge_client, judge_model, prompt, timeout=args.timeout)
        score = parse_score(raw)

        scored.append({
            'conv_id': conv_id,
            'score': score,
            'raw': raw,
        })
        done_score_ids.add(conv_id)
        scores_path.write_text(json.dumps(scored, indent=2, ensure_ascii=False))
        print(f'  [{len(scored)}/{len(results)}] {conv_id}: {score}')

    # ── 结果汇总 ──────────────────────────────────────────────────────
    valid = [s['score'] for s in scored if s['score'] is not None]
    avg = round(sum(valid) / len(valid), 3) if valid else None

    summary = {
        'model': args.checkpoint,
        'n_conv': len(scored),
        'n_valid': len(valid),
        'avg_score': avg,
        'reference': {
            'claude': 3.833,
            'qwen3_baseline': 2.833,
            'aloe_gold': 2.600,
        }
    }
    (out_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print('\n' + '='*50)
    print(f'新模型（Claude SFT Stage1）: {avg}')
    print(f'参考分：Claude 3.833 / Qwen3 baseline 2.833 / ALOE 2.600')
    if avg is not None:
        if avg > 2.833:
            print(f'✅ 突破 baseline！(+{avg - 2.833:.3f})')
        else:
            print(f'❌ 未超过 baseline（差 {2.833 - avg:.3f}）')
    print(f'输出: {out_dir}')


if __name__ == '__main__':
    main()
