#!/usr/bin/env python
"""Big Five 模型完整评估

用训练未见过的用户输入 × 多个 Big Five personality 生成回复并评分。

用法：
  python scripts/eval_big5_model.py --device cuda:3
"""
from __future__ import annotations

import json, re, sys, time, random, torch, yaml
from pathlib import Path
from openai import OpenAI

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.persona_steer import PersonaSteerConfig, PersonaSteerModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

JUDGE_PROMPT = """Evaluate if this response shows IRREPLACEABLE personality markers specific to the described personality type.

Personality: {personality}
User: {user_input}
AI Response: {response}

5 = Unique speech habits, experiences, reactions that ONLY this personality would show
4 = Clear personality markers, occasionally generic
3 = Good but generic (any personality could say this)
2 = Conflicts with personality or nearly featureless
1 = Ignores personality completely

Brief analysis, then:
Score: <1-5>"""


def load_model(ckpt_path, base_model, device, inject_layers):
    dev_id = int(device.split(':')[1])
    backbone_cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    backbone = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True, torch_dtype=torch.float16,
        device_map={"": dev_id}, use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    config = PersonaSteerConfig(
        inject_layers=inject_layers, v_dim=1024, hidden_dim=4096,
        layer_dim=backbone_cfg.hidden_size, gate_hidden_dim=256,
        gate_init_bias=-1.0, gate_max=1.0)
    model = PersonaSteerModel(config=config, encoder=backbone.model)
    model.hyper_network._tokenizer = tokenizer
    model.set_backbone(backbone)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd_ckpt = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    sd = model.state_dict()
    loaded = 0
    for k, v in sd_ckpt.items():
        if k in sd and sd[k].shape == v.shape:
            sd[k] = v.to(sd[k].dtype)
            loaded += 1
    model.load_state_dict(sd, strict=False)

    dev_obj = torch.device(device)
    for p in model.hyper_network.parameters():
        if p.device != dev_obj: p.data = p.data.to(dev_obj)
    model.injection.to(dev_obj)
    model.eval()
    return model, tokenizer, loaded


def generate_response(model, tokenizer, personality_desc, user_input, big5_scores, device):
    messages = [
        {"role": "system", "content": f"/no_think\nPersonality: {personality_desc[:300]}"},
        {"role": "user", "content": user_input},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    prompt_len = input_ids.shape[1]
    v_prev = torch.zeros(1, model.v_dim, device=device)
    big5_t = torch.tensor([big5_scores], dtype=torch.float32, device=device)

    with torch.no_grad():
        model.injection.injection_enabled = False
        v_t, _, _ = model.hyper_network(
            [personality_desc], [user_input], v_prev, big5_scores=big5_t)
        model.injection.injection_enabled = True
        model.injection.set_intervention_vector(v_t)

        generated = input_ids.clone()
        for _ in range(150):
            outputs = model.backbone(input_ids=generated, use_cache=False)
            next_logits = outputs.logits[:, -1, :] / 0.7
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() in {151643, 151645}:
                break

    resp = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
    resp = re.sub(r'<think.*?</think\s*>', '', resp, flags=re.DOTALL).strip()
    return resp


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/big5_stage1/best.pt')
    parser.add_argument('--big5_config', default='configs/big5_personalities.json')
    parser.add_argument('--eval_inputs', default='/tmp/eval_inputs.json')
    parser.add_argument('--n_inputs', type=int, default=30)
    parser.add_argument('--k_personas', type=int, default=8)
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--api_config', default='configs/api_config.yaml')
    parser.add_argument('--output', default='results/big5_eval/full_eval.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load personas
    big5_cfg = json.loads(Path(args.big5_config).read_text())
    personas = big5_cfg['personalities']

    # Load eval inputs
    eval_inputs = json.loads(Path(args.eval_inputs).read_text())[:args.n_inputs]
    print(f'{len(eval_inputs)} 评估输入 × {args.k_personas} personas = {len(eval_inputs)*args.k_personas} 条')

    # Load model
    print('加载模型...')
    model, tokenizer, loaded = load_model(
        args.checkpoint,
        '/home/kemove/Desktop/PersonaSteer/Qwen/Qwen3-4B',
        args.device, [16,17,18,19,20,21,22,23])
    print(f'加载 {loaded} tensors')

    # API
    cfg = yaml.safe_load(Path(args.api_config).read_text())
    api_cfg = cfg.get(cfg.get('default', 'blsc'), {})
    client = OpenAI(api_key=api_cfg['api_key'], base_url=api_cfg['base_url'])

    # Resume
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    done_keys = set()
    if out_path.exists():
        results = json.loads(out_path.read_text())
        done_keys = {(r['user_input'][:30], r['persona_id']) for r in results}
        print(f'断点续传: {len(results)} 条已完成')

    total_gen = len(done_keys)
    for i, user_input in enumerate(eval_inputs):
        selected = rng.sample(personas, min(args.k_personas, len(personas)))
        for p in selected:
            key = (user_input[:30], p['id'])
            if key in done_keys:
                continue

            b5 = p['big5']
            big5_scores = [b5['O'], b5['C'], b5['E'], b5['A'], b5['N']]

            # Generate
            resp = generate_response(
                model, tokenizer, p['description'], user_input, big5_scores, args.device)

            # Judge
            score = None
            try:
                judge_resp = client.chat.completions.create(
                    model='GPT-5.4',
                    messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                        personality=p['description'][:200],
                        user_input=user_input[:200],
                        response=resp[:400])}],
                    max_tokens=200, timeout=60)
                raw = judge_resp.choices[0].message.content or ''
                matches = re.findall(r'Score:\s*([1-5])', raw, re.IGNORECASE)
                score = float(matches[-1]) if matches else None
            except Exception as e:
                pass

            results.append({
                'user_input': user_input[:100],
                'persona_id': p['id'],
                'persona_name': p['name'],
                'big5_scores': big5_scores,
                'response': resp,
                'score': score,
            })
            done_keys.add(key)
            total_gen += 1
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

            if total_gen % 10 == 0 or total_gen <= 5:
                print(f'  [{total_gen}] {p["name"]:15s}: {score}  {resp[:50]}...')

    # Summary
    valid = [r['score'] for r in results if r['score'] is not None]
    from collections import Counter
    dist = Counter(int(s) for s in valid)

    print(f'\n{"="*50}')
    print(f'总评估: {len(results)} 条, 有效: {len(valid)}')
    print(f'均分: {sum(valid)/len(valid):.3f}')
    print(f'分布: {dict(sorted(dist.items()))}')

    # Per-persona
    by_persona = {}
    for r in results:
        if r['score'] is not None:
            by_persona.setdefault(r['persona_name'], []).append(r['score'])

    print(f'\n按 persona:')
    for name, scores in sorted(by_persona.items(), key=lambda x: -sum(x[1])/len(x[1])):
        print(f'  {name:18s}: {sum(scores)/len(scores):.2f} ({len(scores)} 条)')

    print(f'\n输出: {args.output}')


if __name__ == '__main__':
    main()
