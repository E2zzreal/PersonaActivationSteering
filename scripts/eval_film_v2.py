#!/usr/bin/env python
"""
FiLM v2 三线对比评估

A. injection-only: 通用 system prompt + FiLM 注入
B. prompt-only:    人格 system prompt + 无注入
C. prompt+injection: 人格 system prompt + FiLM 注入

用法:
  python scripts/eval_film_v2.py --device cuda:0
"""
from __future__ import annotations

import json, re, sys, random, torch, yaml, argparse
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
        gate_init_bias=-1.0, gate_max=1.0,
        injection_type="film",  # FiLM mode
    )
    model = PersonaSteerModel(config=config, encoder=backbone.model)
    model.hyper_network._tokenizer = tokenizer
    model.set_backbone(backbone)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd_ckpt = ckpt.get('model_state_dict', ckpt)
    sd = model.state_dict()
    loaded = 0
    for k, v in sd_ckpt.items():
        if k in sd and sd[k].shape == v.shape:
            sd[k] = v.to(sd[k].dtype)
            loaded += 1
    model.load_state_dict(sd, strict=False)

    dev_obj = torch.device(device)
    for p in model.hyper_network.parameters():
        if p.device != dev_obj:
            p.data = p.data.to(dev_obj)
    model.injection.to(dev_obj)
    model.eval()
    return model, tokenizer, loaded


def generate_response(model, tokenizer, personality_desc, user_input, big5_scores, device,
                      use_prompt=True, use_injection=True):
    """三线生成：控制 prompt 和 injection 开关"""
    if use_prompt:
        system_content = f"/no_think\nPersonality: {personality_desc[:300]}"
    else:
        system_content = "/no_think\nPlease respond to the user naturally."

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_input},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    prompt_len = input_ids.shape[1]
    v_prev = torch.zeros(1, model.v_dim, device=device)
    big5_t = torch.tensor([big5_scores], dtype=torch.float32, device=device)

    with torch.no_grad():
        # 计算 v_t（总是计算，但可以不使用）
        model.injection.injection_enabled = False
        v_t, _, _ = model.hyper_network(
            [personality_desc], [user_input], v_prev, big5_scores=big5_t)

        if use_injection:
            model.injection.injection_enabled = True
            model.injection.set_intervention_vector(v_t)
        else:
            model.injection.injection_enabled = False

        generated = input_ids.clone()
        for _ in range(150):
            outputs = model.backbone(input_ids=generated, use_cache=False)
            next_logits = outputs.logits[:, -1, :] / 0.7
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() in {151643, 151645}:
                break

        # 恢复
        model.injection.injection_enabled = True

    resp = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
    resp = re.sub(r'<think.*?</think\s*>', '', resp, flags=re.DOTALL).strip()
    return resp


def judge_score(client, personality_desc, user_input, response):
    try:
        judge_resp = client.chat.completions.create(
            model='GPT-5.4',
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                personality=personality_desc[:200],
                user_input=user_input[:200],
                response=response[:400])}],
            max_tokens=200, timeout=60)
        raw = judge_resp.choices[0].message.content or ''
        matches = re.findall(r'Score:\s*([1-5])', raw, re.IGNORECASE)
        return float(matches[-1]) if matches else None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/film_v2/phase3/best.pt')
    parser.add_argument('--big5_config', default='configs/big5_personalities.json')
    parser.add_argument('--eval_inputs', default='/tmp/eval_inputs.json')
    parser.add_argument('--n_inputs', type=int, default=10)
    parser.add_argument('--k_personas', type=int, default=5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--api_config', default='configs/api_config.yaml')
    parser.add_argument('--output', default='results/film_v2_eval/three_line.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    personas = json.loads(Path(args.big5_config).read_text())['personalities']
    eval_inputs = json.loads(Path(args.eval_inputs).read_text())[:args.n_inputs]

    print(f'评估规模: {len(eval_inputs)} inputs × {args.k_personas} personas × 3 settings '
          f'= {len(eval_inputs)*args.k_personas*3} 条')

    # Load model
    print('加载 FiLM 模型...')
    model, tokenizer, loaded = load_model(
        args.checkpoint,
        '/home/kemove/Desktop/PersonaSteer/Qwen/Qwen3-4B',
        args.device, [16, 17, 18, 19, 20, 21, 22, 23])
    print(f'加载 {loaded} tensors, injection_type={model.config.injection_type}')

    # API
    cfg = yaml.safe_load(Path(args.api_config).read_text())
    api_cfg = cfg.get(cfg.get('default', 'blsc'), {})
    client = OpenAI(api_key=api_cfg['api_key'], base_url=api_cfg['base_url'])

    # 三种设置
    settings = [
        ("A_injection_only", False, True),   # no prompt, yes injection
        ("B_prompt_only",    True,  False),  # yes prompt, no injection
        ("C_prompt_inject",  True,  True),   # yes prompt, yes injection
    ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    done_keys = set()
    if out_path.exists():
        results = json.loads(out_path.read_text())
        done_keys = {(r['user_input'][:30], r['persona_id'], r['setting']) for r in results}
        print(f'断点续传: {len(results)} 条已完成')

    total = 0
    for i, user_input in enumerate(eval_inputs):
        selected = rng.sample(personas, min(args.k_personas, len(personas)))
        for p in selected:
            b5 = p['big5']
            big5_scores = [b5['O'], b5['C'], b5['E'], b5['A'], b5['N']]

            for setting_name, use_prompt, use_injection in settings:
                key = (user_input[:30], p['id'], setting_name)
                if key in done_keys:
                    continue

                resp = generate_response(
                    model, tokenizer, p['description'], user_input,
                    big5_scores, args.device,
                    use_prompt=use_prompt, use_injection=use_injection)

                score = judge_score(client, p['description'], user_input, resp)

                results.append({
                    'user_input': user_input[:100],
                    'persona_id': p['id'],
                    'persona_name': p['name'],
                    'setting': setting_name,
                    'use_prompt': use_prompt,
                    'use_injection': use_injection,
                    'response': resp,
                    'score': score,
                })
                done_keys.add(key)
                total += 1

                if total % 5 == 0 or total <= 3:
                    print(f'  [{total}] {setting_name:20s} {p["name"]:15s}: {score}  {resp[:50]}...')

                # 定期保存
                if total % 10 == 0:
                    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Summary
    print(f'\n{"="*60}')
    for setting_name, _, _ in settings:
        valid = [r['score'] for r in results
                 if r['setting'] == setting_name and r['score'] is not None]
        if valid:
            from collections import Counter
            dist = Counter(int(s) for s in valid)
            mean = sum(valid) / len(valid)
            print(f'{setting_name:20s}: mean={mean:.3f}, n={len(valid)}, dist={dict(sorted(dist.items()))}')

    print(f'\n输出: {args.output}')


if __name__ == '__main__':
    main()
