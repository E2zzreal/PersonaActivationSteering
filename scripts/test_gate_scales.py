#!/usr/bin/env python3
"""测试不同gate强度的生成质量"""
import sys
sys.path.insert(0, '/home/kemove/Desktop/Projects/3-PersonaSteer_V2')

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.persona_steer import PersonaSteerModel, PersonaSteerConfig

cfg = yaml.safe_load(open('configs/train_stage2_qwen3.yaml'))
tokenizer = AutoTokenizer.from_pretrained(cfg['base_model'], trust_remote_code=True)
backbone = AutoModelForCausalLM.from_pretrained(
    cfg['base_model'], trust_remote_code=True, torch_dtype=torch.float16, device_map='cuda:0')

ps_config = PersonaSteerConfig(
    inject_layers=cfg['model']['inject_layers'],
    v_dim=cfg['model']['v_dim'],
    hidden_dim=cfg['model']['hidden_dim'],
    layer_dim=cfg['model']['layer_dim'],
)

model = PersonaSteerModel(config=ps_config, backbone=backbone, encoder=backbone.model)
model._tokenizer = tokenizer
model.hyper_network.set_tokenizer(tokenizer)

ckpt = torch.load('checkpoints/stage2_qwen3_v2/best.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.to('cuda:0')
model._register_injection_hooks()
model.eval()

personality = 'He is empathetic, creative, and enthusiastic.'
query = 'Hey, how are you?'

messages = [{'role': 'user', 'content': query}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer.encode(text, return_tensors='pt').to('cuda:0')
v_prev = torch.zeros(1, ps_config.v_dim, device='cuda:0')

# 测试不同gate强度
for gate_scale in [0.0, 0.1, 0.3, 0.5, 1.0]:
    print(f"\n{'='*60}")
    print(f"Gate Scale: {gate_scale}")
    print('='*60)

    with torch.no_grad():
        # 生成v_t
        v_t_layers, _, _ = model.hyper_network([personality], [query], v_prev)

        # 手动设置gate值
        model.injection.current_v_t = v_t_layers
        model.injection.current_gate_values = torch.full(
            (1, 8), gate_scale, device='cuda:0'
        )

        # 生成
        outputs = model.backbone.generate(
            input_ids,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Response: {response[:150]}")
