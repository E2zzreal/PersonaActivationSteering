#!/usr/bin/env python3
"""调试Stage3的gate值"""
import sys
sys.path.insert(0, '/home/kemove/Desktop/Projects/3-PersonaSteer_V2')

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.persona_steer import PersonaSteerModel, PersonaSteerConfig

config_path = "configs/train_stage3_qwen3.yaml"
ckpt_path = "checkpoints/stage3_qwen3_v2/best.pt"

cfg = yaml.safe_load(open(config_path))
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
if model.hyper_network is not None:
    model.hyper_network.set_tokenizer(tokenizer)

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.to('cuda:0')
model._register_injection_hooks()
model.eval()

personality = "He is empathetic, creative, and enthusiastic."
query = "Hey, how are you?"

messages = [{"role": "user", "content": query}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer.encode(text, return_tensors='pt').to('cuda:0')
v_prev = torch.zeros(1, ps_config.v_dim, device='cuda:0')

print("=== 调试Stage3 Gate值 ===\n")

with torch.no_grad():
    # 运行一次forward查看gate值
    logits, v_t_layers, v_norm = model.forward(
        input_ids=input_ids,
        v_prev=v_prev,
        personality_texts=[personality],
        user_query_texts=[query],
    )

    print(f"v_t_layers shape: {v_t_layers.shape}")
    print(f"v_t_layers norm: {v_t_layers.norm(dim=-1).mean():.4f}")

    if hasattr(model.injection, 'current_gate_values'):
        gate_vals = model.injection.current_gate_values
        print(f"\nGate values shape: {gate_vals.shape}")
        print(f"Gate values: {gate_vals[0].cpu().numpy()}")
        print(f"Gate mean: {gate_vals.mean():.4f}")
        print(f"Gate min: {gate_vals.min():.4f}")
        print(f"Gate max: {gate_vals.max():.4f}")
