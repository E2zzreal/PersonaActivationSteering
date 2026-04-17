#!/usr/bin/env python3
"""调试gate的原始输出"""
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
model.eval()

personality = "He is empathetic, creative, and enthusiastic."
query = "Hey, how are you?"

messages = [{"role": "user", "content": query}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer.encode(text, return_tensors='pt').to('cuda:0')
v_prev = torch.zeros(1, ps_config.v_dim, device='cuda:0')

print("=== 调试Gate原始输出 ===\n")

with torch.no_grad():
    # 生成v_t
    v_t_layers, _, _ = model.hyper_network(
        personality_texts=[personality],
        user_query_texts=[query],
        v_prev=v_prev
    )

    print(f"v_t_layers shape: {v_t_layers.shape}")

    # 手动计算gate
    v_mean = v_t_layers.mean(dim=1).float()
    print(f"v_mean shape: {v_mean.shape}")

    # 逐层查看gate_mlp
    gate_mlp = model.injection.gate.gate_mlp
    x = v_mean
    for i, layer in enumerate(gate_mlp):
        x = layer(x)
        print(f"Layer {i} ({layer.__class__.__name__}): output shape={x.shape}, mean={x.mean():.4f}, min={x.min():.4f}, max={x.max():.4f}")
        if i == len(gate_mlp) - 2:  # sigmoid之前
            print(f"  -> Pre-sigmoid values: {x[0].cpu().numpy()}")
