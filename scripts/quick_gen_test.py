#!/usr/bin/env python
"""快速测试新训练checkpoint的生成质量"""

import sys
import torch
import yaml
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.persona_steer import PersonaSteerModel, PersonaSteerConfig
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, AutoModel

# 测试样本
SAMPLE = {
    "personality": "He is easy-going, articulate, and creative. He enjoys deep conversations.",
    "user_msg": "Hey! How are you today? I've been thinking about starting a new hobby.",
}

# 要测试的checkpoint
CHECKPOINTS = [
    ("Baseline (no injection)", None),
    ("Best Batch1: gate_init_0", "checkpoints/exp_gate_init_0/best.pt"),
    ("Best Batch2: gate_reg_0.01_lr3e5", "checkpoints/exp_gate_reg_0.01_lr3e5/best.pt"),
]

def load_model(checkpoint_path, device):
    """加载模型"""
    base_model_path = "/home/kemove/Desktop/PersonaSteer/Qwen/Qwen3-4B"
    
    backbone_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    persona_config = PersonaSteerConfig(
        inject_layers=[8, 9, 10, 11, 12, 13, 14, 15],
        v_dim=1024,
        hidden_dim=4096,
        layer_dim=backbone_config.hidden_size,
        gate_hidden_dim=256,
    )

    encoder = AutoModel.from_pretrained(base_model_path, trust_remote_code=True, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    model = PersonaSteerModel(config=persona_config, encoder=encoder)
    
    if model.hyper_network is not None:
        model.hyper_network.v_norm_clip = 10.0
        model.hyper_network._tokenizer = tokenizer

    backbone = AutoModelForCausalLM.from_pretrained(
        base_model_path, trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    )
    model.set_backbone(backbone)

    if checkpoint_path:
        print(f"  Loading: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
    else:
        model.baseline_mode = True

    model.backbone.to(device)
    if model.hyper_network is not None:
        model.hyper_network.to(device)
    model.injection.to(device)
    model.eval()
    
    return model, tokenizer

def generate(model, tokenizer, personality, user_msg, device):
    """生成回复"""
    messages = [{"role": "user", "content": user_msg}]
    
    try:
        result = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = result.to(device) if not hasattr(result, 'input_ids') else result.input_ids.to(device)
    except Exception as e:
        return f"[Token Error] {e}"

    try:
        with torch.no_grad():
            v_prev = torch.zeros(1, 1024, dtype=torch.float32, device=device)
            outputs, _ = model.generate(
                input_ids=inputs,
                v_prev=v_prev,
                user_texts=[personality],
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
            )
            new_tokens = outputs[0][inputs.shape[-1]:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    except Exception as e:
        import traceback
        return f"[Gen Error] {e}\n{traceback.format_exc()}"

def main():
    device = "cuda:0"
    
    print("=" * 80)
    print("PersonaSteer 生成质量测试")
    print("=" * 80)
    print(f"\nPersonality: {SAMPLE['personality']}")
    print(f"User: {SAMPLE['user_msg']}")
    print()
    
    for name, ckpt_path in CHECKPOINTS:
        print(f"\n--- {name} ---")
        
        try:
            torch.cuda.empty_cache()
            model, tokenizer = load_model(ckpt_path, device)
            response = generate(model, tokenizer, SAMPLE['personality'], SAMPLE['user_msg'], device)
            
            # 检查是否有退化
            is_degenerate = False
            if len(response) < 10:
                is_degenerate = True
            elif any(pattern in response for pattern in ['\n\n\n', '????', '....', '!!!!']):
                is_degenerate = True
            # 检查重复模式
            words = response.split()
            if len(words) > 5 and len(set(words[-10:])) < 3:
                is_degenerate = True
            
            status = "⚠️ 可能退化" if is_degenerate else "✓ 正常"
            print(f"Status: {status}")
            print(f"Response: {response[:300]}{'...' if len(response)>300 else ''}")
            
            del model, tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[FAILED] {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
