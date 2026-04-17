"""
正确的模型加载方式
从checkpoint加载PersonaSteer模型
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoModel, AutoConfig

def load_persona_steer_model(checkpoint_path: str, base_model_path: str):
    """
    正确加载PersonaSteer模型
    
    Args:
        checkpoint_path: PersonaSteer checkpoint路径
        base_model_path: 骨干模型路径
    
    Returns:
        model: 加载好的模型
        config: 模型配置
    """
    # 1. 加载checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # 2. 获取配置
    config = ckpt.get("config", {})
    
    # 3. 加载骨干模型
    print(f"Loading backbone from {base_model_path}")
    backbone = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    
    # 4. 加载encoder
    print(f"Loading encoder from {base_model_path}")
    encoder = AutoModel.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    # 5. 从backbone获取正确的hidden_size
    backbone_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    layer_dim = backbone_config.hidden_size
    print(f"Detected hidden_size: {layer_dim}")
    
    # 6. 创建PersonaSteerConfig
    from src.models.persona_steer import PersonaSteerConfig
    
    persona_config = PersonaSteerConfig(
        inject_layers=getattr(config, "inject_layers", [10, 11, 12, 13, 14, 15, 16, 17]),
        v_dim=getattr(config, "v_dim", 1024),
        hidden_dim=getattr(config, "hidden_dim", 4096),
        layer_dim=layer_dim,  # 使用实际的hidden_size
        backbone_model_name=base_model_path,
        encoder_model_name=base_model_path,
    )
    
    # 7. 创建模型
    from src.models.persona_steer import PersonaSteerModel
    
    model = PersonaSteerModel(
        config=persona_config,
        backbone=backbone,
        encoder=encoder
    )
    
    # 8. 只加载PersonaSteer权重（不包括backbone）
    model_state_dict = ckpt["model_state_dict"]
    
    # 过滤掉backbone权重（因为backbone已经单独加载）
    persona_weights = {
        k: v for k, v in model_state_dict.items()
        if not k.startswith("backbone.")
    }
    
    # 加载权重
    missing, unexpected = model.load_state_dict(persona_weights, strict=False)
    
    print(f"Loaded PersonaSteer weights")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    
    return model, persona_config


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    checkpoint_path = "checkpoints/stage3/best.pt"
    base_model_path = "/home/kemove/.cache/modelscope/Qwen/Qwen2___5-3B"
    
    model, config = load_persona_steer_model(checkpoint_path, base_model_path)
    
    print(f"\n模型加载成功!")
    print(f"Inject layers: {config.inject_layers}")
    print(f"V dimension: {config.v_dim}")
    print(f"Layer dimension: {config.layer_dim}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")


if __name__ == "__main__":
    main()
