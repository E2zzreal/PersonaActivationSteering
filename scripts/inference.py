#!/usr/bin/env python
"""
PersonaSteer 推理演示脚本
交互式对话演示
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.persona_steer import PersonaSteerModel, PersonaSteerConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Run PersonaSteer inference demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/stage3/best.pt",
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )

    return parser.parse_args()


def load_config(config_path: str | None) -> dict:
    """加载配置文件"""
    if config_path is None:
        # 使用默认配置
        return {
            "model": {
                "inject_layers": [14, 15, 16, 17, 18, 19, 20, 21],
                "v_dim": 1024,
                "hidden_dim": 4096,
                "layer_dim": 2560,
            },
        }

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, config: dict, device: str):
    """加载模型"""
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    device = torch.device(device)

    # 创建模型配置
    model_config = config.get("model", {})
    persona_config = PersonaSteerConfig(
        inject_layers=model_config.get("inject_layers", [14, 15, 16, 17, 18, 19, 20, 21]),
        v_dim=model_config.get("v_dim", 1024),
        hidden_dim=model_config.get("hidden_dim", 4096),
        layer_dim=model_config.get("layer_dim", 2560),
        gate_hidden_dim=model_config.get("gate_hidden_dim", 256),
    )
    model = PersonaSteerModel(config=persona_config)

    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # 加载可训练参数
    model_state = model.state_dict()
    for key in state_dict:
        if key in model_state:
            model_state[key] = state_dict[key]
    model.load_state_dict(model_state)

    model.to(device)
    model.eval()

    logger.info("Model loaded successfully")
    return model


def interactive_chat(
    model: PersonaSteerModel,
    tokenizer,
    device: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
):
    """
    交互式对话

    Args:
        model: PersonaSteer 模型
        tokenizer: 分词器
        device: 计算设备
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度
    """
    print("=" * 60)
    print("PersonaSteer 交互式对话演示")
    print("=" * 60)
    print("\n输入用户画像，然后开始对话")
    print("输入 'quit' 或 'exit' 退出对话")
    print("输入 'reset' 重置对话历史")
    print("-" * 60)

    # 初始化干预向量
    v_t = torch.zeros(1, model.config.v_dim).to(device)

    while True:
        try:
            # 获取用户画像
            if v_t.abs().sum() == 0:  # 首次对话
                profile = input("\n用户画像: ").strip()
                if profile.lower() in ["quit", "exit"]:
                    break
                if not profile:
                    print("请输入有效的用户画像")
                    continue
                print(f"用户画像已设置: {profile}")
                print("-" * 40)

            # 对话输入
            user_input = input("\n你: ").strip()

            if user_input.lower() in ["quit", "exit"]:
                print("再见!")
                break

            if user_input.lower() == "reset":
                v_t = torch.zeros(1, model.config.v_dim).to(device)
                print("对话已重置")
                continue

            if not user_input:
                continue

            # 生成回复
            # Tokenize 输入
            input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)

            # 使用模型生成
            with torch.no_grad():
                outputs = model.backbone.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else 1.0,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # 解码回复
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 去除输入部分，只保留生成的回复
            if response.startswith(user_input):
                response = response[len(user_input):].strip()

            print(f"\nAI: {response}")

        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            logger.exception("Error during inference")


def main():
    """主函数"""
    args = parse_args()

    # 加载配置
    config = load_config(args.config)

    # 加载模型
    model = load_model(args.checkpoint, config, args.device)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-4B",
        trust_remote_code=True,
    )

    # 开始交互式对话
    interactive_chat(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
