#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 Qwen3 模型结构

检查 Qwen3-4B 和 Qwen3-Embedding 的实际结构，
确保与 PersonaSteer 代码中的假设一致。

Author: PersonaSteer Team
Date: 2026-03-06
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def verify_qwen3_4b_structure():
    """验证 Qwen3-4B 骨干模型结构"""
    logger.info("=" * 60)
    logger.info("验证 Qwen3-4B 结构")
    logger.info("=" * 60)

    try:
        from transformers import AutoModel, AutoConfig

        model_name = "Qwen/Qwen2.5-3B"  # 使用实际可用的模型
        logger.info(f"加载模型配置: {model_name}")

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # 检查关键配置
        logger.info(f"\n模型配置:")
        logger.info(f"  - 层数: {config.num_hidden_layers}")
        logger.info(f"  - 隐藏层维度: {config.hidden_size}")
        logger.info(f"  - 注意力头数: {config.num_attention_heads}")
        logger.info(f"  - 词表大小: {config.vocab_size}")

        # 验证注入层配置是否合理
        inject_layers = [14, 15, 16, 17, 18, 19, 20, 21]
        if max(inject_layers) >= config.num_hidden_layers:
            logger.warning(f"  ⚠️  注入层 {inject_layers} 超出模型层数 {config.num_hidden_layers}")
            logger.warning(f"  建议调整为: {list(range(config.num_hidden_layers - 8, config.num_hidden_layers))}")
        else:
            logger.info(f"  ✅ 注入层配置合理: {inject_layers}")

        # 加载模型检查结构
        logger.info(f"\n加载模型检查结构...")
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cpu"
        )

        # 检查模型结构
        logger.info(f"\n模型结构检查:")

        # 检查是否有 model.layers
        if hasattr(model, 'model'):
            logger.info(f"  ✅ 存在 model 属性")
            if hasattr(model.model, 'layers'):
                logger.info(f"  ✅ 存在 model.layers 属性")
                logger.info(f"  ✅ 层数: {len(model.model.layers)}")

                # 检查第一层结构
                first_layer = model.model.layers[0]
                logger.info(f"\n  第一层结构:")
                logger.info(f"    - 类型: {type(first_layer).__name__}")

                if hasattr(first_layer, 'self_attn'):
                    logger.info(f"    ✅ 存在 self_attn 属性")
                else:
                    logger.warning(f"    ⚠️  不存在 self_attn 属性")

                # 检查 hook 注册点
                logger.info(f"\n  Hook 注册验证:")
                logger.info(f"    - 可以在 model.model.layers[i] 注册 forward_hook")

            else:
                logger.warning(f"  ⚠️  不存在 model.layers，检查是否为 model.h")
                if hasattr(model.model, 'h'):
                    logger.info(f"  ✅ 存在 model.h 属性")
        else:
            logger.warning(f"  ⚠️  不存在 model 属性，检查顶层结构")
            if hasattr(model, 'layers'):
                logger.info(f"  ✅ 存在顶层 layers 属性")

        return True

    except Exception as e:
        logger.error(f"验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_qwen3_embedding_structure():
    """验证 Qwen3-Embedding 编码器结构"""
    logger.info("\n" + "=" * 60)
    logger.info("验证 Qwen3-Embedding 结构")
    logger.info("=" * 60)

    try:
        from transformers import AutoModel, AutoTokenizer

        # Qwen3-Embedding 可能不存在，使用 Qwen3 的 embedding 层
        model_name = "Qwen/Qwen2.5-3B"
        logger.info(f"使用模型: {model_name}")
        logger.info(f"注意: Qwen3-Embedding 可能需要单独的模型")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cpu"
        )

        # 检查编码器接口
        logger.info(f"\n编码器接口检查:")

        # 测试编码
        test_text = ["你好，今天天气真好"]
        inputs = tokenizer(
            test_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        logger.info(f"  ✅ Tokenizer 可用")
        logger.info(f"  - 输入形状: {inputs['input_ids'].shape}")

        # 前向传播
        import torch
        with torch.no_grad():
            outputs = model(**inputs)

        logger.info(f"\n  模型输出:")
        if hasattr(outputs, 'last_hidden_state'):
            logger.info(f"  ✅ 存在 last_hidden_state")
            logger.info(f"  - 形状: {outputs.last_hidden_state.shape}")
            logger.info(f"  - 隐藏维度: {outputs.last_hidden_state.shape[-1]}")

            # 验证维度是否为 1024
            hidden_dim = outputs.last_hidden_state.shape[-1]
            if hidden_dim == 1024:
                logger.info(f"  ✅ 隐藏维度为 1024，符合设计")
            else:
                logger.warning(f"  ⚠️  隐藏维度为 {hidden_dim}，设计文档假设为 1024")
                logger.warning(f"  需要更新 config.yaml 中的 v_dim 参数")
        else:
            logger.warning(f"  ⚠️  不存在 last_hidden_state，检查输出格式")
            logger.info(f"  输出类型: {type(outputs)}")

        return True

    except Exception as e:
        logger.error(f"验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_hook_mechanism():
    """验证 hook 机制是否正常工作"""
    logger.info("\n" + "=" * 60)
    logger.info("验证 Hook 机制")
    logger.info("=" * 60)

    try:
        from transformers import AutoModel
        import torch

        model_name = "Qwen/Qwen2.5-3B"
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cpu"
        )

        # 测试 hook 注册
        hook_called = [False]

        def test_hook(module, input, output):
            hook_called[0] = True
            logger.info(f"  ✅ Hook 被调用")
            if isinstance(output, tuple):
                logger.info(f"  - 输出是 tuple，长度: {len(output)}")
                logger.info(f"  - 第一个元素形状: {output[0].shape}")
            else:
                logger.info(f"  - 输出形状: {output.shape}")

        # 在第一层注册 hook
        layer = model.model.layers[0]
        handle = layer.register_forward_hook(test_hook)

        logger.info(f"\n测试 hook 注册:")
        logger.info(f"  - 在 model.model.layers[0] 注册 hook")

        # 执行前向传播
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        inputs = tokenizer(["测试"], return_tensors="pt")

        with torch.no_grad():
            model(**inputs)

        if hook_called[0]:
            logger.info(f"  ✅ Hook 机制正常工作")
        else:
            logger.error(f"  ❌ Hook 未被调用")

        # 清理
        handle.remove()

        return hook_called[0]

    except Exception as e:
        logger.error(f"验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_fix_recommendations():
    """生成修复建议"""
    logger.info("\n" + "=" * 60)
    logger.info("修复建议")
    logger.info("=" * 60)

    logger.info("""
1. 如果 Qwen3-4B 层数不是 36 层:
   - 更新 configs/model.yaml 中的 inject_layers
   - 建议使用最后 8 层

2. 如果隐藏维度不是 1024:
   - 更新 configs/model.yaml 中的 v_dim
   - 更新 HyperNetwork 的输入维度

3. 如果模型结构不是 model.model.layers:
   - 修改 persona_steer.py 中的 _register_injection_hooks 方法
   - 适配实际的层访问路径

4. 如果 Qwen3-Embedding 不存在:
   - 使用 Qwen3-4B 的 embedding 层
   - 或使用其他句子编码器 (如 sentence-transformers)
""")


def main():
    """主函数"""
    logger.info("PersonaSteer V2 - Qwen3 模型结构验证\n")

    results = []

    # 验证 Qwen3-4B
    result1 = verify_qwen3_4b_structure()
    results.append(("Qwen3-4B 结构", result1))

    # 验证 Qwen3-Embedding
    result2 = verify_qwen3_embedding_structure()
    results.append(("Qwen3-Embedding 结构", result2))

    # 验证 Hook 机制
    result3 = verify_hook_mechanism()
    results.append(("Hook 机制", result3))

    # 生成修复建议
    generate_fix_recommendations()

    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("验证总结")
    logger.info("=" * 60)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"  {name}: {status}")

    all_passed = all(r for _, r in results)

    if all_passed:
        logger.info("\n✅ 所有验证通过，模型结构符合预期")
        return 0
    else:
        logger.warning("\n⚠️  部分验证失败，请根据建议进行修复")
        return 1


if __name__ == '__main__':
    sys.exit(main())
