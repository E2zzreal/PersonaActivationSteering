#!/usr/bin/env python3
"""
快速验证 Chat Template 修复效果
测试训练数据格式是否正确
"""

import sys
sys.path.insert(0, '/home/kemove/Desktop/Projects/3-PersonaSteer_V2')

from transformers import AutoTokenizer
from src.data.aloe_dataset import ALOEDataset

def test_chat_template():
    print("=" * 80)
    print("Chat Template 修复验证")
    print("=" * 80)
    print()

    # 加载tokenizer
    print("[1/3] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/kemove/.cache/modelscope/Qwen/Qwen2___5-3B",
        trust_remote_code=True
    )
    print(f"✓ Tokenizer 加载成功")
    print()

    # 加载数据集
    print("[2/3] 加载数据集...")
    dataset = ALOEDataset(
        data_path="data/split/val.jsonl",
        tokenizer=tokenizer,
        max_turns=2
    )
    print(f"✓ 数据集加载成功，共 {len(dataset)} 个样本")
    print()

    # 测试第一个样本
    print("[3/3] 测试第一个样本...")
    sample = dataset[0]

    print(f"User ID: {sample['user_id']}")
    print(f"Profile: {sample['profile'][:100]}...")
    print(f"Turns: {len(sample['turns'])}")
    print()

    if len(sample['turns']) > 0:
        turn = sample['turns'][0]
        input_ids = turn['input_ids']
        labels = turn['labels']

        # 解码查看格式
        decoded = tokenizer.decode(input_ids)

        print("-" * 80)
        print("完整序列 (decoded):")
        print("-" * 80)
        print(decoded)
        print()

        # 检查是否包含chat template标记
        has_im_start = '<|im_start|>' in decoded
        has_im_end = '<|im_end|>' in decoded
        has_user = 'user' in decoded
        has_assistant = 'assistant' in decoded

        print("-" * 80)
        print("Chat Template 检查:")
        print("-" * 80)
        print(f"✓ 包含 <|im_start|>: {has_im_start}")
        print(f"✓ 包含 <|im_end|>: {has_im_end}")
        print(f"✓ 包含 user 角色: {has_user}")
        print(f"✓ 包含 assistant 角色: {has_assistant}")
        print()

        # 检查labels构建
        num_masked = sum(1 for x in labels if x == -100)
        num_valid = len(labels) - num_masked

        print("-" * 80)
        print("Labels 构建:")
        print("-" * 80)
        print(f"总长度: {len(labels)}")
        print(f"Masked (-100): {num_masked} ({num_masked/len(labels)*100:.1f}%)")
        print(f"Valid tokens: {num_valid} ({num_valid/len(labels)*100:.1f}%)")
        print()

        # 最终判断
        print("=" * 80)
        if has_im_start and has_im_end and has_user and has_assistant:
            print("✅ 修复成功！Chat template 已正确应用")
        else:
            print("❌ 修复失败！Chat template 未正确应用")
        print("=" * 80)

        return has_im_start and has_im_end and has_user and has_assistant
    else:
        print("❌ 样本没有对话轮次")
        return False

if __name__ == "__main__":
    try:
        success = test_chat_template()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
