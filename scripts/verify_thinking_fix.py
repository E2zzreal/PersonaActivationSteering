#!/usr/bin/env python
"""
快速验证 thinking 模式修复是否生效
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer
from src.data.aloe_dataset import ALOEDataset
import tempfile
import json
import os


def main():
    print("Loading Qwen3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B', trust_remote_code=True)

    # Create test sample
    sample = {
        "user_id": "verify_test",
        "profile": "Test user",
        "personality": "Friendly",
        "conversations": [
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hello! How can I help you?"}
        ]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write(json.dumps(sample) + '\n')
        temp_path = f.name

    try:
        dataset = ALOEDataset(temp_path, tokenizer, max_turns=1)
        result = dataset[0]

        text = tokenizer.decode(result['turns'][0]['input_ids'])

        print("\n=== Tokenized Training Sample ===")
        print(text)
        print("\n=== Verification ===")

        # Check for thinking markers
        has_think = '<tool_call>' in text
        has_thinking_content = 'Okay,' in text or 'I need to' in text

        print(f"Contains <tool_call> marker: {has_think}")
        print(f"Contains pseudo-thinking: {has_thinking_content}")

        if not has_thinking_content:
            print("\n✅ PASS: No pseudo-thinking content detected")
            return 0
        else:
            print("\n❌ FAIL: Pseudo-thinking content still present")
            return 1

    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    sys.exit(main())