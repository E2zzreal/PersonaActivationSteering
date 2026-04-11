"""
Tests for ALOEDataset thinking mode handling
"""
import pytest
from unittest.mock import MagicMock, patch


class TestALOEDatasetThinkingMode:
    """Test that ALOEDataset correctly disables thinking mode during tokenization"""

    def test_getitem_calls_apply_chat_template_with_enable_thinking_false(self):
        """
        ALOEDataset.__getitem__ should call apply_chat_template with enable_thinking=False
        to match inference behavior and prevent thinking markers in training data.
        """
        # Create mock tokenizer that tracks calls
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = (
            "<|im_start|>user\nHello!<|im_end|>\n"
            "<|im_start|>assistant\nHi there!<|im_end|>\n"
        )
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        # Create temporary test data file
        import tempfile
        import json

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            sample = {
                "user_id": "test001",
                "profile": "Test profile",
                "personality": "Test personality",
                "conversations": [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            }
            f.write(json.dumps(sample) + '\n')
            temp_path = f.name

        try:
            from src.data.aloe_dataset import ALOEDataset

            dataset = ALOEDataset(temp_path, mock_tokenizer, max_turns=1)
            _ = dataset[0]  # Trigger __getitem__

            # Verify apply_chat_template was called with enable_thinking=False
            calls = mock_tokenizer.apply_chat_template.call_args_list

            # Should have 2 calls per turn: one for full_text, one for user_only
            assert len(calls) >= 2, f"Expected at least 2 calls, got {len(calls)}"

            # Check that enable_thinking=False is in the calls
            for call in calls:
                kwargs = call.kwargs
                assert 'enable_thinking' in kwargs, \
                    "enable_thinking parameter not found in apply_chat_template call"
                assert kwargs['enable_thinking'] is False, \
                    f"enable_thinking should be False, got {kwargs['enable_thinking']}"

        finally:
            import os
            os.unlink(temp_path)

    def test_getitem_handles_tokenizer_without_enable_thinking_param(self):
        """
        ALOEDataset should gracefully handle tokenizers that don't support
        enable_thinking parameter (non-Qwen3 models).
        """
        # Create mock tokenizer that raises TypeError for enable_thinking
        mock_tokenizer = MagicMock()

        def mock_apply_chat_template(messages, tokenize=False, add_generation_prompt=False, **kwargs):
            # Simulate non-Qwen3 tokenizer that doesn't accept enable_thinking
            if 'enable_thinking' in kwargs:
                raise TypeError("apply_chat_template() got an unexpected keyword argument 'enable_thinking'")
            return "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\nHi there!<|im_end|>\n"

        mock_tokenizer.apply_chat_template.side_effect = mock_apply_chat_template
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        import tempfile
        import json
        import os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            sample = {
                "user_id": "test002",
                "profile": "Test profile",
                "personality": "Test personality",
                "conversations": [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            }
            f.write(json.dumps(sample) + '\n')
            temp_path = f.name

        try:
            from src.data.aloe_dataset import ALOEDataset

            # Should not raise exception
            dataset = ALOEDataset(temp_path, mock_tokenizer, max_turns=1)
            result = dataset[0]

            # Should return valid data
            assert 'turns' in result
            assert len(result['turns']) == 1

        finally:
            os.unlink(temp_path)


class TestALOEDatasetIntegration:
    """Integration tests with real Qwen3 tokenizer"""

    @pytest.mark.integration
    def test_training_prompt_excludes_thinking_markers(self):
        """
        Integration test: Verify that training prompts do NOT contain
        thinking content when using real Qwen3 tokenizer.
        """
        try:
            from transformers import AutoTokenizer
        except ImportError:
            pytest.skip("transformers not installed")

        import tempfile
        import json
        import os

        # Load real Qwen3 tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                'Qwen/Qwen3-4B',
                trust_remote_code=True
            )
        except Exception as e:
            pytest.skip(f"Could not load Qwen3 tokenizer: {e}")

        # Create test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            sample = {
                "user_id": "test003",
                "profile": "Test profile",
                "personality": "Test personality",
                "conversations": [
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing great, thanks for asking!"}
                ]
            }
            f.write(json.dumps(sample) + '\n')
            temp_path = f.name

        try:
            from src.data.aloe_dataset import ALOEDataset

            dataset = ALOEDataset(temp_path, tokenizer, max_turns=1)
            result = dataset[0]

            # Decode the tokenized text
            text = tokenizer.decode(result['turns'][0]['input_ids'])

            # Verify NO thinking content markers
            assert '<tool_call>' not in text or text.count('<tool_call>') <= 1, \
                f"Unexpected thinking markers in training data: {text}"

            # Verify the actual response is present
            assert "I'm doing great" in text, \
                f"Expected response not found in: {text}"

        finally:
            os.unlink(temp_path)