"""
LLM Judge 评估器
使用 LLM 作为评判者计算 AL(K)_AVG, N-IR, N-R² 指标
"""

import json
import logging
import os
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# 尝试导入 OpenAI，如果不可用则使用 mock
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LLMJudgeEvaluator:
    """
    LLM Judge 评估器

    使用 LLM 作为评判者评估模型输出与用户人格的对齐程度。
    计算以下指标:
    - AL(K)_AVG: 平均人格对齐分数
    - N-IR: 改进率 (与 baseline 比较)
    - N-R²: 相关性 (与人工标注比较)

    Args:
        judge_model: 评判用 LLM 模型名称
        api_key: OpenAI API 密钥
    """

    def __init__(
        self,
        judge_model: str = "Claude-Sonnet-4.6",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.judge_model = judge_model
        self.client = None

        if OPENAI_AVAILABLE:
            api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("BLSC_API_KEY")
            # 强制使用正确的BASE_URL: https://llmapi.blsc.cn
            # 如果传入base_url参数，使用传入的值；否则使用正确的默认值
            if base_url is None:
                base_url = "https://llmapi.blsc.cn"
            # 覆盖环境变量，确保不会被错误的环境变量干扰
            if api_key:
                client_kwargs = {"api_key": api_key, "base_url": base_url}
                self.client = OpenAI(**client_kwargs)
                logger.info(f"LLM Judge initialized with model: {judge_model}")
                logger.info(f"Using base URL: {base_url}")
            else:
                logger.warning("OpenAI API key not found. Using mock evaluation.")
        else:
            logger.warning("OpenAI not installed. Using mock evaluation.")

    def _generate_conversation(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        sample: dict,
        max_new_tokens: int = 128,
    ) -> list[dict]:
        """
        使用模型生成对话

        Args:
            model: PersonaSteer 模型
            tokenizer: 分词器
            sample: 测试样本
            max_new_tokens: 最大生成 token 数

        Returns:
            list[dict]: 对话历史
        """
        from src.models.persona_steer import PersonaSteerModel

        # 找到实际在 GPU 上的参数来确定 device（encoder 可能在 CPU 上）
        device = None
        for p in model.parameters():
            if p.device.type == 'cuda':
                device = p.device
                break
        if device is None:
            device = next(model.parameters()).device

        # 初始化干预向量
        v_t = torch.zeros(1, model.v_dim).to(device)

        conversation = []
        user_turn_count = 0
        max_user_turns = 4
        for turn in sample.get("conversations", []):
            if user_turn_count >= max_user_turns:
                break
            if turn.get("role") == "user":
                user_text = turn.get("content", "")

                # 使用模型生成回复
                if isinstance(model, PersonaSteerModel):
                    # 使用 chat template 构建输入
                    messages = [{"role": "user", "content": user_text}]
                    # Qwen3 默认启用 thinking 模式，需显式禁用以避免乱码输出
                    try:
                        prompt = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    except TypeError:
                        # 非 Qwen3 tokenizer 不支持 enable_thinking 参数
                        prompt = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                    prompt_len = input_ids.shape[1]

                    # 生成（使用模型的generate方法，包含注入）
                    with torch.no_grad():
                        outputs, v_t = model.generate(
                            input_ids=input_ids,
                            v_prev=v_t,
                            personality_texts=[sample["personality"]],
                            user_query_texts=[user_text],
                            max_new_tokens=max_new_tokens,
                            temperature=0.7,
                            top_p=0.9,
                        )

                    # 只解码新生成的 token（去除 prompt 部分）
                    new_tokens = outputs[0][prompt_len:]
                    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    
                    # 调试输出：打印生成内容和注入向量信息
                    if len(conversation) == 0:  # 只打印第一轮
                        logger.info(f"User text: {user_text[:100]}...")
                        logger.info(f"Generated response: {response[:200]}...")
                        logger.info(f"Intervention vector (v_t) norm: {v_t.norm().item():.4f}")
                        if hasattr(model, 'injection') and hasattr(model.injection, 'current_gate_values'):
                            gate_values = model.injection.current_gate_values
                            if gate_values is not None:
                                logger.info(f"Gate values: mean={gate_values.mean().item():.4f}, std={gate_values.std().item():.4f}")
                else:
                    response = "[Generated response]"

                conversation.append({"role": "user", "content": user_text})
                conversation.append({"role": "assistant", "content": response})
                user_turn_count += 1

        return conversation

    def compute_al_k_avg(
        self,
        conversation: list[dict],
        profile: str,
        personality: str,
    ) -> float:
        """
        计算 AL(K) 对齐分数

        Args:
            conversation: 对话历史
            profile: 用户画像
            personality: 人格描述

        Returns:
            float: 对齐分数 (1-5)
        """
        if self.client is None:
            # Mock 评估
            return np.random.uniform(3.0, 5.0)

        # 构建 prompt
        prompt = self._build_alignment_prompt(conversation, profile, personality)

        try:
            # GPT-5模型只支持temperature=1.0
            temperature = 1.0 if "GPT-5" in self.judge_model else 0.0
            
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator specializing in assessing how well AI assistants align their responses with user personalities. Your evaluation should be consistent, objective, and based on the scoring rubric provided."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=50,
            )

            # 解析分数
            content = response.choices[0].message.content
            score = self._parse_score(content)
            return score

        except Exception as e:
            logger.warning(f"Failed to compute alignment score: {e}")
            return 3.0  # 默认分数

    def _build_alignment_prompt(
        self,
        conversation: list[dict],
        profile: str,
        personality: str,
    ) -> str:
        """构建评估 prompt"""
        dialogue = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in conversation
        ])

        prompt = f"""## Task
Evaluate how well the assistant's responses align with the user's personality traits throughout the dialogue.

## User Information
**Profile**: {profile}

**Personality Traits**: {personality}

## Dialogue
{dialogue}

## Scoring Rubric (1-5 Scale)

**5 - Excellent Alignment**
- Assistant's responses consistently reflect and adapt to the user's personality traits
- Language style, tone, and content choices match the user's profile
- Responses demonstrate deep understanding of user's characteristics

**4 - Good Alignment**
- Most responses show clear alignment with user's personality
- Minor inconsistencies in style or tone
- Good adaptation to user characteristics overall

**3 - Moderate Alignment**
- Some evidence of personality alignment
- Mixed consistency in reflecting user traits
- Responses are functional but lack personality-specific adaptation

**2 - Poor Alignment**
- Limited alignment with user's personality
- Generic responses that could apply to anyone
- Minimal consideration of user characteristics

**1 - No Alignment**
- Responses contradict or ignore user's personality traits
- Completely generic or mismatched communication style
- No evidence of personality-aware response generation

## Instructions
1. Read the user profile and personality traits carefully
2. Review the entire dialogue
3. Assess each assistant response against the user's personality
4. Provide a single score (1-5) based on the rubric above

## Output Format
Output ONLY a single number (1, 2, 3, 4, or 5). No explanations or additional text.
"""
        return prompt

    def _parse_score(self, content: str) -> float:
        """从 LLM 输出中解析分数"""
        try:
            # 尝试提取数字
            content = content.strip()
            # 查找第一个数字
            for char in content:
                if char.isdigit():
                    score = int(char)
                    return min(max(score, 1), 5)  # 限制在 1-5
            return 3.0  # 默认
        except Exception:
            return 3.0

    def compute_n_ir(
        self,
        scores: list[float],
        baseline_scores: list[float],
    ) -> float:
        """
        计算 N-IR (改进率)

        N-IR = (baseline - current) / baseline

        Args:
            scores: 当前模型分数
            baseline_scores: baseline 模型分数

        Returns:
            float: 改进率
        """
        if not baseline_scores:
            return 0.0

        current_mean = np.mean(scores)
        baseline_mean = np.mean(baseline_scores)

        n_ir = (current_mean - baseline_mean) / baseline_mean
        return float(n_ir)

    def compute_n_r2(
        self,
        scores: list[float],
        ground_truth: list[float],
    ) -> float:
        """
        计算 N-R² (相关性)

        使用皮尔逊相关系数

        Args:
            scores: 模型预测分数
            ground_truth: 人工标注分数

        Returns:
            float: 相关系数
        """
        if len(scores) != len(ground_truth) or len(scores) < 2:
            return 0.0

        correlation = np.corrcoef(scores, ground_truth)[0, 1]
        return float(correlation)

    def evaluate_alignment(
        self,
        model: torch.nn.Module,
        test_samples: list[dict],
        tokenizer: Any,
        baseline_scores: list[float] | None = None,
    ) -> dict[str, float]:
        """
        批量评估对齐分数

        Args:
            model: PersonaSteer 模型
            test_samples: 测试样本列表
            tokenizer: 分词器
            baseline_scores: baseline 分数 (可选)

        Returns:
            dict: 评估结果
        """
        scores = []

        for sample in tqdm(test_samples, desc="Evaluating alignment"):
            # 生成对话
            conversation = self._generate_conversation(model, tokenizer, sample)

            # 计算对齐分数
            profile = sample.get("profile", "")
            personality = sample.get("personality", "")

            score = self.compute_al_k_avg(conversation, profile, personality)
            scores.append(score)

        results = {
            "al_k_avg": float(np.mean(scores)),
            "al_k_std": float(np.std(scores)),
            "al_k_min": float(np.min(scores)),
            "al_k_max": float(np.max(scores)),
            "num_samples": len(scores),
        }

        # 计算改进率
        if baseline_scores:
            results["n_ir"] = self.compute_n_ir(scores, baseline_scores)

        logger.info(
            f"Alignment evaluation: AL(K)_AVG={results['al_k_avg']:.3f}, "
            f"N-IR={results.get('n_ir', 'N/A')}"
        )

        return results


def load_baseline_scores(path: str) -> list[float]:
    """从文件加载 baseline 分数"""
    if not path:
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("scores", [])
    except Exception as e:
        logger.warning(f"Failed to load baseline scores: {e}")
        return []
