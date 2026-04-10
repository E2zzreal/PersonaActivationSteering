"""
属性提取器 - 从 ALOE 数据集提取人格属性标注

用于从用户画像和对话中提取人格属性，包括:
- age: 年龄 (float)
- gender: 性别 (int, 0=未知, 1=男性, 2=女性)
- introversion: 内向程度 (float, 0-1)
- openness: 开放程度 (float, 0-1)
- conscientiousness: 尽责性 (float, 0-1)
- agreeableness: 宜人性 (float, 0-1)
- neuroticism: 神经质 (float, 0-1)
"""

import re
from typing import Any


class AttributeExtractor:
    """从 ALOE 数据集提取人格属性标注"""

    # 大五人格属性名映射
    PERSONALITY_ATTRS = [
        "introversion",
        "openness",
        "conscientiousness",
        "agreeableness",
        "neuroticism",
    ]

    # 性别关键词映射
    GENDER_KEYWORDS = {
        "男": 1,
        "女性": 2,
        "先生": 1,
        "女士": 2,
        "他": 1,
        "她": 2,
        "boy": 1,
        "girl": 2,
        "man": 1,
        "woman": 2,
    }

    # 年龄正则模式
    AGE_PATTERN = re.compile(r"(\d{1,3})[岁]")
    AGE_KEYWORDS = ["岁", "年龄", "age"]

    def __init__(self):
        """初始化属性提取器"""
        pass

    def extract_attributes(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        从 ALOE 样本中提取人格属性

        Args:
            sample: ALOE 数据样本，格式如下:
                {
                    "user_id": str,
                    "profile": str,  # 用户画像描述
                    "personality": str,  # 人格描述
                    "conversations": [...],
                }

        Returns:
            提取的属性字典，包含:
                - age: float (归一化到 0-1)
                - gender: int (0=未知, 1=男, 2=女)
                - introversion: float (0-1)
                - openness: float (0-1)
                - conscientiousness: float (0-1)
                - agreeableness: float (0-1)
                - neuroticism: float (0-1)
        """
        profile = sample.get("profile", "")
        personality = sample.get("personality", "")
        text = f"{profile} {personality}"

        attributes = {
            "age": self._extract_age(text),
            "gender": self._extract_gender(text),
        }

        # 提取大五人格属性
        for attr in self.PERSONALITY_ATTRS:
            attributes[attr] = self._extract_personality_attr(personality, attr)

        return attributes

    def _extract_age(self, text: str) -> float:
        """
        从文本中提取年龄并归一化

        假设年龄范围 10-90，归一化到 0-1
        """
        match = self.AGE_PATTERN.search(text)
        if match:
            age = int(match.group(1))
            # 归一化到 0-1 (假设范围 10-90)
            return max(0.0, min(1.0, (age - 10) / 80))
        # 默认值: 0.5 (中年)
        return 0.5

    def _extract_gender(self, text: str) -> int:
        """
        从文本中提取性别

        Returns:
            0: 未知, 1: 男性, 2: 女性
        """
        text_lower = text.lower()

        # 优先检测女性特征
        female_keywords = ["她", "女性", "女士", "girl", "woman", "她的"]
        male_keywords = ["他", "男性", "先生", "boy", "man", "他的"]

        female_count = sum(1 for kw in female_keywords if kw in text_lower)
        male_count = sum(1 for kw in male_keywords if kw in text_lower)

        if female_count > male_count:
            return 2
        elif male_count > female_count:
            return 1
        return 0

    def _extract_personality_attr(self, text: str, attr: str) -> float:
        """
        从人格描述中提取特定属性值

        Args:
            text: 人格描述文本
            attr: 属性名 (introversion, openness 等)

        Returns:
            float: 属性值 (0-1)
        """
        text_lower = text.lower()

        # 属性关键词映射 (根据描述推断)
        attr_keywords = {
            "introversion": {
                "high": ["内向", "安静", "保守", "独处", "害羞", "introvert", "quiet", "shy"],
                "low": ["外向", "活泼", "开朗", "健谈", "extrovert", "outgoing", "talkative"],
            },
            "openness": {
                "high": ["开放", "好奇", "创意", "新潮", "open", "creative", "curious"],
                "low": ["传统", "保守", "务实", "传统", "traditional", "conventional"],
            },
            "conscientiousness": {
                "high": ["认真", "负责", "有条理", "勤奋", "conscientious", "organized", "diligent"],
                "low": ["随意", "马虎", "拖延", "随性", "careless", "lazy", "messy"],
            },
            "agreeableness": {
                "high": ["友好", "善良", "温和", "合作", "agreeable", "kind", "cooperative"],
                "low": ["冷漠", "挑剔", "强硬", "competitive", "cold", "critical"],
            },
            "neuroticism": {
                "high": ["敏感", "焦虑", "情绪化", "紧张", "anxious", "emotional", "nervous"],
                "low": ["稳定", "冷静", "沉着", "平和", "stable", "calm", "relaxed"],
            },
        }

        keywords = attr_keywords.get(attr, {"high": [], "low": []})

        # 计算高值和低值关键词匹配数
        high_count = sum(1 for kw in keywords["high"] if kw in text_lower)
        low_count = sum(1 for kw in keywords["low"] if kw in text_lower)

        # 归一化到 0-1
        total = high_count + low_count
        if total > 0:
            return high_count / total
        # 默认值 0.5
        return 0.5

    def extract_batch(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        批量提取属性

        Args:
            samples: ALOE 样本列表

        Returns:
            属性列表
        """
        return [self.extract_attributes(sample) for sample in samples]
