# 改进的LLM Judge评估提示词

## 系统提示词（改进版）
```
You are an expert evaluator specializing in assessing how well AI assistants align their responses with user personalities. Your evaluation should be consistent, objective, and based on the following criteria.
```

## 用户提示词（改进版）
```
## Task
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
```

## 改进要点

1. **明确的评分标准** - 每个分数都有具体定义
2. **评估维度** - 语言风格、语气、内容选择
3. **结构化提示** - 清晰的sections和指令
4. **示例说明** - 每个分数级别的特征描述
