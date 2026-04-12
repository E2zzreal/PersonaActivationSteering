"""思考泄露检测与保守型清理工具。"""

from __future__ import annotations

import re
from dataclasses import dataclass

LEAK_PATTERNS = [
    re.compile(r"\bOkay,?\b", re.IGNORECASE),
    re.compile(r"\bI need to\b", re.IGNORECASE),
    re.compile(r"\bthe user is\b", re.IGNORECASE),
    re.compile(r"\bI should\b", re.IGNORECASE),
    re.compile(r"\blet me think\b", re.IGNORECASE),
    re.compile(r"\bmy response should\b", re.IGNORECASE),
]

THINK_PREFIX_PATTERNS = [
    re.compile(r"^\s*(okay,?|well,?)\b", re.IGNORECASE),
    re.compile(r"\bthe user\b", re.IGNORECASE),
    re.compile(r"\bI need to\b", re.IGNORECASE),
    re.compile(r"\bI should\b", re.IGNORECASE),
    re.compile(r"\blet me think\b", re.IGNORECASE),
    re.compile(r"\bmy response should\b", re.IGNORECASE),
    re.compile(r"\bfirst,\b", re.IGNORECASE),
    re.compile(r"\b首先\b"),
    re.compile(r"\b用户\b"),
    re.compile(r"\b我需要\b"),
]

USER_FACING_START_PATTERNS = [
    re.compile(r"^\s*(hi|hello|hey|oh|that|thanks|thank you|i'm|it sounds|that's)\b", re.IGNORECASE),
    re.compile(r"^\s*(你好|嗨|嘿|谢谢|听起来|那真|这真|我很)"),
]

SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n+")


@dataclass
class LeakProcessResult:
    raw_response: str
    clean_response: str
    leak_detected: bool
    leak_patterns: list[str]
    cleaning_applied: bool
    cleaning_skipped: bool


def detect_patterns(text: str) -> list[str]:
    return [pattern.pattern for pattern in LEAK_PATTERNS if pattern.search(text)]


def is_thinking_sentence(sentence: str) -> bool:
    s = sentence.strip()
    if not s:
        return False
    return any(pattern.search(s) for pattern in THINK_PREFIX_PATTERNS)


def is_user_facing_sentence(sentence: str) -> bool:
    s = sentence.strip()
    if not s:
        return False
    return any(pattern.search(s) for pattern in USER_FACING_START_PATTERNS)


def conservative_clean_response(text: str) -> LeakProcessResult:
    raw = (text or "").strip()
    matched = detect_patterns(raw)
    if not raw or not matched:
        return LeakProcessResult(raw, raw, False, matched, False, False)

    parts = [part.strip() for part in SENTENCE_SPLIT.split(raw) if part.strip()]
    if not parts:
        return LeakProcessResult(raw, raw, True, matched, False, True)

    keep_start = None
    for idx, sentence in enumerate(parts):
        if is_user_facing_sentence(sentence):
            keep_start = idx
            break
        if not is_thinking_sentence(sentence) and not detect_patterns(sentence):
            keep_start = idx
            break

    if keep_start is None:
        return LeakProcessResult(raw, raw, True, matched, False, True)

    if keep_start == 0:
        return LeakProcessResult(raw, raw, True, matched, False, True)

    cleaned = " ".join(parts[keep_start:]).strip()

    if len(cleaned) < 12:
        return LeakProcessResult(raw, raw, True, matched, False, True)

    return LeakProcessResult(raw, cleaned, True, matched, True, False)
