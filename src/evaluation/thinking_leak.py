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

SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")


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


def conservative_clean_response(text: str) -> LeakProcessResult:
    raw = (text or "").strip()
    matched = detect_patterns(raw)
    if not raw or not matched:
        return LeakProcessResult(raw, raw, False, matched, False, False)

    parts = SENTENCE_SPLIT.split(raw)
    if not parts:
        return LeakProcessResult(raw, raw, True, matched, False, True)

    max_prefix_sentences = min(3, len(parts))
    prefix_end = 0
    for idx in range(max_prefix_sentences):
        sentence = parts[idx].strip()
        if detect_patterns(sentence):
            prefix_end = idx + 1
        else:
            break

    if prefix_end == 0:
        return LeakProcessResult(raw, raw, True, matched, False, True)

    cleaned = " ".join(part.strip() for part in parts[prefix_end:] if part.strip()).strip()

    if len(cleaned) < 12:
        return LeakProcessResult(raw, raw, True, matched, False, True)

    return LeakProcessResult(raw, cleaned, True, matched, True, False)
