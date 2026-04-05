from __future__ import annotations

import re
from statistics import mean
from typing import Iterable


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def extraction_confidence(text: str, ocr_warnings: int = 0, unreadable_tokens: int = 0) -> float:
    length_score = min(len(text) / 2000.0, 1.0)
    warning_penalty = min(ocr_warnings * 0.08, 0.4)
    unreadable_penalty = min(unreadable_tokens * 0.03, 0.3)
    score = 0.55 + (0.45 * length_score) - warning_penalty - unreadable_penalty
    return clamp(score)


def summary_confidence(summary: str) -> float:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary.strip()) if s.strip()]
    sentence_count = len(sentences)
    count_score = 1.0 if 2 <= sentence_count <= 4 else 0.45
    truncation_artifacts = int(bool(re.search(r"\.\.\.|\s[-,:;]$|\w-$", summary)))
    truncation_score = 1.0 if truncation_artifacts == 0 else 0.35
    grammar_score = 0.9 if all(s[0].isupper() for s in sentences if s) else 0.7
    return clamp((0.45 * count_score) + (0.4 * truncation_score) + (0.15 * grammar_score))


def entities_confidence(entity_scores: Iterable[float], unique_ratio: float) -> float:
    entity_scores = list(entity_scores)
    if not entity_scores:
        return 0.4
    return clamp((0.75 * mean(entity_scores)) + (0.25 * unique_ratio))


def sentiment_confidence(weighted_score: float, agreement: float) -> float:
    return clamp((0.7 * weighted_score) + (0.3 * agreement))


def overall_confidence(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return clamp(mean(values))
