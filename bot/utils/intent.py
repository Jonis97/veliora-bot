"""
Output intent routing: same source → different learning outputs (one visual template).

Teacher / editor / designer behavior is enforced in the AI layer; this module only classifies intent.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Final


class OutputIntent(str, Enum):
    """What the user wants from the current source (routing only — one HTML template)."""

    CARD = "card"
    SPEAKING = "speaking"
    VOCABULARY = "vocabulary"
    TEST = "test"
    SUMMARY = "summary"


class UnclearIntentError(Exception):
    """Follow-up text is too ambiguous — ask one short clarifying question."""

    def __init__(
        self,
        message: str = (
            "Уточни, будь ласка: що саме зробити з цим матеріалом? Наприклад: «зроби картку»."
        ),
    ) -> None:
        self.user_message = message
        super().__init__(message)


# Phrases that imply a follow-up (used by active_source.followup_intent) — keep in sync when adding intents
FOLLOWUP_EXTRA_KEYWORDS: Final[tuple[str, ...]] = (
    "extract vocabulary",
    "key ideas",
    "main points",
    "make a test",
    "mini quiz",
    "now speaking",
    "speaking task",
    "summarize",
    "summary",
    "recap",
)

# Scoring: (intent, patterns, weight per match)
_INTENT_RULES: Final[
    tuple[tuple[OutputIntent, tuple[str, ...], float], ...]
] = (
    (
        OutputIntent.SUMMARY,
        (
            "summarize",
            "summary",
            "key ideas",
            "main points",
            "tldr",
            "recap",
            "core ideas",
            "gist",
            "outline the",
        ),
        3.0,
    ),
    (
        OutputIntent.VOCABULARY,
        (
            "extract vocabulary",
            "vocabulary only",
            "vocabulary list",
            "vocabulary",
            "vocab",
            "words and phrases",
            "lexical",
            "terms from",
            "key terms",
        ),
        3.0,
    ),
    (
        OutputIntent.TEST,
        (
            "make a test",
            "mini test",
            "practice test",
            "mock test",
            "quiz",
            "exam-style",
            "check yourself",
            "test on",
            "questions on",
        ),
        3.0,
    ),
    (
        OutputIntent.SPEAKING,
        (
            "now speaking",
            "speaking practice",
            "speaking task",
            "speaking exercise",
            "speak about",
            "oral practice",
            "say aloud",
            "pronunciation",
            "conversation practice",
            "let's speak",
            "lets speak",
        ),
        3.0,
    ),
    (
        OutputIntent.CARD,
        (
            "make a card",
            "make card",
            "study card",
            "flashcard",
            "another card",
            "new card",
            "learning card",
        ),
        2.5,
    ),
)

# Follow-ups that still mean “do something” but map to default card layout (translate, simplify, etc.)
_CARD_FALLBACK_HINTS: Final[tuple[str, ...]] = (
    "translate",
    "translation",
    "simplify",
    "simpler",
    "easier",
    "harder",
    "redo",
    "try again",
    "rewrite",
    "again",
    "another",
    "template",
    "flashcard",
    "переклад",
    "спрости",
    "картк",
    "шаблон",
)


_WS = re.compile(r"\s+")


def _normalize(t: str) -> str:
    return _WS.sub(" ", t.strip().lower())


def score_intents(text: str) -> dict[OutputIntent, float]:
    """Soft scores for each intent (multiple patterns can contribute)."""
    n = _normalize(text)
    scores: dict[OutputIntent, float] = {k: 0.0 for k in OutputIntent}
    for intent, patterns, weight in _INTENT_RULES:
        for p in patterns:
            if p in n:
                scores[intent] += weight
    # Single-word shortcuts (messy phrasing OK)
    one = n.strip()
    _single: dict[str, OutputIntent] = {
        "card": OutputIntent.CARD,
        "flashcard": OutputIntent.CARD,
        "speaking": OutputIntent.SPEAKING,
        "speak": OutputIntent.SPEAKING,
        "vocabulary": OutputIntent.VOCABULARY,
        "vocab": OutputIntent.VOCABULARY,
        "test": OutputIntent.TEST,
        "quiz": OutputIntent.TEST,
        "summary": OutputIntent.SUMMARY,
        "summarize": OutputIntent.SUMMARY,
        "tldr": OutputIntent.SUMMARY,
    }
    if one in _single:
        scores[_single[one]] += 4.0
    return scores


def _is_vague_followup(text: str, best_score: float) -> bool:
    if best_score > 0:
        return False
    n = _normalize(text)
    if not n:
        return True
    if any(h in n for h in _CARD_FALLBACK_HINTS):
        return False
    # Short, no recognized intent — ask one clarifying question
    if len(n) <= 40 and len(n.split()) <= 5:
        return True
    return False


def resolve_output_intent(text: str, *, is_follow_up: bool) -> OutputIntent:
    """
    Decide routing intent from the user's words (messy phrasing OK).

    New material: defaults to CARD unless strong intent signals in the message.
    Follow-up: if intent is too vague, raises UnclearIntentError.
    """
    scores = score_intents(text)
    best = max(scores.values()) if scores else 0.0

    if _is_vague_followup(text, best) and is_follow_up:
        raise UnclearIntentError()

    if best == 0.0:
        return OutputIntent.CARD

    best_intent = max(scores, key=lambda k: scores[k])
    return best_intent


def intent_label(intent: OutputIntent) -> str:
    """Human-readable label for captions / logs."""
    return {
        OutputIntent.CARD: "Card",
        OutputIntent.SPEAKING: "Speaking",
        OutputIntent.VOCABULARY: "Vocabulary",
        OutputIntent.TEST: "Test",
        OutputIntent.SUMMARY: "Summary",
    }[intent]
