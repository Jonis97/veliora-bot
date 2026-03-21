"""
Per-user active source memory: latest YouTube transcript, text, or voice transcript.

Follow-up messages reuse this source only; new material replaces it entirely.
"""

from __future__ import annotations

import re
from typing import Any, MutableMapping, Optional

ACTIVE_SOURCE_KEY = "veliora_active_source"

# Short acknowledgments — not treated as follow-up instructions
_ACK_ONLY = re.compile(
    r"^(ok|okay|thanks?|thank you|yes|no|hi|hello|hey|cool|great|nice)\b[!.\s]*$",
    re.IGNORECASE,
)


class NeedActiveSourceError(Exception):
    """Raised when the user asks for a follow-up style action but no source is stored yet."""


def load_active_source(user_data: Optional[MutableMapping[str, Any]]) -> Optional[dict[str, Any]]:
    """Return validated active source dict or None."""
    if not user_data:
        return None
    raw = user_data.get(ACTIVE_SOURCE_KEY)
    if not isinstance(raw, dict):
        return None
    st = raw.get("type")
    text = raw.get("text")
    if st not in ("youtube", "text", "voice") or not isinstance(text, str) or not text.strip():
        return None
    return raw


def save_active_source(
    user_data: Optional[MutableMapping[str, Any]],
    *,
    source_type: str,
    text: str,
    video_id: Optional[str] = None,
) -> None:
    """Replace the active source with new material (strict isolation — never merge)."""
    if user_data is None:
        return
    payload: dict[str, Any] = {"type": source_type, "text": text.strip()}
    if video_id:
        payload["video_id"] = video_id
    user_data[ACTIVE_SOURCE_KEY] = payload


def build_followup_prompt(user_instruction: str, active: dict[str, Any]) -> str:
    """Combine user instruction with stored source; model must use only the source block."""
    instruction = user_instruction.strip() or "Continue with the same material."
    body = (active.get("text") or "").strip()
    return (
        "User instruction (follow this; do not introduce unrelated topics or other sessions):\n"
        f"{instruction}\n\n"
        "---\n"
        "Source material (use ONLY the following; do not blend with anything else):\n"
        f"{body}"
    )


def followup_intent(text: str) -> bool:
    """
    True if the message is likely a meta-instruction about existing material
    (translate, another card, template change, etc.), not new standalone content.
    """
    t = text.strip()
    if not t:
        return False
    if len(t) >= 480:
        return False
    if len(t) < 28 and _ACK_ONLY.match(t):
        return False

    tl = t.lower()
    # Short deictic follow-ups ("translate it", "same", …)
    if len(t) < 48 and tl in ("it", "this", "that", "same"):
        return True
    keywords = (
        "translate",
        "translation",
        "simplify",
        "simpler",
        "easier",
        "harder",
        "another card",
        "new card",
        "make a card",
        "make card",
        "make another",
        "speaking task",
        "speaking practice",
        "speaking exercise",
        "change template",
        "different template",
        "[template:",
        "redo",
        "try again",
        "regenerate",
        "rewrite",
        "flashcard",
        "переклад",
        "спрости",
        "картк",
        "шаблон",
    )
    if any(k in tl for k in keywords):
        return True

    # Short imperative / question likely about "it" / previous material
    if len(t) < 120 and len(t.split()) <= 16:
        cues = (
            "make ",
            "give ",
            "create ",
            "build ",
            "show ",
            "turn ",
            "convert ",
            "what about ",
            "can you ",
            "could you ",
            "please ",
            "do another",
            "one more",
            "same topic",
            "same video",
            "same text",
        )
        if any(c in tl for c in cues):
            return True
    return False
