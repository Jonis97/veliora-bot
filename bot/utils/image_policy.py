"""
Topic image URL policy: reject logos, brand UI, banners, and other non-illustrative assets.

Used by topic_image_service (fetch) and template_service (defense in depth at render).
"""

from __future__ import annotations

import re
from typing import Final

# Substrings that strongly suggest logos, UI, or brand marks (case-insensitive check)
_BLOCKED_SUBSTRINGS: Final[tuple[str, ...]] = (
    "logo",
    "wordmark",
    "banner",
    "favicon",
    "icon.svg",
    "/icons/",
    "twitch",
    "youtube",
    "youtu.be",
    "google",
    "facebook",
    "meta_",
    "twitter",
    "x.com",
    "reddit",
    "instagram",
    "tiktok",
    "linkedin",
    "brand",
    "screenshot",
    "thumbnail_default",
    "avatar",
    "badge",
    "emblem",
    "seal_",
    "flag_of",
    "commons-logo",
    "powered_by",
    "app-icon",
)

# Filename / path patterns (often Wikipedia file names)
_BLOCKED_REGEX = (
    re.compile(r"[_/]logo[_\-]", re.I),
    re.compile(r"[_\-]logo[._]", re.I),
    re.compile(r"\bui[_\-]screenshot", re.I),
)


def title_suggests_logo_or_non_photo(title: str) -> bool:
    """Heuristic for Wikipedia page titles that are usually logos or non-photographic."""
    if not title:
        return True
    t = title.strip().lower()
    bad = (
        "logo",
        "wordmark",
        "banner",
        "icon",
        "favicon",
        "emblem",
        "seal",
        "flag of",
        "coat of arms",
        "symbol",
        "trademark",
        "brand",
    )
    return any(b in t for b in bad)


def is_safe_topic_image_url(url: str) -> bool:
    """True only if URL is https and unlikely to be logo/UI/banner junk."""
    if not url or not isinstance(url, str):
        return False
    u = url.strip()
    if not u.startswith("https://"):
        return False
    lower = u.lower()
    for s in _BLOCKED_SUBSTRINGS:
        if s in lower:
            return False
    for rx in _BLOCKED_REGEX:
        if rx.search(lower):
            return False
    return True
