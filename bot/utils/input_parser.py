import re
from typing import Optional

from bot.services.template_service import ALLOWED_TEMPLATES


TEMPLATE_TAG_REGEX = re.compile(r"\[template:(warm_paper|kitchen_collage|influencer_card)\]", re.IGNORECASE)


def parse_template_hint(text: str) -> tuple[str, Optional[str]]:
    """Returns (clean_text, template_name|None)."""
    match = TEMPLATE_TAG_REGEX.search(text)
    if not match:
        return text.strip(), None
    template = match.group(1).lower()
    if template not in ALLOWED_TEMPLATES:
        template = None
    clean = TEMPLATE_TAG_REGEX.sub("", text).strip()
    return clean, template
