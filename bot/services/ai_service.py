import json
from typing import Any, Optional

from openai import AsyncOpenAI

from bot.services.template_service import DEFAULT_TEMPLATE
from bot.utils.retry import with_retry


SYSTEM_PROMPT = """
You write high-retention study cards that feel like a sharp post, not a textbook page.

Return JSON only (no markdown, no extra keys beyond the schema), with this schema:
{
  "template": "warm_paper | kitchen_collage | influencer_card | warm_paper_v2 | kitchen_collage_v2 | influencer_card_v2 (default when unsure: warm_paper_v2)",
  "title": "Provocative, problem-based headline: sound like a mistake people make, a risk, a trap, or a tension—not a chapter title. No neutral labels (avoid 'Introduction to…', 'Overview of…', 'Learning about…'). Use curiosity, contrast, or stakes.",
  "subtitle": "One short hook line (max ~18 words): who this is for or what situation—still punchy, not explanatory prose.",
  "punchline": "Single memorable line for the 'save this' moment (max ~16 words). No lecture tone. No list. Like a tweet people quote.",
  "contrast": { "wrong": "short typical mistake or weak version (max ~20 words)", "better": "sharp fix or stronger line (max ~20 words)" },
  "bullets": ["exactly 3 or 4 items only—each one short, scannable, one idea per line"],
  "cta": "One punchy micro-action (max ~14 words), concrete outcome, no homework-speak"
}

Title rules:
- NEVER sound like a course unit: ban phrases like 'Understanding', 'Basics of', 'Key concepts', 'Important aspects', 'Exploring', 'An introduction to'.
- Prefer patterns like: 'Why X backfires', 'The Y mistake in …', 'Stop doing Z when …', 'You sound less fluent when …', 'The trap: …'.

Content rules:
- Less text beats more: fewer words, higher signal.
- Bullets: max 4 lines, each under ~22 words, one idea each. Use fragments allowed.
- Where useful, encode contrast inside a bullet as 'Weak: … → Strong: …' or 'Don’t: … / Do: …' in one line.
- punchline is the emotional 'save' line; contrast.wrong vs contrast.better is the visual anchor—keep both tight.
- BAN generic motivation and filler that applies to any topic.

If the source is thin, infer tightly—never pad with generic advice.

If the source only includes a YouTube URL (no transcript), infer topic from URL and still follow all rules.

Output valid JSON only. Always include "contrast" with both "wrong" and "better" strings (can be short placeholders only if unavoidable).
""".strip()


class AIContentService:
    def __init__(self, openai_client: AsyncOpenAI, model: str) -> None:
        self._openai_client = openai_client
        self._model = model

    async def generate_card_content(self, source_text: str, template: Optional[str] = None) -> dict[str, Any]:
        user_prompt = (
            f"Source material:\n{source_text}\n\n"
            f"Preferred template: {template or DEFAULT_TEMPLATE}.\n\n"
            "Produce one card: provocative problem-style title, short hook subtitle, punchy punchline, "
            "tight wrong/better contrast, 3–4 razor bullets, punchy CTA. "
            "Optimize for 'I'd save this'—not for classroom neutrality."
        )

        async def _generate() -> dict[str, Any]:
            response = await self._openai_client.chat.completions.create(
                model=self._model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or "{}"
            data = json.loads(content)
            data.setdefault("template", template or DEFAULT_TEMPLATE)
            co = data.get("contrast")
            if not isinstance(co, dict):
                data["contrast"] = {"wrong": "", "better": ""}
            else:
                data["contrast"] = {
                    "wrong": str(co.get("wrong", "") or ""),
                    "better": str(co.get("better", "") or ""),
                }
            return data

        return await with_retry(_generate, attempts=3, operation_name="GPT card generation")
