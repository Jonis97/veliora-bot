import json
from typing import Any, Optional

from openai import AsyncOpenAI

from bot.services.template_service import DEFAULT_TEMPLATE
from bot.utils.retry import with_retry


SYSTEM_PROMPT = """
You write high-retention study cards that feel like a sharp post, not a textbook page.

Return JSON only (no markdown), with this schema:
{
  "template": "warm_paper | kitchen_collage | influencer_card | warm_paper_v2 | kitchen_collage_v2 | influencer_card_v2 (default when unsure: warm_paper_v2)",
  "title": "Provocative, problem-based headline: mistake, risk, trap, or tension—not a chapter title.",
  "subtitle": "One short hook line (max ~16 words).",
  "punchline": "Single memorable save-line (max ~14 words).",
  "contrast": { "wrong": "…", "better": "…" },
  "vocabulary": ["4 strings ONLY in form: English word or short phrase — Ukrainian translation (Cyrillic). Example: aesthetic — естетика. No full sentences, no definitions in English."],
  "mcq_brackets": ["3 or 4 strings: each is ONE sentence or clause with a real bracket choice using (option / option). Example: Focus on (your unique voice / copying trends) to build connections. Must use parentheses and slash. Do NOT repeat or copy vocabulary lines—different wording and task."],
  "bullets": ["3 or 4 short supporting ideas—do not duplicate lines from vocabulary or mcq_brackets"],
  "cta": "One punchy micro-action (max ~12 words)"
}

Title rules:
- BAN neutral course titles: no 'Introduction to', 'Overview', 'Learning about', 'Key concepts'.

When template is warm_paper_v2 (or default warm_paper_v2):
- vocabulary[] is ONLY EN–UA pairs as specified—never full sentences.
- mcq_brackets[] is ONLY bracket-style exercises—never vocabulary duplicates.
- Keep total text lean: short lines, high signal.

For other templates, still include vocabulary and mcq_brackets if the layout benefits; otherwise use empty arrays [].

If the source only includes a YouTube URL (no transcript), infer topic and still follow rules.

Output valid JSON only. Always include "contrast" with wrong and better. Use [] for vocabulary/mcq_brackets only if impossible.
""".strip()


class AIContentService:
    def __init__(self, openai_client: AsyncOpenAI, model: str) -> None:
        self._openai_client = openai_client
        self._model = model

    async def generate_card_content(self, source_text: str, template: Optional[str] = None) -> dict[str, Any]:
        eff = template or DEFAULT_TEMPLATE
        user_prompt = (
            f"Source material:\n{source_text}\n\n"
            f"Preferred template: {eff}.\n\n"
            "Produce one premium card: provocative title, hook subtitle, punchline, contrast pair, "
            "lean bullets, punchy CTA. "
        )
        if eff == "warm_paper_v2":
            user_prompt += (
                "For warm_paper_v2 you MUST fill vocabulary (4× EN — UA only) and mcq_brackets "
                "(3–4 bracket exercises, different content from vocabulary). "
            )
        user_prompt += "Optimize for save-worthiness and clarity, not textbook length."

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
            data.setdefault("template", eff)
            co = data.get("contrast")
            if not isinstance(co, dict):
                data["contrast"] = {"wrong": "", "better": ""}
            else:
                data["contrast"] = {
                    "wrong": str(co.get("wrong", "") or ""),
                    "better": str(co.get("better", "") or ""),
                }
            for key in ("vocabulary", "mcq_brackets"):
                v = data.get(key)
                if not isinstance(v, list):
                    data[key] = []
            return data

        return await with_retry(_generate, attempts=3, operation_name="GPT card generation")
