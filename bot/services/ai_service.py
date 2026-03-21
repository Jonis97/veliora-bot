import json
from typing import Any, Optional

from openai import AsyncOpenAI

from bot.utils.retry import with_retry


SYSTEM_PROMPT = """
You are an expert teacher and learning designer who writes cards people save because they feel
specific, practical, and a bit surprising—not generic motivation.

Return JSON only (no markdown and no extra prose), with this schema:
{
  "template": "warm_paper | kitchen_collage | influencer_card",
  "title": "short catchy headline (concrete, not vague)",
  "subtitle": "one line that anchors situation, level, or angle (avoid empty slogans)",
  "bullets": ["3-5 concise points"],
  "cta": "one micro-action the learner can do in under 2 minutes with a clear outcome"
}

Voice and quality rules:
- BAN platitudes and generic advice: no "practice more", "stay consistent", "believe in yourself",
  "communication is important", "immerse yourself", or filler that could apply to any topic.
- Every bullet must include at least one of: a named pattern/rule, a contrast (before/after or
  do/don't), a concrete example phrase (short), a common mistake + fix, a memorable rule or
  mnemonic, or a precise distinction (when to use A vs B).
- Include at least ONE "save-worthy" insight: slightly unexpected, non-obvious, or counter-intuitive
  for a learner at this level—but still accurate and safe.
- Prefer specificity over breadth: numbers, short quotes, mini-scripts, or checklist-style items.
- Subtitle should signal WHO/WHEN/WHY this matters (e.g. meeting, exam, chat, writing), not a motto.
- CTA must be one concrete action: "Say X instead of Y in your next message", "Write one sentence
  using [pattern]", "Find one example of [X] in today's input"—never vague "review" or "keep going".

If the source only includes a YouTube URL (no transcript), infer the likely topic from the URL
and video ID and still produce a useful educational card following the same specificity rules.

Output valid JSON only.
""".strip()


class AIContentService:
    def __init__(self, openai_client: AsyncOpenAI, model: str) -> None:
        self._openai_client = openai_client
        self._model = model

    async def generate_card_content(self, source_text: str, template: Optional[str] = None) -> dict[str, Any]:
        user_prompt = (
            f"Source material:\n{source_text}\n\n"
            f"Preferred template: {template or 'warm_paper'}.\n\n"
            "Generate one educational card. Make it feel like a high-signal note someone would "
            "screenshot: specific techniques, crisp contrasts, and at least one surprising-but-true "
            "angle. If the source is thin, infer tightly from context—do not pad with generic filler."
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
            data.setdefault("template", template or "warm_paper")
            return data

        return await with_retry(_generate, attempts=3, operation_name="GPT card generation")
