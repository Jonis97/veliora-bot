import json
from typing import Any, Optional

from openai import AsyncOpenAI

from bot.utils.retry import with_retry


SYSTEM_PROMPT = """
You are an educational flashcard content designer.
Return JSON only (no markdown and no extra prose), with this schema:
{
  "template": "warm_paper | kitchen_collage | influencer_card",
  "title": "short catchy headline",
  "subtitle": "one-line context",
  "bullets": ["3-5 concise learning points"],
  "cta": "one practical action"
}
Constraints:
- Keep text concise and clear for social sharing.
- Output valid JSON object.
- If the source only includes a YouTube URL (no transcript), infer the likely topic from the URL
  and video ID and still produce a useful educational card.
""".strip()


class AIContentService:
    def __init__(self, openai_client: AsyncOpenAI, model: str) -> None:
        self._openai_client = openai_client
        self._model = model

    async def generate_card_content(self, source_text: str, template: Optional[str] = None) -> dict[str, Any]:
        user_prompt = (
            f"Source material:\n{source_text}\n\n"
            f"Preferred template: {template or 'warm_paper'}.\n"
            "Generate one educational card."
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
