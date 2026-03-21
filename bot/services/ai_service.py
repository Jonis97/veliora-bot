import json
from typing import Any, Optional

from openai import AsyncOpenAI

from bot.services.template_service import DEFAULT_TEMPLATE
from bot.utils.intent import OutputIntent
from bot.utils.retry import with_retry


SYSTEM_PROMPT = f"""
You are an expert teacher and editor. Output ONE save-worthy educational card in JSON only (no markdown).

Template is always: "{DEFAULT_TEMPLATE}" (warm paper v2 layout).

JSON schema:
{{
  "template": "{DEFAULT_TEMPLATE}",
  "title": "Sharp, specific headline tied to the source — never generic course titles. No: Introduction, Overview, Learning about, Key concepts.",
  "subtitle": "One hook line (max ~16 words).",
  "punchline": "The single most memorable insight from the source (max ~14 words) — this is the 'top insight'.",
  "contrast": {{ "wrong": "typical mistake or weak habit FROM THE SOURCE", "better": "clear better move FROM THE SOURCE" }},
  "vocabulary": ["Exactly 4 lines: English word or short phrase — Ukrainian (Cyrillic). Only terms grounded in the source."],
  "mcq_brackets": ["3 or 4 lines: real bracket exercises (option / option). New sentences; do not copy vocabulary lines."],
  "bullets": ["3 or 4 short supporting points from the source only — no filler, no obvious generic advice"],
  "cta": "One short Let's speak style action (max ~12 words), practical for this topic."
}}

Hard rules:
- Grounding: every field must reflect the provided source. If the source is thin, stay humble — do NOT invent facts or generic life advice.
- Ban filler, banality, and content that could apply to any topic.
- vocabulary: strict EN — UA format only.
- mcq_brackets: must use parentheses and slash; exercises must differ from vocabulary wording.
- contrast.wrong and contrast.better must both be non-empty when the source allows; otherwise use the weakest honest pair you can anchor in the source.

Output valid JSON only.
""".strip()


class AIContentService:
    def __init__(self, openai_client: AsyncOpenAI, model: str) -> None:
        self._openai_client = openai_client
        self._model = model

    async def generate_card_content(
        self,
        source_text: str,
        template: Optional[str] = None,
        *,
        output_intent: OutputIntent = OutputIntent.CARD,
        is_followup: bool = False,
    ) -> dict[str, Any]:
        eff = template or DEFAULT_TEMPLATE
        user_prompt = (
            f"Source material:\n{source_text}\n\n"
            f"Produce exactly one premium card for template {eff}. "
            "Prioritize usefulness and clarity — strong teacher + strong editor. "
            "The card must feel worth saving and sharing.\n"
        )
        if is_followup:
            user_prompt += (
                "Follow-up: use ONLY the source block in the message. "
                "Honor the user's short instruction if present; still obey schema and grounding rules.\n"
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
            data.setdefault("template", eff)
            data["template"] = eff
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
