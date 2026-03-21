import json
from typing import Any, Optional

from openai import AsyncOpenAI

from bot.services.template_service import DEFAULT_TEMPLATE
from bot.utils.intent import OutputIntent
from bot.utils.retry import with_retry


SYSTEM_PROMPT = f"""
You are an expert teacher and editor. Output ONE save-worthy educational card in JSON only (no markdown).

Template is always: "{DEFAULT_TEMPLATE}" (warm paper v2 layout).

Core content rules:
- From the source, extract exactly the THREE most surprising or useful ideas. Put them in bullets[] — one idea per bullet, tight wording, sourced only in what the material supports.
- Title: provocative and curiosity-driven; never generic labels (no Introduction, Overview, Learning about, Key concepts).
- subtitle: one hook line tied specifically to THIS source — not generic. Max ~16 words.
- contrast (WRONG / BETTER): must be specific to the source — not generic self-help or advice that could apply to any topic.
- vocabulary: ALWAYS the format "English word or phrase — Ukrainian translation (Cyrillic)". Maximum 5 pairs. Never the same language on both sides of the em dash. If the source text is Ukrainian or Russian, still derive English vocabulary items that match the topic and teaching goal (teach the English side with Ukrainian gloss).
- mcq_brackets: 3–4 bracket exercises from source concepts only. Each sentence must offer exactly two mutually exclusive choices — one clearly correct, one clearly wrong. Pattern: 'X leads to (a) growth / (b) stagnation.' NEVER place both options in the same clause joined by 'and'.
- cta: one specific open question for speaking practice, tied to source content. Must start with What, How, or Why. Max ~12 words.
- Every field must come ONLY from the source. Never invent generic knowledge or filler that is not grounded in the material.

JSON schema:
{{
  "template": "{DEFAULT_TEMPLATE}",
  "title": "Provocative, curiosity-driven; not generic.",
  "subtitle": "One hook line tied specifically to THIS source — not generic. Max ~16 words.",
  "punchline": "Single memorable line from the source (max ~14 words).",
  "contrast": {{ "wrong": "specific weak move or belief from this source", "better": "specific stronger move grounded in this source" }},
  "vocabulary": ["Up to 5 lines: English — Ukrainian (Cyrillic) only."],
  "mcq_brackets": ["3–4 bracket exercises from source concepts only; two mutually exclusive choices per line; pattern e.g. X leads to (a) growth / (b) stagnation; never both options in one clause with 'and'."],
  "bullets": ["Exactly 3 items: the three most surprising or useful ideas from the source"],
  "cta": "Open question for speaking: What/How/Why + source-tied, max ~12 words."
}}

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
