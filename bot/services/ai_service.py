import json
from typing import Any, Optional

from openai import AsyncOpenAI

from bot.services.template_service import DEFAULT_TEMPLATE
from bot.utils.intent import OutputIntent
from bot.utils.retry import with_retry


CORE_ROLES = """
You combine three roles at once:
- Teacher: clear, accurate, learner-friendly; no fluff.
- Editor: tight wording; remove generic filler, hedging, and repeated ideas.
- Designer: keep each JSON field compact so one fixed card layout stays balanced and premium (Canva-like density).
""".strip()

SYSTEM_PROMPT = f"""
{CORE_ROLES}

Return JSON only (no markdown), with this schema:
{{
  "template": "{DEFAULT_TEMPLATE}",
  "title": "Provocative, problem-based headline: mistake, risk, trap, or tension—not a chapter title.",
  "subtitle": "One short hook line (max ~16 words).",
  "punchline": "Single memorable save-line (max ~14 words).",
  "contrast": {{ "wrong": "…", "better": "…" }},
  "vocabulary": ["4 strings ONLY in form: English word or short phrase — Ukrainian translation (Cyrillic). Example: aesthetic — естетика. No full sentences, no definitions in English."],
  "mcq_brackets": ["3 or 4 strings: each is ONE sentence or clause with a real bracket choice using (option / option). Example: Focus on (your unique voice / copying trends) to build connections. Must use parentheses and slash. Do NOT repeat or copy vocabulary lines—different wording and task."],
  "bullets": ["3 or 4 short supporting ideas—do not duplicate lines from vocabulary or mcq_brackets"],
  "cta": "One punchy micro-action (max ~12 words)"
}}

Title rules:
- BAN neutral course titles: no 'Introduction to', 'Overview', 'Learning about', 'Key concepts'.

Always use template "{DEFAULT_TEMPLATE}" in the JSON (single product layout).

When template is warm_paper_v2, follow these layout rules:
- vocabulary[] is ONLY EN–UA pairs as specified—never full sentences.
- mcq_brackets[] is ONLY bracket-style exercises—never vocabulary duplicates.
- Keep total text lean: short lines, high signal.

If the source only includes a YouTube URL (no transcript), infer topic and still follow rules.

Output valid JSON only. Always include "contrast" with wrong and better. Use [] for vocabulary/mcq_brackets only if impossible.
""".strip()


INTENT_INSTRUCTIONS: dict[OutputIntent, str] = {
    OutputIntent.CARD: (
        "Output intent: default study card. Balance vocabulary, bracket MCQs, bullets, and CTA. "
        "Extract only what matters from the source."
    ),
    OutputIntent.SPEAKING: (
        "Output intent: SPEAKING. Prioritize usable spoken practice: bullets should be short prompts, "
        "lines, or discussion questions the learner can say aloud. CTA must be a concrete speaking micro-task. "
        "Still fill vocabulary and mcq_brackets from the source where useful, but bias bullets + punchline toward oral practice. "
        "Subtitle should signal 'speaking / oral practice'."
    ),
    OutputIntent.VOCABULARY: (
        "Output intent: VOCABULARY. Maximize high-value EN–UA pairs from the source; skip low-signal words. "
        "mcq_brackets should drill those words in new sentences (not copy-paste from vocabulary lines). "
        "Bullets = short usage notes or collocations only. Title/subtitle should signal vocabulary focus."
    ),
    OutputIntent.TEST: (
        "Output intent: TEST / QUIZ. Make mcq_brackets the star: challenging but fair checks of understanding from the source. "
        "Vocabulary lines should support terms under test. Bullets = brief rationale or reminders, not long explanations. "
        "Contrast pair should highlight a common mistake from the material."
    ),
    OutputIntent.SUMMARY: (
        "Output intent: SUMMARY. Distill key ideas only: bullets are the tightest possible takeaways (no fluff). "
        "Punchline = one-line thesis of the source. Title/subtitle reflect synthesis, not a generic label. "
        "Vocabulary = only terms that appear central in the source; mcq_brackets check comprehension of those ideas."
    ),
}


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
        intent_line = INTENT_INSTRUCTIONS.get(output_intent, INTENT_INSTRUCTIONS[OutputIntent.CARD])
        user_prompt = (
            f"Source material:\n{source_text}\n\n"
            f"Preferred template (locked): {eff}.\n"
            f"{intent_line}\n\n"
            "Produce one premium card: provocative title, hook subtitle, punchline, contrast pair, "
            "lean bullets, punchy CTA. "
        )
        if is_followup:
            user_prompt += (
                "This is a follow-up: the block labeled 'Source material' is the ONLY content you may use. "
                "Follow the user instruction and output_intent exactly; do not add facts from other topics or earlier chats. "
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
