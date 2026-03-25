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
- mcq_brackets: 3–4 bracket exercises from source concepts only. Each sentence must offer exactly two mutually exclusive choices — one clearly correct, one clearly wrong. Pattern: 'X leads to (a) growth / (b) stagnation.' NEVER place both options in the same clause joined by 'and'.
- cta: one specific open question for speaking practice, tied to source content. Must start with What, How, or Why. Max ~12 words.
- Every field must come ONLY from the source. Never invent generic knowledge or filler that is not grounded in the material.

EXTRACTION LAW — applies to ALL intents:
Every word, phrase, or example must come directly from the source text or be a minimal cleanup of source wording.

Test before including any item:
'Can I find this exact phrase or idea in the transcript?'
If NO -> do not include it. Period.

For vocabulary specifically:
- Take phrases that actually appear in the source
- Clean them slightly if needed (remove filler words)
- Never paraphrase into generic advice
- Never use phrases that could apply to ANY topic

Violation examples (never do this):
- learn vocabulary
- watch movies
- study consistently

Correct examples:
- watch 15 films in English
- learn 1500 words at A1
- find a native speaker

Learning logic:
- Every card must help BOTH:
  1) understand (EN → UA)
  2) use in speech (UA → EN)
- Each vocabulary item must include:
  • English phrase
  • Ukrainian translation
  • Example sentence in English
- The final block must force speaking:
  • real question (What / How / Why)
- The goal is not translation, but usage

Vocabulary selection rules (think like an experienced teacher):
1. Does this word/phrase actually help the student understand the topic?
2. Is it not too obvious or too basic?
3. Is it clearly connected to the lesson theme?
4. Can it be used in speech or in a task?
5. Does it help understand the video — not just exist in the text?

Prefer phrases and collocations over single words (e.g. 'make progress' not just 'progress').
Avoid isolated basic verbs unless they are part of a useful phrase or collocation.
Avoid overly academic terms and random low-value nouns.
Always use format: English phrase — Ukrainian translation.

For vocab_card template:
- title: must be the specific topic from source, never 'Learning Card'
- vocabulary: extract 6-8 most useful specific words from source text
- cta: must be a real speaking question starting with What/How/Why

JSON schema (warm_paper_v2 only — do not use this shape for vocab_card):
{{
  "template": "warm_paper_v2",
  "title": "Provocative, curiosity-driven; not generic.",
  "subtitle": "One hook line tied specifically to THIS source — not generic. Max ~16 words.",
  "punchline": "Single memorable line from the source (max ~14 words).",
  "contrast": {{ "wrong": "specific weak move or belief from this source", "better": "specific stronger move grounded in this source" }},
  "vocabulary": ["English phrase — Ukrainian translation; one string per item. Pair with vocabulary_examples (same order): English example sentence per item."],
  "vocabulary_examples": ["English example sentences, same length and order as vocabulary[]"],
  "mcq_brackets": ["3–4 bracket exercises from source concepts only; two mutually exclusive choices per line; pattern e.g. X leads to (a) growth / (b) stagnation; never both options in one clause with 'and'."],
  "bullets": ["Exactly 3 items: the three most surprising or useful ideas from the source"],
  "cta": "Open question for speaking: What/How/Why + source-tied, max ~12 words."
}}

JSON schema for vocab_card only:
{{
  "template": "vocab_card",
  "title": "Specific topic from source (never 'Learning Card').",
  "subtitle": "One short line tied to this source.",
  "punchline": "",
  "contrast": {{ "wrong": "", "better": "" }},
  "vocabulary": [
    {{ "term": "English phrase", "translation": "Ukrainian translation", "example": "English example sentence" }}
  ],
  "vocabulary_examples": [],
  "mcq_brackets": [],
  "bullets": [],
  "cta": "Real speaking question starting with What, How, or Why."
}}

For vocab_card: vocabulary must be an array of 6–8 objects; each object has exactly term, translation, and example (three separate fields). Do not use string lines or vocabulary_examples for vocab_card.

When intent is questions:
- Generate 6–9 questions depending on source richness
- Mix: simple → medium → deeper
- Prefer quality over quantity

EXTRACTION LAW for questions:
Every question must come from a specific idea, fact or statement in source.
Test: "Can I point to the exact part of the transcript this comes from?"
If NO → do not include it.
Never generic questions. Never invented content.

JSON schema for questions_card only:
{{
  "template": "questions_card",
  "title": "Topic of the video (from source).",
  "subtitle": "",
  "handle": "Optional @instagram handle or empty string.",
  "questions": ["6–9 questions as strings; source-grounded only."],
  "punchline": "",
  "contrast": {{ "wrong": "", "better": "" }},
  "vocabulary": [],
  "vocabulary_examples": [],
  "mcq_brackets": [],
  "bullets": [],
  "cta": ""
}}

For questions_card: fill questions[] with 6–9 items; title is the video topic; handle optional; image_url is set by the app for YouTube thumbnails.

When intent is lesson:
- All content must come from the source only. Do not invent facts, examples, or situations not supported by the material.
- topic: the main theme / title as it follows from the source (video topic).
- lead_in_questions: 2–3 short, engaging questions that activate curiosity and start discussion — not quizzes or knowledge tests.
- choices: 4–6 situational "this or that" prompts for speaking practice — fun, relevant, tied to ideas from the source. Each item is one line in the form "Option A or Option B?" (or equivalent clear pair). You may use strings or objects with two options (e.g. a/b or option_a/option_b).
- No vocabulary lists and no exercises — those belong to other templates (vocab_card, warm_paper_v2, etc.).
- image_url is set by the app for YouTube thumbnails when applicable.

JSON schema for lesson_card_v1 only:
{{
  "template": "lesson_card_v1",
  "topic": "Main theme from source.",
  "lead_in_questions": ["2–3 warmup questions; curiosity and discussion, not testing knowledge."],
  "choices": ["4–6 lines: 'X or Y?' style; speaking practice; source-grounded only."],
  "title": "",
  "subtitle": "",
  "punchline": "",
  "contrast": {{ "wrong": "", "better": "" }},
  "vocabulary": [],
  "vocabulary_examples": [],
  "mcq_brackets": [],
  "bullets": [],
  "cta": ""
}}

For lesson: use template lesson_card_v1 and populate topic, lead_in_questions, and choices only (other keys empty or minimal). Do not invent content not grounded in the source.

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
        intent: str = "card",
    ) -> dict[str, Any]:
        eff = template or DEFAULT_TEMPLATE
        user_prompt = (
            f"Source material:\n{source_text}\n\n"
            f"User intent: {intent}.\n"
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
            for key in (
                "vocabulary",
                "mcq_brackets",
                "vocabulary_examples",
                "questions",
                "lead_in_questions",
                "choices",
            ):
                v = data.get(key)
                if not isinstance(v, list):
                    data[key] = []
            return data

        return await with_retry(_generate, attempts=3, operation_name="GPT card generation")
