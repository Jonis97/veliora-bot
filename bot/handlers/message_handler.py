import json
import logging
from typing import Any, Optional, Union

from openai import AsyncOpenAI
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Message, Update
from telegram.ext import ContextTypes

from bot.services.pipeline_service import ContentPipelineService
from bot.services.youtube_service import YouTubeTranscriptService, extract_video_id
from bot.utils.active_source import NeedActiveSourceError
from bot.utils.dedup import MessageDeduplicator
from bot.utils.errors import GenerationFailedError, TranscriptUnavailableError


LOGGER = logging.getLogger(__name__)

# Onboarding only — separate from active_source / pipeline memory.
user_state: dict[int, dict[str, Optional[str]]] = {}

# Temporary guided-preview state — separate from active_source.
preview_state: dict[int, dict[str, Any]] = {}

_PREVIEW_TRANSCRIPT_MAX = 14_000

_PREVIEW_SYSTEM_DEFAULT = (
    "You are a helpful teacher. Output ONE JSON object only, no markdown.\n"
    "From the source transcript below ONLY (no invented facts):\n"
    '- "topic": one short line (teacher-friendly)\n'
    '- "key_ideas": exactly 3 short strings, each one bullet-worthy idea from the source\n'
    '- "words": 3 to 5 useful English words or short phrases that appear in or are clearly grounded in the source\n'
    "Keep everything short. No card layout, no images.\n"
    "Follow any additional instruction without breaking source-only rules."
)

_PREVIEW_SYSTEM_LESSON = (
    "You are a helpful teacher. Output ONE JSON object only, no markdown.\n"
    "Use the transcript below ONLY as the source (no invented facts).\n"
    'Return ONLY these keys for a lesson preview:\n'
    '- "topic": one short line (teacher-friendly)\n'
    '- "warmup_questions": exactly 5 short warm-up questions grounded in source\n'
    '- "choices": 3 to 4 this-or-that choices grounded in source, format: "Option A or Option B?"\n'
    '- "support_words": at least 5 simple English words or short chunks from source\n'
    "HARD RULES:\n"
    "- Never return '—' or an empty string as a question or choice.\n"
    "- warmup_questions must be exactly 5 real questions.\n"
    "- support_words must have at least 5 real items (words or chunks).\n"
    "- If the source has limited content, derive questions, choices, and words from the topic while staying consistent with the transcript.\n"
    "Do not include key_ideas, discussion_questions, or vocabulary_items."
)

_PREVIEW_SYSTEM_LESSON_A1 = (
    "You are a helpful teacher. Output ONE JSON object only, no markdown.\n"
    "Use the A1 FILTERED SOURCE in the user message ONLY as the source (no invented facts beyond it). "
    "It is a filtered daily-life topic and scenes — not a raw transcript; do not use or assume any other text.\n"
    "This preview is for CEFR level A1 only. Follow ALL rules below.\n\n"
    "GENERAL:\n"
    "- Use only CEFR A1 vocabulary.\n"
    "- Use ONLY Present Simple.\n"
    "- Max sentence length: 6 words (for every English sentence you output, including topic line if possible).\n"
    '- No abstract words (e.g. "freedom", "success").\n'
    "- No idioms or phrasal verbs.\n"
    "- All content must relate to ONE topic.\n"
    '- No placeholders like "—".\n'
    "- No empty fields.\n"
    "- Do not regenerate existing content; only extend if asked (initial generation: fill all fields from source).\n"
    "- Follow exact counts strictly (no more, no less).\n\n"
    "COHERENCE:\n"
    "- All questions, choices, and vocabulary must stay inside the same daily situation as the topic. "
    "Do not introduce unrelated elements.\n"
    "- Example: if topic is 'morning routine at home', every question, word, and choice must relate "
    "only to morning routine at home.\n"
    "- Forbidden: neighbors, strangers, unrelated places or people.\n\n"
    "TOPIC LINE:\n"
    "- The topic must always be concrete, personal, and easy to answer (everyday life).\n"
    "- If the filtered source suggests abstract, psychological, emotional, or conceptual themes, "
    "express the topic as a simple daily-life situation, not an abstract label.\n"
    "- Examples: self-talk → talking to yourself at home; motivation → things that help you every day; "
    "stress → feeling bad at school or work.\n\n"
    'Return ONLY these keys for a lesson preview:\n'
    '- "topic": one short line (teacher-friendly), one concrete daily-life situation, same ONE topic as all items below\n'
    '- "warmup_questions": exactly 5 questions\n'
    '- "core_questions": exactly 4 questions\n'
    '- "choices": exactly 4 items\n'
    '- "support_words": exactly 6 items\n\n'
    "WARM-UP (warmup_questions): exactly 5 questions\n"
    "- Each max 6 words.\n"
    '- Patterns: "Do you...?" / "Is it...?" / "Can you...?"\n'
    "- Personal, about daily life.\n"
    '- Forbidden: "Why" questions, abstract topics.\n\n'
    "CORE QUESTIONS (core_questions): exactly 4 questions\n"
    "- Each max 7 words.\n"
    '- Patterns: "What do you...?" / "Where do you...?" / "When do you...?"\n'
    "- Directly related to topic.\n"
    "- Forbidden: hypothetical questions, complex grammar.\n\n"
    "THIS OR THAT (choices): exactly 4 items\n"
    '- Format ONLY: "X or Y?" (question mark at end).\n'
    "- X and Y: single words or max 2-word phrases.\n"
    '- Concrete only (e.g. "coffee or tea?").\n'
    "- Forbidden: abstract choices, long phrases.\n\n"
    "VOCABULARY (support_words): exactly 6 items\n"
    '- Format each string: "English — Ukrainian" (em dash between English and Ukrainian).\n'
    "- Nouns or verbs only.\n"
    "- Directly related to topic.\n"
    "- Forbidden: rare or abstract words.\n\n"
    "Do NOT include neutral or generic daily routine elements that are not directly related to the main situation "
    "and main problem from the source.\n"
    "If a word, question, or choice does not clearly reflect the core situation and core problem, it must be excluded.\n"
    "Prioritize relevance over variety.\n\n"
    "Example:\n"
    "If topic is 'waking up early and feeling tired':\n"
    "KEEP: tired, sleep, wake up, feel bad, school\n"
    "REMOVE: breakfast, toast, cereal, coffee\n"
    "(unless they directly appear in the source)\n\n"
    "QUALITY CONTROL:\n"
    "- Every single element — each question, each choice, each vocabulary word — must directly relate to "
    "the simplified topic AND the core problem.\n"
    "- Test before including any element:\n"
    "  'Does this word/question help student talk about THIS specific situation?'\n"
    "- If answer is NO → remove it.\n"
    "- Do not add generic filler just to make the lesson feel complete.\n"
    "- Prioritize relevant content over filler content.\n\n"
    "If the filtered source has limited detail, derive items from the topic and scenes while staying consistent with them.\n"
    "Do not include key_ideas, discussion_questions, or vocabulary_items."
)

_PREVIEW_SYSTEM_LESSON_A2 = (
    "You are a helpful teacher. Output ONE JSON object only, no markdown.\n"
    "Use the A2 FILTERED SOURCE in the user message ONLY as the source (no invented facts beyond it). "
    "It is a filtered topic and scenes — not a raw transcript; do not use or assume any other text.\n"
    "This preview is for CEFR level A2 only. Apply ONLY these rules for lesson + A2.\n\n"
    "GLOBAL:\n"
    "- Every element must reflect the main situation and core meaning from the filtered source.\n"
    "- Do NOT introduce unrelated elements.\n"
    "- Do NOT add generic filler.\n"
    "- Simplify the language, NOT the topic.\n"
    "- Follow exact counts strictly (no more, no less).\n"
    "- Do not regenerate existing content; only extend if asked (initial generation: fill all fields from source).\n"
    '- No placeholders like "—".\n'
    "- No empty fields.\n\n"
    'Return ONLY these keys for a lesson preview:\n'
    '- "topic": one short line (teacher-friendly), same situation as the source (simplified, not replaced)\n'
    '- "warmup_questions": exactly 5 questions\n'
    '- "core_questions": exactly 4 questions\n'
    '- "choices": exactly 4 items\n'
    '- "support_words": exactly 6 items (each "English — Ukrainian")\n\n'
    "WARM-UP (warmup_questions): exactly 5 questions\n"
    '- Patterns ONLY: "Do you...?" / "Is it...?" / "Can you...?"\n'
    "- Must be personal and concrete.\n"
    '- Forbidden: "Have you ever", abstract topics.\n\n'
    "CORE QUESTIONS (core_questions): exactly 4 questions\n"
    "- Why and How questions are allowed.\n"
    "- BUT: must stay concrete, simple, and directly tied to the source situation.\n"
    "- Must relate directly to topic.\n"
    "- Do NOT move into abstract reasoning or theory.\n"
    "- Simple cause-effect is allowed (because, so, helps, makes).\n"
    "- Why and How questions must be about personal experience or simple real-life situations.\n"
    "- Do NOT generate scientific, academic, or theoretical questions.\n"
    "- Wrong: 'What are the phases of hair growth cycle?'\n"
    "- Wrong: 'How does genetics affect hair loss?'\n"
    "- Right: 'Do you know why some people lose hair?'\n"
    "- Right: 'Why is hair loss a problem for some people?'\n"
    "- Questions must be answerable from personal experience, not from scientific knowledge.\n\n"
    "THIS OR THAT (choices): exactly 4 items\n"
    '- Format ONLY: "X or Y?" (question mark at end).\n'
    "- May reflect simple contrast from the source.\n\n"
    "VOCABULARY (support_words): exactly 6 items\n"
    '- Format each string: "English — Ukrainian".\n'
    "- Slightly wider than A1 but still practical.\n"
    "- Must be directly related to topic.\n"
    "- Vocabulary must stay everyday and student-usable.\n"
    "- Do NOT include scientific or academic terms.\n"
    "- All English vocabulary words must be lowercase.\n\n"
    "If the filtered source has limited detail, derive items from the topic and scenes while staying consistent with them.\n"
    "Do not include key_ideas, discussion_questions, or vocabulary_items."
)

_PREVIEW_A2_FILTER_SYSTEM = (
    "You follow the user instructions exactly. Output ONE JSON object only, no markdown."
)

_PREVIEW_A2_FILTER_USER = (
    "Filter this content for A2 English students.\n\n"
    "Output ONLY:\n"
    "- 1 simplified topic (same situation as source, not replaced)\n"
    "- 3–5 scenes or cause-effect points from the source\n\n"
    "Rules:\n"
    "- Stay semantically close to the original source\n"
    "- Do NOT replace topic with generic daily routine\n"
    "- Do NOT invent context that does not exist in source\n"
    "- Topic may include simple cause-effect (why/how)\n"
    "- Language must be simple and concrete\n"
    "- If topic cannot be simplified → keep closest real-life version\n"
    "- NEVER switch to unrelated situations\n\n"
    'Return one JSON object with keys "topic" (string) and "scenes" (array of 3 to 5 strings). '
    "Each scene or cause-effect point is one short English line. No extra keys."
)

_PREVIEW_A1_FILTER_SYSTEM = (
    "You follow the user instructions exactly. Output ONE JSON object only, no markdown."
)

_PREVIEW_A1_FILTER_USER = (
    "Filter this content for A1 English students.\n\n"
    "Output ONLY:\n"
    "- 1 simplified daily-life topic (concrete situation)\n"
    "- 3–5 daily-life scenes or actions from the source\n\n"
    "Rules:\n"
    "- No abstract or mental concepts\n"
    "- Do not copy abstract or complex wording from the source\n"
    "- No psychology, theory, or emotions explanations\n"
    "- Use only: actions, places, times\n"
    "  (talk, go, eat / home, school, work / morning, evening)\n"
    "- Keep only concrete, visible, real-life situations\n"
    "- Each scene must be a simple physical action or situation\n"
    "- Output must be something an A1 student\n"
    "  can imagine, answer, and talk about\n"
    "- Scenes are better than ideas\n\n"
    "CRITICAL: Do NOT replace the topic with a more common or easier scenario.\n"
    "Do NOT generalize.\n"
    "Do NOT invent context that does not exist in the source "
    "(e.g. gym, routine, lifestyle).\n\n"
    "You must simplify the SAME situation from the source.\n"
    "Not a different situation.\n\n"
    "Wrong: source about muscles → output 'exercising at gym'\n"
    "Right: source about muscles → output 'muscles get tired and grow'\n\n"
    "Stay strictly inside the meaning of the source.\n"
    "Simplify the language, not the topic.\n\n"
    "The simplified topic MUST stay semantically close to the original source.\n"
    "Do NOT replace the topic with a generic daily routine.\n"
    "Do NOT invent a new situation unrelated to the source.\n"
    "You must simplify the SAME situation, not change it.\n\n"
    "If the topic cannot be simplified into A1-level daily-life scenes without losing meaning:\n"
    "- keep only the closest possible real-life version\n"
    "- do NOT switch to unrelated topics\n\n"
    "Example:\n"
    "Source topic: teenagers not getting enough sleep\n"
    "A1 version: going to bed late and feeling tired\n"
    "NOT: morning routine or waking up\n\n"
    "All questions, choices, and vocabulary must reflect the main situation AND the main problem from the source.\n"
    "For A1, keep the core difficulty or feeling in a simple, concrete form.\n\n"
    "Example:\n"
    "Source: teenagers not getting enough sleep\n"
    "A1 focus: tired, sleep, wake up early, want to sleep\n"
    "Questions: Do you feel tired in the morning?\n"
    "           Is it hard to wake up for school?\n"
    "NOT: neutral morning routine without the core problem\n\n"
    "Do NOT shift to generic daily routine if the source has a clear main problem.\n\n"
    "If no usable daily-life scenes are found in the source:\n"
    "- ignore the transcript completely\n"
    "- use only the simplified topic\n"
    "- generate 3–5 basic daily-life scenes that\n"
    "  an A1 student can imagine and talk about\n\n"
    "Example output:\n"
    "Topic: talking to yourself at home\n"
    "Scenes:\n"
    "- you talk before sleep\n"
    "- you say words in the mirror\n"
    "- you repeat things you need to do\n\n"
    "Output only the simplified topic and scenes.\n"
    "Nothing else.\n\n"
    'Return one JSON object with keys "topic" (string) and "scenes" (array of 3 to 5 strings). '
    "Each scene is one short English line. No extra keys."
)

_PREVIEW_SYSTEM_QUESTIONS = (
    "You are a helpful teacher. Output ONE JSON object only, no markdown.\n"
    "Use the transcript below ONLY as the source (no invented facts).\n"
    'Return ONLY these keys for a speaking / discussion preview:\n'
    '- "topic": one short line (teacher-friendly)\n'
    '- "discussion_questions": 3 to 5 short discussion questions grounded in the source\n'
    "Do not include key_ideas, words, warmup_questions, vocabulary_items, or grammar_patterns."
)

_PREVIEW_SYSTEM_VOCABULARY = (
    "You are a helpful teacher. Output ONE JSON object only, no markdown.\n"
    "Use the transcript below ONLY as the source (no invented facts).\n"
    'Return ONLY these keys for a vocabulary preview:\n'
    '- "topic": one short line (teacher-friendly)\n'
    '- "vocabulary_items": 8 to 10 items. Each item is an object with:\n'
    '  "english": word or short chunk from the source, and "note": short Ukrainian or English meaning/gloss\n'
    "Do not include key_ideas, warmup_questions, discussion_questions, or grammar_patterns."
)

_PREVIEW_SYSTEM_PHRASES = (
    "You are a helpful teacher. Output ONE JSON object only, no markdown.\n"
    "Use the transcript below ONLY as the source (no invented facts).\n"
    'Return ONLY these keys for a grammar / phrases preview:\n'
    '- "topic": one short line (teacher-friendly)\n'
    '- "grammar_patterns": 2 to 3 objects, each with:\n'
    '  "structure": short name of the pattern, and "formula": short formula (e.g. "used to + verb")\n'
    "Do not include key_ideas, words, vocabulary_items, or discussion_questions."
)

_PREVIEW_INSTR_EASY = (
    "Make the material easier. Simplify vocabulary and ideas. "
    "Keep source meaning unchanged."
)

_PREVIEW_INSTR_DEEP = (
    "Deepen the material. Give stronger ideas and richer vocabulary. "
    "Stay within source content only. Do not invent new topics."
)

_PREVIEW_PATCH_SYSTEM = (
    "You are a helpful teacher. Output ONE JSON object only, no markdown.\n"
    "The user message includes Original transcript + Current preview (complete JSON). "
    "You MUST PATCH that JSON: start from it, do NOT regenerate the preview from the transcript alone. "
    "Output the SAME keys and structure as the current preview (format-specific). "
    "Do not add keys from other formats. Ground new facts only in the transcript. "
    "Preserve every block unless the teacher explicitly asks to change it; default = minimal edit. "
    "For [Простіше]/[Глибше] the output MUST differ measurably from the current preview JSON "
    "(simpler or richer wording), not a verbatim copy."
)

_MEMORY_PATCH_RULES = """MEMORY RULES (never violate):
1. Do not change topic unless explicitly asked
2. Do not change level unless explicitly asked
3. Do not change format unless explicitly asked
4. Do not delete existing questions, words, or patterns
5. If teacher asks to ADD, only add new content, never replace old content
6. Always work on top of current preview
7. Return full result = existing content + additions
8. Do not rewrite, paraphrase, simplify, reorder, or modify existing items unless explicitly asked
9. Never add duplicates or near-duplicates of existing questions, words, or patterns
10. If teacher asks to simplify, keep the same topic and structure, but make the language easier
11. If teacher asks to deepen, keep the same topic and structure, but expand with more depth and new content
"""


def _frozen_base_snapshot_for_patch(
    patch_kind: str, pd: dict[str, Any]
) -> tuple[Any, Any, Any, Any]:
    topic = pd.get("topic", "")
    if patch_kind == "lesson":
        questions: Any = {
            "warmup_questions": pd.get("warmup_questions", []),
            "core_questions": pd.get("core_questions", []),
            "choices": pd.get("choices", []),
        }
        words = pd.get("support_words", [])
        patterns: Any = []
    elif patch_kind == "questions":
        questions = pd.get("discussion_questions", [])
        words = []
        patterns = []
    elif patch_kind == "vocabulary":
        questions = []
        words = pd.get("vocabulary_items", [])
        patterns = []
    elif patch_kind == "phrases":
        questions = []
        words = []
        patterns = pd.get("grammar_patterns", [])
    else:
        q_raw = pd.get("questions")
        if isinstance(q_raw, list) and q_raw:
            questions = q_raw
        else:
            questions = pd.get("key_ideas", [])
        words = pd.get("words", [])
        patterns = pd.get("exercises", [])
    return topic, questions, words, patterns


def _memory_frozen_teacher_section(
    patch_kind: str,
    preview_data: dict[str, Any],
    instruction: str,
) -> str:
    topic, questions, words, patterns = _frozen_base_snapshot_for_patch(
        patch_kind, preview_data
    )
    return (
        f"{_MEMORY_PATCH_RULES}\n"
        "Current preview (FROZEN BASE):\n"
        f"topic: {topic}\n"
        f"questions: {json.dumps(questions, ensure_ascii=False, default=str)}\n"
        f"words: {json.dumps(words, ensure_ascii=False, default=str)}\n"
        f"patterns: {json.dumps(patterns, ensure_ascii=False, default=str)}\n\n"
        f"Teacher instruction: {instruction}\n"
    )


_INTENT_BIAS_BY_KIND: dict[str, str] = {
    "lesson": (
        "VELIORA_ONBOARDING_INTENT_HINT: урок lesson warm up lead in розігрів розпочати навчання"
    ),
    "questions": (
        "VELIORA_ONBOARDING_INTENT_HINT: питання discussion обговорення запитання speaking діалог"
    ),
    "vocabulary": (
        "VELIORA_ONBOARDING_INTENT_HINT: слова vocabulary лексика переклад словниковий vocab лексичний"
    ),
    "phrases": (
        "VELIORA_ONBOARDING_INTENT_HINT: граматика grammar фрази phrases sentence pattern структура морфологія"
    ),
}


def _patch_hard_constraints_block(
    kind: str, level: Optional[str] = None
) -> str:
    if kind == "vocabulary":
        return (
            "HARD CONSTRAINTS (this format):\n"
            "- vocabulary_items: MUST contain 8–10 items (never fewer than 8). Each item: english + note (meaning).\n"
            '- Command "додай більше слів" / "більше слів": ADD new grounded items so the list reaches 9–10 entries; '
            "keep existing pairs; do not replace the whole list unless asked.\n"
        )
    if kind == "questions":
        return (
            "HARD CONSTRAINTS (this format):\n"
            "- discussion_questions: MUST contain 3–5 items.\n"
            '- Command "додай більше питань" / "більше питань": ADD questions until there are 4–5 total; '
            "keep existing questions unless asked to remove.\n"
        )
    if kind == "phrases":
        return (
            "HARD CONSTRAINTS (this format):\n"
            "- grammar_patterns: MUST contain 2–3 objects (structure + formula each).\n"
        )
    if kind == "lesson":
        if _is_lesson_cefr_a1(level):
            return (
                "HARD CONSTRAINTS (CEFR A1 lesson):\n"
                "- GENERAL: CEFR A1 vocabulary only; Present Simple only; max 6 words per sentence; "
                "one topic; no \"—\" or empty fields; do not regenerate existing blocks—only extend when asked; "
                "exact counts only.\n"
                "- warmup_questions: exactly 5 (each max 6 words; Do you / Is it / Can you; no Why).\n"
                "- core_questions: exactly 4 (each max 7 words; What/Where/When do you; no hypotheticals).\n"
                '- choices: exactly 4, format only "X or Y?" (concrete; X/Y one word or max 2-word phrase).\n'
                '- support_words: exactly 6, each "English — Ukrainian", nouns/verbs only, topic-related.\n'
            )
        if _is_lesson_cefr_a2(level):
            return (
                "HARD CONSTRAINTS (CEFR A2 lesson):\n"
                "- GENERAL: concrete, simple English; no \"—\" or empty fields; "
                "do not regenerate existing blocks—only extend when asked; exact counts only; "
                "every element tied to main situation and core meaning; no unrelated or generic filler.\n"
                "- warmup_questions: exactly 5; patterns only Do you / Is it / Can you; "
                "no \"Have you ever\"; personal and concrete.\n"
                "- core_questions: exactly 4; Why/How allowed if concrete and tied to source; "
                "no abstract reasoning or theory; simple cause-effect only.\n"
                '- choices: exactly 4, format only "X or Y?".\n'
                '- support_words: exactly 6, each "English — Ukrainian", topic-related, practical A2 level.\n'
            )
        return (
            "HARD CONSTRAINTS (this format):\n"
            "- warmup_questions: exactly 5 real strings (never \"—\" or empty).\n"
            "- choices: 3–4 real \"this or that\" strings (never \"—\" or empty).\n"
            "- support_words: at least 5 real strings from source or topic.\n"
        )
    return (
        "HARD CONSTRAINTS (default format):\n"
        "- key_ideas: exactly 3 strings; words: 3–15 strings as appropriate.\n"
    )


def _preview_patch_rules_easy(kind: str, level: Optional[str] = None) -> str:
    if _is_lesson_cefr_a1(level):
        lesson_easy = (
            "Apply: зроби простіше — simplify wording of warmup_questions, core_questions, choices, "
            "and support_words only; keep exactly 5 warm-ups, 4 core questions, 4 choices, 6 support lines; "
            "same topic, simpler A1 English and Ukrainian glosses.\n"
        )
    elif _is_lesson_cefr_a2(level):
        lesson_easy = (
            "Apply: зроби простіше — simplify wording of warmup_questions, core_questions, choices, "
            "and support_words only; keep exactly 5 warm-ups, 4 core questions, 4 choices, 6 support lines; "
            "same topic and source situation, simpler A2 English and Ukrainian glosses; "
            "keep warmup patterns Do you / Is it / Can you only; core Why/How only if still concrete and tied to source.\n"
        )
    else:
        lesson_easy = (
            "Apply: зроби простіше — simplify wording of warmup_questions, choices, and support_words only; "
            "keep exactly 5 warm-up questions, 3–4 choices, and at least 5 support words; same topics, simpler English.\n"
        )
    spec = {
        "lesson": lesson_easy,
        "questions": (
            "Apply: зроби простіше — simplify discussion_questions wording only; keep 3–5 questions.\n"
        ),
        "vocabulary": (
            "Apply: зроби простіше — simplify `note` (meaning) text only; keep 8–10 vocabulary_items; "
            "same english chunks unless simplification requires tiny edits.\n"
        ),
        "phrases": (
            "Apply: зроби простіше — simplify structure/formula explanations; keep 2–3 patterns.\n"
        ),
        "default": "Apply: зроби простіше — simplify key_ideas and words; keep counts.\n",
    }.get(kind, "Apply: зроби простіше — simplify key_ideas and words.\n")
    return (
        "Rules (button Простіше — PATCH only, NOT full regeneration):\n"
        "- Current preview (complete JSON) is the stable base; copy it forward then edit.\n"
        "- You MUST produce JSON that is not identical to the current preview (measurable simpler text).\n"
        "- Do not rebuild from transcript alone; do not drop unrelated blocks.\n"
        f"- {spec}"
        + _patch_hard_constraints_block(kind, level)
    )


def _preview_patch_rules_deep(kind: str, level: Optional[str] = None) -> str:
    if _is_lesson_cefr_a1(level):
        lesson_deep = (
            "Apply: зроби глибше (CEFR A1 lesson) — warmup_questions and core_questions: do NOT repeat or paraphrase "
            "the existing strings; write NEW questions only, each more specific to daily life, while keeping exactly "
            "5 warm-ups and 4 core questions. Example: \"Do you talk to yourself?\" → "
            "\"Do you talk to yourself at home or outside?\" "
            "For choices and support_words: use NEW concrete daily-life wording where you change them; "
            "keep exactly 4 choices and 6 support lines; stay in source and all A1 rules.\n"
        )
    elif _is_lesson_cefr_a2(level):
        lesson_deep = (
            "Apply: зроби глибше (CEFR A2 lesson) — enrich warmup_questions, core_questions, choices, support_words; "
            "stay in the same source situation and core meaning; keep exactly 5 warm-ups, 4 core questions, "
            "4 choices, 6 support lines; add concrete detail and simple cause-effect where it helps; "
            "do NOT repeat or lightly paraphrase—make wording meaningfully new; no abstract theory; all A2 rules.\n"
        )
    else:
        lesson_deep = (
            "Apply: зроби глибше — enrich warmup_questions, choices, and support_words; stay in source; "
            "keep 5 warm-ups, 3–4 choices, at least 5 support words.\n"
        )
    spec = {
        "lesson": lesson_deep,
        "questions": (
            "Apply: зроби глибше — richer discussion_questions; stay in source; keep 3–5 items.\n"
        ),
        "vocabulary": (
            "Apply: зроби глибше — richer `note` (meanings) or slightly more precise english; "
            "keep 8–10 vocabulary_items.\n"
        ),
        "phrases": (
            "Apply: зроби глибше — sharper structure names/formulas; stay in source; keep 2–3 patterns.\n"
        ),
        "default": "Apply: зроби глибше — enrich key_ideas and words; stay in source.\n",
    }.get(kind, "Apply: зроби глибше — enrich key_ideas and words; stay in source.\n")
    return (
        "Rules (button Глибше — PATCH only, NOT full regeneration):\n"
        "- Current preview (complete JSON) is the stable base; copy it forward then edit.\n"
        "- You MUST produce JSON that is not identical to the current preview (measurable richer detail).\n"
        "- Do not invent topics outside the transcript; do not drop unrelated blocks.\n"
        f"- {spec}"
        + _patch_hard_constraints_block(kind, level)
    )


def _preview_patch_rules_custom(kind: str, level: Optional[str] = None) -> str:
    common = (
        "Rules (custom teacher text — PATCH only):\n"
        "- Current preview (complete JSON) is the ONLY stable base; merge changes into it.\n"
        "- Never return identical JSON when the teacher asked for a change.\n"
        "- Preserve all blocks unless the teacher explicitly asks to remove or replace.\n"
        "- Default: change only the minimal block the instruction targets.\n"
        '- "зроби простіше" / simpler: simplify only the targeted block.\n'
        '- "зроби глибше" / deeper: enrich only the targeted block; stay in source.\n'
    )
    if kind == "vocabulary":
        return (
            common
            + _patch_hard_constraints_block(kind, level)
            + (
                '- "додай більше слів": extend vocabulary_items to 9–10 grounded entries (not a rewrite with the same count).\n'
            )
        )
    if kind == "questions":
        return (
            common
            + _patch_hard_constraints_block(kind, level)
            + (
                '- "додай більше питань": extend discussion_questions to 4–5 grounded questions.\n'
            )
        )
    if kind == "lesson":
        return common + _patch_hard_constraints_block(kind, level)
    if kind == "phrases":
        return common + _patch_hard_constraints_block(kind, level)
    return (
        common
        + _patch_hard_constraints_block(kind, level)
        + (
            '- "більше слів": extend `words` with NEW items.\n'
            '- "більше питань": extend `questions` if present, else adjust key_ideas.\n'
            '- "більше ідей": extend or enrich `key_ideas`.\n'
            '- "більше вправ": extend `exercises` if present.\n'
        )
    )

_PREVIEW_LIMIT_TEXT = "Давай підтвердимо або почнемо з нового 👇"

_MAX_PREVIEW_EDIT_ROUNDS = 5

_ONB_FMT_STEP1_KB = InlineKeyboardMarkup(
    [
        [
            InlineKeyboardButton("📚 Урок", callback_data="onb_fmt_lesson"),
            InlineKeyboardButton("💬 Speaking", callback_data="onb_fmt_questions"),
        ],
        [
            InlineKeyboardButton("📖 Слова", callback_data="onb_fmt_vocabulary"),
            InlineKeyboardButton("✏️ Граматика", callback_data="onb_fmt_phrases"),
        ],
    ]
)

_ONB_LEVEL_KB = InlineKeyboardMarkup(
    [
        [
            InlineKeyboardButton("A1", callback_data="onb_lvl_A1"),
            InlineKeyboardButton("A2", callback_data="onb_lvl_A2"),
            InlineKeyboardButton("B1", callback_data="onb_lvl_B1"),
            InlineKeyboardButton("B2", callback_data="onb_lvl_B2"),
        ],
    ]
)

# Post-card actions (prefix onb_p_ — matches CallbackQueryHandler ^onb_ in main).
_POST_CARD_KB = InlineKeyboardMarkup(
    [
        [
            InlineKeyboardButton("🔄 Змінити формат", callback_data="onb_p_fmt"),
            InlineKeyboardButton("📊 Змінити рівень", callback_data="onb_p_lvl"),
        ],
    ]
)

_POST_CARD_FMT_KB = InlineKeyboardMarkup(
    [
        [
            InlineKeyboardButton("📚 Урок", callback_data="onb_p_f_lesson"),
            InlineKeyboardButton("💬 Speaking", callback_data="onb_p_f_questions"),
        ],
        [
            InlineKeyboardButton("📖 Слова", callback_data="onb_p_f_vocabulary"),
            InlineKeyboardButton("✏️ Граматика", callback_data="onb_p_f_phrases"),
        ],
    ]
)

_POST_CARD_LVL_KB = InlineKeyboardMarkup(
    [
        [
            InlineKeyboardButton("A1", callback_data="onb_p_l_A1"),
            InlineKeyboardButton("A2", callback_data="onb_p_l_A2"),
            InlineKeyboardButton("B1", callback_data="onb_p_l_B1"),
            InlineKeyboardButton("B2", callback_data="onb_p_l_B2"),
        ],
    ]
)

_PREVIEW_KB = InlineKeyboardMarkup(
    [
        [
            InlineKeyboardButton("✅ Все ок", callback_data="onb_prv_ok"),
            InlineKeyboardButton("✏️ Уточнити", callback_data="onb_prv_ref"),
        ],
    ]
)

_PREVIEW_REFINE_KB = InlineKeyboardMarkup(
    [
        [
            InlineKeyboardButton("📉 Простіше", callback_data="onb_prv_r_easy"),
            InlineKeyboardButton("📚 Глибше", callback_data="onb_prv_r_deep"),
            InlineKeyboardButton("✍️ Своє", callback_data="onb_prv_r_own"),
        ],
    ]
)

_PREVIEW_LIMIT_KB = InlineKeyboardMarkup(
    [
        [
            InlineKeyboardButton("✅ Все ок", callback_data="onb_prv_ok"),
            InlineKeyboardButton("🔄 Нове джерело", callback_data="onb_prv_new"),
        ],
    ]
)

_FMT_CHANGE_LABELS = {
    "lesson": "📚 Урок",
    "questions": "💬 Speaking",
    "vocabulary": "📖 Слова",
    "phrases": "✏️ Граматика",
}


def _is_lesson_cefr_a1(level: Optional[str]) -> bool:
    if level is None:
        return False
    return str(level).strip().upper() == "A1"


def _is_lesson_cefr_a2(level: Optional[str]) -> bool:
    if level is None:
        return False
    return str(level).strip().upper() == "A2"


def _preview_format_kind(fmt: Optional[str]) -> str:
    if not fmt:
        return "default"
    f = str(fmt).strip().lower()
    if f == "lesson":
        return "lesson"
    if f in ("speaking", "questions"):
        return "questions"
    if f in ("vocabulary", "words"):
        return "vocabulary"
    if f in ("grammar", "phrases"):
        return "phrases"
    return "default"


def _preview_system_for_initial(kind: str, level: Optional[str] = None) -> str:
    if kind == "lesson" and _is_lesson_cefr_a1(level):
        return _PREVIEW_SYSTEM_LESSON_A1
    if kind == "lesson" and _is_lesson_cefr_a2(level):
        return _PREVIEW_SYSTEM_LESSON_A2
    return {
        "lesson": _PREVIEW_SYSTEM_LESSON,
        "questions": _PREVIEW_SYSTEM_QUESTIONS,
        "vocabulary": _PREVIEW_SYSTEM_VOCABULARY,
        "phrases": _PREVIEW_SYSTEM_PHRASES,
        "default": _PREVIEW_SYSTEM_DEFAULT,
    }.get(kind, _PREVIEW_SYSTEM_DEFAULT)


def _preview_merge_list_keys(
    kind: str, level: Optional[str] = None
) -> tuple[str, ...]:
    if kind == "lesson" and (
        _is_lesson_cefr_a1(level) or _is_lesson_cefr_a2(level)
    ):
        return ("warmup_questions", "core_questions", "support_words", "choices")
    if kind == "lesson":
        return ("warmup_questions", "support_words", "choices")
    return {
        "questions": ("discussion_questions",),
        "vocabulary": ("vocabulary_items",),
        "phrases": ("grammar_patterns",),
        "default": ("questions", "exercises"),
    }.get(kind, ("questions", "exercises"))


def _coerce_vocabulary_items(raw: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(raw, list):
        return []
    for x in raw:
        if isinstance(x, dict):
            en = str(x.get("english") or x.get("en") or "").strip()
            note = str(
                x.get("note") or x.get("ua") or x.get("meaning") or x.get("gloss") or ""
            ).strip()
            if en:
                out.append(f"{en} — {note}" if note else en)
        else:
            s = str(x).strip()
            if s:
                out.append(s)
    return out[:10]


def _coerce_grammar_patterns(raw: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return []
    for x in raw:
        if isinstance(x, dict):
            st = str(
                x.get("structure") or x.get("pattern") or x.get("name") or ""
            ).strip()
            fm = str(x.get("formula") or "").strip()
            if st or fm:
                out.append({"structure": st or "—", "formula": fm})
        else:
            s = str(x).strip()
            if s:
                out.append({"structure": s, "formula": ""})
    return out[:3]


def _lesson_nonempty_strings(items: Any, max_n: int) -> list[str]:
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for x in items:
        s = str(x).strip()
        if s and s != "—":
            out.append(s)
    return out[:max_n]


_A1_FILTER_FALLBACK_TOPIC = "everyday life at home and school"
_A1_FILTER_FALLBACK_SCENES: tuple[str, ...] = (
    "you wake up in the morning",
    "you eat at home",
    "you go to school or work",
    "you talk with family",
    "you sleep at night",
)


def _coerce_a1_filter_output(data: dict[str, Any]) -> tuple[str, list[str]]:
    topic = str(data.get("topic") or "").strip()
    raw_scenes = data.get("scenes")
    scenes: list[str] = []
    if isinstance(raw_scenes, list):
        for x in raw_scenes:
            s = str(x).strip()
            if s:
                scenes.append(s)
    return topic, scenes[:5]


def _a1_filtered_source_user_block(topic: str, scenes: list[str]) -> str:
    lines = [
        "A1 FILTERED SOURCE — use ONLY this text as your source for the lesson preview.",
        "Do not use or rely on any raw transcript.",
        "",
        f"Topic: {topic}",
        "",
        "Scenes:",
    ]
    for s in scenes:
        lines.append(f"- {s}")
    return "\n".join(lines)


def _a1_resolved_filtered_block(topic: str, scenes: list[str]) -> str:
    if not topic:
        return _a1_filtered_source_user_block(
            _A1_FILTER_FALLBACK_TOPIC, list(_A1_FILTER_FALLBACK_SCENES)
        )
    if len(scenes) < 3:
        seen = {s.lower() for s in scenes}
        for s in _A1_FILTER_FALLBACK_SCENES:
            if len(scenes) >= 3:
                break
            if s.lower() not in seen:
                scenes.append(s)
                seen.add(s.lower())
        if len(scenes) < 3:
            return _a1_filtered_source_user_block(
                _A1_FILTER_FALLBACK_TOPIC, list(_A1_FILTER_FALLBACK_SCENES)
            )
    return _a1_filtered_source_user_block(topic, scenes[:5])


_A2_FILTER_FALLBACK_TOPIC = _A1_FILTER_FALLBACK_TOPIC
_A2_FILTER_FALLBACK_SCENES = _A1_FILTER_FALLBACK_SCENES

_coerce_a2_filter_output = _coerce_a1_filter_output


def _a2_filtered_source_user_block(topic: str, scenes: list[str]) -> str:
    lines = [
        "A2 FILTERED SOURCE — use ONLY this text as your source for the lesson preview.",
        "Do not use or rely on any raw transcript.",
        "",
        f"Topic: {topic}",
        "",
        "Scenes:",
    ]
    for s in scenes:
        lines.append(f"- {s}")
    return "\n".join(lines)


def _a2_resolved_filtered_block(topic: str, scenes: list[str]) -> str:
    if not topic:
        return _a2_filtered_source_user_block(
            _A2_FILTER_FALLBACK_TOPIC, list(_A2_FILTER_FALLBACK_SCENES)
        )
    if len(scenes) < 3:
        seen = {s.lower() for s in scenes}
        for s in _A2_FILTER_FALLBACK_SCENES:
            if len(scenes) >= 3:
                break
            if s.lower() not in seen:
                scenes.append(s)
                seen.add(s.lower())
        if len(scenes) < 3:
            return _a2_filtered_source_user_block(
                _A2_FILTER_FALLBACK_TOPIC, list(_A2_FILTER_FALLBACK_SCENES)
            )
    return _a2_filtered_source_user_block(topic, scenes[:5])


class _OnboardingEnrichedMessage:
    """Proxy so pipeline sees enriched text/caption without mutating the real Message."""

    __slots__ = ("_base", "_enriched")

    def __init__(self, base: Message, enriched: str) -> None:
        self._base = base
        self._enriched = enriched

    @property
    def text(self) -> str:
        return self._enriched

    @property
    def caption(self) -> Optional[str]:
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def _normalize_preview_output(
    data: dict[str, Any], kind: str, level: Optional[str] = None
) -> dict[str, Any]:
    topic = str(data.get("topic", "") or "").strip() or "—"

    if kind == "lesson":
        if _is_lesson_cefr_a1(level) or _is_lesson_cefr_a2(level):
            wq = _lesson_nonempty_strings(data.get("warmup_questions"), 5)
            cq = _lesson_nonempty_strings(data.get("core_questions"), 4)
            ch = _lesson_nonempty_strings(data.get("choices"), 4)
            sw = _lesson_nonempty_strings(data.get("support_words"), 6)
            return {
                "topic": topic,
                "warmup_questions": wq,
                "core_questions": cq,
                "choices": ch,
                "support_words": sw,
            }
        wq = _lesson_nonempty_strings(data.get("warmup_questions"), 5)
        ch = _lesson_nonempty_strings(data.get("choices"), 4)
        sw = _lesson_nonempty_strings(data.get("support_words"), 15)
        return {
            "topic": topic,
            "warmup_questions": wq,
            "choices": ch,
            "support_words": sw,
        }

    if kind == "questions":
        dq = data.get("discussion_questions")
        if not isinstance(dq, list):
            dq = []
        dq = [str(x).strip() for x in dq if str(x).strip()][:5]
        while len(dq) < 3:
            dq.append("—")
        return {"topic": topic, "discussion_questions": dq}

    if kind == "vocabulary":
        items = _coerce_vocabulary_items(data.get("vocabulary_items"))
        while len(items) < 8:
            items.append("—")
        return {"topic": topic, "vocabulary_items": items[:10]}

    if kind == "phrases":
        gp = _coerce_grammar_patterns(data.get("grammar_patterns"))
        while len(gp) < 2:
            gp.append({"structure": "—", "formula": ""})
        return {"topic": topic, "grammar_patterns": gp[:3]}

    key_ideas = data.get("key_ideas")
    words = data.get("words")
    if not isinstance(key_ideas, list):
        key_ideas = []
    if not isinstance(words, list):
        words = []
    ki = [str(x).strip() for x in key_ideas if str(x).strip()][:6]
    while len(ki) < 3:
        ki.append("—")
    wd = [str(x).strip() for x in words if str(x).strip()][:15]
    out: dict[str, Any] = {
        "topic": topic,
        "key_ideas": ki,
        "words": wd,
    }
    for extra_key in ("questions", "exercises"):
        extra_val = data.get(extra_key)
        if isinstance(extra_val, list) and extra_val:
            out[extra_key] = [
                str(x).strip() for x in extra_val if str(x).strip()
            ][:25]
    return out


def _normalize_gpt_preview_dict(data: dict[str, Any]) -> dict[str, Any]:
    return _normalize_preview_output(data, "default")


def _enriched_onboarding_transcript_block(
    fmt: Optional[str],
    lvl: Optional[str],
    transcript: str,
) -> str:
    kind = _preview_format_kind(fmt)
    bias = _INTENT_BIAS_BY_KIND.get(kind, "")
    parts = [f"[FORMAT={fmt}]", f"[LEVEL={lvl}]"]
    if bias:
        parts.append(bias)
    parts.extend(["", f"USER CONTENT:\n{transcript}"])
    return "\n".join(parts)


def _preview_blocks_for_prompt(preview_data: dict[str, Any]) -> tuple[str, str, str]:
    topic = str(preview_data.get("topic") or "—").strip() or "—"
    ideas = preview_data.get("key_ideas")
    if not isinstance(ideas, list):
        ideas = []
    ideas = [str(x).strip() for x in ideas if str(x).strip()]
    while len(ideas) < 3:
        ideas.append("—")
    ideas = ideas[:6]
    ideas_str = " | ".join(ideas)
    words = preview_data.get("words")
    if not isinstance(words, list):
        words = []
    words = [str(x).strip() for x in words if str(x).strip()]
    words_str = ", ".join(words) if words else "—"
    return topic, ideas_str, words_str


def _build_preview_patch_user_content(
    transcript_snippet: str,
    preview_data: dict[str, Any],
    teacher_text: str,
    rules_block: str,
    patch_kind: str,
) -> str:
    pd = preview_data if isinstance(preview_data, dict) else {}
    topic, ideas_str, words_str = _preview_blocks_for_prompt(pd)
    preview_json = json.dumps(pd, ensure_ascii=False, default=str)
    memory_frozen = _memory_frozen_teacher_section(
        patch_kind, pd, teacher_text.strip()
    )
    return (
        f"PATCH_FORMAT_KIND: {patch_kind}\n"
        "You are editing the existing preview JSON below — do NOT rebuild preview from transcript only.\n\n"
        f"Original transcript (source of truth for new facts):\n{transcript_snippet}\n\n"
        "Current preview (human summary — JSON below is authoritative):\n"
        f"TOPIC: {topic}\n"
        f"IDEAS: {ideas_str}\n"
        f"WORDS: {words_str}\n\n"
        f"Current preview (complete JSON, all fields — stable base):\n{preview_json}\n\n"
        f"{memory_frozen}\n"
        f"{rules_block}"
    )


def _format_preview_message(
    preview_data: dict[str, Any],
    format_key: Optional[str] = None,
) -> str:
    kind = _preview_format_kind(format_key)
    topic = str(preview_data.get("topic") or "—").strip() or "—"
    header = "📋 Ось що знайшов:\n\n" + f"📌 Тема: {topic}\n\n"

    if kind == "lesson":
        wq = _lesson_nonempty_strings(preview_data.get("warmup_questions"), 10)
        cq = _lesson_nonempty_strings(preview_data.get("core_questions"), 10)
        ch = _lesson_nonempty_strings(preview_data.get("choices"), 10)
        sw = _lesson_nonempty_strings(preview_data.get("support_words"), 20)
        body = "🔥 Розминка:\n" + "\n".join(f"• {x}" for x in wq) + "\n\n"
        if cq:
            body += "❓ Основні питання:\n" + "\n".join(f"• {x}" for x in cq) + "\n\n"
        if ch:
            body += "⚖️ This or that:\n" + "\n".join(f"• {x}" for x in ch) + "\n\n"
        body += "📚 Слова: " + (", ".join(sw) if sw else "—")
        return header + body

    if kind == "questions":
        dq = preview_data.get("discussion_questions")
        if not isinstance(dq, list):
            dq = []
        dq = [str(x).strip() for x in dq if str(x).strip()]
        body = "💬 Питання для обговорення:\n" + "\n".join(f"• {x}" for x in dq)
        return header + body

    if kind == "vocabulary":
        items = preview_data.get("vocabulary_items")
        if not isinstance(items, list):
            items = []
        lines = [str(x).strip() for x in items if str(x).strip()]
        body = "📖 Ключові слова:\n" + "\n".join(f"• {x}" for x in lines)
        return header + body

    if kind == "phrases":
        gp = preview_data.get("grammar_patterns")
        if not isinstance(gp, list):
            gp = []
        parts: list[str] = []
        for x in gp:
            if isinstance(x, dict):
                st = str(x.get("structure") or "—").strip()
                fm = str(x.get("formula") or "").strip()
                parts.append(f"• {st}" + (f": {fm}" if fm else ""))
            else:
                s = str(x).strip()
                if s:
                    parts.append(f"• {s}")
        body = "✏️ Граматика / структури:\n" + "\n".join(parts)
        return header + body

    ideas = preview_data.get("key_ideas")
    if not isinstance(ideas, list):
        ideas = []
    ideas = [str(x).strip() for x in ideas if str(x).strip()]
    while len(ideas) < 3:
        ideas.append("—")
    ideas = ideas[:6]
    words = preview_data.get("words")
    if not isinstance(words, list):
        words = []
    words = [str(x).strip() for x in words if str(x).strip()][:15]
    words_str = ", ".join(words) if words else "—"
    lines = [
        header,
        "💡 Ідеї:\n",
        "\n".join(f"• {x}" for x in ideas) + "\n\n",
        f"📚 Слова:\n{words_str}",
    ]
    qn = preview_data.get("questions")
    if isinstance(qn, list) and qn:
        lines.append(
            "\n\n❓ Питання:\n"
            + "\n".join(f"• {str(x).strip()}" for x in qn if str(x).strip())
        )
    ex = preview_data.get("exercises")
    if isinstance(ex, list) and ex:
        lines.append(
            "\n\n🏋 Вправи:\n"
            + "\n".join(f"• {str(x).strip()}" for x in ex if str(x).strip())
        )
    return "".join(lines)


class MessageHandlerService:
    def __init__(
        self,
        pipeline: ContentPipelineService,
        deduplicator: MessageDeduplicator,
        youtube_service: YouTubeTranscriptService,
        openai_client: AsyncOpenAI,
        openai_model: str,
    ) -> None:
        self._pipeline = pipeline
        self._deduplicator = deduplicator
        self._youtube_service = youtube_service
        self._openai_client = openai_client
        self._openai_model = openai_model

    async def _call_a1_filter_gpt(self, transcript_snippet: str) -> str:
        user_content = (
            f"{_PREVIEW_A1_FILTER_USER}\n\n---\n\nTranscript to filter:\n{transcript_snippet}"
        )
        response = await self._openai_client.chat.completions.create(
            model=self._openai_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _PREVIEW_A1_FILTER_SYSTEM},
                {"role": "user", "content": user_content},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        topic, scenes = _coerce_a1_filter_output(data)
        return _a1_resolved_filtered_block(topic, scenes)

    async def _call_a2_filter_gpt(self, transcript_snippet: str) -> str:
        user_content = (
            f"{_PREVIEW_A2_FILTER_USER}\n\n---\n\nTranscript to filter:\n{transcript_snippet}"
        )
        response = await self._openai_client.chat.completions.create(
            model=self._openai_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _PREVIEW_A2_FILTER_SYSTEM},
                {"role": "user", "content": user_content},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        topic, scenes = _coerce_a2_filter_output(data)
        return _a2_resolved_filtered_block(topic, scenes)

    async def _call_preview_gpt(
        self,
        transcript: str,
        format_key: Optional[str] = None,
        extra_instruction: Optional[str] = None,
        level: Optional[str] = None,
    ) -> dict[str, Any]:
        kind = _preview_format_kind(format_key)
        system = _preview_system_for_initial(kind, level)
        snippet = transcript.strip()
        if len(snippet) > _PREVIEW_TRANSCRIPT_MAX:
            snippet = snippet[:_PREVIEW_TRANSCRIPT_MAX]
        if kind == "lesson" and _is_lesson_cefr_a1(level):
            user_block = await self._call_a1_filter_gpt(snippet)
        elif kind == "lesson" and _is_lesson_cefr_a2(level):
            user_block = await self._call_a2_filter_gpt(snippet)
        else:
            user_block = f"Transcript:\n{snippet}"
        if extra_instruction and extra_instruction.strip():
            user_block += f"\n\nAdditional instruction:\n{extra_instruction.strip()}"
        response = await self._openai_client.chat.completions.create(
            model=self._openai_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_block},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        if not isinstance(data, dict):
            data = {}
        return _normalize_preview_output(data, kind, level)

    async def _call_preview_patch_gpt(
        self,
        transcript: str,
        preview_data: dict[str, Any],
        teacher_text: str,
        *,
        refine_mode: str = "easy",
        custom_correction: bool = False,
        preview_format: Optional[str] = None,
        preview_level: Optional[str] = None,
    ) -> dict[str, Any]:
        snippet = transcript.strip()
        if len(snippet) > _PREVIEW_TRANSCRIPT_MAX:
            snippet = snippet[:_PREVIEW_TRANSCRIPT_MAX]
        pd_in = preview_data if isinstance(preview_data, dict) else {}
        patch_kind = _preview_format_kind(preview_format)
        if custom_correction:
            rules_block = _preview_patch_rules_custom(patch_kind, preview_level)
        elif refine_mode == "deep":
            rules_block = _preview_patch_rules_deep(patch_kind, preview_level)
        else:
            rules_block = _preview_patch_rules_easy(patch_kind, preview_level)
        LOGGER.info(
            "preview_patch_gpt kind=%s refine_mode=%s custom=%s transcript_len=%s "
            "preview_data=%s teacher_instruction=%s",
            patch_kind,
            "custom" if custom_correction else refine_mode,
            custom_correction,
            len(snippet),
            json.dumps(pd_in, ensure_ascii=False, default=str),
            teacher_text.strip()[:2000],
        )
        user_content = _build_preview_patch_user_content(
            snippet,
            pd_in,
            teacher_text.strip(),
            rules_block,
            patch_kind,
        )
        response = await self._openai_client.chat.completions.create(
            model=self._openai_model,
            response_format={"type": "json_object"},
            temperature=0.75,
            messages=[
                {"role": "system", "content": _PREVIEW_PATCH_SYSTEM},
                {"role": "user", "content": user_content},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        if not isinstance(data, dict):
            data = {}
        normalized = _normalize_preview_output(data, patch_kind, preview_level)
        if custom_correction:
            for k in _preview_merge_list_keys(patch_kind, preview_level):
                if k in normalized:
                    continue
                prev = pd_in.get(k)
                if isinstance(prev, list) and prev:
                    if k == "grammar_patterns":
                        normalized[k] = _coerce_grammar_patterns(prev)
                    elif k == "vocabulary_items":
                        normalized[k] = _coerce_vocabulary_items(prev)
                    else:
                        normalized[k] = [
                            str(x).strip()
                            for x in prev
                            if str(x).strip() and str(x).strip() != "—"
                        ][:25]
        return normalized

    def _guided_ready(self, chat_id: int) -> bool:
        st = user_state.get(chat_id)
        return bool(st and st.get("format") and st.get("level"))

    def _preview_state_bootstrap(self) -> dict[str, Any]:
        return {
            "transcript": None,
            "format": None,
            "level": None,
            "preview_data": {},
            "generating": False,
            "preview_message_id": None,
            "awaiting_edit": False,
            "edit_rounds": 0,
            "limit_reached": False,
            "confirmed": False,
        }

    async def _edit_or_reply_preview(
        self,
        bot: Any,
        chat_id: int,
        prv: dict[str, Any],
        anchor_message: Message,
        text: str,
        reply_markup: InlineKeyboardMarkup,
    ) -> None:
        mid = prv.get("preview_message_id")
        try:
            if mid:
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=int(mid),
                    text=text,
                    reply_markup=reply_markup,
                )
                return
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Preview edit failed, sending new message: %s", exc)
        sent = await anchor_message.reply_text(text, reply_markup=reply_markup)
        prv["preview_message_id"] = sent.message_id

    async def _after_refine_increment(
        self,
        bot: Any,
        chat_id: int,
        prv: dict[str, Any],
        anchor_message: Message,
        preview_data: dict[str, Any],
    ) -> None:
        prv["preview_data"] = preview_data
        prv["edit_rounds"] = int(prv.get("edit_rounds") or 0) + 1
        prv["awaiting_edit"] = False
        prv["limit_reached"] = False
        if prv["edit_rounds"] >= _MAX_PREVIEW_EDIT_ROUNDS:
            prv["limit_reached"] = True
            await self._edit_or_reply_preview(
                bot,
                chat_id,
                prv,
                anchor_message,
                _PREVIEW_LIMIT_TEXT,
                _PREVIEW_LIMIT_KB,
            )
        else:
            await self._edit_or_reply_preview(
                bot,
                chat_id,
                prv,
                anchor_message,
                _format_preview_message(preview_data, prv.get("format")),
                _PREVIEW_KB,
            )

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        chat_id = update.message.chat_id
        user_state.pop(chat_id, None)
        preview_state.pop(chat_id, None)
        await update.message.reply_text(
            "👋 Привіт! Я Veliora 🎓\n\n"
            "Допоможу швидко підготувати матеріал для уроку англійської.\n\n"
            "Що хочеш отримати?",
            reply_markup=_ONB_FMT_STEP1_KB,
        )

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if not query or not query.message:
            return
        await query.answer()
        chat_id = query.message.chat_id
        data = (query.data or "").strip()

        if data == "onb_prv_ok":
            prv = preview_state.get(chat_id)
            if not prv or not prv.get("transcript"):
                await query.message.reply_text(
                    "Не знайшов збережене джерело. Надішли посилання на YouTube ще раз."
                )
                return
            if prv.get("generating"):
                return
            prv["generating"] = True
            prv["confirmed"] = True
            st = user_state.get(chat_id)
            fmt = (st or {}).get("format") or prv.get("format")
            lvl = (st or {}).get("level") or prv.get("level")
            transcript = str(prv.get("transcript") or "").strip()
            preview_data = prv.get("preview_data") or {}
            kind = _preview_format_kind(fmt)

            if kind == "lesson":
                cq = preview_data.get("core_questions")
                approved_block = (
                    f"APPROVED PREVIEW:\n"
                    f"TOPIC: {preview_data.get('topic', '')}\n"
                    f"WARMUP QUESTIONS: {preview_data.get('warmup_questions', [])}\n"
                )
                if isinstance(cq, list) and any(
                    str(x).strip() and str(x).strip() != "—" for x in cq
                ):
                    approved_block += f"CORE QUESTIONS: {cq}\n"
                approved_block += (
                    f"CHOICES: {preview_data.get('choices', [])}\n"
                    f"SUPPORT WORDS: {preview_data.get('support_words', [])}\n"
                )
            elif kind == "vocabulary":
                approved_block = (
                    f"APPROVED PREVIEW:\n"
                    f"TOPIC: {preview_data.get('topic', '')}\n"
                    f"VOCABULARY: {preview_data.get('vocabulary_items', [])}\n"
                )
            elif kind == "phrases":
                approved_block = (
                    f"APPROVED PREVIEW:\n"
                    f"TOPIC: {preview_data.get('topic', '')}\n"
                    f"PATTERNS: {preview_data.get('grammar_patterns', [])}\n"
                )
            elif kind == "questions":
                approved_block = (
                    f"APPROVED PREVIEW:\n"
                    f"TOPIC: {preview_data.get('topic', '')}\n"
                    f"DISCUSSION QUESTIONS: {preview_data.get('discussion_questions', [])}\n"
                )
            else:
                approved_block = (
                    f"APPROVED PREVIEW:\n"
                    f"TOPIC: {preview_data.get('topic', '')}\n"
                    f"IDEAS: {preview_data.get('key_ideas', [])}\n"
                    f"WORDS: {preview_data.get('words', [])}\n"
                )

            structured = (
                f"[FORMAT={fmt}]\n[LEVEL={lvl}]\n\n"
                f"{approved_block}\n"
                f"SOURCE TRANSCRIPT (reference only):\n"
                f"{transcript}"
            )
            proxy = _OnboardingEnrichedMessage(query.message, structured)
            try:
                prepare = await self._pipeline.prepare(context.bot, proxy, chat_id)
                if prepare.preface:
                    await query.message.reply_text("Вже готую твій матеріал ✨")
                elif prepare.status_line:
                    await query.message.reply_text(prepare.status_line)
                result = await self._pipeline.execute(prepare)
            except TranscriptUnavailableError as err:
                prv["generating"] = False
                await query.message.reply_text(err.user_message)
                return
            except GenerationFailedError as err:
                prv["generating"] = False
                await query.message.reply_text(err.user_message)
                return
            except Exception as error:  # noqa: BLE001
                LOGGER.exception("Guided confirm pipeline failed: %s", error)
                prv["generating"] = False
                await query.message.reply_text("Не вдалося згенерувати картку. Спробуй ще раз.")
                return
            preview_state.pop(chat_id, None)
            await self._send_pipeline_result(query.message, result)
            return

        if data == "onb_prv_new":
            preview_state.pop(chat_id, None)
            await query.message.reply_text("Надішли посилання на YouTube ще раз 👇")
            try:
                await query.edit_message_reply_markup(reply_markup=None)
            except Exception:  # noqa: BLE001
                pass
            return

        if data == "onb_prv_ref":
            prv = preview_state.get(chat_id)
            if not prv or not prv.get("transcript"):
                await query.message.reply_text(
                    "Не знайшов збережене джерело. Надішли посилання на YouTube ще раз."
                )
                return
            if prv.get("limit_reached") or int(prv.get("edit_rounds") or 0) >= _MAX_PREVIEW_EDIT_ROUNDS:
                return
            prv["awaiting_edit"] = False
            try:
                await query.edit_message_text(
                    "Як адаптувати матеріал?",
                    reply_markup=_PREVIEW_REFINE_KB,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Refine menu edit failed: %s", exc)
                sent = await query.message.reply_text(
                    "Як адаптувати матеріал?",
                    reply_markup=_PREVIEW_REFINE_KB,
                )
                prv["preview_message_id"] = sent.message_id
            return

        if data == "onb_prv_r_easy":
            prv = preview_state.get(chat_id)
            if not prv or not prv.get("transcript"):
                await query.message.reply_text(
                    "Не знайшов збережене джерело. Надішли посилання на YouTube ще раз."
                )
                return
            if int(prv.get("edit_rounds") or 0) >= _MAX_PREVIEW_EDIT_ROUNDS:
                return
            try:
                pd = await self._call_preview_patch_gpt(
                    str(prv["transcript"]),
                    prv.get("preview_data") or {},
                    _PREVIEW_INSTR_EASY,
                    refine_mode="easy",
                    preview_format=prv.get("format"),
                    preview_level=prv.get("level"),
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Preview GPT failed: %s", exc)
                await query.message.reply_text(
                    "Не вдалося оновити перегляд. Надішли джерело знову або спробуй пізніше."
                )
                return
            await self._after_refine_increment(
                context.bot, chat_id, prv, query.message, pd
            )
            return

        if data == "onb_prv_r_deep":
            prv = preview_state.get(chat_id)
            if not prv or not prv.get("transcript"):
                await query.message.reply_text(
                    "Не знайшов збережене джерело. Надішли посилання на YouTube ще раз."
                )
                return
            if int(prv.get("edit_rounds") or 0) >= _MAX_PREVIEW_EDIT_ROUNDS:
                return
            try:
                pd = await self._call_preview_patch_gpt(
                    str(prv["transcript"]),
                    prv.get("preview_data") or {},
                    _PREVIEW_INSTR_DEEP,
                    refine_mode="deep",
                    preview_format=prv.get("format"),
                    preview_level=prv.get("level"),
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Preview GPT failed: %s", exc)
                await query.message.reply_text(
                    "Не вдалося оновити перегляд. Надішли джерело знову або спробуй пізніше."
                )
                return
            await self._after_refine_increment(
                context.bot, chat_id, prv, query.message, pd
            )
            return

        if data == "onb_prv_r_own":
            prv = preview_state.get(chat_id)
            if not prv or not prv.get("transcript"):
                await query.message.reply_text(
                    "Не знайшов збережене джерело. Надішли посилання на YouTube ще раз."
                )
                return
            if int(prv.get("edit_rounds") or 0) >= _MAX_PREVIEW_EDIT_ROUNDS:
                return
            prv["awaiting_edit"] = True
            await query.message.reply_text(
                "Напиши одним реченням що змінити 👇\n"
                "Наприклад: 'фокус на Present Simple' або 'для бізнес теми'"
            )
            return

        if data == "onb_p_fmt":
            await query.edit_message_reply_markup(reply_markup=_POST_CARD_FMT_KB)
            return
        if data == "onb_p_lvl":
            await query.edit_message_reply_markup(reply_markup=_POST_CARD_LVL_KB)
            return
        if data.startswith("onb_p_f_"):
            fmt = data.removeprefix("onb_p_f_")
            label = _FMT_CHANGE_LABELS.get(fmt, fmt)
            st = user_state.setdefault(chat_id, {})
            st["format"] = fmt
            await query.message.reply_text(
                f"Формат змінено на: {label}\n"
                "Скинь YouTube-відео ще раз або напиши тему 👇"
            )
            await query.edit_message_reply_markup(reply_markup=_POST_CARD_KB)
            return
        if data.startswith("onb_p_l_"):
            level = data.removeprefix("onb_p_l_")
            st = user_state.setdefault(chat_id, {})
            st["level"] = level
            await query.message.reply_text(
                f"Рівень змінено на: {level}\n"
                "Скинь YouTube-відео ще раз або напиши тему 👇"
            )
            await query.edit_message_reply_markup(reply_markup=_POST_CARD_KB)
            return

        if data.startswith("onb_fmt_"):
            fmt = data.removeprefix("onb_fmt_")
            user_state[chat_id] = {"format": fmt, "level": None}
            preview_state.pop(chat_id, None)
            await query.edit_message_text(
                "Для якого рівня?",
                reply_markup=_ONB_LEVEL_KB,
            )
            return

        if data.startswith("onb_lvl_"):
            level = data.removeprefix("onb_lvl_")
            st = user_state.get(chat_id)
            if not st or not st.get("format"):
                await query.edit_message_text("Натисни /start, щоб почати спочатку.")
                return
            st["level"] = level
            preview_state.pop(chat_id, None)
            await query.edit_message_text("Супер. Скинь YouTube-відео або напиши тему 👇")
            return

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return

        message = update.message
        chat_id = message.chat_id
        message_id = message.message_id

        if await self._deduplicator.is_duplicate(chat_id, message_id):
            LOGGER.info("Skipping duplicate message chat_id=%s message_id=%s", chat_id, message_id)
            return

        st = user_state.get(chat_id)
        original_content = (message.text or message.caption or "").strip()

        prv_early = preview_state.get(chat_id)
        if prv_early and prv_early.get("awaiting_edit"):
            if message.voice:
                await message.reply_text(
                    "Напиши одним реченням текстом, що змінити 👇"
                )
                return
            if not original_content:
                return
            video_id_early = extract_video_id(original_content)
            if video_id_early:
                prv_early["awaiting_edit"] = False
            else:
                if not prv_early.get("transcript"):
                    preview_state.pop(chat_id, None)
                    await message.reply_text(
                        "Не знайшов збережене джерело. Надішли посилання на YouTube ще раз."
                    )
                    return
                raw_pd = prv_early.get("preview_data")
                pd_for_patch = dict(raw_pd) if isinstance(raw_pd, dict) else {}
                tr = str(prv_early["transcript"])
                LOGGER.info(
                    "handler=guided_preview.awaiting_edit chat_id=%s transcript_len=%s "
                    "preview_data=%s teacher_instruction=%s",
                    chat_id,
                    len(tr),
                    json.dumps(pd_for_patch, ensure_ascii=False, default=str),
                    original_content[:2000],
                )
                try:
                    pd = await self._call_preview_patch_gpt(
                        tr,
                        pd_for_patch,
                        original_content,
                        custom_correction=True,
                        preview_format=prv_early.get("format"),
                        preview_level=prv_early.get("level"),
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Preview GPT failed (awaiting edit): %s", exc)
                    prv_early["awaiting_edit"] = False
                    await message.reply_text(
                        "Не вдалося оновити перегляд. Надішли джерело знову або спробуй пізніше."
                    )
                    return
                await self._after_refine_increment(
                    context.bot, chat_id, prv_early, message, pd
                )
                return

        if (
            self._guided_ready(chat_id)
            and not message.voice
            and original_content
        ):
            video_id = extract_video_id(original_content)
            if video_id:
                base = self._preview_state_bootstrap()
                base["format"] = st.get("format")
                base["level"] = st.get("level")
                preview_state[chat_id] = base
                try:
                    transcript = await self._youtube_service.fetch_transcript(video_id)
                except TranscriptUnavailableError as err:
                    preview_state.pop(chat_id, None)
                    await message.reply_text(err.user_message)
                    return
                preview_state[chat_id]["transcript"] = transcript
                try:
                    preview_data = await self._call_preview_gpt(
                        transcript,
                        format_key=st.get("format"),
                        level=st.get("level"),
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Preview GPT failed: %s", exc)
                    preview_state.pop(chat_id, None)
                    await message.reply_text(
                        "Не вдалося зробити попередній перегляд. Спробуй ще раз."
                    )
                    return
                preview_state[chat_id]["preview_data"] = preview_data
                sent = await message.reply_text(
                    _format_preview_message(preview_data, st.get("format")),
                    reply_markup=_PREVIEW_KB,
                )
                preview_state[chat_id]["preview_message_id"] = sent.message_id
                return

        pipeline_message: Union[Message, _OnboardingEnrichedMessage] = message
        if (
            st
            and st.get("format")
            and st.get("level")
            and not message.voice
        ):
            if original_content:
                enriched = _enriched_onboarding_transcript_block(
                    st.get("format"),
                    st.get("level"),
                    original_content,
                )
                pipeline_message = _OnboardingEnrichedMessage(message, enriched)

        try:
            prepare = await self._pipeline.prepare(context.bot, pipeline_message, chat_id)
        except NeedActiveSourceError:
            await message.reply_text(
                "Спочатку надішли матеріал: посилання YouTube, текст або голосове повідомлення. "
                "Потім можна написати, наприклад: «зроби картку»."
            )
            return
        except TranscriptUnavailableError as err:
            await message.reply_text(err.user_message)
            return
        except GenerationFailedError as err:
            await message.reply_text(err.user_message)
            return
        except ValueError as err:
            LOGGER.warning("Invalid user input message_id=%s: %s", message_id, err)
            await message.reply_text(
                str(err) or "Надішли текст, голос або посилання YouTube."
            )
            return
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("Prepare failed message_id=%s: %s", message_id, error)
            await message.reply_text("Щось пішло не так. Спробуй ще раз.")
            return

        if prepare.preface:
            await message.reply_text("Вже готую твій матеріал ✨")
        elif prepare.status_line:
            await message.reply_text(prepare.status_line)

        try:
            result = await self._pipeline.execute(prepare)
        except GenerationFailedError as err:
            await message.reply_text(err.user_message)
            return
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("Execute failed message_id=%s: %s", message_id, error)
            await message.reply_text("Не вдалося завершити картку. Спробуй ще раз за хвилину.")
            return

        await self._send_pipeline_result(message, result)

    async def _send_pipeline_result(self, message, result) -> None:
        if result.image_bytes:
            image_file = InputFile(result.image_bytes, filename="educard.png")
            await message.reply_photo(
                photo=image_file,
                caption=f"Картка · {result.template_used} · {result.source_type}",
                reply_markup=_POST_CARD_KB,
            )
        elif result.text_fallback:
            await message.reply_text(result.text_fallback)
        else:
            LOGGER.error("Pipeline returned neither image nor text")
            await message.reply_text(
                "Не вдалося показати прев’ю. Спробуй ще раз."
            )
