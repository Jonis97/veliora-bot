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

_PREVIEW_SYSTEM = (
    "You are a helpful teacher. Output ONE JSON object only, no markdown.\n"
    "From the source transcript below ONLY (no invented facts):\n"
    '- "topic": one short line (teacher-friendly)\n'
    '- "key_ideas": exactly 3 short strings, each one bullet-worthy idea from the source\n'
    '- "words": 3 to 5 useful English words or short phrases that appear in or are clearly grounded in the source\n'
    "Keep everything short. No card layout, no images.\n"
    "Follow any additional instruction without breaking source-only rules."
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
    'Keys: "topic" (string), "key_ideas" (exactly 3 short strings), '
    '"words" (3 to 5 strings).\n'
    "Ground facts in the original transcript only. You are PATCHING the current "
    "preview: change only what the teacher instruction targets; for any part not "
    "targeted, output that field exactly as given in the current preview "
    "(same wording)."
)

_PREVIEW_PATCH_RULES = """Rules:
- Patch ONLY what teacher asked to change
- Keep all other blocks exactly as they are
- If teacher says 'більше слів' → extend only words
- If teacher says 'ідеї простіші' → modify only key_ideas
- If teacher says 'слова простіші' → modify only words
- Do not rebuild entire preview unless explicitly requested"""

_PREVIEW_PATCH_RULES_CUSTOM = """Rules (custom teacher correction — PATCH only, not full regeneration):
- Always expand or adjust on top of the "Current preview (complete JSON)" above; that JSON is the source of truth for what exists now.
- Never return the same content unchanged: the output JSON must differ from the current preview in at least one meaningful way when you apply the teacher instruction.
- Do not remove existing content unless the teacher explicitly asks to remove or replace it.
- Modify only the section(s) the teacher instruction targets; keep all other sections unchanged.
- This is PATCH editing: do not rebuild the whole preview from the transcript unless the teacher explicitly asks for a full redo.

Common teacher commands (match meaning, not only exact spelling):
- When the teacher says "більше слів" (or equivalent) → add 3 to 6 NEW words/phrases to `words`, grounded in the transcript; keep existing `words` entries; extend the list as needed.
- When the teacher says "більше питань" (or equivalent) → add 2 to 4 NEW questions to `questions` (create the array if missing); keep existing `questions` entries.
- When the teacher says "більше ідей" (or equivalent) → add 1 to 2 NEW ideas to `key_ideas`; keep existing idea strings unless the teacher asked to replace them.
- When the teacher says "більше вправ" (or equivalent) → add 2 to 3 NEW short exercises to `exercises` (create the array if missing); keep existing `exercises` entries.

Output: one JSON object with `topic`, `key_ideas`, `words`, and include `questions` and/or `exercises` when they exist in the current preview or when the teacher asked to add them — copy those arrays forward from the current preview JSON and extend as above."""

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


def _normalize_gpt_preview_dict(data: dict[str, Any]) -> dict[str, Any]:
    topic = str(data.get("topic", "") or "").strip()
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
        "topic": topic or "—",
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
) -> str:
    pd = preview_data if isinstance(preview_data, dict) else {}
    topic, ideas_str, words_str = _preview_blocks_for_prompt(pd)
    preview_json = json.dumps(pd, ensure_ascii=False, default=str)
    return (
        f"Original transcript:\n{transcript_snippet}\n\n"
        "Current preview (stable base):\n"
        f"TOPIC: {topic}\n"
        f"IDEAS: {ideas_str}\n"
        f"WORDS: {words_str}\n\n"
        f"Current preview (complete JSON, all fields):\n{preview_json}\n\n"
        f"Teacher instruction: {teacher_text}\n\n"
        f"{rules_block}"
    )


def _format_preview_message(preview_data: dict[str, Any]) -> str:
    topic = str(preview_data.get("topic") or "—").strip() or "—"
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
        "📋 Ось що знайшов:\n\n",
        f"📌 Тема: {topic}\n\n",
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

    async def _call_preview_gpt(
        self,
        transcript: str,
        extra_instruction: Optional[str] = None,
    ) -> dict[str, Any]:
        snippet = transcript.strip()
        if len(snippet) > _PREVIEW_TRANSCRIPT_MAX:
            snippet = snippet[:_PREVIEW_TRANSCRIPT_MAX]
        user_block = f"Transcript:\n{snippet}"
        if extra_instruction and extra_instruction.strip():
            user_block += f"\n\nAdditional instruction:\n{extra_instruction.strip()}"
        response = await self._openai_client.chat.completions.create(
            model=self._openai_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _PREVIEW_SYSTEM},
                {"role": "user", "content": user_block},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        if not isinstance(data, dict):
            data = {}
        return _normalize_gpt_preview_dict(data)

    async def _call_preview_patch_gpt(
        self,
        transcript: str,
        preview_data: dict[str, Any],
        teacher_text: str,
        *,
        custom_correction: bool = False,
    ) -> dict[str, Any]:
        snippet = transcript.strip()
        if len(snippet) > _PREVIEW_TRANSCRIPT_MAX:
            snippet = snippet[:_PREVIEW_TRANSCRIPT_MAX]
        pd_in = preview_data if isinstance(preview_data, dict) else {}
        rules_block = (
            _PREVIEW_PATCH_RULES_CUSTOM
            if custom_correction
            else _PREVIEW_PATCH_RULES
        )
        LOGGER.info(
            "preview_patch_gpt transcript_len=%s preview_data=%s teacher_instruction=%s",
            len(snippet),
            json.dumps(pd_in, ensure_ascii=False, default=str),
            teacher_text.strip()[:2000],
        )
        user_content = _build_preview_patch_user_content(
            snippet,
            pd_in,
            teacher_text.strip(),
            rules_block,
        )
        response = await self._openai_client.chat.completions.create(
            model=self._openai_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _PREVIEW_PATCH_SYSTEM},
                {"role": "user", "content": user_content},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        if not isinstance(data, dict):
            data = {}
        normalized = _normalize_gpt_preview_dict(data)
        if custom_correction:
            for k in ("questions", "exercises"):
                if k in normalized:
                    continue
                prev = pd_in.get(k)
                if isinstance(prev, list) and prev:
                    normalized[k] = [
                        str(x).strip() for x in prev if str(x).strip()
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
                _format_preview_message(preview_data),
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
            st = user_state.get(chat_id)
            fmt = (st or {}).get("format") or prv.get("format")
            lvl = (st or {}).get("level") or prv.get("level")
            transcript = str(prv.get("transcript") or "").strip()
            enriched = (
                f"[FORMAT={fmt}]\n[LEVEL={lvl}]\n\n"
                f"USER CONTENT:\n{transcript}"
            )
            proxy = _OnboardingEnrichedMessage(query.message, enriched)
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
                    preview_data = await self._call_preview_gpt(transcript)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Preview GPT failed: %s", exc)
                    preview_state.pop(chat_id, None)
                    await message.reply_text(
                        "Не вдалося зробити попередній перегляд. Спробуй ще раз."
                    )
                    return
                preview_state[chat_id]["preview_data"] = preview_data
                sent = await message.reply_text(
                    _format_preview_message(preview_data),
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
                enriched = (
                    f"[FORMAT={st['format']}]\n[LEVEL={st['level']}]\n\n"
                    f"USER CONTENT:\n{original_content}"
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
