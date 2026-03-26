import logging
from typing import Any, Optional, Union

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Message, Update
from telegram.ext import ContextTypes

from bot.services.pipeline_service import ContentPipelineService
from bot.utils.active_source import NeedActiveSourceError
from bot.utils.dedup import MessageDeduplicator
from bot.utils.errors import GenerationFailedError, TranscriptUnavailableError


LOGGER = logging.getLogger(__name__)

# Onboarding only — separate from active_source / pipeline memory.
user_state: dict[int, dict[str, Optional[str]]] = {}

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


class MessageHandlerService:
    def __init__(self, pipeline: ContentPipelineService, deduplicator: MessageDeduplicator) -> None:
        self._pipeline = pipeline
        self._deduplicator = deduplicator

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        chat_id = update.message.chat_id
        user_state.pop(chat_id, None)
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
        pipeline_message: Union[Message, _OnboardingEnrichedMessage] = message
        if (
            st
            and st.get("format")
            and st.get("level")
            and not message.voice
        ):
            original_content = (message.text or message.caption or "").strip()
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
