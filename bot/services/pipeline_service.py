"""
MVP: one premium warm_paper_v2 card per request from the latest source only.

- Source is stored per chat_id (in-memory).
- New source replaces the old one; no mixing.
- Template is always warm_paper_v2 (no rerender / no multi-template routing here).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from telegram import Bot, Message

from bot.services.ai_service import AIContentService
from bot.services.screenshot_service import ScreenshotService
from bot.services.template_service import DEFAULT_TEMPLATE, TemplateService
from bot.services.topic_image_service import TopicImageService
from bot.services.transcription_service import VoiceTranscriptionService
from bot.services.youtube_service import YouTubeTranscriptService, extract_video_id
from bot.utils.active_source import NeedActiveSourceError, build_followup_prompt, followup_intent
from bot.utils.errors import GenerationFailedError, TranscriptUnavailableError
from bot.utils.input_parser import parse_template_hint
from bot.utils.intent import OutputIntent

LOGGER = logging.getLogger(__name__)

_CHAT_CONTEXT: dict[int, dict[str, Any]] = {}


def _detect_user_intent(message: Message) -> str:
    raw = (message.text or message.caption or "").lower()
    if any(s in raw for s in ("слова", "словник", "vocabulary", "лексика")):
        return "vocabulary"
    if any(s in raw for s in ("вправи", "завдання", "задание", "exercises")):
        return "exercises"
    if any(s in raw for s in ("виправ", "помилки", "fix", "mistakes")):
        return "fix_mistakes"
    return "card"


def _ctx(chat_id: int) -> dict[str, Any]:
    if chat_id not in _CHAT_CONTEXT:
        _CHAT_CONTEXT[chat_id] = {"active_source": None}
    return _CHAT_CONTEXT[chat_id]


def _load_active_source(chat_id: int) -> Optional[dict[str, Any]]:
    raw = _ctx(chat_id).get("active_source")
    if not isinstance(raw, dict):
        return None
    st = raw.get("type")
    text = raw.get("text")
    if st not in ("youtube", "text", "voice") or not isinstance(text, str) or not text.strip():
        return None
    return raw


def _save_active_source(
    chat_id: int,
    *,
    source_type: str,
    text: str,
    video_id: Optional[str] = None,
) -> None:
    payload: dict[str, Any] = {"type": source_type, "text": text.strip()}
    if video_id:
        payload["video_id"] = video_id
    _ctx(chat_id)["active_source"] = payload


def _source_type_label_uk(st: str) -> str:
    return {"youtube": "YouTube", "voice": "голос", "text": "текст"}.get(st, st)


def _ground_for_ai(inner: str) -> str:
    return (
        "ПРАВИЛА ПРИВ’ЯЗКИ ДО ДЖЕРЕЛА (обов’язково):\n"
        "- Увесь зміст картки — ЛИШЕ з матеріалу нижче. Не додавай фактів, яких немає в джерелі.\n"
        "- Не вигадуй загальні поради «для будь-якого випадку» — лише те, що випливає з цього тексту/транскрипту.\n"
        "- Якщо джерела замало для пункту — краще пропусти або сформулюй обережно в межах того, що є.\n\n"
        f"{inner}"
    )


@dataclass
class PipelineResult:
    template_used: str
    source_type: str
    output_intent: str
    image_bytes: Optional[bytes] = None
    text_fallback: Optional[str] = None


@dataclass
class _ResolvedSource:
    text_for_ai: str
    source_type: str
    persist_new_source: bool
    intent_seed: str
    persist_body: Optional[str] = None
    video_id: Optional[str] = None
    youtube_thumbnail_url: Optional[str] = None
    intent: str = "card"


@dataclass
class PrepareResult:
    chat_id: int
    resolved: _ResolvedSource
    preface: Optional[str]
    status_line: Optional[str]


class ContentPipelineService:
    def __init__(
        self,
        youtube_service: YouTubeTranscriptService,
        transcription_service: VoiceTranscriptionService,
        ai_service: AIContentService,
        template_service: TemplateService,
        screenshot_service: ScreenshotService,
        topic_image_service: TopicImageService,
    ) -> None:
        self._youtube_service = youtube_service
        self._transcription_service = transcription_service
        self._ai_service = ai_service
        self._template_service = template_service
        self._screenshot_service = screenshot_service
        self._topic_image_service = topic_image_service

    async def prepare(self, bot: Bot, message: Message, chat_id: int) -> PrepareResult:
        resolved = await self._resolve_source(bot, message, chat_id)

        if resolved.persist_new_source and resolved.persist_body:
            _save_active_source(
                chat_id,
                source_type=resolved.source_type,
                text=resolved.persist_body,
                video_id=resolved.video_id,
            )
            LOGGER.info("Active source updated chat_id=%s type=%s", chat_id, resolved.source_type)
            preface = (
                "Зрозуміло — зберегла це як поточне джерело. "
                "Створюю для тебе одну навчальну картку warm_paper_v2…"
            )
            status_line = None
        else:
            LOGGER.info("Follow-up on active source chat_id=%s", chat_id)
            preface = None
            status_line = (
                f"Беру матеріал з джерела ({_source_type_label_uk(resolved.source_type)}). "
                "Формую картку…"
            )

        return PrepareResult(
            chat_id=chat_id,
            resolved=resolved,
            preface=preface,
            status_line=status_line,
        )

    async def execute(self, prepare: PrepareResult) -> PipelineResult:
        resolved = prepare.resolved
        raw_ai = resolved.text_for_ai
        grounded = _ground_for_ai(raw_ai)

        eff_template = "vocab_card" if resolved.intent == "vocabulary" else DEFAULT_TEMPLATE

        try:
            card_json = await self._ai_service.generate_card_content(
                grounded,
                eff_template,
                output_intent=OutputIntent.CARD,
                is_followup=not resolved.persist_new_source,
                intent=resolved.intent,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("AI card generation failed: %s", exc)
            raise GenerationFailedError(
                "Не вдалося згенерувати картку. Спробуй ще раз за хвилину."
            ) from exc

        card_json["template"] = eff_template

        card_for_render = dict(card_json)
        if resolved.source_type == "youtube" and resolved.youtube_thumbnail_url:
            card_for_render["image_url"] = resolved.youtube_thumbnail_url
        else:
            topic = str(card_json.get("title", "education")).strip() or "education"
            image_url = await self._topic_image_service.fetch_topic_image(topic)
            if image_url:
                card_for_render["image_url"] = image_url

        used_template = eff_template
        source_type = resolved.source_type
        intent_caption = "Картка"
        html = self._template_service.render_html(card_for_render, eff_template)

        try:
            image_bytes = await self._screenshot_service.html_to_image(html)
            return PipelineResult(
                template_used=used_template,
                source_type=source_type,
                output_intent=intent_caption,
                image_bytes=image_bytes,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Screenshot rendering failed; sending text fallback. Error: %s",
                exc,
                exc_info=True,
            )
            text_body = self._format_card_text_reply(
                card_for_render, used_template, source_type, intent_caption
            )
            return PipelineResult(
                template_used=used_template,
                source_type=source_type,
                output_intent=intent_caption,
                text_fallback=text_body,
            )

    @staticmethod
    def _format_card_text_reply(
        card: dict[str, Any],
        template_used: str,
        source_type: str,
        intent_label_s: str,
    ) -> str:
        title = str(card.get("title", "Learning Card")).strip()
        subtitle = str(card.get("subtitle", "")).strip()
        raw_bullets = card.get("bullets") or []
        if not isinstance(raw_bullets, list):
            raw_bullets = []
        cta = str(card.get("cta", "")).strip()
        lines = [
            f"{intent_label_s} · {template_used} · джерело: {source_type}",
            "",
            f"📌 {title}",
        ]
        if subtitle:
            lines.append(subtitle)
        punch = str(card.get("punchline", "") or "").strip()
        if punch:
            lines.extend(["", f"⭐ {punch}"])
        contrast = card.get("contrast")
        if isinstance(contrast, dict):
            w = str(contrast.get("wrong", "") or "").strip()
            b = str(contrast.get("better", "") or "").strip()
            if w or b:
                lines.append("")
                if w:
                    lines.append(f"✗ {w}")
                if b:
                    lines.append(f"✓ {b}")
        lines.append("")
        for item in raw_bullets[:8]:
            lines.append(f"• {str(item).strip()}")
        if cta:
            lines.extend(["", f"💡 {cta}"])
        lines.append("")
        lines.append("(Попередній перегляд зображення недоступний — текст картки вище.)")
        text = "\n".join(lines)
        if len(text) > 4000:
            return text[:3997] + "..."
        return text

    async def _resolve_source(
        self,
        bot: Bot,
        message: Message,
        chat_id: int,
    ) -> _ResolvedSource:
        active = _load_active_source(chat_id)
        intent = _detect_user_intent(message)

        if message.voice:
            transcript = await self._transcription_service.transcribe_voice(bot, message.voice.file_id)
            if not (transcript or "").strip():
                raise TranscriptUnavailableError(
                    "Не вдалось розпізнати голос. Надішли текст або посилання — зроблю картку."
                )
            return _ResolvedSource(
                text_for_ai=transcript,
                source_type="voice",
                persist_new_source=True,
                intent_seed="",
                persist_body=transcript,
                intent=intent,
            )

        raw = (message.text or message.caption or "").strip()
        if not raw:
            raise ValueError("Надішли текст, голосове повідомлення або посилання на YouTube.")

        cleaned_text, _template = parse_template_hint(raw)
        video_id = extract_video_id(cleaned_text)
        if video_id:
            try:
                transcript = await self._youtube_service.fetch_transcript(video_id)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "YouTube transcript fetch failed for video_id=%s: %s",
                    video_id,
                    exc,
                )
                raise TranscriptUnavailableError() from exc
            if not (transcript or "").strip():
                LOGGER.warning("YouTube transcript empty for video_id=%s", video_id)
                raise TranscriptUnavailableError()
            image_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
            return _ResolvedSource(
                text_for_ai=transcript,
                source_type="youtube",
                persist_new_source=True,
                intent_seed="",
                persist_body=transcript,
                video_id=video_id,
                youtube_thumbnail_url=image_url,
                intent=intent,
            )

        if not cleaned_text.strip() and _template:
            if active is None:
                raise NeedActiveSourceError()
            instruction = (
                "The user sent only a template tag. Regenerate ONE warm_paper_v2 study card from the same source material. "
                "Do not change the topic; use only the stored source."
            )
            return _ResolvedSource(
                text_for_ai=build_followup_prompt(instruction, active),
                source_type=str(active.get("type", "text")),
                persist_new_source=False,
                intent_seed="",
                intent=intent,
            )

        if followup_intent(cleaned_text):
            if active is None:
                raise NeedActiveSourceError()
            return _ResolvedSource(
                text_for_ai=build_followup_prompt(cleaned_text, active),
                source_type=str(active.get("type", "text")),
                persist_new_source=False,
                intent_seed=cleaned_text,
                intent=intent,
            )

        return _ResolvedSource(
            text_for_ai=cleaned_text,
            source_type="text",
            persist_new_source=True,
            intent_seed=cleaned_text,
            persist_body=cleaned_text,
            intent=intent,
        )

    async def process_message(
        self,
        bot: Bot,
        message: Message,
        user_data: Optional[Any] = None,
    ) -> PipelineResult:
        chat_id = message.chat_id
        pr = await self.prepare(bot, message, chat_id)
        return await self.execute(pr)
