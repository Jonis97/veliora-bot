import logging
from dataclasses import dataclass
from typing import Any, MutableMapping, Optional

from telegram import Bot, Message

from bot.services.ai_service import AIContentService
from bot.services.screenshot_service import ScreenshotService
from bot.services.template_service import DEFAULT_TEMPLATE, TemplateService
from bot.utils.errors import GenerationFailedError
from bot.utils.intent import OutputIntent, UnclearIntentError, intent_label, resolve_output_intent
from bot.services.topic_image_service import TopicImageService
from bot.services.transcription_service import VoiceTranscriptionService
from bot.services.youtube_service import YouTubeTranscriptService, extract_video_id
from bot.utils.active_source import (
    NeedActiveSourceError,
    build_followup_prompt,
    followup_intent,
    load_active_source,
    save_active_source,
)
from bot.utils.input_parser import parse_template_hint


LOGGER = logging.getLogger(__name__)

# Single locked layout until more templates are wired into the product core.
LOCKED_TEMPLATE = DEFAULT_TEMPLATE


@dataclass
class PipelineResult:
    """Either `image_bytes` (preferred) or `text_fallback` when screenshot API fails."""

    template_used: str
    source_type: str
    output_intent: str
    image_bytes: Optional[bytes] = None
    text_fallback: Optional[str] = None


@dataclass
class _ResolvedSource:
    """What to send to the model and whether to replace stored active source."""

    text_for_ai: str
    template: Optional[str]
    source_type: str
    persist_new_source: bool
    intent_seed: str
    persist_body: Optional[str] = None
    video_id: Optional[str] = None


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

    async def process_message(
        self,
        bot: Bot,
        message: Message,
        user_data: Optional[MutableMapping[str, Any]] = None,
    ) -> PipelineResult:
        ud: MutableMapping[str, Any] = user_data if user_data is not None else {}
        resolved = await self._resolve_source(bot, message, ud)
        # Persist new material before AI so retries / "try again" follow-ups use the right source.
        if resolved.persist_new_source and resolved.persist_body:
            save_active_source(
                ud,
                source_type=resolved.source_type,
                text=resolved.persist_body,
                video_id=resolved.video_id,
            )
            LOGGER.info(
                "Active source updated: type=%s len=%s",
                resolved.source_type,
                len(resolved.persist_body),
            )
        elif not resolved.persist_new_source:
            LOGGER.info("Using follow-up on existing active source (type=%s)", resolved.source_type)

        seed = (resolved.intent_seed or "").strip()
        if not seed:
            output_intent = OutputIntent.CARD
        else:
            output_intent = resolve_output_intent(
                seed,
                is_follow_up=not resolved.persist_new_source,
            )

        try:
            card_json = await self._ai_service.generate_card_content(
                resolved.text_for_ai,
                LOCKED_TEMPLATE,
                output_intent=output_intent,
                is_followup=not resolved.persist_new_source,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("AI card generation failed: %s", exc)
            raise GenerationFailedError() from exc

        topic = str(card_json.get("title", "education")).strip() or "education"
        image_url = await self._topic_image_service.fetch_topic_image(topic)
        card_for_render = dict(card_json)
        card_for_render["template"] = LOCKED_TEMPLATE
        if image_url:
            card_for_render["image_url"] = image_url
        used_template = LOCKED_TEMPLATE
        source_type = resolved.source_type
        intent_caption = intent_label(output_intent)
        html = self._template_service.render_html(card_for_render, LOCKED_TEMPLATE)

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
        """Plain-text card when image rendering is unavailable (Telegram limit ~4096)."""
        title = str(card.get("title", "Learning Card")).strip()
        subtitle = str(card.get("subtitle", "")).strip()
        raw_bullets = card.get("bullets") or []
        if not isinstance(raw_bullets, list):
            raw_bullets = []
        cta = str(card.get("cta", "")).strip()
        lines = [
            f"{intent_label_s} · {template_used} · source: {source_type}",
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
        lines.append("(Image preview unavailable — text card above.)")
        text = "\n".join(lines)
        if len(text) > 4000:
            return text[:3997] + "..."
        return text

    async def _resolve_source(
        self,
        bot: Bot,
        message: Message,
        user_data: MutableMapping[str, Any],
    ) -> _ResolvedSource:
        """Resolve text for the model and whether this message replaces the active source."""
        if message.voice:
            transcript = await self._transcription_service.transcribe_voice(bot, message.voice.file_id)
            if not transcript:
                raise RuntimeError("Voice transcription returned empty text")
            template: Optional[str] = None
            cap = (message.caption or "").strip()
            if cap:
                _, template = parse_template_hint(cap)
            return _ResolvedSource(
                text_for_ai=transcript,
                template=template,
                source_type="voice",
                persist_new_source=True,
                intent_seed="",
                persist_body=transcript,
            )

        raw = (message.text or message.caption or "").strip()
        if not raw:
            raise ValueError("Unsupported message type. Send text, a voice message, or a YouTube link.")

        cleaned_text, template = parse_template_hint(raw)
        active = load_active_source(user_data)

        video_id = extract_video_id(cleaned_text)
        if video_id:
            try:
                transcript = await self._youtube_service.fetch_transcript(video_id)
                return _ResolvedSource(
                    text_for_ai=transcript,
                    template=template,
                    source_type="youtube",
                    persist_new_source=True,
                    intent_seed="",
                    persist_body=transcript,
                    video_id=video_id,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "YouTube transcript fetch failed for video_id=%s: %s. "
                    "Falling back to URL-only context for GPT.",
                    video_id,
                    exc,
                )
                fallback_text = self._youtube_url_only_context(video_id, cleaned_text)
                return _ResolvedSource(
                    text_for_ai=fallback_text,
                    template=template,
                    source_type="youtube",
                    persist_new_source=True,
                    intent_seed="",
                    persist_body=fallback_text,
                    video_id=video_id,
                )

        if not cleaned_text.strip():
            if template is None:
                raise ValueError("Empty message after removing template tag. Send material or a valid request.")
            if active is None:
                raise NeedActiveSourceError()
            instruction = (
                "The user only sent a template preference. Regenerate the learning card for the same "
                "source material using the preferred template."
            )
            return _ResolvedSource(
                text_for_ai=build_followup_prompt(instruction, active),
                template=template,
                source_type=str(active.get("type", "text")),
                persist_new_source=False,
                intent_seed="",
            )

        if followup_intent(cleaned_text):
            if active is None:
                raise NeedActiveSourceError()
            return _ResolvedSource(
                text_for_ai=build_followup_prompt(cleaned_text, active),
                template=template,
                source_type=str(active.get("type", "text")),
                persist_new_source=False,
                intent_seed=cleaned_text,
            )

        return _ResolvedSource(
            text_for_ai=cleaned_text,
            template=template,
            source_type="text",
            persist_new_source=True,
            intent_seed=cleaned_text,
            persist_body=cleaned_text,
        )

    @staticmethod
    def _youtube_url_only_context(video_id: str, user_link_text: str) -> str:
        """When Supadata transcript is unavailable, GPT infers topic from URL only."""
        canonical_url = f"https://www.youtube.com/watch?v={video_id}"
        return (
            "[Transcript unavailable — no transcript text was fetched.]\n\n"
            f"Video URL: {canonical_url}\n"
            f"Video ID: {video_id}\n"
            f"User message / link: {user_link_text}\n\n"
            "Infer the likely topic or theme from the URL, video ID, and any cues in the link text. "
            "Create an educational flashcard about that topic with clear, generally accurate learning points. "
            "If the topic is ambiguous, choose a reasonable educational angle and frame the card helpfully."
        )
