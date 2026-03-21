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
from bot.utils.input_parser import parse_template_hint


LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Either `image_bytes` (preferred) or `text_fallback` when screenshot API fails."""

    template_used: str
    source_type: str
    image_bytes: Optional[bytes] = None
    text_fallback: Optional[str] = None


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

    async def process_message(self, bot: Bot, message: Message) -> PipelineResult:
        text_source, preferred_template, source_type = await self._extract_source(bot, message)
        card_json = await self._ai_service.generate_card_content(text_source, preferred_template)
        topic = str(card_json.get("title", "education")).strip() or "education"
        image_url = await self._topic_image_service.fetch_topic_image(topic)
        card_for_render = dict(card_json)
        if image_url:
            card_for_render["image_url"] = image_url
        html = self._template_service.render_html(card_for_render, preferred_template)
        used_template = preferred_template or str(card_json.get("template", DEFAULT_TEMPLATE))

        try:
            image_bytes = await self._screenshot_service.html_to_image(html)
            return PipelineResult(
                template_used=used_template,
                source_type=source_type,
                image_bytes=image_bytes,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Screenshot rendering failed; sending text fallback. Error: %s",
                exc,
                exc_info=True,
            )
            text_body = self._format_card_text_reply(card_for_render, used_template, source_type)
            return PipelineResult(
                template_used=used_template,
                source_type=source_type,
                text_fallback=text_body,
            )

    @staticmethod
    def _format_card_text_reply(card: dict[str, Any], template_used: str, source_type: str) -> str:
        """Plain-text card when image rendering is unavailable (Telegram limit ~4096)."""
        title = str(card.get("title", "Learning Card")).strip()
        subtitle = str(card.get("subtitle", "")).strip()
        raw_bullets = card.get("bullets") or []
        if not isinstance(raw_bullets, list):
            raw_bullets = []
        cta = str(card.get("cta", "")).strip()
        lines = [
            f"Template: {template_used} | Source: {source_type}",
            "",
            f"📌 {title}",
        ]
        if subtitle:
            lines.append(subtitle)
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

    async def _extract_source(self, bot: Bot, message: Message) -> tuple[str, Optional[str], str]:
        if message.voice:
            transcript = await self._transcription_service.transcribe_voice(bot, message.voice.file_id)
            if not transcript:
                raise RuntimeError("Voice transcription returned empty text")
            return transcript, None, "voice"

        text = (message.text or message.caption or "").strip()
        if not text:
            raise ValueError("Unsupported message type. Send text, a voice message, or a YouTube link.")

        cleaned_text, template = parse_template_hint(text)
        video_id = extract_video_id(cleaned_text)
        if video_id:
            try:
                transcript = await self._youtube_service.fetch_transcript(video_id)
                return transcript, template, "youtube"
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "YouTube transcript fetch failed for video_id=%s: %s. "
                    "Falling back to URL-only context for GPT.",
                    video_id,
                    exc,
                )
                fallback_text = self._youtube_url_only_context(video_id, cleaned_text)
                return fallback_text, template, "youtube"
        return cleaned_text, template, "text"

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
