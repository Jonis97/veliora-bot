import logging
from dataclasses import dataclass
from typing import Optional

from telegram import Bot, Message

from bot.services.ai_service import AIContentService
from bot.services.screenshot_service import ScreenshotService
from bot.services.template_service import TemplateService
from bot.services.transcription_service import VoiceTranscriptionService
from bot.services.youtube_service import YouTubeTranscriptService, extract_video_id
from bot.utils.input_parser import parse_template_hint


LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    image_bytes: bytes
    template_used: str
    source_type: str


class ContentPipelineService:
    def __init__(
        self,
        youtube_service: YouTubeTranscriptService,
        transcription_service: VoiceTranscriptionService,
        ai_service: AIContentService,
        template_service: TemplateService,
        screenshot_service: ScreenshotService,
    ) -> None:
        self._youtube_service = youtube_service
        self._transcription_service = transcription_service
        self._ai_service = ai_service
        self._template_service = template_service
        self._screenshot_service = screenshot_service

    async def process_message(self, bot: Bot, message: Message) -> PipelineResult:
        text_source, preferred_template, source_type = await self._extract_source(bot, message)
        card_json = await self._ai_service.generate_card_content(text_source, preferred_template)
        html = self._template_service.render_html(card_json, preferred_template)
        image_bytes = await self._screenshot_service.html_to_image(html)

        used_template = preferred_template or str(card_json.get("template", "warm_paper"))
        return PipelineResult(
            image_bytes=image_bytes,
            template_used=used_template,
            source_type=source_type,
        )

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
            transcript = await self._youtube_service.fetch_transcript(video_id)
            return transcript, template, "youtube"
        return cleaned_text, template, "text"
