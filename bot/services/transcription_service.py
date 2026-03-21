import io

from openai import AsyncOpenAI
from telegram import Bot

from bot.utils.retry import with_retry


class VoiceTranscriptionService:
    def __init__(self, openai_client: AsyncOpenAI) -> None:
        self._openai_client = openai_client

    async def transcribe_voice(self, bot: Bot, file_id: str) -> str:
        async def _transcribe() -> str:
            telegram_file = await bot.get_file(file_id)
            audio_bytes = await telegram_file.download_as_bytearray()
            audio_buffer = io.BytesIO(audio_bytes)
            audio_buffer.name = "voice.ogg"

            result = await self._openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_buffer,
            )
            return result.text.strip()

        return await with_retry(
            _transcribe,
            attempts=3,
            operation_name="Whisper transcription",
        )
