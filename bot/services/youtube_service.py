import json
import logging
import re
from typing import Any, Optional

import httpx

from bot.utils.errors import TranscriptUnavailableError
from bot.utils.retry import with_retry


LOGGER = logging.getLogger(__name__)
YOUTUBE_ID_REGEX = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([A-Za-z0-9_-]{11})"
)


def extract_video_id(text: str) -> Optional[str]:
    match = YOUTUBE_ID_REGEX.search(text)
    return match.group(1) if match else None


class YouTubeTranscriptService:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._endpoint = "https://api.supadata.ai/v1/youtube/transcript"

    async def fetch_transcript(self, video_id: str) -> str:
        async def _request() -> str:
            async with httpx.AsyncClient(timeout=25.0) as client:
                response = await client.get(
                    self._endpoint,
                    params={"videoId": video_id, "lang": "en"},
                    headers={"x-api-key": self._api_key},
                )
                response.raise_for_status()
                response_body = response.text
                payload: Any = json.loads(response_body)
                normalized = self._normalize_transcript(payload)
                if not normalized.strip():
                    LOGGER.warning(f"Supadata raw response: {response_body}")
                return normalized

        text = await with_retry(
            _request,
            attempts=3,
            operation_name=f"Supadata transcript fetch ({video_id})",
        )
        if text.strip():
            return text

        raise TranscriptUnavailableError()

    def _normalize_transcript(self, payload: Any) -> str:
        # Handle Supadata format: {"content": [...]}
        if isinstance(payload, dict):
            content = payload.get("content")
            if isinstance(content, list):
                lines = [item.get("text", "") for item in content if item.get("text")]
                return " ".join(lines).strip()

        # Handle list of transcript chunks with "lang" and "text" fields
        if isinstance(payload, list):
            lines = [item.get("text", "") for item in payload if item.get("text")]
            return " ".join(lines).strip()

        # Supadata may return either "content" or a list of transcript chunks.
        if isinstance(payload.get("content"), str):
            return payload["content"].strip()

        lines: list[str] = []
        for item in payload.get("transcript", []) or []:
            text = item.get("text")
            if text:
                lines.append(text)

        transcript = " ".join(lines).strip()
        if not transcript:
            LOGGER.warning("Supadata transcript payload did not contain text.")
        return transcript
