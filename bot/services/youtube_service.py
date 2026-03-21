import logging
import re
from typing import Any, Optional

import httpx

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
                    params={"videoId": video_id},
                    headers={"x-api-key": self._api_key},
                )
                response.raise_for_status()
                payload: dict[str, Any] = response.json()
                return self._normalize_transcript(payload)

        return await with_retry(
            _request,
            attempts=3,
            operation_name=f"Supadata transcript fetch ({video_id})",
        )

    def _normalize_transcript(self, payload: dict[str, Any]) -> str:
        # Supadata may return either "content" or a list of transcript chunks.
        if isinstance(payload.get("content"), str):
            return payload["content"]

        lines: list[str] = []
        for item in payload.get("transcript", []) or []:
            text = item.get("text")
            if text:
                lines.append(text)

        transcript = " ".join(lines).strip()
        if not transcript:
            LOGGER.warning("Supadata transcript payload did not contain text.")
            raise RuntimeError("Could not parse transcript from Supadata response")
        return transcript
