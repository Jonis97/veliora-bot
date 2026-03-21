"""
Resolve a topic-related illustration URL for card heroes.

- Wikipedia first (skips logo-like titles; filters URL patterns; minimum thumb width).
- Optional Unsplash/Pexels when TOPIC_IMAGE_ALLOW_STOCK=1 (still URL-filtered).
- Optional DALL·E when TOPIC_IMAGE_ENABLE_DALLE=1 — abstract background only, no text/logos.

If nothing passes policy, returns None — templates use a gradient hero (no random stock).
"""

import logging
import os
from typing import Any, Awaitable, Callable, List, Optional, Tuple

import httpx

from bot.utils.image_policy import is_safe_topic_image_url, title_suggests_logo_or_non_photo
from bot.utils.retry import with_retry

LOGGER = logging.getLogger(__name__)


def _accept_url(url: Optional[str]) -> Optional[str]:
    if url and is_safe_topic_image_url(url):
        return url
    if url:
        LOGGER.info("Rejected topic image URL (policy): %s", url[:120])
    return None


class TopicImageService:
    def __init__(self, openai_client: Optional[Any] = None) -> None:
        self._openai = openai_client

    async def fetch_topic_image(self, topic: str) -> Optional[str]:
        """Return https URL for a topic-related image, or None."""
        clean = (topic or "").strip()
        if not clean:
            clean = "education"
        q = clean[:100]

        steps: List[Tuple[str, Any]] = [("wikipedia", self._wikipedia_thumbnail)]
        if self._stock_allowed():
            steps.extend(
                (
                    ("unsplash", self._unsplash_first_photo),
                    ("pexels", self._pexels_first_photo),
                )
            )
        for name, step in steps:
            try:

                async def _call(
                    st: Callable[[str], Awaitable[Optional[str]]] = step,
                    query: str = q,
                ) -> Optional[str]:
                    return await st(query)

                url = await with_retry(
                    _call,
                    attempts=2,
                    delay_seconds=0.5,
                    operation_name=f"Topic image ({name})",
                )
                if url:
                    accepted = _accept_url(url)
                    if accepted:
                        LOGGER.info("Topic image from %s", name)
                        return accepted
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Topic image %s failed: %s", name, exc)

        if self._dalle_enabled() and self._openai:
            try:
                url = await self._dalle_generate(clean)
                if url:
                    LOGGER.info("Topic image from dall-e")
                    return url
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("DALL-E topic image failed: %s", exc)

        LOGGER.info("No topic image found; template will use insight placeholder.")
        return None

    def _dalle_enabled(self) -> bool:
        return os.getenv("TOPIC_IMAGE_ENABLE_DALLE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

    def _stock_allowed(self) -> bool:
        """Unsplash/Pexels are off by default to avoid generic stock photos."""
        return os.getenv("TOPIC_IMAGE_ALLOW_STOCK", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )

    async def _wikipedia_thumbnail(self, query: str) -> Optional[str]:
        async with httpx.AsyncClient(
            timeout=18.0,
            headers={"User-Agent": "VelioraBot/1.0 (educational; +https://example.invalid)"},
        ) as client:
            r = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": "5",
                    "format": "json",
                },
            )
            r.raise_for_status()
            search = r.json().get("query", {}).get("search") or []
            if not search:
                return None
            for hit in search[:5]:
                title = hit.get("title")
                if not title or title_suggests_logo_or_non_photo(title):
                    continue
                r2 = await client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "titles": title,
                        "prop": "pageimages",
                        "format": "json",
                        "pithumbsize": "640",
                    },
                )
                r2.raise_for_status()
                pages = r2.json().get("query", {}).get("pages") or {}
                for _pid, page in pages.items():
                    if page.get("missing"):
                        continue
                    th = page.get("thumbnail") or {}
                    thumb = th.get("source")
                    width = int(th.get("width") or 0)
                    if not thumb or not thumb.startswith("http"):
                        continue
                    if 0 < width < 320:
                        continue
                    accepted = _accept_url(thumb)
                    if accepted:
                        return accepted
        return None

    async def _unsplash_first_photo(self, query: str) -> Optional[str]:
        key = os.getenv("UNSPLASH_ACCESS_KEY", "").strip()
        if not key:
            return None
        async with httpx.AsyncClient(timeout=18.0) as client:
            r = await client.get(
                "https://api.unsplash.com/search/photos",
                params={"query": query, "per_page": "1", "orientation": "landscape"},
                headers={"Authorization": f"Client-ID {key}"},
            )
            r.raise_for_status()
            results = r.json().get("results") or []
            if not results:
                return None
            urls = results[0].get("urls") or {}
            u = urls.get("regular") or urls.get("small")
            if u and u.startswith("http"):
                return _accept_url(u)
        return None

    async def _pexels_first_photo(self, query: str) -> Optional[str]:
        key = os.getenv("PEXELS_API_KEY", "").strip()
        if not key:
            return None
        async with httpx.AsyncClient(timeout=18.0) as client:
            r = await client.get(
                "https://api.pexels.com/v1/search",
                params={"query": query, "per_page": "1", "size": "medium"},
                headers={"Authorization": key},
            )
            r.raise_for_status()
            photos = r.json().get("photos") or []
            if not photos:
                return None
            src = photos[0].get("src") or {}
            u = src.get("large") or src.get("medium")
            if u and u.startswith("http"):
                return _accept_url(u)
        return None

    async def _dalle_generate(self, topic: str) -> Optional[str]:
        if not self._openai:
            return None
        prompt = (
            f"Abstract soft atmospheric background suggesting the mood of: {topic}. "
            "Premium editorial style, subtle gradients, no text, no letters, no logos, no brand marks, "
            "no screenshots, no UI, no app icons, no watermarks — suitable as a quiet card background only."
        )[:3900]

        async def _run() -> Optional[str]:
            response = await self._openai.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                n=1,
            )
            data = response.data
            if data and data[0].url:
                return data[0].url
            return None

        return await with_retry(_run, attempts=2, operation_name="DALL-E topic image")
