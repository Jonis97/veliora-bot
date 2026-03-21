"""
Resolve a topic-related illustration URL for card heroes.

- Wikipedia first (skips logo-like titles; filters URL patterns; minimum thumb width).
- Optional Unsplash/Pexels when TOPIC_IMAGE_ALLOW_STOCK=1 (still URL-filtered).
- Optional DALL·E when TOPIC_IMAGE_ENABLE_DALLE=1 — abstract background only, no text/logos.

Before any image is used, optional vision validation (OpenAI) requires ALL of:
  1) matches the topic, 2) improves clarity (not distracting), 3) looks premium/calm.
If any check fails → image is rejected.

If nothing passes, returns None — templates use a gradient hero (no random stock).

Env: TOPIC_IMAGE_VISION_VALIDATE=1 (default) when OpenAI client is set; set to 0 to skip vision.
     TOPIC_IMAGE_VISION_MODEL=gpt-4o-mini (default)
"""

import json
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

    def _vision_enabled(self) -> bool:
        if not self._openai:
            return False
        return os.getenv("TOPIC_IMAGE_VISION_VALIDATE", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )

    async def _vision_validate(self, topic: str, image_url: str) -> bool:
        """
        Reject unless ALL are true: topic match, clarity (helps learning), premium look.
        Any failure → False.
        """
        t = (topic or "").strip()[:500]
        prompt = (
            f'Learning card topic: "{t}"\n\n'
            "This image may be used as a wide hero background on a premium study card (not a sticker).\n\n"
            "Answer these three yes/no questions about THIS image:\n"
            "1) topic_match — Does the image clearly relate to the topic (subject, setting, or concrete visual "
            "anchor)? Not generic unrelated wallpaper or random decor.\n"
            "2) clarity — Would it help a learner understand or remember the topic, or would it distract, confuse, "
            "or add visual noise?\n"
            "3) premium — Does it look calm, editorial, premium as a background (soft photo or tasteful illustration)? "
            "Reject if you see logos, brand UI, screenshots, memes, clipart, heavy text overlays, watermarks, or "
            "cheap/tacky stock.\n\n"
            "The image is approved ONLY if ALL three are true. Set \"approved\" false if ANY is false.\n\n"
            'Return JSON only with keys: approved (boolean), topic_match (boolean), clarity (boolean), premium (boolean).'
        )
        model = os.getenv("TOPIC_IMAGE_VISION_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"

        async def _call() -> bool:
            response = await self._openai.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict visual editor for educational products. Output JSON only.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
                max_tokens=220,
                temperature=0.1,
            )
            raw = (response.choices[0].message.content or "").strip()
            data = json.loads(raw)
            approved = bool(data.get("approved"))
            tm = bool(data.get("topic_match"))
            cl = bool(data.get("clarity"))
            pr = bool(data.get("premium"))
            ok = approved and tm and cl and pr
            if not ok:
                LOGGER.info(
                    "Vision gate: rejected image (approved=%s topic_match=%s clarity=%s premium=%s)",
                    approved,
                    tm,
                    cl,
                    pr,
                )
            return ok

        try:
            return await with_retry(_call, attempts=2, delay_seconds=0.6, operation_name="Topic image vision gate")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Vision validation failed; rejecting image: %s", exc)
            return False

    async def _finalize_candidate(self, topic: str, url: Optional[str]) -> Optional[str]:
        """URL policy, then vision gate (if enabled). Any failure → None."""
        u = _accept_url(url)
        if not u:
            return None
        if self._vision_enabled():
            if not await self._vision_validate(topic, u):
                LOGGER.info("Topic image rejected after validation: %s", u[:100])
                return None
        return u

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
                    final = await self._finalize_candidate(clean, url)
                    if final:
                        LOGGER.info("Topic image from %s (passed validation)", name)
                        return final
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Topic image %s failed: %s", name, exc)

        if self._dalle_enabled() and self._openai:
            try:
                url = await self._dalle_generate(clean)
                if url:
                    final = await self._finalize_candidate(clean, url)
                    if final:
                        LOGGER.info("Topic image from dall-e (passed validation)")
                        return final
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("DALL-E topic image failed: %s", exc)

        LOGGER.info("No topic image passed validation; template will use gradient hero.")
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
                    # _finalize_candidate applies URL policy + vision gate
                    finalized = await self._finalize_candidate(query, thumb)
                    if finalized:
                        return finalized
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
