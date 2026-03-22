import httpx

from bot.utils.retry import with_retry


class ScreenshotService:
    """
    ScreenshotOne API: https://api.screenshotone.com/take
    Use `access_key` from env `SCREENSHOTONE_API_KEY` via `load_settings()` in main.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._endpoint = "https://api.screenshotone.com/take"

    async def take_screenshot(self, html_content: str, orientation: str = "portrait") -> bytes:
        """
        Render HTML to PNG. `orientation` is "landscape" (1280×720) or "portrait" (600×920).
        Templates must include a root `.page` element for `selector=.page`.
        """
        if orientation == "landscape":
            width, height = 1280, 720
        else:
            width, height = 600, 920

        async def _render() -> bytes:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(
                    self._endpoint,
                    params={
                        "access_key": self._api_key,
                        "html": html_content,
                        "format": "png",
                        "viewport_width": width,
                        "viewport_height": height,
                        "selector": ".page",
                    },
                )
                response.raise_for_status()
                return response.content

        return await with_retry(_render, attempts=3, operation_name="Screenshot rendering")

    async def html_to_image(
        self,
        html: str,
        *,
        orientation: str = "portrait",
    ) -> bytes:
        """Backward-compatible alias for `take_screenshot`."""
        return await self.take_screenshot(html, orientation=orientation)
