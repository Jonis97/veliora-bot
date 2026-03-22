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

    async def html_to_image(
        self,
        html: str,
        *,
        viewport_width: int = 600,
        viewport_height: int = 920,
    ) -> bytes:
        """
        Render HTML to PNG using only the query parameters required by ScreenshotOne.
        The `html` value is passed as a request param; httpx URL-encodes it correctly.
        Templates must include a root `.page` element for `selector=.page`.
        """

        async def _render() -> bytes:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(
                    self._endpoint,
                    params={
                        "access_key": self._api_key,
                        "html": html,
                        "format": "png",
                        "viewport_width": viewport_width,
                        "viewport_height": viewport_height,
                        "selector": ".page",
                    },
                )
                response.raise_for_status()
                return response.content

        return await with_retry(_render, attempts=3, operation_name="Screenshot rendering")
