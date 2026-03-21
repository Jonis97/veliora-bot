from urllib.parse import quote

import httpx

from bot.utils.retry import with_retry


class ScreenshotService:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._endpoint = "https://api.screenshotone.com/take"

    async def html_to_image(self, html: str) -> bytes:
        data_url = "data:text/html;charset=utf-8," + quote(html)

        async def _render() -> bytes:
            async with httpx.AsyncClient(timeout=35.0) as client:
                response = await client.get(
                    self._endpoint,
                    params={
                        "access_key": self._api_key,
                        "url": data_url,
                        "viewport_width": 1080,
                        "viewport_height": 1350,
                        "device_scale_factor": 1,
                        "format": "png",
                    },
                )
                response.raise_for_status()
                return response.content

        return await with_retry(_render, attempts=3, operation_name="Screenshot rendering")
