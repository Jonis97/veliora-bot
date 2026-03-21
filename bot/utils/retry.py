import asyncio
import logging
from typing import Awaitable, Callable, Optional, TypeVar


LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


async def with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    delay_seconds: float = 0.8,
    operation_name: str = "operation",
) -> T:
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt >= attempts:
                break
            backoff = delay_seconds * attempt
            LOGGER.warning(
                "%s failed (%s/%s): %s. Retrying in %.1fs",
                operation_name,
                attempt,
                attempts,
                error,
                backoff,
            )
            await asyncio.sleep(backoff)
    raise RuntimeError(f"{operation_name} failed after {attempts} attempts") from last_error
