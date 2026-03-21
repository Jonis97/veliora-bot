import asyncio
import time


class MessageDeduplicator:
    """In-memory message deduplication by chat_id + message_id."""

    def __init__(self, ttl_seconds: int = 600) -> None:
        self._ttl_seconds = ttl_seconds
        self._seen: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def is_duplicate(self, chat_id: int, message_id: int) -> bool:
        key = f"{chat_id}:{message_id}"
        now = time.time()
        async with self._lock:
            self._cleanup(now)
            if key in self._seen:
                return True
            self._seen[key] = now
            return False

    def _cleanup(self, now: float) -> None:
        expired = [k for k, ts in self._seen.items() if now - ts > self._ttl_seconds]
        for key in expired:
            self._seen.pop(key, None)
