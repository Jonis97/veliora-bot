"""User-facing pipeline errors with short, clear messages."""


class TranscriptUnavailableError(Exception):
    """YouTube transcript missing/unavailable or empty voice transcription — do not generate a card."""

    def __init__(
        self,
        message: str | None = None,
    ) -> None:
        self.user_message = message or (
            "Не вдалось отримати транскрипт цього відео.\n"
            "Скопіюй текст з відео або встав свій матеріал — зроблю картку."
        )
        super().__init__(self.user_message)


class GenerationFailedError(Exception):
    """AI or rendering failed after retries."""

    def __init__(
        self,
        message: str = "Не вдалося згенерувати картку. Спробуй ще раз за хвилину.",
    ) -> None:
        self.user_message = message
        super().__init__(message)
