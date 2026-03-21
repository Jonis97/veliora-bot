"""User-facing pipeline errors with short, clear messages."""


class GenerationFailedError(Exception):
    """AI or rendering failed after retries."""

    def __init__(
        self,
        message: str = "Не вдалося згенерувати картку. Спробуй ще раз за хвилину.",
    ) -> None:
        self.user_message = message
        super().__init__(message)
