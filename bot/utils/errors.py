"""User-facing pipeline errors with short, clear messages."""


class GenerationFailedError(Exception):
    """AI or rendering failed after retries."""

    def __init__(
        self,
        message: str = "Couldn’t generate your study content. Please try again in a moment.",
    ) -> None:
        self.user_message = message
        super().__init__(message)
