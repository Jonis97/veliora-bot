import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    openai_api_key: str
    supadata_api_key: str
    screenshotone_api_key: str
    webhook_url: str
    port: int
    webhook_path: str
    webhook_secret_token: str
    openai_model: str
    unsplash_access_key: Optional[str] = None


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_settings() -> Settings:
    return Settings(
        telegram_token=_required("TELEGRAM_TOKEN"),
        openai_api_key=_required("OPENAI_API_KEY"),
        supadata_api_key=_required("SUPADATA_API_KEY"),
        screenshotone_api_key=_required("SCREENSHOTONE_API_KEY"),
        webhook_url=_required("WEBHOOK_URL"),
        port=int(os.getenv("PORT", "8080")),
        webhook_path=os.getenv("WEBHOOK_PATH", "/telegram/webhook"),
        webhook_secret_token=os.getenv("WEBHOOK_SECRET_TOKEN", "veliora-secret-token"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        unsplash_access_key=os.getenv("UNSPLASH_ACCESS_KEY"),
    )
