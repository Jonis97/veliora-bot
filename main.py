import logging

from openai import AsyncOpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from bot.handlers.message_handler import MessageHandlerService
from bot.services.ai_service import AIContentService
from bot.services.pipeline_service import ContentPipelineService
from bot.services.screenshot_service import ScreenshotService
from bot.services.template_service import TemplateService
from bot.services.topic_image_service import TopicImageService
from bot.services.transcription_service import VoiceTranscriptionService
from bot.services.youtube_service import YouTubeTranscriptService
from bot.utils.config import load_settings
from bot.utils.dedup import MessageDeduplicator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text(
        "Send text, voice, or a YouTube link.\n"
        "Optional template tag: [template:warm_paper], [template:kitchen_collage], "
        "[template:influencer_card], or v2: [template:warm_paper_v2], "
        "[template:kitchen_collage_v2], [template:influencer_card_v2]."
    )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    LOGGER.exception("Unhandled Telegram error", exc_info=context.error)


def build_application() -> tuple[Application, str, int, str, str]:
    settings = load_settings()
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

    pipeline = ContentPipelineService(
        youtube_service=YouTubeTranscriptService(settings.supadata_api_key),
        transcription_service=VoiceTranscriptionService(openai_client),
        ai_service=AIContentService(openai_client, settings.openai_model),
        template_service=TemplateService(),
        screenshot_service=ScreenshotService(settings.screenshotone_api_key),
        topic_image_service=TopicImageService(openai_client),
    )
    deduplicator = MessageDeduplicator(ttl_seconds=600)
    message_handler = MessageHandlerService(pipeline, deduplicator)

    app = Application.builder().token(settings.telegram_token).build()
    app.add_handler(CommandHandler("start", start_command))
    accepted_inputs = (filters.TEXT | filters.VOICE | filters.CAPTION) & ~filters.COMMAND
    app.add_handler(MessageHandler(accepted_inputs, message_handler.handle_message))
    app.add_error_handler(error_handler)

    return (
        app,
        settings.webhook_url,
        settings.port,
        settings.webhook_path,
        settings.webhook_secret_token,
    )


def main() -> None:
    app, webhook_url, port, webhook_path, secret_token = build_application()
    full_webhook_url = webhook_url.rstrip("/") + webhook_path
    LOGGER.info("Starting bot with webhook: %s", full_webhook_url)
    app.run_webhook(
        listen="0.0.0.0",
        port=port,
        url_path=webhook_path.lstrip("/"),
        webhook_url=full_webhook_url,
        secret_token=secret_token,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
