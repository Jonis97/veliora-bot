import asyncio
import logging

import uvicorn
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from bot.api.routes import api, init_api
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


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    LOGGER.exception("Unhandled Telegram error", exc_info=context.error)


def build_application():
    settings = load_settings()
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    youtube_service = YouTubeTranscriptService(settings.supadata_api_key)
    pipeline = ContentPipelineService(
        youtube_service=youtube_service,
        transcription_service=VoiceTranscriptionService(openai_client),
        ai_service=AIContentService(anthropic_client),
        template_service=TemplateService(),
        screenshot_service=ScreenshotService(settings.screenshotone_api_key),
        topic_image_service=TopicImageService(openai_client),
        unsplash_access_key=settings.unsplash_access_key,
    )
    deduplicator = MessageDeduplicator(ttl_seconds=600)
    message_handler = MessageHandlerService(
        pipeline,
        deduplicator,
        youtube_service,
        anthropic_client,
    )

    app = Application.builder().token(settings.telegram_token).build()
    app.add_handler(CommandHandler("start", message_handler.start_command))
    app.add_handler(CallbackQueryHandler(message_handler.handle_callback, pattern=r"^onb_"))
    accepted_inputs = (filters.TEXT | filters.VOICE | filters.CAPTION) & ~filters.COMMAND
    app.add_handler(MessageHandler(accepted_inputs, message_handler.handle_message))
    app.add_error_handler(error_handler)

    return app, settings.webhook_url, settings.port, settings.webhook_path, settings.webhook_secret_token, pipeline


def main() -> None:
    app, webhook_url, port, webhook_path, secret_token, pipeline = build_application()
    init_api(pipeline, app.bot)

    # Нормалізація webhook_path
    normalized_path = "/" + webhook_path.strip("/")
    full_webhook_url = webhook_url.rstrip("/") + normalized_path
    LOGGER.info("Starting bot + API on port %s", port)

    async def run():
        await app.initialize()
        await app.bot.set_webhook(
            url=full_webhook_url,
            secret_token=secret_token if secret_token else None,
            drop_pending_updates=True,
        )
        await app.start()

        from fastapi import Request, Response
        from telegram import Update
        import json as _json

        @api.post(normalized_path)
        async def telegram_webhook(request: Request):
            data = await request.body()
            update = Update.de_json(_json.loads(data), app.bot)
            await app.process_update(update)
            return Response(status_code=200)

        config = uvicorn.Config(api, host="0.0.0.0", port=port, log_level="info")
        server = uvicorn.Server(config)
        try:
            await server.serve()
        finally:
            await app.stop()
            await app.shutdown()

    asyncio.run(run())


if __name__ == "__main__":
    main()
