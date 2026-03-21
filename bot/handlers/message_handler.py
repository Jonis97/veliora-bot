import logging

from telegram import InputFile, Update
from telegram.ext import ContextTypes

from bot.services.pipeline_service import ContentPipelineService
from bot.utils.dedup import MessageDeduplicator


LOGGER = logging.getLogger(__name__)


class MessageHandlerService:
    def __init__(self, pipeline: ContentPipelineService, deduplicator: MessageDeduplicator) -> None:
        self._pipeline = pipeline
        self._deduplicator = deduplicator

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return

        message = update.message
        chat_id = message.chat_id
        message_id = message.message_id

        if await self._deduplicator.is_duplicate(chat_id, message_id):
            LOGGER.info("Skipping duplicate message chat_id=%s message_id=%s", chat_id, message_id)
            return

        await message.reply_text("Processing your content, this can take a few seconds...")

        try:
            result = await self._pipeline.process_message(context.bot, message)
            if result.image_bytes:
                image_file = InputFile(result.image_bytes, filename="educard.png")
                await message.reply_photo(
                    photo=image_file,
                    caption=f"Template: {result.template_used} | Source: {result.source_type}",
                )
            elif result.text_fallback:
                await message.reply_text(result.text_fallback)
            else:
                LOGGER.error("Pipeline returned neither image nor text for message_id=%s", message_id)
                await message.reply_text(
                    "Could not produce a card preview. Please try again in a moment."
                )
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("Failed to process message_id=%s: %s", message_id, error)
            await message.reply_text(
                "Something went wrong while generating the card. Please try again in a moment."
            )
