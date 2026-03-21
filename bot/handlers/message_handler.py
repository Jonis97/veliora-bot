import logging

from telegram import InputFile, Update
from telegram.ext import ContextTypes

from bot.services.pipeline_service import ContentPipelineService
from bot.utils.active_source import NeedActiveSourceError
from bot.utils.dedup import MessageDeduplicator
from bot.utils.errors import GenerationFailedError
from bot.utils.intent import UnclearIntentError


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
            result = await self._pipeline.process_message(context.bot, message, context.user_data)
            if result.image_bytes:
                image_file = InputFile(result.image_bytes, filename="educard.png")
                await message.reply_photo(
                    photo=image_file,
                    caption=f"{result.output_intent} · {result.template_used} · {result.source_type}",
                )
            elif result.text_fallback:
                await message.reply_text(result.text_fallback)
            else:
                LOGGER.error("Pipeline returned neither image nor text for message_id=%s", message_id)
                await message.reply_text(
                    "Could not produce a card preview. Please try again in a moment."
                )
        except NeedActiveSourceError:
            await message.reply_text(
                "Send a YouTube link, voice note, or paste text first — then I can build cards, "
                "vocabulary, speaking tasks, tests, or summaries from it."
            )
        except UnclearIntentError as err:
            await message.reply_text(str(err.user_message))
        except GenerationFailedError as err:
            await message.reply_text(err.user_message)
        except ValueError as err:
            LOGGER.warning("Invalid user input message_id=%s: %s", message_id, err)
            await message.reply_text(
                str(err) or "That message doesn’t support. Send text, voice, or a YouTube link."
            )
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("Failed to process message_id=%s: %s", message_id, error)
            await message.reply_text(
                "Something went wrong. Please try again in a moment."
            )
