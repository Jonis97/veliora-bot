"""
Core product flow: per-chat memory, source grounding, intent routing, re-render without re-fetch.

All chat state lives in an in-memory dict keyed by chat_id (MVP). template_service / screenshot_service are only called, not modified here.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Optional

from telegram import Bot, Message

from bot.services.ai_service import AIContentService
from bot.services.screenshot_service import ScreenshotService
from bot.services.template_service import ALLOWED_TEMPLATES, DEFAULT_TEMPLATE, TemplateService
from bot.services.topic_image_service import TopicImageService
from bot.services.transcription_service import VoiceTranscriptionService
from bot.services.youtube_service import YouTubeTranscriptService, extract_video_id
from bot.utils.active_source import (
    NeedActiveSourceError,
    build_followup_prompt,
    followup_intent,
)
from bot.utils.errors import GenerationFailedError
from bot.utils.input_parser import parse_template_hint
from bot.utils.intent import OutputIntent, UnclearIntentError, intent_label, resolve_output_intent

LOGGER = logging.getLogger(__name__)

# --- In-memory per-chat context (MVP). Not persisted across restarts. ---
_CHAT_CONTEXT: dict[int, dict[str, Any]] = {}


def _ctx(chat_id: int) -> dict[str, Any]:
    if chat_id not in _CHAT_CONTEXT:
        _CHAT_CONTEXT[chat_id] = {
            "active_source": None,
            "generated_card": None,
            "last_template": DEFAULT_TEMPLATE,
            "last_intent": None,
        }
    return _CHAT_CONTEXT[chat_id]


def _load_active_source(chat_id: int) -> Optional[dict[str, Any]]:
    raw = _ctx(chat_id).get("active_source")
    if not isinstance(raw, dict):
        return None
    st = raw.get("type")
    text = raw.get("text")
    if st not in ("youtube", "text", "voice") or not isinstance(text, str) or not text.strip():
        return None
    return raw


def _save_active_source(
    chat_id: int,
    *,
    source_type: str,
    text: str,
    video_id: Optional[str] = None,
) -> None:
    payload: dict[str, Any] = {"type": source_type, "text": text.strip()}
    if video_id:
        payload["video_id"] = video_id
    _ctx(chat_id)["active_source"] = payload


def _save_generated_card(chat_id: int, card: dict[str, Any]) -> None:
    """Deep copy of last rendered card JSON for template re-renders."""
    _ctx(chat_id)["generated_card"] = copy.deepcopy(card)
    _ctx(chat_id)["last_template"] = str(card.get("template") or DEFAULT_TEMPLATE)


def _effective_template(user_hint: Optional[str]) -> str:
    if user_hint and user_hint in ALLOWED_TEMPLATES:
        return user_hint
    return DEFAULT_TEMPLATE


def _source_type_label(st: str) -> str:
    return {"youtube": "YouTube", "voice": "voice", "text": "text"}.get(st, st)


def _ground_for_ai(inner: str) -> str:
    """Keep model on the source topic; user ops (translate, speaking) apply to this material only."""
    return (
        "GROUNDING (critical):\n"
        "- Base EVERYTHING only on the source material below.\n"
        "- The user request is an operation ON that material (translate words, speaking, quiz, etc.). "
        "Do NOT replace the topic with a generic lesson about that operation "
        "(e.g. if the source is a French alphabet video and the user says “translate words”, "
        "work with words from that source — not a new essay about translation).\n\n"
        f"{inner}"
    )


def _combined_request_note(seed: str) -> str:
    n = seed.lower()
    if (" and " in n or " & " in n or " also " in n) and len(n) > 12:
        return (
            "\n\n[User asked for multiple things in one message: produce ONE coherent combined output "
            "from the same source, unless impossible — then prioritize the clearest part.]"
        )
    return ""


def _post_route_intent(seed: str, intent: OutputIntent) -> OutputIntent:
    """Map messy phrases without editing intent.py."""
    tl = seed.lower()
    if "translate" in tl or "translation" in tl or "переклад" in tl:
        return OutputIntent.VOCABULARY
    if "simplify" in tl or "simpler" in tl or "easier to understand" in tl:
        return OutputIntent.SUMMARY
    return intent


def _resolve_intent_safe(seed: str, *, is_follow_up: bool) -> OutputIntent:
    try:
        intent = resolve_output_intent(seed, is_follow_up=is_follow_up)
    except UnclearIntentError:
        # Safe default when the message has substance; re-raise only for genuinely empty/vague follow-ups.
        s = seed.strip()
        if len(s) < 6 or s.lower() in {"do it", "go", "??", "help", "hmm"}:
            raise
        return _post_route_intent(seed, OutputIntent.CARD)
    return _post_route_intent(seed, intent)


@dataclass
class PipelineResult:
    """Screenshot or text fallback; optional style tip after first generation."""

    template_used: str
    source_type: str
    output_intent: str
    image_bytes: Optional[bytes] = None
    text_fallback: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class _ResolvedSource:
    text_for_ai: str
    template: Optional[str]
    source_type: str
    persist_new_source: bool
    intent_seed: str
    persist_body: Optional[str] = None
    video_id: Optional[str] = None
    rerender_only: bool = False


@dataclass
class PrepareResult:
    """First phase: resolve source, optional instant re-render, UX lines for the handler."""

    chat_id: int
    resolved: _ResolvedSource
    output_intent: OutputIntent
    preface: Optional[str]
    status_line: Optional[str]
    rerender_complete: Optional[PipelineResult] = None


class ContentPipelineService:
    def __init__(
        self,
        youtube_service: YouTubeTranscriptService,
        transcription_service: VoiceTranscriptionService,
        ai_service: AIContentService,
        template_service: TemplateService,
        screenshot_service: ScreenshotService,
        topic_image_service: TopicImageService,
    ) -> None:
        self._youtube_service = youtube_service
        self._transcription_service = transcription_service
        self._ai_service = ai_service
        self._template_service = template_service
        self._screenshot_service = screenshot_service
        self._topic_image_service = topic_image_service

    async def prepare(self, bot: Bot, message: Message, chat_id: int) -> PrepareResult:
        """
        Resolve source / optional template-only re-render (no transcript/API re-fetch).
        Returns preface + status for the handler, or a finished PipelineResult if re-render only.
        """
        resolved = await self._resolve_source(bot, message, chat_id)

        if resolved.rerender_only:
            ctx = _ctx(chat_id)
            base = ctx.get("generated_card")
            if not isinstance(base, dict):
                raise ValueError("No saved card to re-style. Generate a card first, then add a template tag.")
            tmpl = _effective_template(resolved.template)
            card = copy.deepcopy(base)
            card["template"] = tmpl
            html = self._template_service.render_html(card, tmpl)
            try:
                image_bytes = await self._screenshot_service.html_to_image(html)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Re-render screenshot failed: %s", exc, exc_info=True)
                raise GenerationFailedError("Couldn’t render that layout. Try another template or ask again.") from exc
            ctx["last_template"] = tmpl
            _save_generated_card(chat_id, card)
            done = PipelineResult(
                template_used=tmpl,
                source_type=resolved.source_type,
                output_intent="Re-style",
                image_bytes=image_bytes,
                suggestion=self._style_tip(tmpl),
            )
            return PrepareResult(
                chat_id=chat_id,
                resolved=resolved,
                output_intent=OutputIntent.CARD,
                preface=None,
                status_line=None,
                rerender_complete=done,
            )

        if resolved.persist_new_source and resolved.persist_body:
            _save_active_source(
                chat_id,
                source_type=resolved.source_type,
                text=resolved.persist_body,
                video_id=resolved.video_id,
            )
            LOGGER.info("Active source updated chat_id=%s type=%s", chat_id, resolved.source_type)
        elif not resolved.persist_new_source:
            LOGGER.info("Follow-up on active source chat_id=%s", chat_id)

        seed = (resolved.intent_seed or "").strip()
        if not seed:
            output_intent = OutputIntent.CARD
        else:
            output_intent = _resolve_intent_safe(
                seed,
                is_follow_up=not resolved.persist_new_source,
            )

        preface: Optional[str] = None
        status_line: Optional[str] = None
        if resolved.persist_new_source:
            preface = (
                "Got it — I saved this as your current source. "
                "Now building your study card (you can also ask for vocabulary, speaking, test, or summary next)."
            )
        else:
            status_line = (
                f"Using your {_source_type_label(resolved.source_type)} source. "
                f"Now creating: {intent_label(output_intent)}."
            )

        return PrepareResult(
            chat_id=chat_id,
            resolved=resolved,
            output_intent=output_intent,
            preface=preface,
            status_line=status_line,
            rerender_complete=None,
        )

    async def execute(self, prepare: PrepareResult) -> PipelineResult:
        """AI generation + render + persist structured card (not used for re-render path)."""
        resolved = prepare.resolved
        chat_id = prepare.chat_id
        output_intent = prepare.output_intent

        raw_ai = resolved.text_for_ai
        grounded = _ground_for_ai(raw_ai) + _combined_request_note(resolved.intent_seed or "")

        user_template = _effective_template(resolved.template)

        try:
            card_json = await self._ai_service.generate_card_content(
                grounded,
                user_template,
                output_intent=output_intent,
                is_followup=not resolved.persist_new_source,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("AI card generation failed: %s", exc)
            raise GenerationFailedError() from exc

        eff = _effective_template(str(card_json.get("template") or user_template))
        card_json["template"] = eff

        topic = str(card_json.get("title", "education")).strip() or "education"
        image_url = await self._topic_image_service.fetch_topic_image(topic)
        card_for_render = dict(card_json)
        if image_url:
            card_for_render["image_url"] = image_url

        used_template = eff
        source_type = resolved.source_type
        intent_caption = intent_label(output_intent)
        html = self._template_service.render_html(card_for_render, used_template)

        _ctx(chat_id)["last_intent"] = intent_caption
        _save_generated_card(chat_id, card_for_render)

        try:
            image_bytes = await self._screenshot_service.html_to_image(html)
            return PipelineResult(
                template_used=used_template,
                source_type=source_type,
                output_intent=intent_caption,
                image_bytes=image_bytes,
                suggestion=self._style_tip(used_template),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Screenshot rendering failed; sending text fallback. Error: %s",
                exc,
                exc_info=True,
            )
            text_body = self._format_card_text_reply(
                card_for_render, used_template, source_type, intent_caption
            )
            return PipelineResult(
                template_used=used_template,
                source_type=source_type,
                output_intent=intent_caption,
                text_fallback=text_body,
                suggestion=self._style_tip(used_template),
            )

    @staticmethod
    def _style_tip(current: str) -> str:
        """Optional nudge — does not block generation."""
        alts = [t for t in sorted(ALLOWED_TEMPLATES) if t != current][:3]
        if not alts:
            return ""
        tags = " ".join(f"[template:{a}]" for a in alts)
        return f"Other looks (same content): {tags}"

    @staticmethod
    def _format_card_text_reply(
        card: dict[str, Any],
        template_used: str,
        source_type: str,
        intent_label_s: str,
    ) -> str:
        title = str(card.get("title", "Learning Card")).strip()
        subtitle = str(card.get("subtitle", "")).strip()
        raw_bullets = card.get("bullets") or []
        if not isinstance(raw_bullets, list):
            raw_bullets = []
        cta = str(card.get("cta", "")).strip()
        lines = [
            f"{intent_label_s} · {template_used} · source: {source_type}",
            "",
            f"📌 {title}",
        ]
        if subtitle:
            lines.append(subtitle)
        punch = str(card.get("punchline", "") or "").strip()
        if punch:
            lines.extend(["", f"⭐ {punch}"])
        contrast = card.get("contrast")
        if isinstance(contrast, dict):
            w = str(contrast.get("wrong", "") or "").strip()
            b = str(contrast.get("better", "") or "").strip()
            if w or b:
                lines.append("")
                if w:
                    lines.append(f"✗ {w}")
                if b:
                    lines.append(f"✓ {b}")
        lines.append("")
        for item in raw_bullets[:8]:
            lines.append(f"• {str(item).strip()}")
        if cta:
            lines.extend(["", f"💡 {cta}"])
        lines.append("")
        lines.append("(Image preview unavailable — text card above.)")
        text = "\n".join(lines)
        if len(text) > 4000:
            return text[:3997] + "..."
        return text

    async def _resolve_source(
        self,
        bot: Bot,
        message: Message,
        chat_id: int,
    ) -> _ResolvedSource:
        active = _load_active_source(chat_id)

        if message.voice:
            transcript = await self._transcription_service.transcribe_voice(bot, message.voice.file_id)
            if not transcript:
                raise RuntimeError("Voice transcription returned empty text")
            template: Optional[str] = None
            cap = (message.caption or "").strip()
            if cap:
                _, template = parse_template_hint(cap)
            return _ResolvedSource(
                text_for_ai=transcript,
                template=template,
                source_type="voice",
                persist_new_source=True,
                intent_seed="",
                persist_body=transcript,
            )

        raw = (message.text or message.caption or "").strip()
        if not raw:
            raise ValueError("Unsupported message type. Send text, a voice message, or a YouTube link.")

        cleaned_text, template = parse_template_hint(raw)
        video_id = extract_video_id(cleaned_text)
        if video_id:
            try:
                transcript = await self._youtube_service.fetch_transcript(video_id)
                return _ResolvedSource(
                    text_for_ai=transcript,
                    template=template,
                    source_type="youtube",
                    persist_new_source=True,
                    intent_seed="",
                    persist_body=transcript,
                    video_id=video_id,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "YouTube transcript fetch failed for video_id=%s: %s. Falling back to URL-only context.",
                    video_id,
                    exc,
                )
                fallback_text = self._youtube_url_only_context(video_id, cleaned_text)
                return _ResolvedSource(
                    text_for_ai=fallback_text,
                    template=template,
                    source_type="youtube",
                    persist_new_source=True,
                    intent_seed="",
                    persist_body=fallback_text,
                    video_id=video_id,
                )

        ctx = _ctx(chat_id)
        stored_card = ctx.get("generated_card")

        if not cleaned_text.strip() and template:
            if stored_card and isinstance(stored_card, dict) and template in ALLOWED_TEMPLATES:
                if active is None:
                    raise NeedActiveSourceError()
                return _ResolvedSource(
                    text_for_ai="",
                    template=template,
                    source_type=str(active.get("type", "text")),
                    persist_new_source=False,
                    intent_seed="",
                    rerender_only=True,
                )
            if active is None:
                raise NeedActiveSourceError()
            instruction = (
                "The user only sent a template preference. Regenerate the learning card for the same "
                "source material using the preferred template."
            )
            return _ResolvedSource(
                text_for_ai=build_followup_prompt(instruction, active),
                template=template,
                source_type=str(active.get("type", "text")),
                persist_new_source=False,
                intent_seed="",
            )

        if followup_intent(cleaned_text):
            if active is None:
                raise NeedActiveSourceError()
            return _ResolvedSource(
                text_for_ai=build_followup_prompt(cleaned_text, active),
                template=template,
                source_type=str(active.get("type", "text")),
                persist_new_source=False,
                intent_seed=cleaned_text,
            )

        return _ResolvedSource(
            text_for_ai=cleaned_text,
            template=template,
            source_type="text",
            persist_new_source=True,
            intent_seed=cleaned_text,
            persist_body=cleaned_text,
        )

    @staticmethod
    def _youtube_url_only_context(video_id: str, user_link_text: str) -> str:
        canonical_url = f"https://www.youtube.com/watch?v={video_id}"
        return (
            "[Transcript unavailable — no transcript text was fetched.]\n\n"
            f"Video URL: {canonical_url}\n"
            f"Video ID: {video_id}\n"
            f"User message / link: {user_link_text}\n\n"
            "Infer the likely topic or theme from the URL, video ID, and any cues in the link text. "
            "Create an educational flashcard about that topic with clear, generally accurate learning points. "
            "If the topic is ambiguous, choose a reasonable educational angle and frame the card helpfully."
        )

    # Backwards-compatible entry (tests / callers): single-shot without UX split
    async def process_message(
        self,
        bot: Bot,
        message: Message,
        user_data: Optional[Any] = None,
    ) -> PipelineResult:
        chat_id = message.chat_id
        pr = await self.prepare(bot, message, chat_id)
        if pr.rerender_complete:
            return pr.rerender_complete
        return await self.execute(pr)
