"""
MVP: one premium warm_paper_v2 card per request from the latest source only.

- Source is stored per chat_id (in-memory).
- New source replaces the old one; no mixing.
- Template is always warm_paper_v2 (no rerender / no multi-template routing here).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from telegram import Bot, Message

from bot.services.ai_service import AIContentService
from bot.services.screenshot_service import ScreenshotService
from bot.services.template_service import DEFAULT_TEMPLATE, TemplateService
from bot.services.topic_image_service import TopicImageService
from bot.services.transcription_service import VoiceTranscriptionService
from bot.services.youtube_service import YouTubeTranscriptService, extract_video_id
from bot.utils.active_source import NeedActiveSourceError, build_followup_prompt, followup_intent
from bot.utils.errors import GenerationFailedError, TranscriptUnavailableError
from bot.utils.input_parser import parse_template_hint
from bot.utils.intent import OutputIntent

LOGGER = logging.getLogger(__name__)

_CHAT_CONTEXT: dict[int, dict[str, Any]] = {}

_UNSPLASH_GENERIC_WORDS = frozenset(
    {"understanding", "why", "how", "the", "and", "its", "of"}
)


def _first_meaningful_topic_word(title: str) -> Optional[str]:
    """Strip generic words; return first remaining topic token (e.g. for Unsplash query)."""
    for raw in re.findall(r"[A-Za-zА-Яа-яЇїІіЄєҐґ']+", title.lower()):
        w = raw.strip("'")
        if len(w) < 2:
            continue
        if w in _UNSPLASH_GENERIC_WORDS:
            continue
        return w
    return None


def _lesson_visual_keyword_from_card_fields(card_json: dict[str, Any]) -> Optional[str]:
    """Prefer GPT `title`, then `topic`, for the same word-extraction rule as the spec."""
    raw = str(card_json.get("title", "") or card_json.get("topic", "") or "").strip()
    if not raw:
        return None
    return _first_meaningful_topic_word(raw)


def _approved_preview_topic_for_unsplash(source_text: str) -> Optional[str]:
    """Topic line from guided APPROVED PREVIEW block (matches preview_data topic)."""
    if "APPROVED PREVIEW:" not in source_text:
        return None
    start = source_text.find("APPROVED PREVIEW:")
    window = source_text[start : start + 6000]
    m = re.search(r"(?m)^TOPIC:\s*(.+)$", window)
    if not m:
        return None
    line = m.group(1).strip()
    return line or None


_LESSON_CARD_V1_EXTRA_RULES = """
LESSON CARD (lesson_card_v1) — HARD RULES:
- choices: exactly 3–4 real \"this or that\" / \"Option A or Option B?\" items for speaking practice.
- Never use \"—\", placeholders, or empty strings in choices.
- Ground every choice in the approved topic and source; if the transcript is thin, derive options from the topic.
- lead_in_questions: only real questions — never \"—\" or empty strings.
"""


async def _fetch_unsplash_regular_image_url(keyword: str, access_key: str) -> Optional[str]:
    if not access_key or not keyword:
        return None
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                "https://api.unsplash.com/photos/random",
                params={
                    "query": keyword,
                    "orientation": "landscape",
                    "content_filter": "high",
                },
                headers={"Authorization": f"Client-ID {access_key}"},
            )
            response.raise_for_status()
            data = response.json()
            urls = data.get("urls")
            if isinstance(urls, dict):
                out = str(urls.get("regular") or "").strip()
                return out or None
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Unsplash random photo fetch failed: %s", exc)
    return None


_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "phrases": (
        "грамат",
        "grammar",
        "фраз",
        "phrase",
        "структур",
        "sentence pattern",
        "речен",
    ),
    "vocabulary": ("слов", "vocab", "лексик", "перекла", "word", "translat"),
    "questions": (
        "питан",
        "вопрос",
        "question",
        "запитан",
        "обговор",
        "discussion",
    ),
    "lesson": (
        "урок",
        "lesson",
        "розпочат",
        "warm up",
        "розігр",
        "lead in",
    ),
    "exercises": ("вправи", "завдання", "задание", "exercises"),
    "fix_mistakes": ("виправ", "помилки", "fix", "mistakes"),
}

# When two intents have the same keyword match count, prefer earlier in this tuple.
_INTENT_TIEBREAK_ORDER = (
    "phrases",
    "vocabulary",
    "questions",
    "lesson",
    "exercises",
    "fix_mistakes",
)


def _detect_user_intent(message: Message) -> str:
    full_text = (message.text or message.caption or "")
    import re as _re
    fmt_match = _re.search(r'\[FORMAT=(\w+)\]', full_text[:200])
    if fmt_match:
        fmt = fmt_match.group(1).lower()
        fmt_map = {
            "lesson": "lesson",
            "vocabulary": "vocabulary",
            "questions": "questions",
            "phrases": "phrases",
        }
        if fmt in fmt_map:
            return fmt_map[fmt]
    user_text = full_text[:500].lower()
    scores: dict[str, int] = {}
    for intent, keywords in _INTENT_KEYWORDS.items():
        scores[intent] = sum(
            1 for keyword in keywords if keyword in user_text
        )
    best = max(scores.values(), default=0)
    if best == 0:
        return "card"
    leaders = [intent for intent, n in scores.items() if n == best]
    if len(leaders) == 1:
        return leaders[0]
    for intent in _INTENT_TIEBREAK_ORDER:
        if intent in leaders:
            return intent
    return leaders[0]


def _ctx(chat_id: int) -> dict[str, Any]:
    if chat_id not in _CHAT_CONTEXT:
        _CHAT_CONTEXT[chat_id] = {"active_source": None}
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


def _source_type_label_uk(st: str) -> str:
    return {"youtube": "YouTube", "voice": "голос", "text": "текст"}.get(st, st)


def _ground_for_ai(inner: str) -> str:
    return (
        "ПРАВИЛА ПРИВ’ЯЗКИ ДО ДЖЕРЕЛА (обов’язково):\n"
        "- Увесь зміст картки — ЛИШЕ з матеріалу нижче. Не додавай фактів, яких немає в джерелі.\n"
        "- Не вигадуй загальні поради «для будь-якого випадку» — лише те, що випливає з цього тексту/транскрипту.\n"
        "- Якщо джерела замало для пункту — краще пропусти або сформулюй обережно в межах того, що є.\n\n"
        f"{inner}"
    )


@dataclass
class PipelineResult:
    template_used: str
    source_type: str
    output_intent: str
    image_bytes: Optional[bytes] = None
    text_fallback: Optional[str] = None


@dataclass
class _ResolvedSource:
    text_for_ai: str
    source_type: str
    persist_new_source: bool
    intent_seed: str
    persist_body: Optional[str] = None
    video_id: Optional[str] = None
    youtube_thumbnail_url: Optional[str] = None
    intent: str = "card"


@dataclass
class PrepareResult:
    chat_id: int
    resolved: _ResolvedSource
    preface: Optional[str]
    status_line: Optional[str]


class ContentPipelineService:
    def __init__(
        self,
        youtube_service: YouTubeTranscriptService,
        transcription_service: VoiceTranscriptionService,
        ai_service: AIContentService,
        template_service: TemplateService,
        screenshot_service: ScreenshotService,
        topic_image_service: TopicImageService,
        unsplash_access_key: Optional[str] = None,
    ) -> None:
        self._youtube_service = youtube_service
        self._transcription_service = transcription_service
        self._ai_service = ai_service
        self._template_service = template_service
        self._screenshot_service = screenshot_service
        self._topic_image_service = topic_image_service
        self._unsplash_access_key = unsplash_access_key

    async def prepare(self, bot: Bot, message: Message, chat_id: int) -> PrepareResult:
        resolved = await self._resolve_source(bot, message, chat_id)

        if resolved.persist_new_source and resolved.persist_body:
            _save_active_source(
                chat_id,
                source_type=resolved.source_type,
                text=resolved.persist_body,
                video_id=resolved.video_id,
            )
            LOGGER.info("Active source updated chat_id=%s type=%s", chat_id, resolved.source_type)
            preface = (
                "Зрозуміло — зберегла це як поточне джерело. "
                "Створюю для тебе одну навчальну картку warm_paper_v2…"
            )
            status_line = None
        else:
            LOGGER.info("Follow-up on active source chat_id=%s", chat_id)
            preface = None
            status_line = (
                f"Беру матеріал з джерела ({_source_type_label_uk(resolved.source_type)}). "
                "Формую картку…"
            )

        return PrepareResult(
            chat_id=chat_id,
            resolved=resolved,
            preface=preface,
            status_line=status_line,
        )

    async def execute(self, prepare: PrepareResult) -> PipelineResult:
        resolved = prepare.resolved
        raw_ai = resolved.text_for_ai
        grounded = _ground_for_ai(raw_ai)

        if resolved.intent == "vocabulary":
            eff_template = "vocab_card"
        elif resolved.intent == "questions":
            eff_template = "questions_card"
        elif resolved.intent == "lesson":
            eff_template = "lesson_card_v1"
        elif resolved.intent == "phrases":
            eff_template = "phrases_card"
        else:
            eff_template = DEFAULT_TEMPLATE

        if eff_template == "lesson_card_v1":
            grounded = grounded + _LESSON_CARD_V1_EXTRA_RULES

        try:
            card_json = await self._ai_service.generate_card_content(
                grounded,
                eff_template,
                output_intent=OutputIntent.CARD,
                is_followup=not resolved.persist_new_source,
                intent=resolved.intent,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("AI card generation failed: %s", exc)
            raise GenerationFailedError(
                "Не вдалося згенерувати картку. Спробуй ще раз за хвилину."
            ) from exc

        card_json["template"] = eff_template

        card_for_render = dict(card_json)
        if eff_template == "lesson_card_v1":
            merged_title = str(
                card_json.get("topic", "") or card_json.get("title", "") or ""
            ).strip()
            if merged_title:
                card_for_render["title"] = merged_title

        if eff_template == "lesson_card_v1":
            unsplash_url: Optional[str] = None
            preview_topic_line = _approved_preview_topic_for_unsplash(raw_ai)
            preview_kw = (
                _first_meaningful_topic_word(preview_topic_line)
                if preview_topic_line
                else None
            )
            iq = str(card_json.get("image_query") or "").strip()
            if preview_kw:
                query = preview_kw
            elif iq:
                query = iq
            else:
                query = _lesson_visual_keyword_from_card_fields(card_json) or ""
            if query and self._unsplash_access_key:
                unsplash_url = await _fetch_unsplash_regular_image_url(
                    query, self._unsplash_access_key
                )
            if unsplash_url:
                card_for_render["image_url"] = unsplash_url
            elif resolved.source_type == "youtube" and resolved.youtube_thumbnail_url:
                card_for_render["image_url"] = resolved.youtube_thumbnail_url
            else:
                topic = str(card_json.get("title", "education")).strip() or "education"
                image_url = await self._topic_image_service.fetch_topic_image(topic)
                if image_url:
                    card_for_render["image_url"] = image_url
        elif resolved.source_type == "youtube" and resolved.youtube_thumbnail_url:
            card_for_render["image_url"] = resolved.youtube_thumbnail_url
        else:
            topic = str(card_json.get("title", "education")).strip() or "education"
            image_url = await self._topic_image_service.fetch_topic_image(topic)
            if image_url:
                card_for_render["image_url"] = image_url

        used_template = eff_template
        source_type = resolved.source_type
        intent_caption = "Картка"
        html = self._template_service.render_html(card_for_render, eff_template)

        try:
            image_bytes = await self._screenshot_service.html_to_image(html)
            return PipelineResult(
                template_used=used_template,
                source_type=source_type,
                output_intent=intent_caption,
                image_bytes=image_bytes,
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
            )

    @staticmethod
    def _format_card_text_reply(
        card: dict[str, Any],
        template_used: str,
        source_type: str,
        intent_label_s: str,
    ) -> str:
        if template_used == "lesson_card_v1":
            topic = str(card.get("topic", "") or card.get("title", "Lesson")).strip()
            lines_l = [
                f"{intent_label_s} · {template_used} · джерело: {source_type}",
                "",
                f"📌 {topic}",
                "",
                "Lead-in:",
            ]
            li = card.get("lead_in_questions")
            if isinstance(li, list):
                for q in li[:3]:
                    lines_l.append(f"• {str(q).strip()}")
            else:
                lines_l.append("• —")
            lines_l.extend(["", "This or that:"])
            ch = card.get("choices")
            if isinstance(ch, list):
                for c in ch[:6]:
                    if isinstance(c, dict):
                        a = str(
                            c.get("a", "")
                            or c.get("option_a", "")
                            or c.get("A", "")
                        ).strip()
                        b = str(
                            c.get("b", "")
                            or c.get("option_b", "")
                            or c.get("B", "")
                        ).strip()
                        line = f"{a} or {b}?" if a and b else str(
                            c.get("text", "") or c.get("line", "") or ""
                        ).strip()
                        lines_l.append(f"• {line or '—'}")
                    else:
                        lines_l.append(f"• {str(c).strip()}")
            else:
                lines_l.append("• —")
            lines_l.extend(["", "(Попередній перегляд зображення недоступний — текст картки вище.)"])
            text = "\n".join(lines_l)
            if len(text) > 4000:
                return text[:3997] + "..."
            return text

        if template_used == "phrases_card":
            title_p = str(card.get("title", "Learning Card")).strip()
            sub_p = str(card.get("subtitle", "")).strip()
            lines_p = [
                f"{intent_label_s} · {template_used} · джерело: {source_type}",
                "",
                f"📌 {title_p}",
            ]
            if sub_p:
                lines_p.append(sub_p)
            lines_p.append("")
            ph = card.get("phrases")
            if isinstance(ph, list):
                for i, item in enumerate(ph[:5], 1):
                    if not isinstance(item, dict):
                        continue
                    lines_p.append(f"{i}. {str(item.get('phrase', '')).strip()}")
                    tr = str(item.get("translation", "")).strip()
                    if tr:
                        lines_p.append(f"   ({tr})")
                    fo = str(item.get("formula", "")).strip()
                    if fo:
                        lines_p.append(f"   [{fo}]")
                    ex = item.get("examples")
                    if isinstance(ex, list):
                        for exs in ex[:2]:
                            lines_p.append(f"   • {str(exs).strip()}")
                    lines_p.append("")
            lines_p.append("(Попередній перегляд зображення недоступний — текст картки вище.)")
            text_p = "\n".join(lines_p)
            if len(text_p) > 4000:
                return text_p[:3997] + "..."
            return text_p

        title = str(card.get("title", "Learning Card")).strip()
        subtitle = str(card.get("subtitle", "")).strip()
        raw_bullets = card.get("bullets") or []
        if not isinstance(raw_bullets, list):
            raw_bullets = []
        cta = str(card.get("cta", "")).strip()
        lines = [
            f"{intent_label_s} · {template_used} · джерело: {source_type}",
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
        lines.append("(Попередній перегляд зображення недоступний — текст картки вище.)")
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
        intent = _detect_user_intent(message)

        if message.voice:
            transcript = await self._transcription_service.transcribe_voice(bot, message.voice.file_id)
            if not (transcript or "").strip():
                raise TranscriptUnavailableError(
                    "Не вдалось розпізнати голос. Надішли текст або посилання — зроблю картку."
                )
            return _ResolvedSource(
                text_for_ai=transcript,
                source_type="voice",
                persist_new_source=True,
                intent_seed="",
                persist_body=transcript,
                intent=intent,
            )

        raw = (message.text or message.caption or "").strip()
        if not raw:
            raise ValueError("Надішли текст, голосове повідомлення або посилання на YouTube.")

        cleaned_text, _template = parse_template_hint(raw)
        video_id = extract_video_id(cleaned_text)
        if video_id:
            try:
                transcript = await self._youtube_service.fetch_transcript(video_id)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "YouTube transcript fetch failed for video_id=%s: %s",
                    video_id,
                    exc,
                )
                raise TranscriptUnavailableError() from exc
            if not (transcript or "").strip():
                LOGGER.warning("YouTube transcript empty for video_id=%s", video_id)
                raise TranscriptUnavailableError()
            image_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
            return _ResolvedSource(
                text_for_ai=transcript,
                source_type="youtube",
                persist_new_source=True,
                intent_seed="",
                persist_body=transcript,
                video_id=video_id,
                youtube_thumbnail_url=image_url,
                intent=intent,
            )

        if not cleaned_text.strip() and _template:
            if active is None:
                raise NeedActiveSourceError()
            instruction = (
                "The user sent only a template tag. Regenerate ONE warm_paper_v2 study card from the same source material. "
                "Do not change the topic; use only the stored source."
            )
            return _ResolvedSource(
                text_for_ai=build_followup_prompt(instruction, active),
                source_type=str(active.get("type", "text")),
                persist_new_source=False,
                intent_seed="",
                intent=intent,
            )

        if followup_intent(cleaned_text):
            if active is None:
                raise NeedActiveSourceError()
            return _ResolvedSource(
                text_for_ai=build_followup_prompt(cleaned_text, active),
                source_type=str(active.get("type", "text")),
                persist_new_source=False,
                intent_seed=cleaned_text,
                intent=intent,
            )

        return _ResolvedSource(
            text_for_ai=cleaned_text,
            source_type="text",
            persist_new_source=True,
            intent_seed=cleaned_text,
            persist_body=cleaned_text,
            intent=intent,
        )

    async def process_message(
        self,
        bot: Bot,
        message: Message,
        user_data: Optional[Any] = None,
    ) -> PipelineResult:
        chat_id = message.chat_id
        pr = await self.prepare(bot, message, chat_id)
        return await self.execute(pr)
