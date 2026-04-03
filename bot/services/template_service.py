import re
from html import escape
from typing import Any, List, Optional, Tuple

from bot.utils.image_policy import is_safe_topic_image_url


ALLOWED_TEMPLATES = {
    "warm_paper",
    "kitchen_collage",
    "influencer_card",
    "warm_paper_v2",
    "kitchen_collage_v2",
    "influencer_card_v2",
    "vocab_card",
    "questions_card",
    "speaking_card_v2",
    "speaking_poster_card",
    "lesson_card_v1",
    "lesson_art_v1",
    "phrases_card",
}

# Default when no template tag and AI omits template: test v2 first; v1 still selectable explicitly.
DEFAULT_TEMPLATE = "warm_paper_v2"

# Hero: subtle photo treatment + gradient-only fallback (no random/sticker imagery).
HERO_LAYER_CSS = """
    .hero-media.hero-has-img .hero-img-stack {
      position: relative;
      width: 100%;
      height: 100%;
      border-radius: inherit;
      overflow: hidden;
    }
    .hero-media.hero-has-img .hero-img {
      filter: saturate(0.9) contrast(1.04);
    }
    .hero-img-scrim {
      position: absolute;
      inset: 0;
      pointer-events: none;
      border-radius: inherit;
      background: linear-gradient(180deg, rgba(255, 252, 248, 0.12) 0%, rgba(35, 28, 22, 0.18) 100%);
    }
    .hero-media.hero-no-img .hero-bg-fill {
      width: 100%;
      height: 100%;
      min-height: 140px;
      border-radius: inherit;
      background:
        radial-gradient(ellipse 100% 80% at 50% 30%, rgba(255, 252, 248, 0.55) 0%, transparent 45%),
        linear-gradient(165deg, #ebe3d9 0%, #d8cec4 55%, #c9beb3 100%);
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.48);
    }
    .hero-media.hero-no-img.hero-warm .hero-bg-fill {
      background:
        radial-gradient(ellipse 90% 70% at 45% 35%, rgba(255, 248, 240, 0.5) 0%, transparent 50%),
        linear-gradient(165deg, #efe6dc 0%, #e0d4c8 100%);
    }
    .hero-media.hero-no-img.hero-kitchen .hero-bg-fill {
      background:
        radial-gradient(ellipse 90% 70% at 45% 35%, rgba(255, 248, 240, 0.5) 0%, transparent 50%),
        linear-gradient(165deg, #efe6dc 0%, #e0d4c8 100%);
    }
    .hero-media.hero-no-img.hero-influencer .hero-bg-fill {
      background:
        radial-gradient(ellipse 85% 65% at 55% 25%, rgba(255, 255, 255, 0.35) 0%, transparent 50%),
        linear-gradient(155deg, #f0ede8 0%, #e2ddd6 100%);
    }
    .hero-media.hero-no-img.hero-warm_v2 .hero-bg-fill {
      min-height: 156px;
      background:
        radial-gradient(ellipse 90% 70% at 45% 35%, rgba(255, 248, 242, 0.65) 0%, transparent 50%),
        radial-gradient(ellipse 60% 50% at 85% 80%, rgba(230, 210, 190, 0.22) 0%, transparent 45%),
        linear-gradient(175deg, #ebe3d9 0%, #dfd4c8 45%, #d2c7bc 100%);
    }
    .hero-media.hero-no-img.hero-kitchen_v2 .hero-bg-fill {
      min-height: 148px;
      background:
        radial-gradient(ellipse 90% 70% at 45% 35%, rgba(255, 250, 245, 0.55) 0%, transparent 48%),
        linear-gradient(175deg, #ebe3d9 0%, #dfd4c8 70%, #d2c7bc 100%);
    }
    .hero-media.hero-no-img.hero-influencer_v2 .hero-bg-fill {
      min-height: 156px;
      background:
        radial-gradient(ellipse 90% 60% at 50% 0%, rgba(255, 252, 248, 0.5) 0%, transparent 50%),
        linear-gradient(165deg, #efe8e0 0%, #e0d4c8 100%);
    }
""".strip()


def _phrases_example_line_html(raw: str) -> str:
    """**segments** in example strings become bold blue spans; escape the rest."""
    if not raw:
        return ""
    parts = re.split(r"(\*\*[\s\S]*?\*\*)", raw)
    chunks: List[str] = []
    for p in parts:
        if p.startswith("**") and p.endswith("**") and len(p) >= 4:
            inner = escape(p[2:-2].strip())
            chunks.append(f'<span class="pc-ex-bold">{inner}</span>')
        else:
            chunks.append(escape(p))
    return "".join(chunks)


def _normalize_card(raw: dict[str, Any]) -> dict[str, Any]:
    bullets = raw.get("bullets", [])
    if not isinstance(bullets, list):
        bullets = []
    # Short, sharp points (AI targets 3–4; cap at 4 for layout)
    cleaned_bullets = [escape(str(item)) for item in bullets[:4] if str(item).strip()]
    raw_url = str(raw.get("image_url", "") or raw.get("photo", "") or "").strip()
    image_url = ""
    if raw_url.startswith("https://") and is_safe_topic_image_url(raw_url):
        image_url = raw_url
    pl_raw = str(raw.get("punchline", "") or raw.get("takeaway", "") or "").strip()
    punchline = escape(pl_raw[:220]) if pl_raw else ""
    contrast = raw.get("contrast")
    cw = cb = ""
    if isinstance(contrast, dict):
        cw = escape(str(contrast.get("wrong", "") or contrast.get("weak", "")).strip()[:220])
        cb = escape(str(contrast.get("better", "") or contrast.get("strong", "")).strip()[:220])
    voc_in: Any = raw.get("vocabulary")
    vocabulary_lines: List[str] = []
    if isinstance(voc_in, list):
        for x in voc_in[:4]:
            if isinstance(x, dict):
                t = str(x.get("term", "") or "").strip()
                tr = str(x.get("translation", "") or x.get("gloss", "") or "").strip()
                if t or tr:
                    line = f"{t} — {tr}" if t and tr else (t or tr)
                    vocabulary_lines.append(escape(line)[:200])
            elif str(x).strip():
                vocabulary_lines.append(escape(str(x).strip())[:200])
    mc_in: Any = raw.get("mcq_brackets") or raw.get("mcq_exercises")
    mcq_bracket_lines: List[str] = []
    if isinstance(mc_in, list):
        mcq_bracket_lines = [escape(str(x).strip())[:320] for x in mc_in[:4] if str(x).strip()]

    vocab_examples_in: Any = raw.get("vocabulary_examples")
    if not isinstance(vocab_examples_in, list):
        vocab_examples_in = []
    vocab_card_rows: List[dict[str, str]] = []
    if isinstance(voc_in, list):
        for i, item in enumerate(voc_in[:8]):
            ex_raw = ""
            if i < len(vocab_examples_in):
                ex_raw = str(vocab_examples_in[i]).strip()
            if isinstance(item, dict):
                term_u = str(item.get("term", "") or item.get("english", "") or "").strip()
                trans_u = str(
                    item.get("translation", "")
                    or item.get("gloss", "")
                    or item.get("uk", "")
                    or ""
                ).strip()
                if not ex_raw and item.get("example"):
                    ex_raw = str(item.get("example", "")).strip()
            else:
                s = str(item).strip()
                pairs = _split_term_translation([s])
                term_u, trans_u = pairs[0] if pairs else (s, "")
            term_e = escape(term_u[:120])
            trans_e = escape(trans_u[:200])
            ex_e = escape(ex_raw[:280]) if ex_raw else ""
            vocab_card_rows.append({"term": term_e, "translation": trans_e, "example": ex_e})

    ques_in: Any = raw.get("questions")
    questions_lines: List[str] = []
    if isinstance(ques_in, list):
        questions_lines = [escape(str(x).strip())[:500] for x in ques_in[:9] if str(x).strip()]
    handle_raw = str(raw.get("handle", "") or raw.get("instagram_handle", "") or "").strip()
    handle_display = ""
    if handle_raw:
        h = handle_raw if handle_raw.startswith("@") else f"@{handle_raw.lstrip('@')}"
        handle_display = escape(h[:80])

    topic_src = str(raw.get("topic", "") or "").strip()
    title_src = str(raw.get("title", "") or "").strip()
    lesson_topic_src = topic_src or title_src or "Lesson"
    lesson_topic = escape(lesson_topic_src[:220])

    lead_in_in: Any = raw.get("lead_in_questions")
    lead_in_questions_lines: List[str] = []
    if isinstance(lead_in_in, list):
        lead_in_questions_lines = [
            escape(str(x).strip())[:500] for x in lead_in_in[:3] if str(x).strip()
        ]

    disc_in: Any = raw.get("discussion_questions")
    discussion_questions_lines: List[str] = []
    if isinstance(disc_in, list):
        discussion_questions_lines = [
            escape(str(x).strip())[:500] for x in disc_in[:3] if str(x).strip()
        ]

    vocab_lesson_in: Any = raw.get("vocab")
    lesson_vocab_lines: List[str] = []
    if isinstance(vocab_lesson_in, list):
        lesson_vocab_lines = [
            escape(str(x).strip())[:200] for x in vocab_lesson_in[:6] if str(x).strip()
        ]

    choices_in: Any = raw.get("choices")
    choice_lines: List[str] = []
    if isinstance(choices_in, list):
        for c in choices_in[:6]:
            if isinstance(c, dict):
                a = str(
                    c.get("a", "")
                    or c.get("option_a", "")
                    or c.get("A", "")
                    or c.get("left", "")
                ).strip()
                b = str(
                    c.get("b", "")
                    or c.get("option_b", "")
                    or c.get("B", "")
                    or c.get("right", "")
                ).strip()
                if a and b:
                    choice_lines.append(escape(f"{a} or {b}?")[:500])
                else:
                    t = str(c.get("text", "") or c.get("line", "") or "").strip()
                    if t:
                        choice_lines.append(escape(t)[:500])
            else:
                s = str(c).strip()
                if s:
                    choice_lines.append(escape(s)[:500])

    phrases_in: Any = raw.get("phrases")
    phrases_blocks: List[dict[str, str]] = []
    if isinstance(phrases_in, list):
        for i, item in enumerate(phrases_in[:5]):
            if not isinstance(item, dict):
                continue
            ph = str(item.get("phrase", "") or "").strip()
            tr = str(item.get("translation", "") or "").strip()
            form = str(item.get("formula", "") or "").strip()
            ex_raw = item.get("examples")
            ex_list: List[str] = []
            if isinstance(ex_raw, list):
                ex_list = [str(x).strip() for x in ex_raw[:2] if str(x).strip()]
            while len(ex_list) < 2:
                ex_list.append("")
            phrases_blocks.append(
                {
                    "phrase_e": escape(ph[:400]),
                    "translation_e": escape(tr[:200]),
                    "formula_e": escape(form[:400]),
                    "ex1_html": _phrases_example_line_html(ex_list[0][:500]),
                    "ex2_html": _phrases_example_line_html(ex_list[1][:500]),
                    "num": str(i + 1),
                }
            )

    return {
        "template": raw.get("template", DEFAULT_TEMPLATE),
        "title": escape(str(raw.get("title", "Learning Card"))),
        "subtitle": escape(str(raw.get("subtitle", ""))),
        "bullets": cleaned_bullets,
        "cta": escape(str(raw.get("cta", "Try this today."))),
        "image_url": image_url,
        "punchline": punchline,
        "contrast_wrong": cw,
        "contrast_better": cb,
        "vocabulary_lines": vocabulary_lines,
        "mcq_bracket_lines": mcq_bracket_lines,
        "vocab_card_rows": vocab_card_rows,
        "questions_lines": questions_lines,
        "handle_display": handle_display,
        "lesson_topic": lesson_topic,
        "lead_in_questions_lines": lead_in_questions_lines,
        "discussion_questions_lines": discussion_questions_lines,
        "lesson_vocab_lines": lesson_vocab_lines,
        "choice_lines": choice_lines,
        "phrases_blocks": phrases_blocks,
    }


def _hero_media_block(card: dict[str, Any], variant: str) -> str:
    """
    Premium hero: topic photo with soft scrim, or gradient-only background (no random/sticker imagery).
    Never overlays text on the image; no-image uses quiet gradient + texture via CSS.
    """
    url = (card.get("image_url") or "").strip()
    if url and not is_safe_topic_image_url(url):
        url = ""
    if url:
        safe = escape(url, quote=True)
        return (
            f'<div class="hero-media hero-{variant} hero-has-img">'
            f'<div class="hero-img-stack">'
            f'<img class="hero-img" src="{safe}" alt="" loading="lazy" />'
            f'<div class="hero-img-scrim" aria-hidden="true"></div>'
            f"</div></div>"
        )
    return (
        f'<div class="hero-media hero-{variant} hero-no-img" aria-hidden="true">'
        f'<div class="hero-bg-fill"></div></div>'
    )


def _contrast_strip_html(card: dict[str, Any], prefix: str) -> str:
    """High-contrast wrong vs better block — main visual anchor when present."""
    w = card.get("contrast_wrong", "")
    b = card.get("contrast_better", "")
    if not w and not b:
        return ""
    bad_blk = ""
    good_blk = ""
    if w:
        bad_blk = (
            f'<div class="{prefix}-bad">'
            f'<span class="{prefix}-tag">Wrong</span>'
            f'<p class="{prefix}-txt">{w}</p></div>'
        )
    if b:
        good_blk = (
            f'<div class="{prefix}-good">'
            f'<span class="{prefix}-tag">Better</span>'
            f'<p class="{prefix}-txt">{b}</p></div>'
        )
    return (
        f'<div class="{prefix}-strip" role="group" aria-label="Wrong versus better">'
        f"{bad_blk}{good_blk}</div>"
    )


def _badge_level(subtitle_escaped: str) -> str:
    """Short label for the circular badge; defaults to A2/B1."""
    s = subtitle_escaped.strip()
    if not s:
        return "A2/B1"
    if len(s) <= 12:
        return s
    return "A2/B1"


def _split_term_translation(bullets: List[str]) -> List[Tuple[str, str]]:
    """Split bullets into (term, gloss) when separators are present."""
    pairs: List[Tuple[str, str]] = []
    for b in bullets:
        for sep in (" — ", " – ", " - ", ":", "—"):
            if sep in b:
                a, c = b.split(sep, 1)
                pairs.append((a.strip(), c.strip()))
                break
        else:
            pairs.append((b.strip(), ""))
    return pairs


def _placeholder_line() -> str:
    return '<span class="muted">—</span>'


def _highlight_bracket_choice(escaped_line: str) -> str:
    """Emphasize first (option / option) chunk in MCQ line (line is already escaped)."""
    if "(" not in escaped_line or ")" not in escaped_line:
        return escaped_line
    return re.sub(
        r"\(([^)]+)\)",
        r'<span class="mcq-gap">(<span class="mcq-or">\1</span>)</span>',
        escaped_line,
        count=1,
    )


def _vocab_rows_from_lines(lines: List[str]) -> List[str]:
    """Vocabulary: one row per line, English — Ukrainian only."""
    rows: List[str] = []
    for line in lines:
        pairs = _split_term_translation([line])
        if not pairs:
            continue
        term, trans = pairs[0]
        if trans:
            rows.append(
                f'<div class="vr-v2"><span class="tv2">{term}</span>'
                f'<span class="sv2">—</span><span class="gv2 ua">{trans}</span></div>'
            )
        else:
            rows.append(f'<div class="vr-v2"><span class="tv2 full">{term}</span></div>')
    return rows


class TemplateService:
    def render_html(self, card: dict[str, Any], forced_template: Optional[str] = None) -> str:
        normalized = _normalize_card(card)
        template_name = forced_template or normalized["template"]
        if template_name not in ALLOWED_TEMPLATES:
            template_name = DEFAULT_TEMPLATE
        normalized["template"] = template_name
        if template_name == "kitchen_collage_v2":
            return self._kitchen_collage_v2_template(normalized)
        if template_name == "kitchen_collage":
            return self._kitchen_collage_template(normalized)
        if template_name == "influencer_card_v2":
            return self._influencer_card_v2_template(normalized)
        if template_name == "influencer_card":
            return self._influencer_card_template(normalized)
        if template_name == "vocab_card":
            return self._vocab_card_template(normalized)
        if template_name == "questions_card":
            return self._questions_card_template(normalized)
        if template_name == "speaking_card_v2":
            return self._speaking_card_v2_template(normalized)
        if template_name == "speaking_poster_card":
            return self._speaking_poster_card_template(normalized)
        if template_name == "lesson_card_v1":
            return self._lesson_card_v1_template(normalized)
        if template_name == "lesson_art_v1":
            return self._lesson_art_v1_template(normalized)
        if template_name == "phrases_card":
            return self._phrases_card_template(normalized)
        if template_name == "warm_paper_v2":
            return self._warm_paper_v2_template(normalized)
        return self._warm_paper_template(normalized)

    # --- Template 1: warm_paper (idioms / vocabulary / phrases) ---

    def _warm_paper_template(self, card: dict[str, Any]) -> str:
        topic = card["title"]
        level = _badge_level(card["subtitle"])
        bullets = card["bullets"]
        cta = card["cta"]
        pairs = _split_term_translation(bullets)
        vocab_rows = []
        for term, trans in pairs[:6]:
            if trans:
                vocab_rows.append(
                    f'<div class="vocab-row"><span class="term">{term}</span>'
                    f'<span class="sep">—</span><span class="trans">{trans}</span></div>'
                )
            else:
                vocab_rows.append(f'<div class="vocab-row"><span class="term full">{term}</span></div>')
        if not vocab_rows:
            vocab_rows.append(
                '<div class="vocab-row muted">Add vocabulary pairs (e.g. idiom — meaning).</div>'
            )

        # Exercises: use bullets as options or placeholders
        ex_lines = bullets[:4] if bullets else []
        mcq_items = []
        for i, line in enumerate(ex_lines, 1):
            mcq_items.append(f'<div class="mcq"><span class="mcq-n">{i}.</span> {line}</div>')
        while len(mcq_items) < 3:
            mcq_items.append(
                f'<div class="mcq muted"><span class="mcq-n">{len(mcq_items) + 1}.</span> '
                f"Choose the best option.</div>"
            )

        content_sections = f"""
        <div class="cols-2">
          <section class="sticky note-tilt note-warm" aria-label="Vocabulary">
            <div class="pin" aria-hidden="true"></div>
            <h3 class="sec-title">Vocabulary</h3>
            <div class="vocab-list">{"".join(vocab_rows)}</div>
          </section>
          <section class="sticky note-tilt note-cream" aria-label="Choose the correct option">
            <div class="pin" aria-hidden="true"></div>
            <h3 class="sec-title">Choose the Correct Option</h3>
            <div class="mcq-list">{"".join(mcq_items[:4])}</div>
          </section>
        </div>
        <section class="speak-bar" aria-label="Let's speak">
          <h3 class="speak-title">Let’s Speak</h3>
          <p class="speak-text">{cta if cta else "Practice aloud in a short sentence."}</p>
        </section>
        """

        hero_block = _hero_media_block(card, "warm")
        return self._wrap_warm_paper_html(topic, level, hero_block, content_sections)

    def _wrap_warm_paper_html(
        self, topic: str, level: str, hero_block: str, content_sections: str
    ) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=600, initial-scale=1.0" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Caveat:wght@600&family=DM+Serif+Display:ital@0;1&display=swap" rel="stylesheet" />
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; }}
    body {{
      background: #e8d5b0;
      font-family: "DM Serif Display", Georgia, serif;
      color: #3d2c2b;
      -webkit-font-smoothing: antialiased;
    }}
    .page {{
      width: 600px;
      min-height: 920px;
      position: relative;
      overflow: hidden;
      background-color: #f5f0e8;
      background-image:
        linear-gradient(90deg, rgba(200, 180, 150, 0.12) 1px, transparent 1px),
        linear-gradient(rgba(200, 180, 150, 0.1) 1px, transparent 1px),
        radial-gradient(ellipse at 30% 20%, rgba(253, 248, 240, 0.95) 0%, #f5f0e8 55%);
      background-size: 24px 24px, 24px 24px, auto;
      padding: 18px 16px 22px;
    }}
    .level-badge {{
      position: absolute;
      top: 14px;
      left: 14px;
      width: 52px;
      height: 52px;
      border-radius: 50%;
      background: #8b2323;
      color: #fdf8f0;
      font-family: "DM Serif Display", Georgia, serif;
      font-size: 11px;
      line-height: 1.1;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: 4px;
      box-shadow: 0 3px 8px rgba(0,0,0,0.18);
      z-index: 5;
      word-wrap: break-word;
      overflow-wrap: anywhere;
    }}
    .header-block {{
      padding: 8px 64px 10px 72px;
      text-align: center;
    }}
    .topic-title {{
      font-family: "Caveat", cursive;
      font-size: 38px;
      line-height: 1.05;
      margin: 0;
      color: #4a3028;
      word-wrap: break-word;
      overflow-wrap: anywhere;
    }}
    .hero-media {{
      margin: 10px auto 12px;
      width: 92%;
      max-width: 520px;
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid rgba(139, 69, 19, 0.3);
      box-shadow: 0 4px 12px rgba(60, 40, 30, 0.12);
    }}
    .hero-img {{
      display: block;
      width: 100%;
      height: auto;
      max-height: 160px;
      object-fit: cover;
    }}
    .insight-card {{
      margin: 10px auto 12px;
      width: 92%;
      max-width: 520px;
      min-height: 96px;
      padding: 14px 16px 16px;
      border-radius: 14px;
      position: relative;
      overflow: hidden;
    }}
    .insight-card.insight-warm {{
      background: linear-gradient(145deg, #fffef9 0%, #f5f0e8 55%, #fdf8f0 100%);
      border: 1px solid rgba(139, 69, 19, 0.32);
      box-shadow: 0 5px 16px rgba(60, 40, 30, 0.1);
    }}
    .insight-kicker {{
      font-family: "Caveat", cursive;
      font-size: 20px;
      line-height: 1.1;
      margin: 0 0 8px;
      color: #8b4513;
    }}
    .insight-body {{
      margin: 0;
      font-size: 13px;
      line-height: 1.45;
      color: #3d2c2b;
      overflow-wrap: anywhere;
      word-wrap: break-word;
    }}
    .insight-accent {{
      position: absolute;
      left: 0;
      top: 10px;
      bottom: 10px;
      width: 4px;
      border-radius: 2px;
      background: linear-gradient(180deg, #c62828 0%, #8b2323 100%);
    }}
    .cols-2 {{
      display: flex;
      gap: 10px;
      align-items: stretch;
      margin-bottom: 10px;
    }}
    .sticky {{
      flex: 1;
      min-width: 0;
      padding: 12px 10px 14px;
      border-radius: 10px;
      position: relative;
      box-shadow: 4px 6px 14px rgba(60, 40, 30, 0.12);
    }}
    .note-tilt {{ transform: rotate(-0.6deg); }}
    .note-warm {{ background: #fdf8f0; border: 1px solid rgba(200, 170, 130, 0.45); }}
    .note-cream {{ background: #fffdf8; border: 1px solid rgba(180, 160, 130, 0.4); transform: rotate(0.5deg); }}
    .pin {{
      position: absolute;
      top: -6px;
      right: 14px;
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: radial-gradient(circle at 30% 30%, #ff6b4a, #c62828);
      box-shadow: 0 2px 4px rgba(0,0,0,0.25);
    }}
    .sec-title {{
      font-family: "Caveat", cursive;
      font-size: 20px;
      margin: 0 0 8px;
      color: #5c3d32;
    }}
    .vocab-list {{ font-size: 12px; line-height: 1.35; }}
    .vocab-row {{ margin-bottom: 6px; overflow-wrap: anywhere; word-wrap: break-word; }}
    .term {{ font-weight: 600; color: #3d2c2b; }}
    .term.full {{ display: block; }}
    .sep {{ margin: 0 4px; opacity: 0.5; }}
    .trans {{ color: #5c4038; }}
    .mcq-list {{ font-size: 11.5px; line-height: 1.35; }}
    .mcq {{ margin-bottom: 6px; overflow-wrap: anywhere; word-wrap: break-word; }}
    .mcq-n {{ font-weight: 700; color: #8b4513; margin-right: 4px; }}
    .muted {{ color: rgba(61, 44, 43, 0.45); font-style: italic; }}
    .speak-bar {{
      width: 100%;
      margin-top: 4px;
      padding: 12px 14px;
      border-radius: 12px;
      background: linear-gradient(135deg, #2e7d4a 0%, #1b5e32 100%);
      color: #f4fff8;
      box-shadow: 0 4px 12px rgba(30, 80, 50, 0.2);
    }}
    .speak-title {{
      font-family: "Caveat", cursive;
      font-size: 22px;
      margin: 0 0 6px;
    }}
    .speak-text {{
      margin: 0;
      font-size: 12.5px;
      line-height: 1.4;
      overflow-wrap: anywhere;
      word-wrap: break-word;
    }}
    {HERO_LAYER_CSS}
  </style>
</head>
<body>
  <div class="page">
    <div class="level-badge">{level}</div>
    <header class="header-block">
      <h1 class="topic-title">{topic}</h1>
    </header>
    {hero_block}
    {content_sections}
  </div>
</body>
</html>"""

    # --- Template 2: kitchen_collage (collocations / grammar / mixed) ---

    def _kitchen_collage_template(self, card: dict[str, Any]) -> str:
        topic = card["title"]
        level = _badge_level(card["subtitle"])
        bullets = card["bullets"]
        cta = card["cta"]
        b = bullets + [""] * 5
        sec_vocab = b[0] if b[0] else _placeholder_line()
        sec_fix = b[1] if b[1] else _placeholder_line()
        sec_choose = b[2] if b[2] else _placeholder_line()
        sec_gap = b[3] if b[3] else _placeholder_line()
        sec_speak = cta if cta else (b[4] if b[4] else "Short speaking cue.")

        content_sections = f"""
    <section class="panel panel-tan">
      <div class="pin-sm"></div>
      <h3 class="ptitle">Vocabulary List</h3>
      <p class="pbody">{sec_vocab}</p>
    </section>
    <section class="panel panel-ivory">
      <div class="pin-sm"></div>
      <h3 class="ptitle">Correct the Mistake</h3>
      <p class="pbody">{sec_fix}</p>
    </section>
    <section class="panel panel-tan tilt-r">
      <div class="pin-sm"></div>
      <h3 class="ptitle">Choose the Option</h3>
      <p class="pbody">{sec_choose}</p>
    </section>
    <section class="panel panel-ivory tilt-l">
      <div class="pin-sm"></div>
      <h3 class="ptitle">Fill in the Gaps</h3>
      <p class="pbody">{sec_gap}</p>
    </section>
    <section class="panel panel-speak">
      <div class="pin-sm pin-light"></div>
      <h3 class="ptitle light">Speaking</h3>
      <p class="pbody light">{sec_speak}</p>
    </section>
        """

        hero_block = _hero_media_block(card, "kitchen")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=600, initial-scale=1.0" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Caveat:wght@600&family=DM+Serif+Display:ital@0;1&display=swap" rel="stylesheet" />
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; }}
    body {{
      background: #d4c4a8;
      font-family: "DM Serif Display", Georgia, serif;
      color: #3d2c2b;
      -webkit-font-smoothing: antialiased;
    }}
    .page {{
      width: 600px;
      min-height: 920px;
      position: relative;
      background: #fdf8f0;
      background-image:
        linear-gradient(0deg, rgba(245, 240, 232, 0.9) 1px, transparent 1px),
        linear-gradient(90deg, rgba(220, 200, 170, 0.15) 1px, transparent 1px);
      background-size: 100% 28px, 18px 100%;
      padding: 16px 14px 20px;
      overflow: hidden;
    }}
    .level-badge {{
      position: absolute;
      top: 12px;
      left: 12px;
      width: 50px;
      height: 50px;
      border-radius: 50%;
      background: #8b2323;
      color: #fdf8f0;
      font-size: 10px;
      line-height: 1.1;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: 4px;
      box-shadow: 0 3px 8px rgba(0,0,0,0.15);
      z-index: 5;
      overflow-wrap: anywhere;
    }}
    .head-k {{
      padding: 6px 58px 8px 64px;
      text-align: center;
    }}
    .topic-title {{
      font-family: "Caveat", cursive;
      font-size: 36px;
      line-height: 1.05;
      margin: 0;
      color: #4a3028;
      word-wrap: break-word;
      overflow-wrap: anywhere;
    }}
    .hero-media {{
      margin: 8px auto 10px;
      width: 94%;
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid rgba(139, 69, 19, 0.28);
      box-shadow: 0 3px 10px rgba(50, 35, 25, 0.1);
    }}
    .hero-img {{
      display: block;
      width: 100%;
      height: auto;
      max-height: 140px;
      object-fit: cover;
    }}
    .insight-card {{
      margin: 8px auto 10px;
      width: 94%;
      min-height: 88px;
      padding: 12px 14px 14px;
      border-radius: 12px;
      position: relative;
      overflow: hidden;
    }}
    .insight-card.insight-kitchen {{
      background: linear-gradient(160deg, #fffdf9 0%, #f5f0e8 50%, #e8d5b0 100%);
      border: 1px solid rgba(180, 140, 100, 0.4);
      box-shadow: 0 4px 12px rgba(50, 35, 25, 0.1);
    }}
    .insight-kicker {{
      font-family: "Caveat", cursive;
      font-size: 19px;
      margin: 0 0 6px;
      color: #8b4513;
    }}
    .insight-body {{
      margin: 0;
      font-size: 12px;
      line-height: 1.45;
      overflow-wrap: anywhere;
      word-wrap: break-word;
    }}
    .insight-accent {{
      position: absolute;
      left: 0;
      top: 8px;
      bottom: 8px;
      width: 3px;
      border-radius: 2px;
      background: linear-gradient(180deg, #c62828 0%, #5d4037 100%);
    }}
    .stack {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .panel {{
      position: relative;
      padding: 10px 12px 12px;
      border-radius: 10px;
      box-shadow: 3px 5px 12px rgba(50, 35, 25, 0.1);
    }}
    .panel-tan {{ background: #f5f0e8; border: 1px solid rgba(180, 150, 120, 0.4); }}
    .panel-ivory {{ background: #fffdf9; border: 1px solid rgba(200, 175, 140, 0.35); }}
    .panel-speak {{
      background: linear-gradient(120deg, #5d4037 0%, #3e2723 100%);
      border: 1px solid rgba(0,0,0,0.08);
    }}
    .tilt-r {{ transform: rotate(0.4deg); }}
    .tilt-l {{ transform: rotate(-0.35deg); }}
    .pin-sm {{
      position: absolute;
      top: -5px;
      right: 12px;
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: radial-gradient(circle at 30% 30%, #ff7961, #c62828);
      box-shadow: 0 2px 3px rgba(0,0,0,0.2);
    }}
    .pin-light {{
      background: radial-gradient(circle at 30% 30%, #ffd180, #e65100);
    }}
    .ptitle {{
      font-family: "Caveat", cursive;
      font-size: 19px;
      margin: 0 0 6px;
      color: #5c3d32;
    }}
    .ptitle.light {{ color: #ffe0b2; }}
    .pbody {{
      margin: 0;
      font-size: 11.5px;
      line-height: 1.4;
      overflow-wrap: anywhere;
      word-wrap: break-word;
    }}
    .pbody.light {{ color: #fff8f0; }}
    .muted {{ opacity: 0.5; font-style: italic; }}
    {HERO_LAYER_CSS}
  </style>
</head>
<body>
  <div class="page">
    <div class="level-badge">{level}</div>
    <header class="head-k">
      <h1 class="topic-title">{topic}</h1>
    </header>
    {hero_block}
    <div class="stack">
      {content_sections}
    </div>
  </div>
</body>
</html>"""

    # --- Template 3: influencer_card (social / trends / modern) ---

    def _influencer_card_template(self, card: dict[str, Any]) -> str:
        topic = card["title"]
        level = _badge_level(card["subtitle"])
        bullets = card["bullets"]
        cta = card["cta"]
        b = bullets + [""] * 5
        dq = b[0] if b[0] else "What is your take on this topic?"
        vocab = b[1] if b[1] else "Key term — short definition."
        tf = b[2] if b[2] else "True or false: [statement]."
        gram = b[3] if b[3] else "Pattern: subject + verb + …"
        write_pills = [p for p in bullets[:3] if p] or ["Brainstorm", "Draft", "Polish"]

        pills_html = "".join(f'<span class="pill">{p}</span>' for p in write_pills[:4])

        content_sections = f"""
    <section class="card-mod discussion">
      <h3 class="mtitle">Discussion Questions</h3>
      <p class="mbody">{dq}</p>
    </section>
    <section class="card-mod vocab">
      <h3 class="mtitle">Vocabulary</h3>
      <p class="mbody small">{vocab}</p>
    </section>
    <div class="row-2">
      <section class="card-mod tf">
        <h3 class="mtitle">True / False</h3>
        <p class="mbody">{tf}</p>
      </section>
      <section class="card-mod bubble-wrap">
        <h3 class="mtitle">Grammar</h3>
        <div class="bubble">{gram}</div>
      </section>
    </div>
    <section class="card-mod writing">
      <h3 class="mtitle">Writing Prompt</h3>
      <p class="mbody prompt">{cta if cta else "Write 3–5 sentences."}</p>
      <div class="pills">{pills_html}</div>
    </section>
        """

        hero_block = _hero_media_block(card, "influencer")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=600, initial-scale=1.0" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Caveat:wght@600&family=DM+Serif+Display:ital@0;1&display=swap" rel="stylesheet" />
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; }}
    body {{
      background: #1a1a24;
      font-family: "DM Serif Display", Georgia, serif;
      color: #2b2b35;
      -webkit-font-smoothing: antialiased;
    }}
    .page {{
      width: 600px;
      min-height: 920px;
      position: relative;
      background: linear-gradient(165deg, #fdf8f0 0%, #f5f0e8 40%, #e8d5b0 100%);
      padding: 16px 14px 20px;
      overflow: hidden;
    }}
    .page::before {{
      content: "";
      position: absolute;
      inset: 0;
      background: radial-gradient(circle at 100% 0%, rgba(255, 120, 90, 0.12) 0%, transparent 45%),
                  radial-gradient(circle at 0% 100%, rgba(80, 120, 200, 0.1) 0%, transparent 40%);
      pointer-events: none;
    }}
    .level-badge {{
      position: absolute;
      top: 12px;
      left: 12px;
      width: 50px;
      height: 50px;
      border-radius: 50%;
      background: #8b2323;
      color: #fdf8f0;
      font-size: 10px;
      line-height: 1.1;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: 4px;
      box-shadow: 0 4px 10px rgba(0,0,0,0.2);
      z-index: 5;
      overflow-wrap: anywhere;
    }}
    .head-i {{
      padding: 4px 56px 6px 60px;
      text-align: center;
      position: relative;
      z-index: 1;
    }}
    .topic-title {{
      font-family: "Caveat", cursive;
      font-size: 40px;
      line-height: 1.05;
      margin: 0;
      color: #3d2c2b;
      text-shadow: 0 1px 0 rgba(255,255,255,0.6);
      word-wrap: break-word;
      overflow-wrap: anywhere;
    }}
    .hero-media {{
      margin: 10px auto 12px;
      width: 94%;
      border-radius: 16px;
      overflow: hidden;
      border: 2px solid rgba(139, 35, 35, 0.25);
      box-shadow: 0 6px 18px rgba(40, 30, 20, 0.12);
      position: relative;
      z-index: 1;
    }}
    .hero-img {{
      display: block;
      width: 100%;
      height: auto;
      max-height: 150px;
      object-fit: cover;
    }}
    .insight-card {{
      margin: 10px auto 12px;
      width: 94%;
      min-height: 96px;
      padding: 14px 16px 16px;
      border-radius: 16px;
      position: relative;
      z-index: 1;
      overflow: hidden;
    }}
    .insight-card.insight-influencer {{
      background: linear-gradient(135deg, rgba(255, 253, 248, 0.98) 0%, #fdf8f0 45%, #f5f0e8 100%);
      border: 2px solid rgba(198, 40, 40, 0.22);
      box-shadow: 0 8px 22px rgba(60, 40, 30, 0.12);
    }}
    .insight-kicker {{
      font-family: "Caveat", cursive;
      font-size: 22px;
      margin: 0 0 8px;
      color: #c62828;
    }}
    .insight-body {{
      margin: 0;
      font-size: 12.5px;
      line-height: 1.45;
      color: #3d2c2b;
      overflow-wrap: anywhere;
      word-wrap: break-word;
    }}
    .insight-accent {{
      position: absolute;
      left: 0;
      top: 10px;
      bottom: 10px;
      width: 4px;
      border-radius: 2px;
      background: linear-gradient(180deg, #1565c0 0%, #6a1b9a 50%, #c62828 100%);
    }}
    .card-mod {{
      position: relative;
      margin-bottom: 8px;
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(255, 253, 248, 0.92);
      border: 1px solid rgba(200, 170, 140, 0.35);
      box-shadow: 0 6px 16px rgba(40, 30, 20, 0.08);
      z-index: 1;
    }}
    .discussion {{
      border-left: 4px solid #c62828;
    }}
    .vocab {{
      border-left: 4px solid #8d6e63;
    }}
    .row-2 {{
      display: flex;
      gap: 8px;
    }}
    .row-2 .card-mod {{
      flex: 1;
      min-width: 0;
    }}
    .tf {{
      border-left: 4px solid #2e7d32;
    }}
    .bubble-wrap {{
      border-left: 4px solid #1565c0;
    }}
    .bubble {{
      font-size: 11px;
      line-height: 1.35;
      padding: 8px 10px;
      border-radius: 18px;
      background: linear-gradient(145deg, #e3f2fd 0%, #fff 100%);
      border: 1px solid rgba(21, 101, 192, 0.25);
      overflow-wrap: anywhere;
      word-wrap: break-word;
    }}
    .writing {{
      border-left: 4px solid #6a1b9a;
      padding-bottom: 12px;
    }}
    .mtitle {{
      font-family: "Caveat", cursive;
      font-size: 20px;
      margin: 0 0 6px;
      color: #4e342e;
    }}
    .mbody {{
      margin: 0;
      font-size: 11.5px;
      line-height: 1.4;
      overflow-wrap: anywhere;
      word-wrap: break-word;
    }}
    .mbody.small {{ font-size: 11px; }}
    .mbody.prompt {{ font-weight: 600; color: #3d2c2b; }}
    .pills {{
      margin-top: 8px;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .pill {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 10px;
      background: #fff;
      border: 1px solid rgba(200, 160, 120, 0.5);
      color: #5d4037;
      box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }}
    {HERO_LAYER_CSS}
  </style>
</head>
<body>
  <div class="page">
    <div class="level-badge">{level}</div>
    <header class="head-i">
      <h1 class="topic-title">{topic}</h1>
    </header>
    {hero_block}
    {content_sections}
  </div>
</body>
</html>"""

    def _questions_card_template(self, card: dict[str, Any]) -> str:
        title = card["title"]
        handle = (card.get("handle_display") or "").strip()
        handle_html = f'<p class="qc-handle">{handle}</p>' if handle else ""
        questions = list(card.get("questions_lines") or [])[:9]
        if not questions:
            questions = [escape("—")]
        cells = "".join(
            f'<div class="qc-card" role="article"><p class="qc-q">{q}</p></div>' for q in questions
        )
        grid_html = f'<div class="qc-grid">{cells}</div>'

        thumb_url = (card.get("image_url") or "").strip()
        if thumb_url and is_safe_topic_image_url(thumb_url):
            safe_u = escape(thumb_url, quote=True)
            thumb_html = f'<figure class="qc-thumb"><img src="{safe_u}" alt="" loading="lazy" /></figure>'
        else:
            thumb_html = ""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=600, initial-scale=1.0" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&display=swap" rel="stylesheet" />
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; }}
    body {{ background: #e8e8e8; font-family: "DM Serif Display", Georgia, serif; -webkit-font-smoothing: antialiased; }}
    .qc-page {{
      width: 600px;
      min-height: 920px;
      margin: 0 auto;
      background: #f5f5f5;
      padding: 36px 28px 32px;
      position: relative;
    }}
    .qc-title {{
      margin: 0;
      font-size: 32px;
      font-weight: 700;
      line-height: 1.12;
      color: #0a0a0a;
      text-align: center;
      letter-spacing: -0.02em;
      word-wrap: break-word;
      overflow-wrap: anywhere;
    }}
    .qc-handle {{
      margin: 10px 0 0;
      text-align: center;
      font-size: 11px;
      letter-spacing: 0.06em;
      color: #555;
    }}
    .qc-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(248px, 1fr));
      gap: 14px;
      margin-top: 28px;
      align-content: start;
    }}
    .qc-card {{
      background: #fff;
      border-radius: 20px;
      border: 1px solid #1a1a1a;
      padding: 18px 16px;
      min-height: 72px;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 1px 0 rgba(255,255,255,0.9) inset, 0 8px 24px rgba(0,0,0,0.06);
    }}
    .qc-q {{
      margin: 0;
      font-size: 13px;
      line-height: 1.5;
      font-weight: 500;
      color: #141414;
      text-align: center;
      overflow-wrap: anywhere;
      word-wrap: break-word;
    }}
    .qc-thumb {{
      margin: 28px auto 0;
      max-width: 100%;
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid rgba(0,0,0,0.08);
    }}
    .qc-thumb img {{
      display: block;
      width: 100%;
      height: auto;
      vertical-align: middle;
    }}
  </style>
</head>
<body>
  <div class="qc-page page">
    <header>
      <h1 class="qc-title">{title}</h1>
      {handle_html}
    </header>
    {grid_html}
    {thumb_html}
  </div>
</body>
</html>"""

    def _speaking_card_v2_template(self, card: dict[str, Any]) -> str:
        title = card["title"]
        handle = (card.get("handle_display") or "").strip()
        handle_html = f'<p class="sc2-handle">{handle}</p>' if handle else ""
        items = [
            q
            for q in list(card.get("questions_lines") or [])[:8]
            if str(q).strip()
        ]
        if not items:
            items = [escape("—")]
        rhythm_classes = (" sc2-cell--lift", " sc2-cell--soft", " sc2-cell--cream")
        hero_q = items[0]
        rest = items[1:]
        hero_html = (
            f'<div class="sc2-hero-slot">'
            f'<article class="sc2-cell sc2-cell--hero" role="article">'
            f'<p class="sc2-q sc2-q--hero">{hero_q}</p></article></div>'
        )
        if not rest:
            grid_html = f'<div class="sc2-poster-flow">{hero_html}</div>'
        else:
            cell_parts: List[str] = []
            for i, q in enumerate(rest):
                rhythm = rhythm_classes[(i + 1) % 3]
                cell_parts.append(
                    f'<div class="sc2-cell{rhythm}" role="article">'
                    f'<p class="sc2-q">{q}</p></div>'
                )
            r = len(rest)
            grid_html = (
                f'<div class="sc2-poster-flow">{hero_html}'
                f'<div class="sc2-cluster sc2-cluster--r{r}">{"".join(cell_parts)}</div>'
                f"</div>"
            )

        thumb_url = (card.get("image_url") or "").strip()
        if thumb_url and is_safe_topic_image_url(thumb_url):
            safe_u = escape(thumb_url, quote=True)
            deco_html = (
                f'<figure class="sc2-deco" aria-hidden="true">'
                f'<img src="{safe_u}" alt="" loading="lazy" /></figure>'
            )
        else:
            deco_html = ""

        header_mod = " sc2-header--with-visual" if deco_html else ""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=600, initial-scale=1.0" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Lora:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet" />
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; }}
    body {{
      background: #c4baa8;
      font-family: "Lora", "DM Serif Display", Georgia, serif;
      -webkit-font-smoothing: antialiased;
    }}
    .sc2-page {{
      width: 600px;
      min-height: 920px;
      margin: 0 auto;
      padding: 34px 32px 48px;
      position: relative;
      overflow: hidden;
      background-color: #f3e9dc;
      background-image:
        radial-gradient(ellipse 100% 70% at 50% -15%, rgba(255, 255, 255, 0.82) 0%, transparent 48%),
        radial-gradient(ellipse 70% 45% at 0% 20%, rgba(196, 165, 116, 0.14) 0%, transparent 55%),
        radial-gradient(ellipse 55% 40% at 100% 85%, rgba(180, 150, 120, 0.12) 0%, transparent 50%),
        linear-gradient(90deg, rgba(120, 100, 75, 0.028) 1px, transparent 1px),
        linear-gradient(rgba(120, 100, 75, 0.022) 1px, transparent 1px),
        linear-gradient(168deg, #fdfbf7 0%, #f0e4d4 42%, #e5d8c6 100%);
      background-size: 100% 100%, 100% 100%, 100% 100%, 24px 24px, 24px 24px, auto;
      box-shadow:
        inset 0 0 0 1px rgba(255, 255, 255, 0.45),
        inset 0 1px 0 rgba(255, 255, 255, 0.65),
        0 22px 56px rgba(42, 36, 28, 0.1);
    }}
    .sc2-page::after {{
      content: "";
      position: absolute;
      pointer-events: none;
      inset: 0;
      opacity: 0.35;
      background-image: radial-gradient(rgba(90, 70, 50, 0.04) 1px, transparent 1px);
      background-size: 5px 5px;
      mix-blend-mode: multiply;
    }}
    .sc2-accent-bar {{
      position: relative;
      z-index: 1;
      height: 6px;
      margin: -34px -32px 28px -32px;
      border-radius: 0 0 12px 12px;
      background: linear-gradient(
        92deg,
        #a67c52 0%,
        #d4b896 18%,
        #9a7b58 48%,
        #c9a876 78%,
        #b8956a 100%
      );
      box-shadow: 0 3px 12px rgba(90, 70, 45, 0.2);
    }}
    .sc2-header {{
      position: relative;
      z-index: 1;
      display: flex;
      flex-direction: row;
      align-items: flex-start;
      justify-content: space-between;
      gap: 20px 26px;
      margin-bottom: 34px;
      padding: 4px 4px 28px 2px;
      border-bottom: none;
      background: linear-gradient(
        to right,
        transparent 0%,
        rgba(139, 115, 85, 0.1) 50%,
        transparent 100%
      );
      background-size: 100% 1px;
      background-position: 0 100%;
      background-repeat: no-repeat;
    }}
    .sc2-header--with-visual {{
      align-items: stretch;
    }}
    .sc2-header--with-visual .sc2-title-block {{
      padding-top: 6px;
    }}
    .sc2-title-block {{
      flex: 1;
      min-width: 0;
      text-align: left;
      padding-right: 10px;
    }}
    .sc2-title-wrap {{
      position: relative;
      padding: 16px 20px 18px 22px;
      border-radius: 20px;
      background: linear-gradient(
        145deg,
        rgba(255, 252, 248, 0.95) 0%,
        rgba(248, 240, 228, 0.55) 55%,
        rgba(255, 250, 244, 0.75) 100%
      );
      box-shadow:
        inset 0 0 0 1px rgba(255, 255, 255, 0.75),
        0 6px 28px rgba(42, 36, 28, 0.07),
        0 2px 8px rgba(42, 36, 28, 0.04);
    }}
    .sc2-title-wrap::before {{
      content: "";
      position: absolute;
      left: 10px;
      top: 14px;
      bottom: 14px;
      width: 5px;
      border-radius: 6px;
      background: linear-gradient(180deg, #d4b896 0%, #8b6f4a 55%, #c4a574 100%);
      box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.35);
    }}
    .sc2-title {{
      margin: 0;
      padding-left: 8px;
      font-family: "DM Serif Display", Georgia, serif;
      font-size: 30px;
      font-weight: 700;
      line-height: 1.08;
      color: #16120e;
      letter-spacing: -0.028em;
      word-wrap: break-word;
      overflow-wrap: anywhere;
      text-shadow: 0 1px 0 rgba(255, 255, 255, 0.5);
    }}
    .sc2-handle {{
      margin: 14px 0 0 10px;
      font-size: 11.5px;
      line-height: 1.5;
      letter-spacing: 0.09em;
      text-transform: uppercase;
      color: #6d604e;
      font-weight: 600;
    }}
    .sc2-deco {{
      flex: 0 1 auto;
      width: min(48%, 278px);
      min-width: 208px;
      height: 158px;
      margin: 0 0 0 auto;
      align-self: center;
      padding: 6px;
      border-radius: 28px;
      background: linear-gradient(148deg, #ffffff 0%, #ebe3d6 42%, #d8cdc0 100%);
      box-shadow:
        0 18px 44px rgba(42, 36, 28, 0.18),
        0 6px 14px rgba(42, 36, 28, 0.08),
        inset 0 2px 0 rgba(255, 255, 255, 0.95),
        inset 0 0 0 1px rgba(255, 255, 255, 0.5);
      overflow: hidden;
    }}
    .sc2-deco img {{
      display: block;
      width: 100%;
      height: 100%;
      min-height: 142px;
      object-fit: cover;
      border-radius: 22px;
      vertical-align: middle;
      filter: saturate(0.96) contrast(1.03) brightness(1.02);
    }}
    .sc2-poster-flow {{
      position: relative;
      z-index: 1;
      width: 100%;
    }}
    .sc2-hero-slot {{
      margin: 0 0 6px 0;
      padding: 0 1px 18px 3px;
    }}
    .sc2-cell--hero {{
      min-height: 118px;
      padding: 26px 26px 28px;
      border-radius: 28px;
      background: linear-gradient(
        168deg,
        #fffefb 0%,
        #f2e6dc 48%,
        #e4d6c8 100%
      );
      border: 1px solid rgba(105, 82, 58, 0.3);
      box-shadow:
        0 3px 0 rgba(255, 255, 255, 0.92) inset,
        0 18px 46px rgba(42, 36, 28, 0.12),
        0 5px 14px rgba(42, 36, 28, 0.06);
    }}
    .sc2-cluster {{
      position: relative;
      display: grid;
      grid-template-columns: 1fr 1fr;
      width: 100%;
      column-gap: 20px;
      row-gap: 22px;
      padding-top: 22px;
      margin-top: 8px;
      align-content: start;
      justify-content: center;
    }}
    .sc2-cluster::before {{
      content: "";
      position: absolute;
      left: 8%;
      right: 8%;
      top: 0;
      height: 1px;
      background: linear-gradient(
        90deg,
        transparent,
        rgba(107, 93, 74, 0.18) 20%,
        rgba(107, 93, 74, 0.18) 80%,
        transparent
      );
    }}
    .sc2-cluster .sc2-cell:nth-child(2n) {{
      transform: translateY(6px);
    }}
    .sc2-cluster .sc2-cell:nth-child(4n+3),
    .sc2-cluster .sc2-cell:nth-child(4n+4) {{
      margin-top: 8px;
    }}
    .sc2-cluster--r1 .sc2-cell {{
      grid-column: 1 / -1;
      justify-self: center;
      max-width: 400px;
      width: 100%;
    }}
    .sc2-cluster--r3 .sc2-cell:nth-child(3) {{
      grid-column: 1 / -1;
      justify-self: center;
      width: 100%;
      max-width: 352px;
    }}
    .sc2-cluster--r5 .sc2-cell:nth-child(5) {{
      grid-column: 1 / -1;
      justify-self: center;
      width: 100%;
      max-width: calc(50% - 10px);
    }}
    .sc2-cluster--r7 .sc2-cell:nth-child(7) {{
      grid-column: 1 / -1;
      justify-self: center;
      width: 100%;
      max-width: calc(50% - 10px);
    }}
    .sc2-cell {{
      border-radius: 24px;
      padding: 20px 18px;
      min-height: 100px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .sc2-cell--lift {{
      background: linear-gradient(168deg, #fffefb 0%, #f3e8dc 52%, #ebe0d4 100%);
      border: 1px solid rgba(130, 105, 78, 0.22);
      box-shadow:
        0 2px 0 rgba(255, 255, 255, 0.88) inset,
        0 10px 28px rgba(42, 36, 28, 0.08),
        0 2px 6px rgba(42, 36, 28, 0.04);
    }}
    .sc2-cell--soft {{
      background: linear-gradient(185deg, rgba(255, 252, 248, 0.98) 0%, rgba(244, 234, 222, 0.65) 100%);
      border: 1.5px dashed rgba(115, 95, 72, 0.32);
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.75) inset,
        0 6px 20px rgba(42, 36, 28, 0.05);
    }}
    .sc2-cell--cream {{
      background: linear-gradient(160deg, #faf6ef 0%, #efe4d8 100%);
      border: 1px dashed rgba(140, 118, 92, 0.38);
      border-top: 1px solid rgba(255, 255, 255, 0.65);
      box-shadow:
        0 8px 24px rgba(42, 36, 28, 0.06),
        inset 0 -1px 0 rgba(120, 95, 70, 0.05);
    }}
    .sc2-q {{
      margin: 0;
      font-size: 14px;
      line-height: 1.62;
      font-weight: 500;
      color: #252018;
      text-align: center;
      overflow-wrap: anywhere;
      word-wrap: break-word;
      max-width: 100%;
      letter-spacing: 0.012em;
    }}
    .sc2-q--hero {{
      font-size: 15.5px;
      line-height: 1.58;
      font-weight: 600;
      letter-spacing: 0.018em;
      color: #1a1510;
    }}
  </style>
</head>
<body>
  <div class="sc2-page page">
    <div class="sc2-accent-bar" aria-hidden="true"></div>
    <header class="sc2-header{header_mod}">
      <div class="sc2-title-block">
        <div class="sc2-title-wrap">
          <h1 class="sc2-title">{title}</h1>
        </div>
        {handle_html}
      </div>
      {deco_html}
    </header>
    {grid_html}
  </div>
</body>
</html>"""

    def _speaking_poster_card_template(self, card: dict[str, Any]) -> str:
        title = card["title"]
        handle = (card.get("handle_display") or "").strip()
        sub_html = f'<p class="sp-sub">{handle}</p>' if handle else ""
        items = [
            q
            for q in list(card.get("questions_lines") or [])[:8]
            if str(q).strip()
        ]
        if not items:
            items = [escape("—")]
        n = len(items)
        styles = (" sp-card--a", " sp-card--b", " sp-card--c")
        cells = "".join(
            f'<article class="sp-card{styles[i % 3]}" role="article">'
            f'<p class="sp-q">{items[i]}</p></article>'
            for i in range(n)
        )
        grid_mod = f" sp-grid--n{n}"
        grid_html = f'<div class="sp-grid{grid_mod}">{cells}</div>'

        thumb_url = (card.get("image_url") or "").strip()
        if thumb_url and is_safe_topic_image_url(thumb_url):
            safe_u = escape(thumb_url, quote=True)
            deco_html = (
                f'<figure class="sp-deco" aria-hidden="true">'
                f'<img src="{safe_u}" alt="" loading="lazy" /></figure>'
            )
            head_cls = "sp-top sp-top--visual"
        else:
            deco_html = ""
            head_cls = "sp-top"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=600, initial-scale=1.0" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Lora:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet" />
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; }}
    body {{
      background: #b8a994;
      font-family: "Lora", "DM Serif Display", Georgia, serif;
      -webkit-font-smoothing: antialiased;
    }}
    .sp-page {{
      width: 600px;
      min-height: 920px;
      margin: 0 auto;
      padding: 38px 34px 48px;
      position: relative;
      overflow: hidden;
      background-color: #f2e8dc;
      background-image:
        radial-gradient(ellipse 110% 65% at 50% -8%, rgba(255, 255, 255, 0.78) 0%, transparent 50%),
        radial-gradient(ellipse 50% 40% at 92% 8%, rgba(200, 170, 130, 0.18) 0%, transparent 55%),
        linear-gradient(90deg, rgba(100, 82, 60, 0.025) 1px, transparent 1px),
        linear-gradient(rgba(100, 82, 60, 0.02) 1px, transparent 1px),
        linear-gradient(172deg, #fdf9f3 0%, #ebe0d2 45%, #dfd2c2 100%);
      background-size: 100% 100%, 100% 100%, 26px 26px, 26px 26px, auto;
      box-shadow:
        inset 0 0 0 1px rgba(255, 255, 255, 0.42),
        0 24px 60px rgba(38, 32, 24, 0.12);
    }}
    .sp-page::before {{
      content: "";
      position: absolute;
      pointer-events: none;
      inset: 0;
      opacity: 0.28;
      background-image: radial-gradient(rgba(70, 55, 40, 0.035) 1px, transparent 1px);
      background-size: 6px 6px;
    }}
    .sp-ribbon {{
      position: relative;
      z-index: 1;
      height: 7px;
      margin: -38px -34px 30px -34px;
      border-radius: 0 0 14px 14px;
      background: linear-gradient(93deg, #9a7a58 0%, #d4b896 25%, #7d6246 55%, #c9a876 100%);
      box-shadow: 0 4px 14px rgba(60, 48, 32, 0.18);
    }}
    .sp-top {{
      position: relative;
      z-index: 1;
      display: flex;
      flex-direction: row;
      align-items: flex-start;
      justify-content: space-between;
      gap: 20px 28px;
      margin-bottom: 36px;
      padding-bottom: 8px;
    }}
    .sp-top--visual {{
      align-items: stretch;
    }}
    .sp-headline {{
      flex: 1;
      min-width: 0;
      padding: 4px 8px 0 4px;
    }}
    .sp-title {{
      margin: 0;
      font-family: "DM Serif Display", Georgia, serif;
      font-size: 34px;
      font-weight: 700;
      line-height: 1.05;
      color: #141008;
      letter-spacing: -0.03em;
      word-wrap: break-word;
      overflow-wrap: anywhere;
      text-shadow: 0 1px 0 rgba(255, 255, 255, 0.45);
    }}
    .sp-sub {{
      margin: 14px 0 0;
      font-size: 12px;
      line-height: 1.5;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #6a5c4a;
      font-weight: 600;
    }}
    .sp-deco {{
      flex: 0 0 auto;
      width: min(46%, 276px);
      min-width: 200px;
      height: 152px;
      padding: 6px;
      border-radius: 26px;
      align-self: center;
      background: linear-gradient(150deg, #ffffff 0%, #e8dfd2 50%, #d4c8b8 100%);
      box-shadow:
        0 16px 40px rgba(38, 32, 24, 0.15),
        inset 0 2px 0 rgba(255, 255, 255, 0.9);
      overflow: hidden;
    }}
    .sp-deco img {{
      display: block;
      width: 100%;
      height: 100%;
      object-fit: cover;
      border-radius: 20px;
      filter: saturate(0.96) contrast(1.03);
    }}
    .sp-grid {{
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-columns: 1fr 1fr;
      width: 100%;
      column-gap: 22px;
      row-gap: 26px;
      align-content: start;
      justify-items: stretch;
      padding-top: 8px;
    }}
    .sp-grid--n1 .sp-card {{
      grid-column: 1 / -1;
      max-width: 440px;
      justify-self: center;
    }}
    .sp-grid--n3 .sp-card:nth-child(3),
    .sp-grid--n5 .sp-card:nth-child(5),
    .sp-grid--n7 .sp-card:nth-child(7) {{
      grid-column: 1 / -1;
      justify-self: center;
      max-width: calc(50% - 11px);
    }}
    .sp-card {{
      border-radius: 22px;
      padding: 20px 18px;
      min-height: 102px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .sp-card--a {{
      background: linear-gradient(175deg, #fffefb 0%, #f4ebe2 100%);
      border: 1.5px dashed rgba(120, 98, 72, 0.4);
      box-shadow:
        0 2px 0 rgba(255, 255, 255, 0.85) inset,
        0 10px 26px rgba(38, 32, 24, 0.07);
    }}
    .sp-card--b {{
      background: linear-gradient(185deg, rgba(255, 252, 248, 0.98) 0%, rgba(242, 232, 220, 0.75) 100%);
      border: 1px solid rgba(130, 108, 78, 0.22);
      box-shadow: 0 8px 22px rgba(38, 32, 24, 0.06);
    }}
    .sp-card--c {{
      background: linear-gradient(165deg, #faf6ef 0%, #ebe0d4 100%);
      border: 1.5px dashed rgba(100, 85, 65, 0.32);
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.8) inset,
        0 6px 18px rgba(38, 32, 24, 0.05);
    }}
    .sp-q {{
      margin: 0;
      font-size: 14px;
      line-height: 1.62;
      font-weight: 500;
      color: #221a12;
      text-align: center;
      overflow-wrap: anywhere;
      word-wrap: break-word;
      letter-spacing: 0.015em;
    }}
  </style>
</head>
<body>
  <div class="sp-page page">
    <div class="sp-ribbon" aria-hidden="true"></div>
    <header class="{head_cls}">
      <div class="sp-headline">
        <h1 class="sp-title">{title}</h1>
        {sub_html}
      </div>
      {deco_html}
    </header>
    {grid_html}
  </div>
</body>
</html>"""

    def _lesson_card_v1_template(self, card: dict[str, Any]) -> str:
        topic = (card.get("lesson_topic") or card.get("title") or "Lesson").strip()
        lead_in = list(card.get("lead_in_questions_lines") or [])[:3]
        if not lead_in:
            lead_in = [escape("—")]
        lead_items = "".join(f'<li class="lc-li">{q}</li>' for q in lead_in)

        choices = list(card.get("choice_lines") or [])[:6]
        if not choices:
            choices = [escape("—")]
        choice_items = "".join(f'<li class="lc-li lc-choice">{c}</li>' for c in choices)

        raw_img = str(card.get("image_url") or "").strip()
        if raw_img and is_safe_topic_image_url(raw_img):
            safe_img = escape(raw_img, quote=True)
            media_html = (
                f'<figure class="lc-media lc-media-img">'
                f'<img src="{safe_img}" alt="" loading="lazy" /></figure>'
            )
        else:
            media_html = '<div class="lc-media lc-media-placeholder" aria-hidden="true"></div>'

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=600, initial-scale=1.0" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Caveat:wght@600&family=DM+Serif+Display:ital@0;1&display=swap" rel="stylesheet" />
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; }}
    body {{
      background: linear-gradient(165deg, #d8cdc2 0%, #c9beb3 50%, #d2c7bc 100%);
      font-family: "DM Serif Display", Georgia, serif;
      color: #2a1f1c;
      -webkit-font-smoothing: antialiased;
    }}
    .lc-page.page {{
      width: 600px;
      min-height: 920px;
      margin: 0 auto;
      position: relative;
      overflow: hidden;
      background-color: #e5d9cc;
      background-image:
        linear-gradient(90deg, rgba(140, 120, 100, 0.04) 1px, transparent 1px),
        linear-gradient(rgba(140, 120, 100, 0.034) 1px, transparent 1px),
        radial-gradient(ellipse 85% 65% at 12% 25%, rgba(255, 255, 255, 0.5) 0%, transparent 48%),
        radial-gradient(ellipse 70% 55% at 88% 75%, rgba(235, 218, 200, 0.45) 0%, transparent 50%),
        radial-gradient(ellipse 110% 55% at 50% -5%, rgba(255, 252, 248, 0.92) 0%, transparent 52%),
        linear-gradient(178deg, #fdfaf5 0%, #f4ebe1 35%, #ebe1d6 70%, #e5d9cc 100%);
      background-size:
        24px 24px,
        24px 24px,
        100% 100%,
        100% 100%,
        100% 100%,
        100% 100%;
      background-position: 0 0, 0 0, 0 0, 0 0, 0 0, 0 0;
      box-shadow: inset 0 0 100px rgba(255, 250, 245, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.55);
      padding: 32px 28px 36px;
    }}
    .lc-header {{
      margin: 0 auto 20px;
      width: 100%;
      max-width: 504px;
      padding: 26px 24px 24px;
      border-radius: 20px;
      text-align: center;
      background: linear-gradient(168deg, #f2e8de 0%, #e9ddd2 45%, #e0d2c6 100%);
      border: 1px solid rgba(100, 78, 58, 0.16);
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.7) inset,
        0 14px 36px rgba(45, 35, 28, 0.08),
        0 5px 14px rgba(45, 35, 28, 0.05);
    }}
    .lc-topic-title {{
      font-family: "Caveat", cursive;
      font-size: 40px;
      font-weight: 600;
      line-height: 1.08;
      margin: 0;
      color: #3d2a22;
      letter-spacing: -0.02em;
      word-wrap: break-word;
      overflow-wrap: anywhere;
    }}
    .lc-header-divider {{
      height: 2px;
      width: min(220px, 72%);
      margin: 18px auto 0;
      border-radius: 2px;
      background: linear-gradient(90deg, transparent 0%, rgba(90, 65, 48, 0.42) 50%, transparent 100%);
    }}
    .lc-panel {{
      margin: 0 auto 18px;
      width: 100%;
      max-width: 504px;
      padding: 18px 20px 20px;
      border-radius: 18px;
      position: relative;
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.7) inset,
        0 14px 36px rgba(45, 35, 28, 0.09),
        0 5px 12px rgba(45, 35, 28, 0.04);
    }}
    .lc-panel-a {{
      background: linear-gradient(168deg, #fffcf9 0%, #faf5ee 100%);
      border: 1px solid rgba(200, 175, 150, 0.2);
      transform: rotate(-0.55deg);
    }}
    .lc-panel-b {{
      background: linear-gradient(168deg, #fffef9 0%, #f8f2ea 100%);
      border: 1px solid rgba(190, 165, 140, 0.2);
      transform: rotate(0.45deg);
    }}
    .lc-pin {{
      position: absolute;
      top: -6px;
      right: 18px;
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: radial-gradient(circle at 30% 30%, #ff8a80, #c62828);
      box-shadow: 0 3px 6px rgba(0, 0, 0, 0.22);
    }}
    .lc-h2 {{
      margin: 0 0 14px;
      font-size: 9.5px;
      font-weight: 700;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: #5c4a42;
    }}
    .lc-ul {{
      margin: 0;
      padding: 0 0 0 20px;
    }}
    .lc-li {{
      margin: 0 0 12px;
      font-size: 14px;
      line-height: 1.52;
      color: #2f2420;
      font-weight: 500;
    }}
    .lc-li:last-child {{ margin-bottom: 0; }}
    .lc-choice {{
      font-size: 13.5px;
      line-height: 1.48;
    }}
    .lc-media {{
      margin: 10px auto 0;
      width: 100%;
      max-width: 504px;
      border-radius: 18px;
      overflow: hidden;
      border: 1px solid rgba(100, 78, 58, 0.14);
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.65) inset,
        0 10px 28px rgba(45, 35, 28, 0.09),
        0 4px 12px rgba(45, 35, 28, 0.05);
    }}
    .lc-media-img img {{
      display: block;
      width: 100%;
      height: auto;
      vertical-align: middle;
    }}
    .lc-media-placeholder {{
      min-height: 168px;
      background: linear-gradient(168deg, #f2e8de 0%, #e9ddd2 45%, #e0d2c6 100%);
    }}
  </style>
</head>
<body>
  <div class="lc-page page">
    <header class="lc-header" aria-label="Topic">
      <h1 class="lc-topic-title">{topic}</h1>
      <div class="lc-header-divider" aria-hidden="true"></div>
    </header>
    <section class="lc-panel lc-panel-a" aria-labelledby="lc-leadin">
      <div class="lc-pin" aria-hidden="true"></div>
      <h2 id="lc-leadin" class="lc-h2">Lead-in</h2>
      <ul class="lc-ul">{lead_items}</ul>
    </section>
    <section class="lc-panel lc-panel-b" aria-labelledby="lc-tot">
      <div class="lc-pin" aria-hidden="true"></div>
      <h2 id="lc-tot" class="lc-h2">This or that</h2>
      <ul class="lc-ul">{choice_items}</ul>
    </section>
    {media_html}
  </div>
</body>
</html>"""

    def _lesson_art_v1_template(self, card: dict[str, Any]) -> str:
        topic = (card.get("lesson_topic") or card.get("title") or "Lesson").strip()

        # Flexible caps: lead-in 2–3, discussion 2–4, choices 3–4, vocab 4–6 (use what exists).
        lead_raw = list(card.get("lead_in_questions_lines") or [])
        lead_in = lead_raw[:3]
        if not lead_in:
            lead_in = [escape("—")]
        lead_items = "".join(f'<li class="la-li">{q}</li>' for q in lead_in)

        disc_raw = list(card.get("discussion_questions_lines") or [])
        discussion = disc_raw[:4]
        if not discussion:
            discussion = [escape("—")]
        disc_items = "".join(f'<li class="la-li">{q}</li>' for q in discussion)

        choice_raw = list(card.get("choice_lines") or [])
        choices = choice_raw[:4]
        if not choices:
            choices = [escape("—")]
        choice_items = "".join(
            f'<li class="la-li la-li-choice">{c}</li>' for c in choices
        )

        vocab_raw = list(card.get("lesson_vocab_lines") or [])
        vocab_lines = vocab_raw[:6]
        if not vocab_lines:
            vocab_lines = [escape("—")]
        vocab_items = "".join(
            f'<li class="la-vocab-pill"><span class="la-vocab-text">{w}</span></li>'
            for w in vocab_lines
        )

        raw_img = str(card.get("image_url") or "").strip()
        if raw_img and is_safe_topic_image_url(raw_img):
            safe_img = escape(raw_img, quote=True)
            media_html = (
                f'<figure class="la-media la-media-img" aria-label="Visual">'
                f'<img src="{safe_img}" alt="" loading="lazy" /></figure>'
            )
        else:
            media_html = (
                '<div class="la-media la-media-placeholder" aria-hidden="true"></div>'
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=600, initial-scale=1.0" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap" rel="stylesheet" />
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; }}
    body {{
      background: #e4d9cc;
      font-family: "Plus Jakarta Sans", system-ui, sans-serif;
      color: #2b221d;
      -webkit-font-smoothing: antialiased;
    }}
    .la-page.page {{
      width: 600px;
      min-height: 1120px;
      margin: 0 auto;
      position: relative;
      overflow: hidden;
      background:
        radial-gradient(ellipse 140% 90% at 50% -10%, rgba(255, 252, 247, 0.95) 0%, transparent 55%),
        linear-gradient(175deg, #faf6ef 0%, #f3ebe1 38%, #ebe0d4 72%, #e3d6c8 100%);
      padding: 40px 30px 36px;
    }}
    .la-page::before {{
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      opacity: 0.55;
      background:
        radial-gradient(ellipse 80% 50% at 15% 20%, rgba(255, 255, 255, 0.5) 0%, transparent 50%),
        radial-gradient(ellipse 70% 45% at 92% 88%, rgba(210, 185, 160, 0.28) 0%, transparent 50%);
    }}
    .la-page::after {{
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      opacity: 0.035;
      background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    }}
    .la-inner {{
      position: relative;
      z-index: 1;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }}
    .la-title-card {{
      text-align: center;
      padding: 34px 28px 32px;
      border-radius: 26px;
      background:
        linear-gradient(165deg, rgba(255, 255, 255, 0.92) 0%, #fffdfb 48%, #faf5ee 100%);
      border: 1px solid rgba(130, 105, 85, 0.14);
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 1) inset,
        0 2px 0 rgba(255, 250, 245, 0.6) inset,
        0 22px 50px rgba(55, 38, 28, 0.09),
        0 8px 20px rgba(45, 32, 24, 0.05);
    }}
    .la-title-eyebrow {{
      margin: 0 0 10px;
      font-size: 9.5px;
      font-weight: 600;
      letter-spacing: 0.28em;
      text-transform: uppercase;
      color: #9a8274;
    }}
    .la-title {{
      font-family: "Cormorant Garamond", Georgia, serif;
      font-size: 42px;
      font-weight: 600;
      line-height: 1.1;
      margin: 0;
      color: #231a15;
      letter-spacing: -0.03em;
      text-shadow: 0 1px 0 rgba(255, 255, 255, 0.5);
    }}
    .la-title-line {{
      width: min(240px, 62%);
      height: 3px;
      margin: 20px auto 0;
      border-radius: 3px;
      background: linear-gradient(
        90deg,
        transparent,
        rgba(165, 130, 105, 0.45) 15%,
        rgba(195, 155, 125, 0.55) 50%,
        rgba(165, 130, 105, 0.45) 85%,
        transparent
      );
      box-shadow: 0 1px 2px rgba(255, 255, 255, 0.6);
    }}
    .la-title-deco {{
      margin: 14px auto 0;
      width: 48px;
      height: 6px;
      opacity: 0.45;
      background: radial-gradient(circle, rgba(160, 125, 100, 0.5) 0%, transparent 70%);
    }}
    .la-block {{
      padding: 20px 22px 22px;
      border-radius: 20px;
      background: #fffefb;
      border: 1px solid rgba(125, 100, 80, 0.1);
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.98) inset,
        0 14px 36px rgba(42, 30, 22, 0.055),
        0 4px 14px rgba(42, 30, 22, 0.04);
    }}
    .la-h {{
      margin: 0 0 13px;
      font-size: 9.5px;
      font-weight: 600;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: #8a7264;
    }}
    .la-ul {{
      margin: 0;
      padding: 0 0 0 20px;
    }}
    .la-li {{
      margin: 0 0 12px;
      font-size: 14.5px;
      line-height: 1.58;
      color: #342a24;
      font-weight: 500;
    }}
    .la-li:last-child {{ margin-bottom: 0; }}
    .la-li-choice {{
      font-size: 14px;
      line-height: 1.52;
      color: #3d322c;
    }}
    .la-vocab-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 9px 10px;
      margin: 0;
      padding: 0;
      list-style: none;
    }}
    .la-vocab-pill {{
      display: block;
      padding: 9px 16px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 500;
      color: #3d332c;
      background: linear-gradient(180deg, #faf7f3 0%, #f0e8df 100%);
      border: 1px solid rgba(125, 100, 82, 0.13);
      box-shadow: 0 1px 2px rgba(38, 28, 20, 0.05);
    }}
    .la-vocab-text {{
      display: inline;
    }}
    .la-media {{
      margin-top: 6px;
      width: 100%;
      border-radius: 22px;
      overflow: hidden;
      border: 1px solid rgba(110, 88, 70, 0.14);
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.75) inset,
        0 20px 48px rgba(38, 28, 22, 0.12),
        0 8px 22px rgba(38, 28, 22, 0.08);
    }}
    .la-media.la-media-img {{
      position: relative;
      min-height: 280px;
      max-height: 320px;
      background: #e8dfd6;
    }}
    .la-media-img img {{
      display: block;
      width: 100%;
      height: 100%;
      min-height: 280px;
      object-fit: cover;
      object-position: center;
      vertical-align: middle;
    }}
    .la-media-placeholder {{
      min-height: 280px;
      background:
        linear-gradient(145deg, #efe6dc 0%, #e2d5c8 50%, #d8cbc0 100%);
    }}
  </style>
</head>
<body>
  <div class="la-page page">
    <div class="la-inner">
      <header class="la-title-card" aria-label="Topic">
        <p class="la-title-eyebrow">Lesson</p>
        <h1 class="la-title">{topic}</h1>
        <div class="la-title-line" aria-hidden="true"></div>
        <div class="la-title-deco" aria-hidden="true"></div>
      </header>
      <section class="la-block" aria-labelledby="la-lead">
        <h2 id="la-lead" class="la-h">Lead-in</h2>
        <ul class="la-ul">{lead_items}</ul>
      </section>
      <section class="la-block" aria-labelledby="la-disc">
        <h2 id="la-disc" class="la-h">Discussion</h2>
        <ul class="la-ul">{disc_items}</ul>
      </section>
      <section class="la-block" aria-labelledby="la-tot">
        <h2 id="la-tot" class="la-h">This or that</h2>
        <ul class="la-ul">{choice_items}</ul>
      </section>
      <section class="la-block la-block-vocab" aria-labelledby="la-voc">
        <h2 id="la-voc" class="la-h">Vocabulary</h2>
        <ul class="la-vocab-grid">{vocab_items}</ul>
      </section>
      {media_html}
    </div>
  </div>
</body>
</html>"""

    def _phrases_card_template(self, card: dict[str, Any]) -> str:
        title = card["title"]
        sub_raw = str(card.get("subtitle") or "").strip()
        subtitle_html = f'<p class="pc-subtitle">{sub_raw}</p>' if sub_raw else ""
        handle = (card.get("handle_display") or "").strip()
        handle_html = f'<p class="pc-handle">{handle}</p>' if handle else ""

        blocks_in = list(card.get("phrases_blocks") or [])
        if not blocks_in:
            blocks_in = [
                {
                    "phrase_e": escape("—"),
                    "translation_e": "",
                    "formula_e": escape("—"),
                    "ex1_html": "",
                    "ex2_html": "",
                    "num": "1",
                }
            ]

        plane_svg = """<svg class="pc-plane-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="32" height="32" aria-hidden="true"><path fill="rgba(110, 140, 168, 0.42)" d="M2 21l20-9L2 3v7l12 2-12 2v7z"/></svg>"""

        block_parts: List[str] = []
        for b in blocks_in:
            tr = (b.get("translation_e") or "").strip()
            trans_part = (
                f' <span class="pc-trans">({tr})</span>' if tr else ""
            )
            ex1 = b.get("ex1_html") or ""
            ex2 = b.get("ex2_html") or ""
            li1c = ex1 if ex1 else escape("—")
            li2c = ex2 if ex2 else escape("—")
            block_parts.append(
                f"""<article class="pc-block">
  <div class="pc-num-circle" aria-hidden="true"><span class="pc-num-inner">{b.get("num", "?")}</span></div>
  <div class="pc-block-main">
    <p class="pc-phrase-line"><span class="pc-phrase-text">{b.get("phrase_e", "")}</span>{trans_part}</p>
    <div class="pc-formula">{b.get("formula_e", "")}</div>
    <ul class="pc-ex-ul">
      <li class="pc-ex-li">{li1c}</li>
      <li class="pc-ex-li">{li2c}</li>
    </ul>
  </div>
</article>"""
            )

        blocks_html = "\n".join(block_parts)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=600, initial-scale=1.0" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&display=swap" rel="stylesheet" />
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; }}
    body {{
      background: #e8e4de;
      font-family: "DM Serif Display", Georgia, serif;
      -webkit-font-smoothing: antialiased;
    }}
    .pc-page {{
      width: 600px;
      min-height: 920px;
      margin: 0 auto;
      position: relative;
      background: #fdf8f0;
      padding: 36px 36px 44px;
      overflow: hidden;
    }}
    .pc-decor {{
      position: absolute;
      inset: 0;
      pointer-events: none;
      z-index: 0;
    }}
    .pc-line {{
      position: absolute;
      border: 1px solid rgba(140, 170, 195, 0.28);
      border-radius: 50%;
    }}
    .pc-line-a {{ width: 240px; height: 150px; top: 70px; left: -90px; transform: rotate(-26deg); }}
    .pc-line-b {{ width: 200px; height: 120px; bottom: 140px; right: -50px; transform: rotate(22deg); }}
    .pc-line-c {{ width: 280px; height: 170px; top: 42%; left: 8%; transform: rotate(12deg); opacity: 0.75; }}
    .pc-line-d {{ width: 160px; height: 100px; top: 18%; right: 12%; transform: rotate(-15deg); opacity: 0.55; }}
    .pc-plane {{
      position: absolute;
      z-index: 2;
      pointer-events: none;
    }}
    .pc-plane-tr {{ top: 20px; right: 24px; }}
    .pc-plane-br {{ bottom: 32px; right: 28px; }}
    .pc-plane-svg {{ display: block; }}
    .pc-handle {{
      position: absolute;
      top: 28px;
      left: 28px;
      z-index: 3;
      margin: 0;
      font-size: 11px;
      font-style: italic;
      color: #888;
      letter-spacing: 0.02em;
      max-width: 42%;
      overflow-wrap: anywhere;
    }}
    .pc-head {{
      position: relative;
      z-index: 1;
      text-align: center;
      padding: 8px 48px 28px;
    }}
    .pc-title {{
      margin: 0;
      font-size: 32px;
      font-weight: 700;
      color: #0a0a0a;
      line-height: 1.12;
      letter-spacing: -0.02em;
      word-wrap: break-word;
      overflow-wrap: anywhere;
    }}
    .pc-subtitle {{
      margin: 12px 0 0;
      font-size: 13px;
      font-style: italic;
      color: #777;
      line-height: 1.45;
    }}
    .pc-phrases {{
      position: relative;
      z-index: 1;
    }}
    .pc-block {{
      display: flex;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 36px;
    }}
    .pc-block:last-child {{ margin-bottom: 0; }}
    .pc-num-circle {{
      flex-shrink: 0;
      width: 30px;
      height: 30px;
      border-radius: 50%;
      background: #2d6fa8;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-top: 2px;
      box-shadow: 0 2px 6px rgba(45, 111, 168, 0.25);
    }}
    .pc-num-inner {{
      font-size: 14px;
      font-weight: 700;
      color: #fff;
      line-height: 1;
    }}
    .pc-block-main {{ flex: 1; min-width: 0; }}
    .pc-phrase-line {{
      margin: 0 0 10px;
      font-size: 15px;
      line-height: 1.45;
    }}
    .pc-phrase-text {{ font-weight: 700; color: #0a0a0a; }}
    .pc-trans {{
      font-weight: 400;
      font-style: italic;
      color: #777;
    }}
    .pc-formula {{
      display: inline-block;
      max-width: 100%;
      padding: 10px 14px;
      border-radius: 10px;
      background: #f0e8d8;
      font-size: 14px;
      font-weight: 700;
      color: #1a1a1a;
      line-height: 1.4;
      margin-bottom: 4px;
    }}
    .pc-ex-ul {{
      margin: 12px 0 0;
      padding: 0 0 0 18px;
    }}
    .pc-ex-li {{
      margin: 0 0 8px;
      font-size: 13px;
      line-height: 1.5;
      color: #1a1a1a;
    }}
    .pc-ex-li:last-child {{ margin-bottom: 0; }}
    .pc-ex-bold {{
      color: #2d6fa8;
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <div class="pc-page page">
    <div class="pc-decor" aria-hidden="true">
      <div class="pc-line pc-line-a"></div>
      <div class="pc-line pc-line-b"></div>
      <div class="pc-line pc-line-c"></div>
      <div class="pc-line pc-line-d"></div>
    </div>
    <div class="pc-plane pc-plane-tr">{plane_svg}</div>
    <div class="pc-plane pc-plane-br">{plane_svg}</div>
    {handle_html}
    <header class="pc-head">
      <h1 class="pc-title">{title}</h1>
      {subtitle_html}
    </header>
    <div class="pc-phrases">
      {blocks_html}
    </div>
  </div>
</body>
</html>"""

    # --- vocab_card: vocabulary-focused layout (warm paper family) ---

    def _vocab_card_template(self, card: dict[str, Any]) -> str:
        topic = card["title"]
        level = _badge_level(card["subtitle"])
        sub = (card.get("subtitle") or "").strip()
        subtitle_html = (
            f'<p class="vc-sub">{sub}</p>' if sub else '<p class="vc-sub muted">Vocabulary</p>'
        )
        cta = card["cta"]
        rows_in = list(card.get("vocab_card_rows") or [])
        if not rows_in and card.get("vocabulary_lines"):
            for vl in (card.get("vocabulary_lines") or [])[:8]:
                rows_in.append({"term": vl, "translation": "", "example": ""})

        item_html: List[str] = []
        for row in rows_in[:8]:
            term = row.get("term", "")
            translation = row.get("translation") or row.get("gloss", "")
            ex = row.get("example", "")
            pair = (
                f'<p class="vc-pair"><span class="vc-term">{term}</span>'
                f'<span class="vc-dash">—</span><span class="vc-trans">{translation}</span></p>'
            )
            ex_blk = f'<p class="vc-ex">{ex}</p>' if ex else ""
            item_html.append(f'<div class="vc-item">{pair}{ex_blk}</div>')
        while len(item_html) < 5:
            item_html.append(
                '<div class="vc-item muted"><p class="vc-pair">word — слово</p></div>'
            )

        vocab_block = f'<div class="vc-list" aria-label="Vocabulary">{"".join(item_html[:8])}</div>'
        try_block = f"""
    <section class="vc-try" aria-label="Try it">
      <p class="vc-try-label">Try it:</p>
      <p class="vc-try-body">{cta if cta else "Say one sentence using two of the words above."}</p>
    </section>
        """
        hero_block = _hero_media_block(card, "warm_v2")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=600, initial-scale=1.0" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Caveat:wght@600&family=DM+Serif+Display:ital@0;1&display=swap" rel="stylesheet" />
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; }}
    body {{
      background: linear-gradient(165deg, #d8cdc2 0%, #c9beb3 50%, #d2c7bc 100%);
      font-family: "DM Serif Display", Georgia, serif;
      color: #2a1f1c;
      -webkit-font-smoothing: antialiased;
    }}
    .page {{
      width: 600px;
      min-height: 880px;
      position: relative;
      overflow: hidden;
      background-color: #e5d9cc;
      background-image:
        linear-gradient(90deg, rgba(140, 120, 100, 0.04) 1px, transparent 1px),
        linear-gradient(rgba(140, 120, 100, 0.034) 1px, transparent 1px),
        radial-gradient(ellipse 85% 65% at 12% 25%, rgba(255, 255, 255, 0.5) 0%, transparent 48%),
        radial-gradient(ellipse 70% 55% at 88% 75%, rgba(235, 218, 200, 0.45) 0%, transparent 50%),
        radial-gradient(ellipse 110% 55% at 50% -5%, rgba(255, 252, 248, 0.92) 0%, transparent 52%),
        linear-gradient(178deg, #fdfaf5 0%, #f4ebe1 35%, #ebe1d6 70%, #e5d9cc 100%);
      background-size: 24px 24px, 24px 24px, 100% 100%, 100% 100%, 100% 100%, 100% 100%;
      box-shadow: inset 0 0 100px rgba(255, 250, 245, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.55);
      padding: 32px 28px 36px;
    }}
    .level-badge {{
      position: absolute; top: 18px; left: 18px; width: 50px; height: 50px; border-radius: 50%;
      background: linear-gradient(145deg, #8b2323 0%, #6a1a1a 100%);
      color: #fffaf3; font-size: 10px; line-height: 1.05; display: flex;
      align-items: center; justify-content: center; text-align: center; padding: 4px;
      box-shadow: 0 6px 16px rgba(60, 20, 20, 0.25), 0 2px 4px rgba(0,0,0,0.12);
      z-index: 5; overflow-wrap: anywhere;
    }}
    .vc-hdr {{ padding: 8px 58px 12px 62px; text-align: center; }}
    .vc-title {{
      font-family: "DM Serif Display", Georgia, serif;
      font-size: 34px;
      font-weight: 700;
      line-height: 1.08;
      margin: 0;
      color: #2a1f1c;
      letter-spacing: -0.02em;
      word-wrap: break-word;
      overflow-wrap: anywhere;
    }}
    .vc-sub {{ margin: 10px auto 0; max-width: 92%; font-size: 12.5px; line-height: 1.5; color: #5c4a42; font-weight: 500; }}
    .hero-media {{
      margin: 14px auto 16px;
      width: 88%;
      max-width: 504px;
      height: 140px;
      border-radius: 20px;
      overflow: hidden;
      border: 1px solid rgba(70, 50, 38, 0.1);
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.55) inset,
        0 14px 40px rgba(35, 24, 18, 0.1),
        0 5px 14px rgba(35, 24, 18, 0.06);
    }}
    .hero-img {{
      display: block;
      width: 100%;
      height: 100%;
      min-height: 140px;
      object-fit: cover;
      object-position: center;
    }}
    .vc-list {{
      margin: 8px auto 0;
      width: 92%;
      max-width: 520px;
      padding: 16px 18px 18px;
      border-radius: 18px;
      background: linear-gradient(168deg, #fffcf9 0%, #faf5ee 100%);
      border: 1px solid rgba(200, 175, 150, 0.22);
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.7) inset,
        0 12px 32px rgba(45, 35, 28, 0.08);
    }}
    .vc-item {{ margin-bottom: 14px; }}
    .vc-item:last-child {{ margin-bottom: 0; }}
    .vc-pair {{ margin: 0; font-size: 13px; line-height: 1.45; }}
    .vc-term {{ font-weight: 700; color: #1f1612; }}
    .vc-dash {{ margin: 0 6px; opacity: 0.45; }}
    .vc-trans {{ color: #3e2723; }}
    .vc-ex {{
      margin: 6px 0 0;
      padding-left: 2px;
      font-size: 10.5px;
      line-height: 1.4;
      color: rgba(60, 48, 40, 0.72);
      font-style: italic;
    }}
    .muted {{ color: rgba(47, 36, 32, 0.42); font-style: italic; }}
    .vc-try {{
      width: 100%;
      margin-top: 18px;
      padding: 16px 18px 18px;
      border-radius: 18px;
      background: linear-gradient(138deg, #1b5e20 0%, #2e7d32 48%, #1b5e20 100%);
      color: #f6fcf7;
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.12) inset,
        0 12px 32px rgba(15, 50, 25, 0.22),
        0 4px 12px rgba(15, 50, 25, 0.12);
    }}
    .vc-try-label {{
      margin: 0 0 8px;
      font-family: "Caveat", cursive;
      font-size: 22px;
      line-height: 1.1;
      color: #e8f5e9;
    }}
    .vc-try-body {{
      margin: 0;
      font-size: 12px;
      line-height: 1.55;
      overflow-wrap: anywhere;
      word-wrap: break-word;
      opacity: 0.97;
    }}
    {HERO_LAYER_CSS}
  </style>
</head>
<body>
  <div class="page">
    <div class="level-badge">{level}</div>
    <header class="vc-hdr">
      <h1 class="vc-title">{topic}</h1>
      {subtitle_html}
    </header>
    {hero_block}
    {vocab_block}
    {try_block}
  </div>
</body>
</html>"""

    # --- v2: refined layout / typography (v1 retained); ScreenshotOne: .page 600×920, selector .page ---

    def _warm_paper_v2_template(self, card: dict[str, Any]) -> str:
        topic = card["title"]
        level = _badge_level(card["subtitle"])
        sub = (card.get("subtitle") or "").strip()
        subtitle_html = (
            f'<p class="topic-sub-v2">{sub}</p>' if sub else '<p class="topic-sub-v2 muted">Study note</p>'
        )
        bullets = card["bullets"]
        cta = card["cta"]
        voc_lines = card.get("vocabulary_lines") or []
        mcq_lines = card.get("mcq_bracket_lines") or []

        if isinstance(voc_lines, list) and voc_lines:
            vocab_rows = _vocab_rows_from_lines([str(x) for x in voc_lines if str(x).strip()])
        else:
            pairs = _split_term_translation(bullets)
            vocab_rows = []
            for term, trans in pairs[:4]:
                if trans:
                    vocab_rows.append(
                        f'<div class="vr-v2"><span class="tv2">{term}</span>'
                        f'<span class="sv2">—</span><span class="gv2 ua">{trans}</span></div>'
                    )
                else:
                    vocab_rows.append(f'<div class="vr-v2"><span class="tv2 full">{term}</span></div>')
        if not vocab_rows:
            vocab_rows.append(
                '<div class="vr-v2 muted">word — слово</div>'
            )

        has_contrast = bool(card.get("contrast_wrong") or card.get("contrast_better"))

        if isinstance(mcq_lines, list) and mcq_lines:
            ex_lines = [str(x) for x in mcq_lines if str(x).strip()][:4]
        else:
            ex_lines = []
            for b in bullets:
                if "(" in b and ")" in b and "/" in b:
                    ex_lines.append(b)
            if len(ex_lines) < 2:
                ex_lines = list(bullets)[:4]

        mcq_items = []
        for i, line in enumerate(ex_lines, 1):
            hit = " key-hit" if i == 1 and not has_contrast else ""
            styled = _highlight_bracket_choice(line)
            mcq_items.append(f'<div class="mcq-v2{hit}"><span class="nv2">{i}.</span> {styled}</div>')
        while len(mcq_items) < 3:
            mcq_items.append(
                f'<div class="mcq-v2 muted"><span class="nv2">{len(mcq_items) + 1}.</span> '
                f"Template: one sentence with (option / option).</div>"
            )

        if vocab_rows and not has_contrast:
            vr0 = vocab_rows[0].replace('<div class="vr-v2"', '<div class="vr-v2 key-hit"', 1)
            vocab_rows[0] = vr0

        content_sections = f"""
    <div class="cols-v2">
      <section class="panel-v2 pv2-a" aria-label="Vocabulary">
        <div class="pin-v2"></div>
        <p class="label-v2">Vocabulary</p>
        <p class="hint-v2">English — українська · phrases only</p>
        <div class="body-v2 body-v2-vocab">{"".join(vocab_rows)}</div>
      </section>
      <section class="panel-v2 pv2-b" aria-label="Choose the correct option">
        <div class="pin-v2"></div>
        <p class="label-v2">Choose the correct option</p>
        <p class="hint-v2">Bracket choices — different from vocabulary</p>
        <div class="body-v2 body-v2-mcq">{"".join(mcq_items[:4])}</div>
      </section>
    </div>
    <section class="speak-v2" aria-label="Let's speak">
      <p class="label-v2 light">Let’s speak</p>
      <p class="speak-body-v2">{cta if cta else "Practice aloud in a short sentence."}</p>
    </section>
        """

        hero_block = _hero_media_block(card, "warm_v2")
        contrast_strip = _contrast_strip_html(card, "wp2")
        return self._wrap_warm_paper_v2_html(
            topic, level, subtitle_html, hero_block, contrast_strip, content_sections
        )

    def _wrap_warm_paper_v2_html(
        self,
        topic: str,
        level: str,
        subtitle_html: str,
        hero_block: str,
        contrast_strip: str,
        content_sections: str,
    ) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=600, initial-scale=1.0" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Caveat:wght@600&family=DM+Serif+Display:ital@0;1&display=swap" rel="stylesheet" />
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; }}
    body {{
      background: linear-gradient(165deg, #d8cdc2 0%, #c9beb3 50%, #d2c7bc 100%);
      font-family: "DM Serif Display", Georgia, serif;
      color: #2a1f1c;
      -webkit-font-smoothing: antialiased;
    }}
    .page {{
      width: 600px;
      min-height: 880px;
      position: relative;
      overflow: hidden;
      background-color: #e5d9cc;
      background-image:
        linear-gradient(90deg, rgba(140, 120, 100, 0.04) 1px, transparent 1px),
        linear-gradient(rgba(140, 120, 100, 0.034) 1px, transparent 1px),
        radial-gradient(ellipse 85% 65% at 12% 25%, rgba(255, 255, 255, 0.5) 0%, transparent 48%),
        radial-gradient(ellipse 70% 55% at 88% 75%, rgba(235, 218, 200, 0.45) 0%, transparent 50%),
        radial-gradient(ellipse 110% 55% at 50% -5%, rgba(255, 252, 248, 0.92) 0%, transparent 52%),
        linear-gradient(178deg, #fdfaf5 0%, #f4ebe1 35%, #ebe1d6 70%, #e5d9cc 100%);
      background-size:
        24px 24px,
        24px 24px,
        100% 100%,
        100% 100%,
        100% 100%,
        100% 100%;
      background-position: 0 0, 0 0, 0 0, 0 0, 0 0, 0 0;
      box-shadow: inset 0 0 100px rgba(255, 250, 245, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.55);
      padding: 32px 28px 36px;
    }}
    .level-badge {{
      position: absolute; top: 18px; left: 18px; width: 50px; height: 50px; border-radius: 50%;
      background: linear-gradient(145deg, #8b2323 0%, #6a1a1a 100%);
      color: #fffaf3; font-size: 10px; line-height: 1.05; display: flex;
      align-items: center; justify-content: center; text-align: center; padding: 4px;
      box-shadow: 0 6px 16px rgba(60, 20, 20, 0.25), 0 2px 4px rgba(0,0,0,0.12);
      z-index: 5; overflow-wrap: anywhere;
    }}
    .hdr-v2 {{ padding: 8px 58px 18px 62px; text-align: center; }}
    .topic-title-v2 {{
      font-family: "Caveat", cursive; font-size: 40px; line-height: 1.02; margin: 0; color: #3d2a22;
      letter-spacing: -0.02em; word-wrap: break-word; overflow-wrap: anywhere;
    }}
    .topic-sub-v2 {{ margin: 12px auto 0; max-width: 88%; font-size: 12px; line-height: 1.55; color: #6d5c54; font-weight: 500; }}
    .hero-media {{
      margin: 18px auto 18px;
      width: 88%;
      max-width: 504px;
      height: 156px;
      border-radius: 20px;
      overflow: hidden;
      border: 1px solid rgba(70, 50, 38, 0.1);
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.55) inset,
        0 18px 48px rgba(35, 24, 18, 0.12),
        0 6px 16px rgba(35, 24, 18, 0.07);
    }}
    .hero-img {{
      display: block;
      width: 100%;
      height: 100%;
      min-height: 156px;
      object-fit: cover;
      object-position: center;
    }}
    .insight-card {{
      margin: 18px auto 18px;
      width: 88%;
      max-width: 504px;
      min-height: 100px;
      padding: 22px 24px 24px 26px;
      border-radius: 22px;
      position: relative;
      overflow: hidden;
      transform: rotate(-0.4deg);
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.65) inset,
        0 14px 40px rgba(40, 28, 20, 0.09),
        0 5px 14px rgba(40, 28, 20, 0.05);
    }}
    .insight-card.insight-warm_v2 {{
      background: linear-gradient(168deg, #fffffe 0%, #f7f0e8 50%, #f0e8de 100%);
      border: 1px solid rgba(100, 78, 58, 0.14);
    }}
    .insight-card.insight-warm_v2.insight-premium {{
      background: linear-gradient(158deg, #fffefb 0%, #f5ebe3 42%, #ebe2d8 100%);
      border: 1px solid rgba(85, 65, 50, 0.18);
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.75) inset,
        0 20px 52px rgba(35, 26, 18, 0.1),
        0 6px 18px rgba(35, 26, 18, 0.06);
    }}
    .insight-kicker {{ font-family: "Caveat", cursive; font-size: 22px; margin: 0 0 10px; color: #7a4a3a; letter-spacing: 0.02em; }}
    .insight-body {{ margin: 0; font-size: 15px; font-weight: 600; line-height: 1.32; overflow-wrap: anywhere; word-wrap: break-word; color: #241a16; }}
    .insight-accent {{ position: absolute; left: 0; top: 12px; bottom: 12px; width: 4px; border-radius: 2px;
      background: linear-gradient(180deg, #b71c1c, #5d4037); }}
    .wp2-strip {{
      display: flex;
      margin: 0 auto 18px;
      width: 88%;
      max-width: 504px;
      border-radius: 18px;
      overflow: hidden;
      transform: rotate(0.35deg);
      box-shadow:
        0 12px 32px rgba(30, 20, 15, 0.1),
        0 4px 10px rgba(30, 20, 15, 0.05);
      border: 1px solid rgba(42, 31, 26, 0.12);
    }}
    .wp2-bad {{ flex: 1; background: linear-gradient(175deg, #fff5f5 0%, #ffcdd2 100%); padding: 14px 14px 16px; }}
    .wp2-good {{ flex: 1; background: linear-gradient(175deg, #f5fff7 0%, #c8e6c9 100%); padding: 14px 14px 16px;
      border-left: 2px dashed rgba(0,0,0,0.1); }}
    .wp2-tag {{ font-size: 9px; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase;
      color: #b71c1c; display: block; margin-bottom: 6px; }}
    .wp2-good .wp2-tag {{ color: #1b5e20; }}
    .wp2-txt {{ margin: 0; font-size: 12px; line-height: 1.38; font-weight: 600; color: #1a1a1a; overflow-wrap: anywhere; }}
    .mcq-v2.key-hit, .vr-v2.key-hit {{ border-left: 4px solid #c62828; padding-left: 10px; margin-left: -2px; background: rgba(255, 245, 240, 0.65); border-radius: 0 10px 10px 0; }}
    .mcq-gap {{ font-weight: 600; color: #4a3728; }}
    .mcq-or {{ color: #b71c1c; font-weight: 700; }}
    .cols-v2 {{ display: flex; gap: 18px; align-items: stretch; margin-bottom: 20px; }}
    .panel-v2 {{
      flex: 1;
      min-width: 0;
      padding: 18px 16px 20px;
      border-radius: 18px;
      position: relative;
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.7) inset,
        0 14px 36px rgba(45, 35, 28, 0.09),
        0 5px 12px rgba(45, 35, 28, 0.04);
    }}
    .pv2-a {{
      background: linear-gradient(168deg, #fffcf9 0%, #faf5ee 100%);
      border: 1px solid rgba(200, 175, 150, 0.2);
      transform: rotate(-1.05deg);
    }}
    .pv2-b {{
      background: linear-gradient(168deg, #fffef9 0%, #f8f2ea 100%);
      border: 1px solid rgba(190, 165, 140, 0.2);
      transform: rotate(0.95deg);
    }}
    .pin-v2 {{
      position: absolute; top: -6px; right: 18px; width: 14px; height: 14px; border-radius: 50%;
      background: radial-gradient(circle at 30% 30%, #ff8a80, #c62828);
      box-shadow: 0 3px 6px rgba(0,0,0,0.22);
    }}
    .label-v2 {{
      font-family: "DM Serif Display", Georgia, serif; font-size: 9.5px; font-weight: 700; letter-spacing: 0.16em;
      text-transform: uppercase; color: #5c4a42; margin: 0 0 4px;
    }}
    .hint-v2 {{ margin: 0 0 14px; font-size: 9px; letter-spacing: 0.07em; color: rgba(90, 72, 62, 0.48); font-weight: 500; }}
    .label-v2.light {{ color: #e8f5e9; }}
    .body-v2 {{ font-size: 11.5px; line-height: 1.48; }}
    .body-v2-vocab .vr-v2 {{ margin-bottom: 10px; }}
    .body-v2-mcq .mcq-v2 {{ margin-bottom: 12px; }}
    .vr-v2 {{ margin-bottom: 8px; overflow-wrap: anywhere; word-wrap: break-word; }}
    .tv2 {{ font-weight: 700; color: #2f2420; }}
    .tv2.full {{ display: block; }}
    .sv2 {{ margin: 0 5px; opacity: 0.45; }}
    .gv2 {{ color: #4e3d36; }}
    .gv2.ua {{ font-style: normal; color: #3e2723; letter-spacing: 0.01em; }}
    .mcq-v2 {{ margin-bottom: 8px; overflow-wrap: anywhere; word-wrap: break-word; }}
    .nv2 {{ font-weight: 700; color: #8b4513; margin-right: 5px; }}
    .muted {{ color: rgba(47, 36, 32, 0.42); font-style: italic; }}
    .speak-v2 {{
      width: 100%;
      padding: 18px 20px 20px;
      border-radius: 18px;
      transform: rotate(-0.35deg);
      background: linear-gradient(138deg, #1b5e20 0%, #2e7d32 48%, #1b5e20 100%);
      color: #f6fcf7;
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.12) inset,
        0 12px 32px rgba(15, 50, 25, 0.22),
        0 4px 12px rgba(15, 50, 25, 0.12);
    }}
    .speak-body-v2 {{ margin: 10px 0 0; font-size: 12px; line-height: 1.55; overflow-wrap: anywhere; word-wrap: break-word; opacity: 0.97; }}
    {HERO_LAYER_CSS}
  </style>
</head>
<body>
  <div class="page">
    <div class="level-badge">{level}</div>
    <header class="hdr-v2">
      <h1 class="topic-title-v2">{topic}</h1>
      {subtitle_html}
    </header>
    {hero_block}
    {contrast_strip}
    {content_sections}
  </div>
</body>
</html>"""

    def _kitchen_collage_v2_template(self, card: dict[str, Any]) -> str:
        topic = card["title"]
        level = _badge_level(card["subtitle"])
        sub = (card.get("subtitle") or "").strip()
        subtitle_html = (
            f'<p class="ksub-v2">{sub}</p>' if sub else '<p class="ksub-v2 muted">Mixed practice</p>'
        )
        bullets = card["bullets"]
        cta = card["cta"]
        b = bullets + [""] * 5
        sec_vocab = b[0] if b[0] else _placeholder_line()
        sec_fix = b[1] if b[1] else _placeholder_line()
        sec_choose = b[2] if b[2] else _placeholder_line()
        sec_gap = b[3] if b[3] else _placeholder_line()
        sec_speak = cta if cta else (b[4] if b[4] else "Short speaking cue.")

        has_contrast = bool(card.get("contrast_wrong") or card.get("contrast_better"))
        sections = (
            ("01", "Vocabulary list", sec_vocab),
            ("02", "Correct the mistake", sec_fix),
            ("03", "Choose the option", sec_choose),
            ("04", "Fill in the gaps", sec_gap),
        )
        blocks = []
        for idx, (num, label, body) in enumerate(sections):
            kf = " kfocus" if idx == 0 and not has_contrast else ""
            blocks.append(
                f'<section class="kpanel-v2{kf}"><span class="knum-v2">{num}</span>'
                f'<p class="klab-v2">{label}</p><p class="kbody-v2">{body}</p></section>'
            )
        blocks.append(
            f'<section class="kpanel-v2 kspeak-v2"><span class="knum-v2 klight">05</span>'
            f'<p class="klab-v2 klight">Speaking</p><p class="kbody-v2 klight">{sec_speak}</p></section>'
        )
        content_sections = '<div class="kstack-v2">' + "".join(blocks) + "</div>"

        hero_block = _hero_media_block(card, "kitchen_v2")
        contrast_strip = _contrast_strip_html(card, "kc2")
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=600, initial-scale=1.0" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Caveat:wght@600&family=DM+Serif+Display:ital@0;1&display=swap" rel="stylesheet" />
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; }}
    body {{
      background: linear-gradient(168deg, #d4c8ba 0%, #bfb3a4 48%, #c9beb0 100%);
      font-family: "DM Serif Display", Georgia, serif; color: #2c221e; -webkit-font-smoothing: antialiased;
    }}
    .kc2-strip {{
      display: flex; margin: 0 auto 16px; width: 90%; max-width: 520px; border-radius: 16px; overflow: hidden;
      transform: rotate(-0.45deg);
      box-shadow: 0 12px 28px rgba(30, 20, 15, 0.1), 0 4px 10px rgba(30, 20, 15, 0.06);
      border: 1px solid rgba(62, 39, 35, 0.2);
    }}
    .kc2-bad {{ flex: 1; background: linear-gradient(175deg, #fff5f5 0%, #ffcdd2 100%); padding: 12px 14px; }}
    .kc2-good {{ flex: 1; background: linear-gradient(175deg, #f3fff5 0%, #c8e6c9 100%); padding: 12px 14px; border-left: 2px dashed rgba(0,0,0,0.08); }}
    .kc2-tag {{ font-size: 8.5px; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; color: #b71c1c; display: block; margin-bottom: 4px; }}
    .kc2-good .kc2-tag {{ color: #1b5e20; }}
    .kc2-txt {{ margin: 0; font-size: 11.5px; line-height: 1.35; font-weight: 600; overflow-wrap: anywhere; }}
    .kpanel-v2.kfocus {{
      border: 2px solid #c62828;
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.5) inset,
        0 10px 28px rgba(198, 40, 40, 0.18),
        0 4px 12px rgba(198, 40, 40, 0.1);
    }}
    .page {{
      width: 600px; min-height: 920px; position: relative;
      overflow: hidden;
      background-color: #efe6dc;
      background-image:
        linear-gradient(90deg, rgba(130, 110, 90, 0.038) 1px, transparent 1px),
        linear-gradient(rgba(130, 110, 90, 0.032) 1px, transparent 1px),
        linear-gradient(0deg, rgba(235, 225, 210, 0.42) 1px, transparent 1px),
        radial-gradient(ellipse 90% 60% at 20% 15%, rgba(255, 255, 255, 0.55) 0%, transparent 45%),
        radial-gradient(ellipse 70% 55% at 85% 80%, rgba(230, 210, 190, 0.4) 0%, transparent 48%),
        linear-gradient(175deg, #fdfaf5 0%, #f2e8dc 45%, #e8dfd2 100%);
      background-size: 22px 22px, 22px 22px, 100% 26px, 100% 100%, 100% 100%, 100% 100%;
      background-position: 0 0, 0 0, 0 0, 0 0, 0 0, 0 0;
      box-shadow: inset 0 0 80px rgba(255, 248, 240, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.5);
      padding: 28px 22px 30px;
    }}
    .level-badge {{
      position: absolute; top: 16px; left: 16px; width: 50px; height: 50px; border-radius: 50%;
      background: linear-gradient(145deg, #8b2323 0%, #6a1a1a 100%);
      color: #fffaf3; font-size: 10px; display: flex; align-items: center;
      justify-content: center; text-align: center; padding: 4px;
      box-shadow: 0 6px 16px rgba(60, 20, 20, 0.22), 0 2px 4px rgba(0,0,0,0.1);
      z-index: 5; overflow-wrap: anywhere; line-height: 1.05;
    }}
    .khdr-v2 {{ padding: 6px 56px 16px 60px; text-align: center; }}
    .ktop-v2 {{ font-family: "Caveat", cursive; font-size: 38px; line-height: 1.05; margin: 0; color: #3d2a22;
      word-wrap: break-word; overflow-wrap: anywhere; }}
    .ksub-v2 {{ margin: 10px auto 0; max-width: 90%; font-size: 12.5px; line-height: 1.5; color: #5a4a42; }}
    .hero-media {{
      margin: 16px auto 16px;
      width: 90%;
      max-width: 520px;
      height: 148px;
      border-radius: 18px;
      overflow: hidden;
      border: 1px solid rgba(110, 85, 65, 0.14);
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.5) inset,
        0 16px 40px rgba(40, 30, 22, 0.11),
        0 5px 14px rgba(40, 30, 22, 0.06);
    }}
    .hero-img {{
      display: block;
      width: 100%;
      height: 100%;
      min-height: 148px;
      object-fit: cover;
      object-position: center;
    }}
    .insight-card {{
      margin: 16px auto 16px;
      width: 90%;
      max-width: 520px;
      min-height: 94px;
      padding: 18px 18px 18px 20px;
      border-radius: 18px;
      position: relative;
      overflow: hidden;
      transform: rotate(-0.5deg);
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.65) inset,
        0 12px 32px rgba(45, 32, 24, 0.09),
        0 4px 12px rgba(45, 32, 24, 0.05);
    }}
    .insight-card.insight-kitchen_v2 {{
      background: linear-gradient(168deg, #fffffe 0%, #f5efe6 55%, #ebe3d8 100%);
      border: 1px solid rgba(150, 120, 90, 0.22);
    }}
    .insight-kicker {{ font-family: "Caveat", cursive; font-size: 20px; margin: 0 0 6px; color: #8b4513; }}
    .insight-body {{ margin: 0; font-size: 12.5px; line-height: 1.48; overflow-wrap: anywhere; word-wrap: break-word; }}
    .insight-accent {{ position: absolute; left: 0; top: 10px; bottom: 10px; width: 3px; border-radius: 2px;
      background: linear-gradient(180deg, #bf360c, #4e342e); }}
    .kstack-v2 {{ display: flex; flex-direction: column; gap: 14px; }}
    .kpanel-v2 {{
      position: relative;
      padding: 14px 16px 16px 46px;
      border-radius: 16px;
      background: linear-gradient(168deg, #fffefb 0%, #faf5ee 100%);
      border: 1px solid rgba(190, 165, 135, 0.28);
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.75) inset,
        0 12px 28px rgba(45, 35, 28, 0.08),
        0 4px 10px rgba(45, 35, 28, 0.04);
    }}
    .kpanel-v2:nth-of-type(odd) {{ transform: rotate(-0.85deg); }}
    .kpanel-v2:nth-of-type(even) {{ transform: rotate(0.75deg); }}
    .kspeak-v2 {{
      background: linear-gradient(128deg, #5d4037 0%, #3e2723 100%);
      border: 1px solid rgba(0,0,0,0.08);
      transform: rotate(-0.4deg) !important;
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.08) inset,
        0 14px 32px rgba(20, 14, 10, 0.22),
        0 5px 12px rgba(20, 14, 10, 0.12);
    }}
    .knum-v2 {{
      position: absolute; left: 12px; top: 12px; width: 24px; height: 24px; border-radius: 50%;
      background: #efe6dc; color: #5d4037; font-size: 9px; font-weight: 700; display: flex;
      align-items: center; justify-content: center;
    }}
    .knum-v2.klight {{ background: rgba(255, 220, 180, 0.25); color: #ffe0b2; }}
    .klab-v2 {{
      margin: 0 0 6px; font-size: 10px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
      color: #6d4c41;
    }}
    .klab-v2.klight {{ color: #ffe0b2; }}
    .kbody-v2 {{ margin: 0; font-size: 11.5px; line-height: 1.45; overflow-wrap: anywhere; word-wrap: break-word; }}
    .kbody-v2.klight {{ color: #fff8f0; }}
    .muted {{ opacity: 0.5; font-style: italic; }}
    {HERO_LAYER_CSS}
  </style>
</head>
<body>
  <div class="page">
    <div class="level-badge">{level}</div>
    <header class="khdr-v2">
      <h1 class="ktop-v2">{topic}</h1>
      {subtitle_html}
    </header>
    {hero_block}
    {contrast_strip}
    {content_sections}
  </div>
</body>
</html>"""

    def _influencer_card_v2_template(self, card: dict[str, Any]) -> str:
        topic = card["title"]
        level = _badge_level(card["subtitle"])
        sub = (card.get("subtitle") or "").strip()
        subtitle_html = (
            f'<p class="isub-v2">{sub}</p>' if sub else '<p class="isub-v2 muted">Social learning</p>'
        )
        bullets = card["bullets"]
        cta = card["cta"]
        b = bullets + [""] * 5
        dq = b[0] if b[0] else "What is your take on this topic?"
        vocab = b[1] if b[1] else "Key term — short definition."
        tf = b[2] if b[2] else "True or false: [statement]."
        gram = b[3] if b[3] else "Pattern: subject + verb + …"
        write_pills = [p for p in bullets[:3] if p] or ["Brainstorm", "Draft", "Polish"]
        pills_html = "".join(f'<span class="pill-v2">{p}</span>' for p in write_pills[:4])

        has_contrast = bool(card.get("contrast_wrong") or card.get("contrast_better"))
        disc_cls = "imod-v2 imod-span imod-star" if not has_contrast else "imod-v2 imod-span"

        content_sections = f"""
    <div class="igrid-v2">
      <section class="{disc_cls}">
        <p class="ilab-v2">Discussion</p>
        <p class="itxt-v2">{dq}</p>
      </section>
      <section class="imod-v2 imod-span">
        <p class="ilab-v2">Vocabulary</p>
        <p class="itxt-v2 small">{vocab}</p>
      </section>
      <div class="irow-v2">
        <section class="imod-v2 imod-half">
          <p class="ilab-v2">True / false</p>
          <p class="itxt-v2">{tf}</p>
        </section>
        <section class="imod-v2 imod-bubble imod-half">
          <p class="ilab-v2">Grammar</p>
          <div class="bubble-v2">{gram}</div>
        </section>
      </div>
      <section class="imod-v2 imod-span imod-write">
        <p class="ilab-v2">Writing</p>
        <p class="itxt-v2 strong">{cta if cta else "Write 3–5 sentences."}</p>
        <div class="pills-v2">{pills_html}</div>
      </section>
    </div>
        """

        hero_block = _hero_media_block(card, "influencer_v2")
        contrast_strip = _contrast_strip_html(card, "inf2")
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=600, initial-scale=1.0" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Caveat:wght@600&family=DM+Serif+Display:ital@0;1&display=swap" rel="stylesheet" />
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; }}
    body {{
      background: linear-gradient(165deg, #3d3832 0%, #1e1e26 55%, #12121a 100%);
      font-family: "DM Serif Display", Georgia, serif; color: #2a2420; -webkit-font-smoothing: antialiased;
    }}
    .inf2-strip {{
      display: flex; margin: 0 auto 16px; width: 90%; max-width: 520px; border-radius: 18px; overflow: hidden;
      transform: rotate(0.4deg);
      box-shadow: 0 14px 36px rgba(20, 14, 10, 0.2), 0 4px 12px rgba(20, 14, 10, 0.1);
      border: 1px solid rgba(26, 26, 26, 0.35); z-index: 1; position: relative;
    }}
    .inf2-bad {{ flex: 1; background: linear-gradient(175deg, #fff5f5 0%, #ffcdd2 100%); padding: 14px 16px; }}
    .inf2-good {{ flex: 1; background: linear-gradient(175deg, #f1fff4 0%, #c8e6c9 100%); padding: 14px 16px; border-left: 2px dashed rgba(0,0,0,0.1); }}
    .inf2-tag {{ font-size: 9px; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase; color: #b71c1c; display: block; margin-bottom: 5px; }}
    .inf2-good .inf2-tag {{ color: #1b5e20; }}
    .inf2-txt {{ margin: 0; font-size: 12px; line-height: 1.38; font-weight: 700; color: #0d0d0d; overflow-wrap: anywhere; }}
    .imod-star {{
      border: 2px solid #c62828 !important;
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.45) inset,
        0 12px 30px rgba(198, 40, 40, 0.2) !important,
        0 4px 12px rgba(198, 40, 40, 0.12) !important;
    }}
    .page {{
      width: 600px; min-height: 920px; position: relative;
      background-color: #ebe3d9;
      background-image:
        linear-gradient(90deg, rgba(120, 100, 85, 0.035) 1px, transparent 1px),
        linear-gradient(rgba(120, 100, 85, 0.03) 1px, transparent 1px),
        linear-gradient(168deg, #fefbf6 0%, #f3ebe3 38%, #e8ddd4 100%);
      background-size: 24px 24px, 24px 24px, 100% 100%;
      padding: 28px 20px 30px;
      overflow: hidden;
      box-shadow:
        0 24px 60px rgba(0, 0, 0, 0.28),
        inset 0 0 90px rgba(255, 250, 245, 0.35),
        inset 0 1px 0 rgba(255, 255, 255, 0.55);
    }}
    .page::before {{
      content: ""; position: absolute; inset: 0; pointer-events: none;
      background:
        radial-gradient(ellipse 80% 50% at 100% 0%, rgba(255, 140, 100, 0.12) 0%, transparent 52%),
        radial-gradient(ellipse 70% 45% at 0% 100%, rgba(80, 120, 200, 0.09) 0%, transparent 48%);
    }}
    .level-badge {{
      position: absolute; top: 16px; left: 16px; width: 50px; height: 50px; border-radius: 50%;
      background: linear-gradient(145deg, #2a2a30 0%, #121218 100%);
      color: #faf6f0; font-size: 10px; display: flex; align-items: center;
      justify-content: center; text-align: center; padding: 4px;
      box-shadow: 0 6px 18px rgba(0,0,0,0.35), 0 2px 4px rgba(0,0,0,0.2);
      z-index: 5; overflow-wrap: anywhere; line-height: 1.05;
    }}
    .ihdr-v2 {{ padding: 8px 54px 18px 58px; text-align: center; position: relative; z-index: 1; }}
    .ititle-v2 {{
      font-family: "Caveat", cursive; font-size: 42px; line-height: 1.02; margin: 0; color: #1f1612;
      letter-spacing: -0.02em; word-wrap: break-word; overflow-wrap: anywhere;
    }}
    .isub-v2 {{ margin: 10px auto 0; max-width: 90%; font-size: 12.5px; line-height: 1.5; color: #4a3f38; }}
    .hero-media {{
      margin: 16px auto 16px;
      width: 90%;
      max-width: 520px;
      height: 156px;
      border-radius: 20px;
      overflow: hidden;
      border: 1px solid rgba(40, 30, 25, 0.12);
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.55) inset,
        0 18px 48px rgba(25, 18, 14, 0.14),
        0 6px 16px rgba(25, 18, 14, 0.08);
      position: relative; z-index: 1;
    }}
    .hero-img {{
      display: block;
      width: 100%;
      height: 100%;
      min-height: 156px;
      object-fit: cover;
      object-position: center;
    }}
    .insight-card {{
      margin: 16px auto 16px;
      width: 90%;
      max-width: 520px;
      min-height: 100px;
      padding: 18px 20px 20px 24px;
      border-radius: 20px;
      position: relative;
      z-index: 1;
      overflow: hidden;
      transform: rotate(-0.55deg);
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.7) inset,
        0 14px 40px rgba(40, 30, 24, 0.1),
        0 5px 14px rgba(40, 30, 24, 0.06);
    }}
    .insight-card.insight-influencer_v2 {{
      background: linear-gradient(148deg, rgba(255,255,255,0.98) 0%, #faf6f1 50%, #f2ebe4 100%);
      border: 1px solid rgba(180, 140, 110, 0.22);
    }}
    .insight-kicker {{ font-family: "Caveat", cursive; font-size: 22px; margin: 0 0 8px; color: #c62828; }}
    .insight-body {{ margin: 0; font-size: 14px; font-weight: 600; line-height: 1.35; overflow-wrap: anywhere; word-wrap: break-word; }}
    .insight-accent {{ position: absolute; left: 0; top: 12px; bottom: 12px; width: 4px; border-radius: 2px;
      background: linear-gradient(180deg, #0d47a1, #6a1b9a, #c62828); }}
    .igrid-v2 {{
      display: flex; flex-direction: column; gap: 14px; position: relative; z-index: 1;
    }}
    .irow-v2 {{ display: flex; gap: 14px; align-items: stretch; }}
    .imod-half {{ flex: 1; min-width: 0; }}
    .imod-v2 {{
      padding: 14px 16px 16px;
      border-radius: 16px;
      background: linear-gradient(168deg, #fffefb 0%, #faf6f0 100%);
      border: 1px solid rgba(200, 175, 150, 0.28);
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.75) inset,
        0 12px 28px rgba(35, 26, 20, 0.08),
        0 4px 10px rgba(35, 26, 20, 0.04);
    }}
    .imod-v2:nth-of-type(odd) {{ transform: rotate(-0.9deg); }}
    .imod-v2:nth-of-type(even) {{ transform: rotate(0.7deg); }}
    .imod-write {{
      border-left: 4px solid #4527a0;
      transform: rotate(-0.35deg) !important;
    }}
    .imod-bubble {{ border-left: 4px solid #1565c0; }}
    .ilab-v2 {{
      margin: 0 0 6px; font-size: 9.5px; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase;
      color: #6d4c41;
    }}
    .itxt-v2 {{ margin: 0; font-size: 11.5px; line-height: 1.45; overflow-wrap: anywhere; word-wrap: break-word; }}
    .itxt-v2.small {{ font-size: 11px; }}
    .itxt-v2.strong {{ font-weight: 600; color: #1f1612; }}
    .bubble-v2 {{
      font-size: 11px; line-height: 1.4; padding: 10px 12px; border-radius: 20px;
      background: linear-gradient(160deg, #e8eaf6 0%, #fff 100%);
      border: 1px solid rgba(21, 101, 192, 0.2);
      overflow-wrap: anywhere; word-wrap: break-word;
    }}
    .pills-v2 {{ margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px; }}
    .pill-v2 {{
      display: inline-block; padding: 5px 12px; border-radius: 999px; font-size: 10px;
      background: #fff; border: 1px solid rgba(180, 150, 120, 0.45); color: #4e342e;
      box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }}
    .muted {{ opacity: 0.55; font-style: italic; }}
    {HERO_LAYER_CSS}
  </style>
</head>
<body>
  <div class="page">
    <div class="level-badge">{level}</div>
    <header class="ihdr-v2">
      <h1 class="ititle-v2">{topic}</h1>
      {subtitle_html}
    </header>
    {hero_block}
    {contrast_strip}
    {content_sections}
  </div>
</body>
</html>"""
