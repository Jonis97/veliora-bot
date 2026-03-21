from html import escape
from typing import Any, List, Optional, Tuple


ALLOWED_TEMPLATES = {
    "warm_paper",
    "kitchen_collage",
    "influencer_card",
    "warm_paper_v2",
    "kitchen_collage_v2",
    "influencer_card_v2",
}


def _normalize_card(raw: dict[str, Any]) -> dict[str, Any]:
    bullets = raw.get("bullets", [])
    if not isinstance(bullets, list):
        bullets = []
    cleaned_bullets = [escape(str(item)) for item in bullets[:5] if str(item).strip()]
    raw_url = str(raw.get("image_url", "") or raw.get("photo", "") or "").strip()
    image_url = raw_url if raw_url.startswith(("http://", "https://")) else ""
    return {
        "template": raw.get("template", "warm_paper"),
        "title": escape(str(raw.get("title", "Learning Card"))),
        "subtitle": escape(str(raw.get("subtitle", ""))),
        "bullets": cleaned_bullets,
        "cta": escape(str(raw.get("cta", "Try this today."))),
        "image_url": image_url,
    }


def _build_takeaway_text(card: dict[str, Any]) -> str:
    """Primary line for insight card when no photo is available."""
    bullets = card.get("bullets") or []
    if isinstance(bullets, list) and bullets:
        return str(bullets[0])
    sub = str(card.get("subtitle", "")).strip()
    if sub:
        return sub
    cta = str(card.get("cta", "")).strip()
    if cta:
        return cta
    return (
        "Pick one term from the card and use it in a new sentence within the next hour—"
        "same context you’ll face in real life (message, email, or aloud)."
    )


def _hero_media_block(card: dict[str, Any], variant: str) -> str:
    """
    Photo area: real image if `image_url` is set on the raw card; otherwise a styled insight card.
    variant: warm | kitchen | influencer
    """
    url = (card.get("image_url") or "").strip()
    if url:
        safe = escape(url, quote=True)
        return (
            f'<div class="hero-media hero-{variant}">'
            f'<img class="hero-img" src="{safe}" alt="" />'
            f"</div>"
        )
    takeaway = _build_takeaway_text(card)
    return (
        f'<aside class="insight-card insight-{variant}" aria-label="Key takeaway">'
        f'<div class="insight-kicker">Key takeaway</div>'
        f'<p class="insight-body">{takeaway}</p>'
        f'<div class="insight-accent" aria-hidden="true"></div>'
        f"</aside>"
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


class TemplateService:
    def render_html(self, card: dict[str, Any], forced_template: Optional[str] = None) -> str:
        normalized = _normalize_card(card)
        template_name = forced_template or normalized["template"]
        if template_name not in ALLOWED_TEMPLATES:
            template_name = "warm_paper"
        normalized["template"] = template_name
        if template_name == "kitchen_collage_v2":
            return self._kitchen_collage_v2_template(normalized)
        if template_name == "kitchen_collage":
            return self._kitchen_collage_template(normalized)
        if template_name == "influencer_card_v2":
            return self._influencer_card_v2_template(normalized)
        if template_name == "influencer_card":
            return self._influencer_card_template(normalized)
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
        pairs = _split_term_translation(bullets)
        vocab_rows = []
        for term, trans in pairs[:6]:
            if trans:
                vocab_rows.append(
                    f'<div class="vr-v2"><span class="tv2">{term}</span>'
                    f'<span class="sv2">—</span><span class="gv2">{trans}</span></div>'
                )
            else:
                vocab_rows.append(f'<div class="vr-v2"><span class="tv2 full">{term}</span></div>')
        if not vocab_rows:
            vocab_rows.append('<div class="vr-v2 muted">Add pairs (term — gloss).</div>')

        ex_lines = bullets[:4] if bullets else []
        mcq_items = []
        for i, line in enumerate(ex_lines, 1):
            mcq_items.append(f'<div class="mcq-v2"><span class="nv2">{i}.</span> {line}</div>')
        while len(mcq_items) < 3:
            mcq_items.append(
                f'<div class="mcq-v2 muted"><span class="nv2">{len(mcq_items) + 1}.</span> '
                f"Choose the best option.</div>"
            )

        content_sections = f"""
    <div class="cols-v2">
      <section class="panel-v2 pv2-a" aria-label="Vocabulary">
        <div class="pin-v2"></div>
        <p class="label-v2">Vocabulary</p>
        <div class="body-v2">{"".join(vocab_rows)}</div>
      </section>
      <section class="panel-v2 pv2-b" aria-label="Choose the correct option">
        <div class="pin-v2"></div>
        <p class="label-v2">Choose the correct option</p>
        <div class="body-v2">{"".join(mcq_items[:4])}</div>
      </section>
    </div>
    <section class="speak-v2" aria-label="Let's speak">
      <p class="label-v2 light">Let’s speak</p>
      <p class="speak-body-v2">{cta if cta else "Practice aloud in a short sentence."}</p>
    </section>
        """

        hero_block = _hero_media_block(card, "warm_v2")
        return self._wrap_warm_paper_v2_html(topic, level, subtitle_html, hero_block, content_sections)

    def _wrap_warm_paper_v2_html(
        self, topic: str, level: str, subtitle_html: str, hero_block: str, content_sections: str
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
    body {{ background: #e0d4c4; font-family: "DM Serif Display", Georgia, serif; color: #2f2420; -webkit-font-smoothing: antialiased; }}
    .page {{
      width: 600px; min-height: 920px; position: relative; overflow: hidden;
      background: #f7f2ea;
      background-image:
        linear-gradient(90deg, rgba(160, 140, 120, 0.08) 1px, transparent 1px),
        linear-gradient(rgba(160, 140, 120, 0.06) 1px, transparent 1px);
      background-size: 20px 20px, 20px 20px;
      padding: 22px 20px 26px;
    }}
    .level-badge {{
      position: absolute; top: 16px; left: 16px; width: 48px; height: 48px; border-radius: 50%;
      background: #7a1f1f; color: #fffaf3; font-size: 10px; line-height: 1.05; display: flex;
      align-items: center; justify-content: center; text-align: center; padding: 4px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.14); z-index: 5; overflow-wrap: anywhere;
    }}
    .hdr-v2 {{ padding: 6px 56px 14px 60px; text-align: center; }}
    .topic-title-v2 {{
      font-family: "Caveat", cursive; font-size: 40px; line-height: 1.02; margin: 0; color: #3d2a22;
      letter-spacing: -0.02em; word-wrap: break-word; overflow-wrap: anywhere;
    }}
    .topic-sub-v2 {{ margin: 8px auto 0; max-width: 92%; font-size: 12.5px; line-height: 1.45; color: #5c4a42; }}
    .hero-media {{ margin: 14px auto 14px; width: 92%; max-width: 528px; border-radius: 16px; overflow: hidden;
      border: 1px solid rgba(100, 70, 50, 0.22); box-shadow: 0 8px 24px rgba(40, 28, 20, 0.1); }}
    .hero-img {{ display: block; width: 100%; height: auto; max-height: 152px; object-fit: cover; }}
    .insight-card {{ margin: 14px auto 14px; width: 92%; max-width: 528px; min-height: 100px; padding: 16px 18px 18px 20px;
      border-radius: 16px; position: relative; overflow: hidden; }}
    .insight-card.insight-warm_v2 {{
      background: linear-gradient(160deg, #fff 0%, #faf5ee 100%);
      border: 1px solid rgba(120, 90, 70, 0.2);
      box-shadow: 0 10px 28px rgba(45, 32, 24, 0.08);
    }}
    .insight-kicker {{ font-family: "Caveat", cursive; font-size: 21px; margin: 0 0 8px; color: #8b4513; }}
    .insight-body {{ margin: 0; font-size: 13px; line-height: 1.5; overflow-wrap: anywhere; word-wrap: break-word; }}
    .insight-accent {{ position: absolute; left: 0; top: 12px; bottom: 12px; width: 4px; border-radius: 2px;
      background: linear-gradient(180deg, #b71c1c, #5d4037); }}
    .cols-v2 {{ display: flex; gap: 12px; align-items: stretch; margin-bottom: 12px; }}
    .panel-v2 {{
      flex: 1; min-width: 0; padding: 14px 12px 16px; border-radius: 14px; position: relative;
      box-shadow: 0 6px 18px rgba(45, 35, 28, 0.09);
    }}
    .pv2-a {{ background: #fffefb; border: 1px solid rgba(190, 165, 135, 0.35); transform: rotate(-0.4deg); }}
    .pv2-b {{ background: #fffdf8; border: 1px solid rgba(175, 155, 130, 0.32); transform: rotate(0.35deg); }}
    .pin-v2 {{ position: absolute; top: -5px; right: 16px; width: 13px; height: 13px; border-radius: 50%;
      background: radial-gradient(circle at 30% 30%, #ff7961, #b71c1c); box-shadow: 0 2px 4px rgba(0,0,0,0.2); }}
    .label-v2 {{
      font-family: "DM Serif Display", Georgia, serif; font-size: 10px; font-weight: 700; letter-spacing: 0.14em;
      text-transform: uppercase; color: #6d4c41; margin: 0 0 10px;
    }}
    .label-v2.light {{ color: #e8f5e9; }}
    .body-v2 {{ font-size: 12px; line-height: 1.42; }}
    .vr-v2 {{ margin-bottom: 8px; overflow-wrap: anywhere; word-wrap: break-word; }}
    .tv2 {{ font-weight: 700; color: #2f2420; }}
    .tv2.full {{ display: block; }}
    .sv2 {{ margin: 0 5px; opacity: 0.45; }}
    .gv2 {{ color: #4e3d36; }}
    .mcq-v2 {{ margin-bottom: 8px; overflow-wrap: anywhere; word-wrap: break-word; }}
    .nv2 {{ font-weight: 700; color: #8b4513; margin-right: 5px; }}
    .muted {{ color: rgba(47, 36, 32, 0.42); font-style: italic; }}
    .speak-v2 {{
      width: 100%; padding: 14px 16px 16px; border-radius: 14px;
      background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 55%, #1b5e20 100%);
      color: #f1f8f4; box-shadow: 0 6px 16px rgba(20, 60, 30, 0.22);
    }}
    .speak-body-v2 {{ margin: 6px 0 0; font-size: 12.5px; line-height: 1.45; overflow-wrap: anywhere; word-wrap: break-word; }}
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

        sections = (
            ("01", "Vocabulary list", sec_vocab),
            ("02", "Correct the mistake", sec_fix),
            ("03", "Choose the option", sec_choose),
            ("04", "Fill in the gaps", sec_gap),
        )
        blocks = []
        for num, label, body in sections:
            blocks.append(
                f'<section class="kpanel-v2"><span class="knum-v2">{num}</span>'
                f'<p class="klab-v2">{label}</p><p class="kbody-v2">{body}</p></section>'
            )
        blocks.append(
            f'<section class="kpanel-v2 kspeak-v2"><span class="knum-v2 klight">05</span>'
            f'<p class="klab-v2 klight">Speaking</p><p class="kbody-v2 klight">{sec_speak}</p></section>'
        )
        content_sections = '<div class="kstack-v2">' + "".join(blocks) + "</div>"

        hero_block = _hero_media_block(card, "kitchen_v2")
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
    body {{ background: #c9b99a; font-family: "DM Serif Display", Georgia, serif; color: #2c221e; -webkit-font-smoothing: antialiased; }}
    .page {{
      width: 600px; min-height: 920px; position: relative;
      background: #fdfaf5;
      background-image: linear-gradient(0deg, rgba(235, 225, 210, 0.5) 1px, transparent 1px);
      background-size: 100% 26px;
      padding: 20px 18px 24px;
      overflow: hidden;
    }}
    .level-badge {{
      position: absolute; top: 14px; left: 14px; width: 48px; height: 48px; border-radius: 50%;
      background: #7a1f1f; color: #fffaf3; font-size: 10px; display: flex; align-items: center;
      justify-content: center; text-align: center; padding: 4px; box-shadow: 0 3px 10px rgba(0,0,0,0.12);
      z-index: 5; overflow-wrap: anywhere; line-height: 1.05;
    }}
    .khdr-v2 {{ padding: 4px 54px 12px 58px; text-align: center; }}
    .ktop-v2 {{ font-family: "Caveat", cursive; font-size: 38px; line-height: 1.05; margin: 0; color: #3d2a22;
      word-wrap: break-word; overflow-wrap: anywhere; }}
    .ksub-v2 {{ margin: 8px auto 0; max-width: 94%; font-size: 12.5px; line-height: 1.45; color: #5a4a42; }}
    .hero-media {{ margin: 12px auto 14px; width: 94%; border-radius: 14px; overflow: hidden;
      border: 1px solid rgba(130, 100, 75, 0.25); box-shadow: 0 6px 20px rgba(40, 30, 22, 0.1); }}
    .hero-img {{ display: block; width: 100%; height: auto; max-height: 136px; object-fit: cover; }}
    .insight-card {{ margin: 12px auto 14px; width: 94%; min-height: 90px; padding: 14px 16px 16px 18px;
      border-radius: 14px; position: relative; overflow: hidden; }}
    .insight-card.insight-kitchen_v2 {{
      background: linear-gradient(165deg, #fff 0%, #f5efe6 100%);
      border: 1px solid rgba(150, 120, 90, 0.28);
      box-shadow: 0 8px 22px rgba(45, 32, 24, 0.08);
    }}
    .insight-kicker {{ font-family: "Caveat", cursive; font-size: 20px; margin: 0 0 6px; color: #8b4513; }}
    .insight-body {{ margin: 0; font-size: 12.5px; line-height: 1.48; overflow-wrap: anywhere; word-wrap: break-word; }}
    .insight-accent {{ position: absolute; left: 0; top: 10px; bottom: 10px; width: 3px; border-radius: 2px;
      background: linear-gradient(180deg, #bf360c, #4e342e); }}
    .kstack-v2 {{ display: flex; flex-direction: column; gap: 10px; }}
    .kpanel-v2 {{
      position: relative; padding: 12px 14px 14px 44px; border-radius: 12px;
      background: #fffdf9; border: 1px solid rgba(190, 165, 135, 0.35);
      box-shadow: 0 4px 14px rgba(45, 35, 28, 0.06);
    }}
    .kspeak-v2 {{
      background: linear-gradient(125deg, #4e342e 0%, #3e2723 100%);
      border: 1px solid rgba(0,0,0,0.06);
      box-shadow: 0 6px 18px rgba(30, 20, 15, 0.2);
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

        content_sections = f"""
    <div class="igrid-v2">
      <section class="imod-v2 imod-span">
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
    body {{ background: #121218; font-family: "DM Serif Display", Georgia, serif; color: #2a2420; -webkit-font-smoothing: antialiased; }}
    .page {{
      width: 600px; min-height: 920px; position: relative;
      background: linear-gradient(168deg, #fefbf6 0%, #f3ebe3 38%, #e8ddd4 100%);
      padding: 20px 16px 24px;
      overflow: hidden;
    }}
    .page::before {{
      content: ""; position: absolute; inset: 0; pointer-events: none;
      background: radial-gradient(ellipse 80% 50% at 100% 0%, rgba(255, 140, 100, 0.14) 0%, transparent 50%),
                  radial-gradient(ellipse 70% 45% at 0% 100%, rgba(80, 120, 200, 0.1) 0%, transparent 45%);
    }}
    .level-badge {{
      position: absolute; top: 14px; left: 14px; width: 48px; height: 48px; border-radius: 50%;
      background: #1a1a1a; color: #faf6f0; font-size: 10px; display: flex; align-items: center;
      justify-content: center; text-align: center; padding: 4px; box-shadow: 0 4px 14px rgba(0,0,0,0.2);
      z-index: 5; overflow-wrap: anywhere; line-height: 1.05;
    }}
    .ihdr-v2 {{ padding: 6px 52px 14px 56px; text-align: center; position: relative; z-index: 1; }}
    .ititle-v2 {{
      font-family: "Caveat", cursive; font-size: 42px; line-height: 1.02; margin: 0; color: #1f1612;
      letter-spacing: -0.02em; word-wrap: break-word; overflow-wrap: anywhere;
    }}
    .isub-v2 {{ margin: 8px auto 0; max-width: 94%; font-size: 12.5px; line-height: 1.45; color: #4a3f38; }}
    .hero-media {{
      margin: 12px auto 14px; width: 94%; border-radius: 18px; overflow: hidden;
      border: 1px solid rgba(40, 30, 25, 0.15); box-shadow: 0 10px 32px rgba(25, 18, 14, 0.15);
      position: relative; z-index: 1;
    }}
    .hero-img {{ display: block; width: 100%; height: auto; max-height: 148px; object-fit: cover; }}
    .insight-card {{
      margin: 12px auto 14px; width: 94%; min-height: 96px; padding: 16px 18px 18px 22px;
      border-radius: 18px; position: relative; z-index: 1; overflow: hidden;
    }}
    .insight-card.insight-influencer_v2 {{
      background: linear-gradient(145deg, rgba(255,255,255,0.97) 0%, #faf6f1 100%);
      border: 1px solid rgba(180, 140, 110, 0.28);
      box-shadow: 0 12px 32px rgba(40, 30, 24, 0.1);
    }}
    .insight-kicker {{ font-family: "Caveat", cursive; font-size: 22px; margin: 0 0 8px; color: #c62828; }}
    .insight-body {{ margin: 0; font-size: 13px; line-height: 1.5; overflow-wrap: anywhere; word-wrap: break-word; }}
    .insight-accent {{ position: absolute; left: 0; top: 12px; bottom: 12px; width: 4px; border-radius: 2px;
      background: linear-gradient(180deg, #0d47a1, #6a1b9a, #c62828); }}
    .igrid-v2 {{
      display: flex; flex-direction: column; gap: 10px; position: relative; z-index: 1;
    }}
    .irow-v2 {{ display: flex; gap: 10px; align-items: stretch; }}
    .imod-half {{ flex: 1; min-width: 0; }}
    .imod-v2 {{
      padding: 12px 14px 14px; border-radius: 14px;
      background: rgba(255, 253, 250, 0.95);
      border: 1px solid rgba(200, 175, 150, 0.35);
      box-shadow: 0 6px 18px rgba(35, 26, 20, 0.07);
    }}
    .imod-write {{ border-left: 4px solid #4527a0; }}
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
    {content_sections}
  </div>
</body>
</html>"""
