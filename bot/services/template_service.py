from html import escape
from typing import Any, Optional


ALLOWED_TEMPLATES = {"warm_paper", "kitchen_collage", "influencer_card"}


def _normalize_card(raw: dict[str, Any]) -> dict[str, Any]:
    bullets = raw.get("bullets", [])
    if not isinstance(bullets, list):
        bullets = []
    cleaned_bullets = [escape(str(item)) for item in bullets[:5] if str(item).strip()]
    return {
        "template": raw.get("template", "warm_paper"),
        "title": escape(str(raw.get("title", "Learning Card"))),
        "subtitle": escape(str(raw.get("subtitle", ""))),
        "bullets": cleaned_bullets,
        "cta": escape(str(raw.get("cta", "Try this today."))),
    }


class TemplateService:
    def render_html(self, card: dict[str, Any], forced_template: Optional[str] = None) -> str:
        normalized = _normalize_card(card)
        template_name = forced_template or normalized["template"]
        if template_name not in ALLOWED_TEMPLATES:
            template_name = "warm_paper"
        normalized["template"] = template_name
        if template_name == "kitchen_collage":
            return self._kitchen_collage_template(normalized)
        if template_name == "influencer_card":
            return self._influencer_card_template(normalized)
        return self._warm_paper_template(normalized)

    def _base_html(self, card: dict[str, Any], *, background: str, card_bg: str, text_color: str, cta_bg: str, cta_color: str) -> str:
        bullets_html = "".join(f"<li>{item}</li>" for item in card["bullets"])
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      width: 1080px;
      height: 1350px;
      font-family: Inter, Arial, sans-serif;
      background: {background};
      color: {text_color};
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .card {{
      width: 90%;
      min-height: 80%;
      border-radius: 36px;
      padding: 56px;
      background: {card_bg};
      box-shadow: 0 24px 60px rgba(0, 0, 0, 0.18);
      border: 1px solid rgba(0, 0, 0, 0.08);
    }}
    h1 {{ margin: 0 0 12px; font-size: 60px; line-height: 1.05; }}
    h2 {{ margin: 0 0 34px; font-size: 30px; font-weight: 500; opacity: 0.92; }}
    ul {{ margin: 0; padding-left: 32px; font-size: 32px; line-height: 1.4; }}
    li {{ margin-bottom: 18px; }}
    .cta {{
      margin-top: 42px;
      font-size: 30px;
      font-weight: 700;
      padding: 18px 24px;
      border-radius: 16px;
      display: inline-block;
      background: {cta_bg};
      color: {cta_color};
    }}
  </style>
</head>
<body>
  <main class="card">
    <h1>{card["title"]}</h1>
    <h2>{card["subtitle"]}</h2>
    <ul>{bullets_html}</ul>
    <div class="cta">{card["cta"]}</div>
  </main>
</body>
</html>
""".strip()

    def _warm_paper_template(self, card: dict[str, Any]) -> str:
        return self._base_html(
            card,
            background="linear-gradient(150deg, #f5ecd7 0%, #efe2c1 100%)",
            card_bg="rgba(255, 255, 255, 0.82)",
            text_color="#3b2f2f",
            cta_bg="#8d6e63",
            cta_color="#ffffff",
        )

    def _kitchen_collage_template(self, card: dict[str, Any]) -> str:
        return self._base_html(
            card,
            background="linear-gradient(125deg, #fff3e0 0%, #ffe0b2 45%, #ffcc80 100%)",
            card_bg="rgba(255, 255, 255, 0.86)",
            text_color="#4e342e",
            cta_bg="#fb8c00",
            cta_color="#ffffff",
        )

    def _influencer_card_template(self, card: dict[str, Any]) -> str:
        return self._base_html(
            card,
            background="linear-gradient(145deg, #8e24aa 0%, #5e35b1 45%, #1e88e5 100%)",
            card_bg="rgba(9, 14, 38, 0.78)",
            text_color="#f8f9ff",
            cta_bg="#00e5ff",
            cta_color="#102027",
        )
