# Veliora Telegram Bot

Production-ready async Telegram bot that converts user text, voice notes, or YouTube links into educational visual cards.

## Project Structure

```text
veliora-bot/
├── bot/
│   ├── handlers/
│   │   └── message_handler.py
│   ├── services/
│   │   ├── ai_service.py
│   │   ├── pipeline_service.py
│   │   ├── screenshot_service.py
│   │   ├── template_service.py
│   │   ├── topic_image_service.py
│   │   ├── transcription_service.py
│   │   └── youtube_service.py
│   └── utils/
│       ├── config.py
│       ├── dedup.py
│       ├── input_parser.py
│       └── retry.py
├── .env.example
├── Procfile
├── main.py
└── requirements.txt
```

## Features

- Async handlers and services end-to-end.
- Input auto-detection:
  - Plain text
  - Voice note (Whisper transcription)
  - YouTube URL (Supadata transcript API)
- AI JSON card generation via GPT-4o-mini.
- HTML card templates (v1 kept as backup; v2 = refined layout):
  - `warm_paper` / `warm_paper_v2`
  - `kitchen_collage` / `kitchen_collage_v2`
  - `influencer_card` / `influencer_card_v2`
- HTML screenshot rendering via ScreenshotOne.
- Retry logic (3 attempts) and structured logging.
- Duplicate update protection via `chat_id:message_id`.
- Webhook deployment ready for Railway.

## Environment Variables

Required:

- `TELEGRAM_TOKEN`
- `OPENAI_API_KEY`
- `SUPADATA_API_KEY`
- `SCREENSHOTONE_API_KEY`
- `WEBHOOK_URL` (public app URL, e.g. Railway domain)

Optional:

- `WEBHOOK_PATH` (default: `/telegram/webhook`)
- `WEBHOOK_SECRET_TOKEN` (default: `veliora-secret-token`)
- `OPENAI_MODEL` (default: `gpt-4o-mini`)
- `PORT` (default: `8080`)

## Run locally

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and set values.
3. Export env vars (or use your process manager).
4. Run:
   - `python main.py`

## Template Selection

In text messages, include an optional template tag:

- `[template:warm_paper]` · `[template:warm_paper_v2]`
- `[template:kitchen_collage]` · `[template:kitchen_collage_v2]`
- `[template:influencer_card]` · `[template:influencer_card_v2]`

If no tag is provided, default template is `warm_paper`.
