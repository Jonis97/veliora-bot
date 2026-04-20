"""
Veliora Mini App API bridge.
Reuses existing bot services via init_api().
Grammar-only first live test.
"""
import io
import json
import logging
import os
import time
from string import Template
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from bot.utils.intent import OutputIntent

LOGGER = logging.getLogger(__name__)

# ── Shared instances ──────────────────────────────────────────────────────────
_ai_service         = None
_screenshot_service = None
_youtube_service    = None
_bot                = None

def init_api(pipeline, bot):
    """Call from main.py after build_application()."""
    global _ai_service, _screenshot_service, _youtube_service, _bot
    _ai_service         = pipeline._ai_service
    _screenshot_service = pipeline._screenshot_service
    _youtube_service    = pipeline._youtube_service
    _bot                = bot

# ── FastAPI app ───────────────────────────────────────────────────────────────
api = FastAPI(title="Veliora Mini App API")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# ── Preset loader ─────────────────────────────────────────────────────────────
# presets/ lives in project root (same level as main.py)
# __file__ is bot/api/routes.py → go up 2 levels
PRESET_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'presets')

def load_preset_html(base_preset: str) -> str:
    path = os.path.abspath(os.path.join(PRESET_DIR, f"{base_preset}.html"))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Preset HTML missing: {path}\n"
            f"Copy {base_preset}.html to the presets/ folder."
        )
    with open(path, encoding='utf-8') as f:
        return f.read()

# ── Scene overrides ───────────────────────────────────────────────────────────
BG_COLORS = {'warm': '#f0ebe3', 'light': '#f8f8f6', 'dark': '#0f1923'}

def apply_scene_overrides(html: str, overrides: dict) -> str:
    bg  = BG_COLORS.get(overrides.get('bg_tone', 'light'), '#f8f8f6')
    d   = overrides.get('density_padding_mult', 1.0)
    op  = overrides.get('decoration_opacity', 0.0)
    css = (
        f"<style>:root{{--scene-bg:{bg};--density-mult:{d};--deco-opacity:{op}}}"
        f"body,.page{{background:{bg}!important}}</style>"
    )
    return html.replace('</head>', css + '\n</head>', 1)

# ── Content → $placeholder bindings ──────────────────────────────────────────
def build_bindings(content: dict) -> dict:
    def li(items):
        if not isinstance(items, list):
            return str(items)
        return ''.join(f'<li>{i}</li>' for i in items if str(i).strip())

    b = {
        'title':            content.get('title', ''),
        'level':            content.get('level', ''),
        'lead_in_items':    li(content.get('lead_in_items', [])),
        'discussion_items': li(content.get('discussion_items', [])),
        'choice_items':     li(content.get('choice_items', [])),
        'vocab_items':      li(content.get('vocab_items', [])),
        'media_block':      content.get('media_block', ''),
    }
    for i in range(1, 13):
        b[f'vocab_item_{i}']      = content.get(f'vocab_item_{i}', '')
        b[f'discussion_item_{i}'] = content.get(f'discussion_item_{i}', '')
        b[f'lead_in_item_{i}']    = content.get(f'lead_in_item_{i}', '')
    return b

# ── Material persistence ──────────────────────────────────────────────────────
DATA_FILE = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'materials.json'
)

def save_material(record: dict) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(DATA_FILE)), exist_ok=True)
    items = []
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, encoding='utf-8') as f:
                items = json.load(f)
        except Exception:
            items = []
    mid = f"mat_{int(time.time() * 1000)}"
    record['material_id'] = mid
    record['created_at']  = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    items.append(record)
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return mid

# ── In-memory render store: material_id → bytes + metadata ───────────────────
_pending_renders: dict = {}

# ── Pydantic models ───────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    teacher_id:   str
    mode:         str           # "grammar" | "lesson" | "speaking" | "vocabulary"
    level:        str           # "A1" | "A2" | "B1" | "B2"
    structure:    list[str]     # section labels (for context only, not used by AI yet)
    source_type:  str           # "custom" | "youtube" | "text"
    source_value: str           # text or YouTube URL

class SceneOverrides(BaseModel):
    bg_tone:              str   = 'light'
    density_padding_mult: float = 1.0
    decoration_opacity:   float = 0.0

class RenderRequest(BaseModel):
    teacher_id:      str
    mode:            Optional[str]  = None
    level:           Optional[str]  = None
    variant_id:      str            = 'v1_grammar_balanced'
    base_preset:     str            = 'grammar_clean_v1'
    block_order:     list[dict]     = []
    scene_overrides: SceneOverrides = SceneOverrides()
    content:         dict[str, Any] = {}

class SendRequest(BaseModel):
    teacher_id:  str
    png_url:     Optional[str] = None
    material_id: Optional[str] = None

class RenderSendRequest(BaseModel):
    teacher_id:      str
    base_preset:     str                    = 'grammar_clean_v1'
    scene_overrides: SceneOverrides         = SceneOverrides()
    content:         dict[str, Any]
    mode:            Optional[str]          = None
    level:           Optional[str]          = None
    variant_id:      str                    = 'v1_grammar_balanced'
    block_order:     list[dict]             = []

# ── POST /api/generate ────────────────────────────────────────────────────────
@api.post('/api/generate')
async def api_generate(req: GenerateRequest):
    try:
        # Bug fix 5: speaking must use lesson_card_v1 (produces choices + lead_in_questions).
        # speaking_card_v2 falls through to warm_paper_v2 schema which has no choices field.
        mode_to_template = {
            'lesson':     'lesson_card_v1',
            'speaking':   'lesson_card_v1',
            'vocabulary': 'vocab_card',
            'grammar':    'lesson_card_v1',
        }
        template = mode_to_template.get(req.mode, 'lesson_card_v1')

        # Bug fix 1: fetch YouTube transcript before passing to Claude.
        # Claude cannot browse URLs — it needs the plain-text transcript.
        source_text = req.source_value
        if req.source_type == 'youtube' and _youtube_service is not None:
            from bot.services.youtube_service import extract_video_id
            video_id = extract_video_id(source_text)
            if video_id:
                try:
                    source_text = await _youtube_service.fetch_transcript(video_id)
                except Exception as yt_err:
                    LOGGER.warning(f'YouTube transcript fetch failed ({video_id}): {yt_err}')
                    raise HTTPException(
                        status_code=422,
                        detail={'status': 'error', 'error': 'source_unavailable',
                                'message': 'Could not fetch YouTube transcript. Try a different video or use Text source.'}
                    )

        source_text = f"[FORMAT={req.mode}]\n[LEVEL={req.level}]\n\nUSER CONTENT:\n{source_text}"

        card_json = await _ai_service.generate_card_content(
            source_text,
            template,
            output_intent=OutputIntent.CARD,
            is_followup=False,
            intent=req.mode,
        )

        # Bug fix 2+3: AI always returns 'vocabulary', never 'vocab'.
        # vocab_card returns objects {term, translation, example}; flatten to strings.
        raw_vocab = card_json.get('vocabulary', [])
        if raw_vocab and isinstance(raw_vocab[0], dict):
            vocab_strings = [
                f"{v.get('term', '')} — {v.get('translation', '')}"
                for v in raw_vocab if isinstance(v, dict)
            ]
        else:
            vocab_strings = [str(v) for v in raw_vocab if v]

        content = {
            'title':            str(card_json.get('topic') or card_json.get('title', '')).strip(),
            'lead_in_items':    card_json.get('lead_in_questions', []),
            'discussion_items': card_json.get('discussion_questions', card_json.get('bullets', [])),
            'choice_items':     card_json.get('choices', []),
            'vocab_items':      vocab_strings,
        }

        # Bug fix 4: vocabulary mode — Mini App getSections() reads vocab_item_1..12,
        # not vocab_items. Populate both for forward compatibility.
        if req.mode == 'vocabulary':
            for i, v in enumerate(vocab_strings[:12], start=1):
                content[f'vocab_item_{i}'] = v

        preset_map = {
            'lesson':     'lesson_warm_v1',
            'speaking':   'speaking_collage_v1',
            'vocabulary': 'vocab_dark_v1',
            'grammar':    'grammar_clean_v1',
        }

        return {
            'status':           'ok',
            'preset_id':        preset_map.get(req.mode, 'grammar_clean_v1'),
            'edit_rounds_used': 0,
            'content':          content,
        }

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception('/api/generate failed')
        raise HTTPException(
            status_code=500,
            detail={'status': 'error', 'error': 'generation_failed', 'message': str(e)}
        )

# ── POST /api/render_and_send ─────────────────────────────────────────────────
@api.post('/api/render_and_send')
async def api_render_and_send(req: RenderSendRequest):
    try:
        # 1. Load preset HTML from presets/ folder
        prod_html = load_preset_html(req.base_preset)

        # 2. Substitute content bindings
        bindings = build_bindings(req.content)
        rendered = Template(prod_html).safe_substitute(bindings)

        # 3. Apply scene overrides (CSS vars)
        rendered = apply_scene_overrides(rendered, req.scene_overrides.dict())

        # 4. Render PNG — exact confirmed signature: html_to_image(html: str) → bytes
        image_bytes = await _screenshot_service.html_to_image(rendered)

        # 5. Send directly to Telegram — no CDN needed
        tg_user_id = int(req.teacher_id.replace('tg_', ''))
        msg = await _bot.send_photo(
            chat_id=tg_user_id,
            photo=io.BytesIO(image_bytes),
            caption="Ваша картка готова 🎓",
        )
        file_id = msg.photo[-1].file_id

        # 6. Persist material record
        material_id = save_material({
            'teacher_id':       req.teacher_id,
            'mode':             req.mode,
            'level':            req.level,
            'variant_id':       req.variant_id,
            'base_preset':      req.base_preset,
            'png_url':          f'tg://{file_id}',
            'content_snapshot': req.content,
        })

        return {'status': 'ok', 'file_id': file_id, 'material_id': material_id}

    except FileNotFoundError as e:
        LOGGER.error(f'/api/render_and_send preset missing: {e}')
        raise HTTPException(
            status_code=500,
            detail={'status': 'error', 'error': 'preset_missing', 'message': str(e)}
        )
    except Exception as e:
        LOGGER.exception('/api/render_and_send failed')
        raise HTTPException(
            status_code=500,
            detail={'status': 'error', 'error': 'failed', 'message': str(e)}
        )

# ── POST /api/render ──────────────────────────────────────────────────────────
@api.post('/api/render')
async def api_render(req: RenderRequest):
    try:
        prod_html   = load_preset_html(req.base_preset)
        bindings    = build_bindings(req.content)
        rendered    = Template(prod_html).safe_substitute(bindings)
        rendered    = apply_scene_overrides(rendered, req.scene_overrides.dict())
        image_bytes = await _screenshot_service.html_to_image(rendered)

        material_id = save_material({
            'teacher_id':       req.teacher_id,
            'mode':             req.mode,
            'level':            req.level,
            'variant_id':       req.variant_id,
            'base_preset':      req.base_preset,
            'content_snapshot': req.content,
        })
        _pending_renders[material_id] = {
            'bytes':      image_bytes,
            'teacher_id': req.teacher_id,
        }
        return {'status': 'ok', 'png_url': None, 'material_id': material_id}

    except FileNotFoundError as e:
        LOGGER.error(f'/api/render preset missing: {e}')
        raise HTTPException(
            status_code=500,
            detail={'status': 'error', 'error': 'preset_missing', 'message': str(e)}
        )
    except Exception as e:
        LOGGER.exception('/api/render failed')
        raise HTTPException(
            status_code=500,
            detail={'status': 'error', 'error': 'failed', 'message': str(e)}
        )

# ── POST /api/send ────────────────────────────────────────────────────────────
@api.post('/api/send')
async def api_send(req: SendRequest):
    try:
        pending = _pending_renders.pop(req.material_id, None)
        if not pending:
            raise HTTPException(
                status_code=404,
                detail={'status': 'error', 'message': 'Material not found or already sent'}
            )
        tg_user_id = int(pending['teacher_id'].replace('tg_', ''))
        await _bot.send_photo(
            chat_id=tg_user_id,
            photo=io.BytesIO(pending['bytes']),
            caption="Ваша картка готова 🎓",
        )
        return {'status': 'ok'}

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception('/api/send failed')
        raise HTTPException(
            status_code=500,
            detail={'status': 'error', 'error': 'failed', 'message': str(e)}
        )