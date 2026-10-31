import os
import io
import time
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import re
import json
from collections import defaultdict
import aiohttp
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import numpy as np
import discord
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =========================================================
# CONFIGURACIÓN DE MODO VISUAL Y FONDO
# =========================================================
# Elige fácilmente cómo quieres presentar los God Packs:
# "TROPHY_GRID"    -> Modo Imagen 1: 3 cartas arriba y 2 abajo con inclinación sutil y luz neón adaptativa
# "STAGGERED_WAVE" -> Modo Imagen 2: 5 cartas en fila escalonada (ola/podio) 100% rectas con luz adaptativa
PRESENTATION_MODE = "TROPHY_GRID"

# False -> Fondo oscuro elegante estilo vitrina donde resalta al máximo la luz de color de cada carta (Recomendado)
# True  -> Fondo 100% transparente (PNG RGBA) manteniendo el resplandor de color alrededor de cada carta
TRANSPARENT_BACKGROUND = False

# =========================================================
# VARIABLES DE ENTORNO DESDE RAILWAY / .ENV
# =========================================================

DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN", "")
CLIENT_ID = int(os.environ.get("CLIENT_ID", "0"))
FORUM_CHANNEL_ID = int(os.environ.get("FORUM_CHANNEL_ID", "0"))
LOG_CHANNEL_ID = int(os.environ.get("LOG_CHANNEL_ID", "0"))
INVALID_LOG_CHANNEL_ID = int(os.environ.get("INVALID_LOG_CHANNEL_ID", "0"))
UPSTASH_REDIS_REST_URL = os.environ.get("UPSTASH_REDIS_REST_URL", "").rstrip("/")
UPSTASH_REDIS_REST_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("gp_showcase_bot")

BG_DARK_COLOR = (14, 16, 24)
BG_CENTER_GLOW = (28, 33, 52)

TRIGGER_PATTERNS = [
    re.compile(r"god\s*pack", re.IGNORECASE),
    re.compile(r"\[\d/[45]\]\[\d+P\]\[[^\]]+\]", re.IGNORECASE),
]

VALID_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
PROCESSED_MESSAGES = set()
INVALID_LOG_COOLDOWN = 10
INVALID_LOG_LAST_SENT = {}

CHANNEL_GROUP_MAP = {
    1497179630027800728: "Elite_Four",
    1497179450960379915: "Trainer",
    1534456693126332568: "Gym_Leader",
}

GROUP_CONFIG = {
    "Trainer": {"FORUM_CHANNEL_ID": 1497177430828384396, "LABEL": "Trainer"},
    "Gym_Leader": {"FORUM_CHANNEL_ID": 1501347022773686483, "LABEL": "Gym Leader"},
    "Elite_Four": {"FORUM_CHANNEL_ID": 1497179653276827720, "LABEL": "Elite Four"},
}

group_locks = defaultdict(asyncio.Lock)
http_session: Optional[aiohttp.ClientSession] = None

# =========================================================
# RUTINAS DE VALIDACIÓN Y LOGS DE ERRORES / REGISTRO
# =========================================================

def is_target_message(message: discord.Message) -> bool:
    if message.webhook_id is None:
        return False
    if message.channel.id not in CHANNEL_GROUP_MAP:
        return False
    content = message.content or ""
    if any(pattern.search(content) for pattern in TRIGGER_PATTERNS):
        return True
    if len(message.attachments) > 0:
        return True
    return False

async def delete_message_later(msg: discord.Message, delay: int):
    await asyncio.sleep(delay)
    try:
        await msg.delete()
    except Exception:
        pass

async def forward_invalid_message(message: discord.Message, reason: str):
    if not INVALID_LOG_CHANNEL_ID:
        return

    original_text = message.content or "(sin texto)"
    owner_id = extract_owner_discord_id_from_first_line(original_text)

    tagged_user = None
    if owner_id:
        try:
            tagged_user = await resolve_gp_owner(client, original_text, CHANNEL_GROUP_MAP.get(message.channel.id, "Elite_Four"))
        except Exception:
            pass

    cooldown_key = f"{message.channel.id}:{owner_id}" if owner_id else str(message.channel.id)
    now = time.monotonic()
    
    if now - INVALID_LOG_LAST_SENT.get(cooldown_key, 0) < INVALID_LOG_COOLDOWN:
        try:
            await message.delete()
        except Exception:
            pass
        return

    INVALID_LOG_LAST_SENT[cooldown_key] = now

    try:
        log_channel = client.get_channel(INVALID_LOG_CHANNEL_ID) or await client.fetch_channel(INVALID_LOG_CHANNEL_ID)
        if not log_channel:
            return

        mention_line = f"{tagged_user['mention']}\n" if tagged_user and tagged_user.get("mention") else ""
        allowed = discord.AllowedMentions(users=True)

        content = (
            f"{mention_line}"
            f"**Rejected message from channel <#{message.channel.id}>**\n"
            f"**Reason:** {reason}\n\n"
            f"**Original message:**\n"
            f"```{original_text[:1300]}```"
        )

        await log_channel.send(content=content, allowed_mentions=allowed)
        try:
            await message.delete()
        except Exception:
            pass
    except Exception as e:
        logger.exception("Error reenviando mensaje inválido: %s", e)

# =========================================================
# DETECCIÓN Y EXTRACCIÓN EXACTA DE CARTAS (4 Y 5 CARTAS)
# =========================================================

def is_already_linear_gp(img: Image.Image) -> bool:
    w, h = img.size
    return (w / max(h, 1)) >= 2.1


def find_segments_1d(proj: np.ndarray, threshold: float = 0.12, min_len: int = 40) -> List[Tuple[int, int]]:
    active = proj > threshold
    padded = np.concatenate(([False], active, [False]))
    starts = np.where(padded[1:] & ~padded[:-1])[0]
    ends = np.where(~padded[1:] & padded[:-1])[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s) >= min_len]


def extract_axis_bounds(mask_slice: np.ndarray, axis: int, expected_count: int, fallback_rel: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
    length = mask_slice.shape[1 - axis]
    proj = mask_slice.mean(axis=axis)

    for thresh in (0.12, 0.18, 0.08, 0.25):
        segs = find_segments_1d(proj, threshold=thresh, min_len=int(length * 0.18))
        if len(segs) == expected_count:
            return segs

    return [(int(round(a * length)), int(round(b * length))) for a, b in fallback_rel]


def detect_and_extract_cards(source_img: Image.Image) -> List[Image.Image]:
    rgb_img = source_img.convert("RGB")
    w, h = rgb_img.size
    arr = np.array(rgb_img, dtype=np.float32)

    if is_already_linear_gp(rgb_img):
        num_c = 4 if (w / max(h, 1)) < 3.15 else 5
        step = w / num_c
        return [rgb_img.crop((int(i * step), 0, int((i + 1) * step), h)) for i in range(num_c)]

    corner_samples = np.concatenate([
        arr[int(h * 0.85):int(h * 0.98), int(w * 0.01):int(w * 0.10)].reshape(-1, 3),
        arr[int(h * 0.85):int(h * 0.98), int(w * 0.90):int(w * 0.99)].reshape(-1, 3),
    ], axis=0)
    bg_rgb = np.median(corner_samples, axis=0)

    diff = np.linalg.norm(arr - bg_rgb, axis=2)
    mask = diff > 22.0

    mid_top, mid_bot = int(h * 0.46), int(h * 0.54)
    row_profile = mask[mid_top:mid_bot, :].mean(axis=1)
    split_y = mid_top + int(np.argmin(row_profile))

    top_y_segs = find_segments_1d(mask[:split_y, :].mean(axis=1), threshold=0.10, min_len=int(h * 0.25))
    top_y1, top_y2 = top_y_segs[0] if top_y_segs else (int(round(0.043 * h)), int(round(0.493 * h)))

    bot_y_segs = find_segments_1d(mask[split_y:, :].mean(axis=1), threshold=0.10, min_len=int(h * 0.25))
    if bot_y_segs:
        bot_y1, bot_y2 = split_y + bot_y_segs[0][0], split_y + bot_y_segs[0][1]
    else:
        bot_y1, bot_y2 = int(round(0.504 * h)), int(round(0.954 * h))

    fallback_top_x = [(0.0078, 0.3310), (0.3388, 0.6612), (0.6690, 0.9922)]
    top_x_bounds = extract_axis_bounds(mask[top_y1:top_y2, :], axis=0, expected_count=3, fallback_rel=fallback_top_x)

    left_outer_check = mask[bot_y1:bot_y2, int(w * 0.18):int(w * 0.30)].mean()
    right_outer_check = mask[bot_y1:bot_y2, int(w * 0.70):int(w * 0.82)].mean()
    is_five_cards = (left_outer_check > 0.25 and right_outer_check > 0.25)

    if is_five_cards:
        fallback_bot_x = [(0.1728, 0.4961), (0.5039, 0.8272)]
        bot_x_bounds = extract_axis_bounds(mask[bot_y1:bot_y2, :], axis=0, expected_count=2, fallback_rel=fallback_bot_x)
    else:
        fallback_bot_x = [(0.3388, 0.6612)]
        bot_x_bounds = extract_axis_bounds(mask[bot_y1:bot_y2, :], axis=0, expected_count=1, fallback_rel=fallback_bot_x)

    cards: List[Image.Image] = []
    for x1, x2 in top_x_bounds:
        cards.append(rgb_img.crop((x1, top_y1, x2, top_y2)))
    for x1, x2 in bot_x_bounds:
        cards.append(rgb_img.crop((x1, bot_y1, x2, bot_y2)))

    return cards


# =========================================================
# MOTOR GRÁFICO: EXTRACCIÓN DE COLOR Y 2 MODOS DE VITRINA
# =========================================================

def extract_vibrant_card_color(card_img: Image.Image) -> Tuple[int, int, int]:
    """
    Analiza la carta para extraer su color elemental/dominante más vibrante
    y saturado para generar el aura de luz adaptativa.
    """
    small = card_img.convert("RGB").resize((48, 64), Image.Resampling.BILINEAR)
    # Recortamos ligeramente los bordes exteriores para enfocarnos en el marco e ilustración
    cropped = small.crop((4, 4, 44, 60))
    arr = np.array(cropped, dtype=np.float32) / 255.0
    pixels = arr.reshape(-1, 3)

    max_c = pixels.max(axis=1)
    min_c = pixels.min(axis=1)
    diff = max_c - min_c

    sat = np.where(max_c > 0, diff / max_c, 0)
    val = max_c

    # Filtramos grises, blancos puros y negros oscuros para quedarnos con el color vivo
    mask = (sat > 0.28) & (val > 0.25) & (val < 0.96)
    vibrant = pixels[mask]

    if len(vibrant) < 15:
        mask2 = (val > 0.20) & (val < 0.95)
        vibrant = pixels[mask2] if np.any(mask2) else pixels

    # Ponderamos los píxeles más saturados y luminosos
    v_max = vibrant.max(axis=1)
    v_min = vibrant.min(axis=1)
    v_sat = np.where(v_max > 0, (v_max - v_min) / v_max, 0)
    weights = (v_sat ** 1.5) * v_max + 1e-4

    avg_rgb = np.average(vibrant, axis=0, weights=weights)

    # Potenciamos el brillo y saturación del color final para que el efecto neón destaque
    c_max = float(avg_rgb.max())
    if c_max > 0:
        avg_rgb = avg_rgb / c_max

    r, g, b = [int(np.clip(c * 255.0, 45, 255)) for c in avg_rgb]
    return (r, g, b)


def create_showcase_canvas(width: int, height: int, transparent: bool = False) -> Image.Image:
    if transparent:
        return Image.new("RGBA", (width, height), (0, 0, 0, 0))

    bg = Image.new("RGBA", (width, height), (*BG_DARK_COLOR, 255))
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow)

    cx, cy = width // 2, height // 2
    max_rx, max_ry = int(width * 0.52), int(height * 0.52)
    for step in range(6, 0, -1):
        ratio = step / 6.0
        rx, ry = int(max_rx * ratio), int(max_ry * ratio)
        alpha = int(42 * (1.0 - ratio + 0.15))
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=(*BG_CENTER_GLOW, alpha))

    glow = glow.filter(ImageFilter.GaussianBlur(radius=50))
    return Image.alpha_composite(bg, glow)


def style_card_with_rounded_corners(card_img: Image.Image, target_w: int, target_h: int, radius: int = 18) -> Image.Image:
    resized = card_img.convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
    sharpened = resized.filter(ImageFilter.UnsharpMask(radius=1.2, percent=115, threshold=2)).convert("RGBA")

    mask = Image.new("L", (target_w, target_h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, target_w - 1, target_h - 1), radius=radius, fill=255)

    out = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    out.paste(sharpened, (0, 0), mask)
    return out


def paste_card_with_adaptive_glow(
    canvas: Image.Image,
    styled_card: Image.Image,
    angle: float,
    center_xy: Tuple[int, int],
    glow_color: Tuple[int, int, int],
    radius: int = 18,
    shadow_offset: Tuple[int, int] = (0, 18),
    shadow_blur: int = 20,
    shadow_opacity: int = 190
) -> None:
    """
    Dibuja una sombra profunda, una doble aura de luz neón del color de la carta
    y un borde luminoso sutil antes de colocar la carta en el lienzo.
    """
    pad = 95
    w, h = styled_card.size
    total_w, total_h = w + pad * 2, h + pad * 2

    # 1. Capa de Sombra Oscura
    shadow_layer = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    s_draw = ImageDraw.Draw(shadow_layer)
    s_draw.rounded_rectangle(
        (pad, pad, pad + w - 1, pad + h - 1),
        radius=radius,
        fill=(0, 0, 0, shadow_opacity)
    )
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=shadow_blur))

    # 2. Capa de Luz Exterior (Aura amplia difuminada del color de la carta)
    outer_glow = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    og_draw = ImageDraw.Draw(outer_glow)
    spread_out = 18
    og_draw.rounded_rectangle(
        (pad - spread_out, pad - spread_out, pad + w + spread_out, pad + h + spread_out),
        radius=radius + spread_out,
        fill=(*glow_color, 155)
    )
    outer_glow = outer_glow.filter(ImageFilter.GaussianBlur(radius=30))

    # 3. Capa de Luz Interior (Resplandor intenso pegado al borde de la carta)
    inner_glow = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    ig_draw = ImageDraw.Draw(inner_glow)
    spread_in = 6
    bright_color = tuple(min(255, int(c * 1.15 + 35)) for c in glow_color)
    ig_draw.rounded_rectangle(
        (pad - spread_in, pad - spread_in, pad + w + spread_in, pad + h + spread_in),
        radius=radius + spread_in,
        fill=(*bright_color, 215)
    )
    inner_glow = inner_glow.filter(ImageFilter.GaussianBlur(radius=11))

    # 4. Combinamos las capas de iluminación y pegamos la carta encima
    combined = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    combined.alpha_composite(shadow_layer, shadow_offset)
    combined.alpha_composite(outer_glow, (0, 0))
    combined.alpha_composite(inner_glow, (0, 0))
    combined.paste(styled_card, (pad, pad), styled_card)

    # Pequeño filete de luz en el contorno para darle acabado de vitrina
    rim_draw = ImageDraw.Draw(combined)
    rim_draw.rounded_rectangle(
        (pad - 1, pad - 1, pad + w, pad + h),
        radius=radius + 1,
        outline=(*bright_color, 185),
        width=2
    )

    if abs(angle) > 0.05:
        combined = combined.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)

    rw, rh = combined.size
    cx, cy = center_xy
    top_left_x = cx - (rw // 2)
    top_left_y = cy - (rh // 2)

    canvas.alpha_composite(combined, (top_left_x, top_left_y))


def build_trophy_grid_image(cards: List[Image.Image], transparent: bool = False) -> Image.Image:
    """
    MODO 1 (TROPHY_GRID): 3 cartas arriba y 2 abajo centradas, con ligera inclinación
    en los extremos y luz adaptativa intensa detrás de cada carta (como en la Imagen 1).
    """
    canvas_w, canvas_h = 1520, 1080
    canvas = create_showcase_canvas(canvas_w, canvas_h, transparent=transparent)
    num_cards = len(cards)

    card_w, card_h = 336, 468

    if num_cards == 4:
        positions = [
            (240, 540, 5.0),
            (586, 540, 1.5),
            (934, 540, -1.5),
            (1280, 540, -5.0),
        ]
    else:
        # 3 cartas arriba (izquierda ligeramente inclinada, centro recta, derecha ligeramente inclinada)
        # 2 cartas abajo centradas en los espacios intermedios
        positions = [
            (330, 290, 6.0),    # Arriba Izquierda
            (760, 280, 0.0),    # Arriba Centro
            (1190, 290, -6.0),  # Arriba Derecha
            (535, 790, 3.5),    # Abajo Izquierda
            (985, 790, -3.5),   # Abajo Derecha
        ]

    for idx, card in enumerate(cards):
        if idx >= len(positions):
            break
        cx, cy, angle = positions[idx]
        glow_rgb = extract_vibrant_card_color(card)
        styled = style_card_with_rounded_corners(card, card_w, card_h, radius=18)
        paste_card_with_adaptive_glow(
            canvas,
            styled,
            angle=angle,
            center_xy=(cx, cy),
            glow_color=glow_rgb,
            radius=18
        )

    return canvas if transparent else canvas.convert("RGB")


def build_staggered_wave_image(cards: List[Image.Image], transparent: bool = False) -> Image.Image:
    """
    MODO 2 (STAGGERED_WAVE): 5 cartas en una sola fila escalonada tipo ola/podio,
    100% verticales (0° de rotación), sin solaparse y con luz adaptativa (como en la Imagen 2).
    """
    canvas_w, canvas_h = 1920, 860
    canvas = create_showcase_canvas(canvas_w, canvas_h, transparent=transparent)
    num_cards = len(cards)

    if num_cards == 4:
        card_w, card_h = 350, 488
        positions = [
            (300, 465, 0.0),
            (673, 395, 0.0),
            (1247, 395, 0.0),
            (1620, 465, 0.0),
        ]
    else:
        card_w, card_h = 332, 463
        # Escalonado simétrico: extremos bajos, intermedios a media altura, centro en la cima
        positions = [
            (228, 485, 0.0),   # Carta 1 (Baja)
            (594, 425, 0.0),   # Carta 2 (Media)
            (960, 365, 0.0),   # Carta 3 (Centro Alta)
            (1326, 425, 0.0),  # Carta 4 (Media)
            (1692, 485, 0.0),  # Carta 5 (Baja)
        ]

    for idx, card in enumerate(cards):
        if idx >= len(positions):
            break
        cx, cy, angle = positions[idx]
        glow_rgb = extract_vibrant_card_color(card)
        styled = style_card_with_rounded_corners(card, card_w, card_h, radius=18)
        paste_card_with_adaptive_glow(
            canvas,
            styled,
            angle=angle,
            center_xy=(cx, cy),
            glow_color=glow_rgb,
            radius=18
        )

    return canvas if transparent else canvas.convert("RGB")


def build_gp_showcase_image(cards: List[Image.Image], transparent: Optional[bool] = None) -> Image.Image:
    use_transparent = TRANSPARENT_BACKGROUND if transparent is None else transparent
    mode = PRESENTATION_MODE.strip().upper()

    if mode == "STAGGERED_WAVE":
        return build_staggered_wave_image(cards, transparent=use_transparent)
    return build_trophy_grid_image(cards, transparent=use_transparent)


def convert_gp_to_showcase_bytes(source_img: Image.Image) -> Tuple[bytes, List[Image.Image]]:
    cards = detect_and_extract_cards(source_img)
    final_img = build_gp_showcase_image(cards, transparent=TRANSPARENT_BACKGROUND)
    out_buf = io.BytesIO()
    final_img.save(out_buf, format="PNG", optimize=True)
    return out_buf.getvalue(), cards


# =========================================================
# HELPERS UPSTASH REDIS
# =========================================================

async def get_http_session() -> aiohttp.ClientSession:
    global http_session
    if http_session is None or http_session.closed:
        http_session = aiohttp.ClientSession()
    return http_session

def redis_headers() -> dict:
    return {"Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}"}


def users_key(group: str) -> str:
    return f"users:{group}"


def online_key(group: str) -> str:
    return f"online:{group}"


def vip_key(group: str) -> str:
    return f"vip:{group}"


def gp_users_key() -> str:
    return "gp_users"


def live_stats_key(group: str) -> str:
    return f"gp_live_stats:{group}"


def vote_state_key(group: str) -> str:
    return f"gp_votes:{group}"


def safe_json_loads(value, default):
    try:
        if value is None:
            return default
        if isinstance(value, (dict, list)):
            return value
        return json.loads(value)
    except Exception:
        return default


async def redis_command(*parts):
    if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
        raise RuntimeError("Missing UPSTASH_REDIS_REST_URL or UPSTASH_REDIS_REST_TOKEN")

    encoded = "/".join(quote(str(p), safe="") for p in parts)
    url = f"{UPSTASH_REDIS_REST_URL}/{encoded}"

    session = await get_http_session()
    async with session.get(url, headers=redis_headers()) as resp:
        text = await resp.text()
        if resp.status not in (200, 201):
            raise RuntimeError(f"Redis command failed {resp.status}: {text}")
        data = json.loads(text)
        return data.get("result")


async def redis_get(key: str, default=None):
    result = await redis_command("get", key)
    return default if result is None else result


async def redis_set_json(key: str, data) -> None:
    await redis_command("set", key, json.dumps(data))


async def redis_get_json(key: str, default):
    raw = await redis_get(key)
    return safe_json_loads(raw, default)


async def redis_hgetall_json(key: str) -> dict:
    result = await redis_command("hgetall", key)
    if not result:
        return {}
    items = result.items() if isinstance(result, dict) else zip(result[0::2], result[1::2])
    return {str(field): safe_json_loads(val, {}) for field, val in items}


async def redis_hset_json(key: str, field: str, value) -> None:
    await redis_command("hset", key, field, json.dumps(value))


async def redis_smembers_ids(key: str) -> List[str]:
    result = await redis_command("smembers", key)
    if not result:
        return []
    return [str(x).strip() for x in result if re.fullmatch(r"\d{16}", str(x).strip())]


async def redis_sadd_id(key: str, value: str) -> bool:
    value = str(value or "").strip()
    if not re.fullmatch(r"\d{16}", value):
        return False
    await redis_command("sadd", key, value)
    return True


# =========================================================
# PARSERS DE METADATOS Y PROPIETARIO DEL GP
# =========================================================

def extract_owner_discord_id_from_first_line(content: str) -> Optional[str]:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return None
    first = lines[0]
    m = re.search(r"<@?!?(\d{17,20})>", first)
    if m:
        return m.group(1)
    m = re.match(r"^@?(\d{17,20})\b", first)
    return m.group(1) if m else None


def extract_first_line_username_hint(content: str) -> Optional[str]:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return None
    m = re.match(r"^@([^\s]+)", lines[0])
    return m.group(1).strip() if m else None


def extract_friend_id(content: str) -> Optional[str]:
    for line in (line.strip() for line in content.splitlines() if line.strip()):
        if "God Pack found" in line or line.startswith("<@"):
            continue
        m = re.search(r"\((\d{16})\)", line)
        if m:
            return m.group(1)
    return None


def parse_heartbeat_metadata(content: str) -> dict:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    result = {
        "obtainer_user": extract_first_line_username_hint(content),
        "owner_discord_id": extract_owner_discord_id_from_first_line(content),
        "bot_name": None,
        "game_id": None,
        "packs_count": None,
        "pack_position": None,
        "pack_total": 5,
        "pack_name": None,
        "filename": None,
        "raw_pack_line": None,
    }

    for line in lines:
        m = re.match(r"^(.+?)\s*\((\d+)\)$", line)
        if m:
            result["bot_name"] = m.group(1).strip()
            result["game_id"] = m.group(2).strip()
            continue

        m = re.search(r"(\[(\d)/([45])\]\[(\d+)P\]\[([^\]]*)\])", line, re.IGNORECASE)
        if m:
            result["raw_pack_line"] = m.group(1)
            result["pack_position"] = int(m.group(2))
            result["pack_total"] = int(m.group(3))
            result["packs_count"] = int(m.group(4))
            result["pack_name"] = m.group(5).strip()
            continue

        m = re.match(r"^File name:\s*(.+)$", line, re.IGNORECASE)
        if m:
            result["filename"] = m.group(1).strip()

    return result


def build_pack_label_from_meta(meta: dict) -> str:
    pos = meta.get("pack_position")
    total = meta.get("pack_total") or 5
    if isinstance(pos, int) and 1 <= pos <= 5:
        return f"[{pos}/{total}]"
    return f"[?/{total}]"


async def load_group_users(group: str) -> dict:
    if group not in GROUP_CONFIG:
        return {}
    return await redis_hgetall_json(users_key(group))


def normalize_name_for_match(value: str) -> str:
    value = str(value or "").lower().strip()
    value = re.sub(r"[\u200b-\u200d\ufeff*_`~|>]", "", value)
    value = re.sub(r"^@+|[:：]+$|[^\w]", "", value)
    return value


def names_match(input_name: str, user_data: dict) -> bool:
    clean_input = normalize_name_for_match(input_name)
    if not clean_input:
        return False

    candidates = [
        user_data.get("name"),
        user_data.get("heartbeatName"),
        user_data.get("displayName"),
        user_data.get("username"),
    ]
    if isinstance(user_data.get("aliases"), list):
        candidates.extend(user_data.get("aliases"))

    for cand in candidates:
        clean_cand = normalize_name_for_match(cand)
        if clean_cand and (clean_cand == clean_input or clean_cand in clean_input or clean_input in clean_cand):
            return True
    return False


async def resolve_rival_owner_by_game_id(game_id: str) -> Optional[dict]:
    if not game_id:
        return None
    try:
        rivals_sessions = await redis_hgetall_json("rivals_sessions")
        for sess in rivals_sessions.values():
            for m in (sess or {}).get("members") or []:
                if str(m.get("gameId") or "").strip() == game_id:
                    uid = str(m.get("discordId"))
                    name = m.get("name") or m.get("heartbeatName") or uid
                    return {"discord_id": uid, "display_name": name, "mention": f"<@{uid}>", "game_id": game_id}
    except Exception:
        pass

    try:
        rival_duos = await redis_hgetall_json("rival_duos")
        for duo in rival_duos.values():
            online_users = (duo or {}).get("onlineUsers") or {}
            members = (duo or {}).get("members") or {}
            for uid, m_data in members.items():
                if str(m_data.get("gameId") or "").strip() == game_id and online_users.get(str(uid), False):
                    name = m_data.get("name") or m_data.get("heartbeatName") or uid
                    return {"discord_id": str(uid), "display_name": name, "mention": f"<@{uid}>", "game_id": game_id}
    except Exception:
        pass
    return None


async def resolve_gp_owner(client: discord.Client, content: str, group: str) -> dict:
    mention_match = re.search(r"<@!?(\d+)>", content)
    owner_discord_id = mention_match.group(1) if mention_match else None

    username_match = re.search(r"@([a-zA-Z0-9_\.]+)", content)
    username_hint = username_match.group(1) if username_match else None

    users = await load_group_users(group)

    if owner_discord_id and owner_discord_id in users:
        u = users[owner_discord_id]
        return {
            "discord_id": owner_discord_id,
            "display_name": u.get("name") or u.get("heartbeatName") or owner_discord_id,
            "mention": f"<@{owner_discord_id}>",
        }

    if username_hint:
        for uid, data in users.items():
            if names_match(username_hint, data):
                return {
                    "discord_id": uid,
                    "display_name": data.get("name") or data.get("heartbeatName") or username_hint,
                    "mention": f"<@{uid}>",
                }

    if owner_discord_id:
        try:
            user = await client.fetch_user(int(owner_discord_id))
            return {
                "discord_id": owner_discord_id,
                "display_name": user.display_name or user.name,
                "mention": f"<@{owner_discord_id}>",
            }
        except Exception:
            pass

    return {
        "discord_id": owner_discord_id,
        "display_name": username_hint or owner_discord_id or "unknown",
        "mention": f"<@{owner_discord_id}>" if owner_discord_id else "@unknown",
    }


# =========================================================
# GESTIÓN DE VIPS, STATS Y MENCIONES ONLINE
# =========================================================

async def add_vip_id(friend_id: str, group: str) -> bool:
    friend_id = str(friend_id or "").strip()
    if not re.fullmatch(r"\d{16}", friend_id) or group not in GROUP_CONFIG:
        return False
    try:
        await redis_sadd_id(vip_key(group), friend_id)
        now_ts = int(datetime.now(timezone.utc).timestamp())
        await redis_command("hset", f"{vip_key(group)}:timestamps", friend_id, now_ts)
        return True
    except Exception:
        return False


async def cleanup_expired_vips(group: str):
    try:
        ts_key = f"{vip_key(group)}:timestamps"
        vip_data = await redis_command("hgetall", ts_key)
        if not vip_data:
            return
        now_ts = int(datetime.now(timezone.utc).timestamp())
        expire_seconds = 48 * 60 * 60
        items = vip_data.items() if isinstance(vip_data, dict) else zip(vip_data[0::2], vip_data[1::2])
        for friend_id, saved_ts in items:
            try:
                if (now_ts - int(saved_ts)) >= expire_seconds:
                    await redis_command("srem", vip_key(group), friend_id)
                    await redis_command("hdel", ts_key, friend_id)
            except Exception:
                pass
    except Exception:
        pass


async def register_user_gp(owner_info: dict) -> None:
    discord_id = owner_info.get("discord_id")
    if not discord_id:
        return
    data = await redis_hgetall_json(gp_users_key())
    entry = data.get(discord_id, {"name": owner_info.get("display_name", "Unknown"), "gp": 0})
    entry["gp"] = int(entry.get("gp", 0)) + 1
    entry["name"] = owner_info.get("display_name", "Unknown")
    await redis_hset_json(gp_users_key(), discord_id, entry)


async def get_online_mentions(group: str) -> List[str]:
    if group not in GROUP_CONFIG:
        return []
    try:
        online_ids = await redis_smembers_ids(online_key(group))
        users = await load_group_users(group)
        mentions = []
        for discord_id, user_info in users.items():
            main_id = str(user_info.get("main_id", "")).strip()
            sec_id = str(user_info.get("sec_id", "")).strip()
            if main_id in online_ids or (sec_id and sec_id in online_ids):
                mentions.append(f"<@{discord_id}>")

        if group in ("Elite_Four", "Gym_Leader"):
            for gid in online_ids:
                owner = await resolve_rival_owner_by_game_id(gid)
                if owner and owner.get("mention") and owner["mention"] not in mentions:
                    mentions.append(owner["mention"])
        return mentions
    except Exception:
        return []


def get_utc6_date_string() -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=6)).date().isoformat()


async def update_stats_safe(group: str, callback):
    async with group_locks[group]:
        stats = await redis_get_json(
            live_stats_key(group),
            {"totalGP": 0, "totalAlive": 0, "currentDay": None, "daily": {"gp": 0, "alive": 0}, "history": [], "processedMessages": []}
        )
        today = get_utc6_date_string()
        if not stats.get("currentDay"):
            stats["currentDay"] = today
        elif stats["currentDay"] != today:
            stats["history"].insert(0, {"date": stats["currentDay"], "gp": stats["daily"]["gp"], "alive": stats["daily"]["alive"]})
            stats["history"] = stats["history"][:5]
            stats["currentDay"] = today
            stats["daily"] = {"gp": 0, "alive": 0}
        stats = await callback(stats)
        await redis_set_json(live_stats_key(group), stats)
        return stats


async def increment_gp_callback(stats: dict) -> dict:
    stats["totalGP"] += 1
    stats["daily"]["gp"] += 1
    return stats


# =========================================================
# PANEL EMBED REDISEÑADO Y MINIMALISTA (SIN LUPAS)
# =========================================================

def build_post_title(meta: dict, pack_label: str) -> str:
    packs_count = meta.get("packs_count")
    packs_text = f"[{packs_count}P]" if packs_count is not None else "[?P]"
    bot_name = meta.get("bot_name") or "UnknownBot"
    game_id = meta.get("game_id") or "UnknownID"
    return f"{pack_label} {packs_text} {bot_name} [{game_id}]"


def build_forum_info_embed(
    meta: dict,
    pack_label: str,
    status: str = "none",
    alive_users: Optional[List[str]] = None,
    dead_users: Optional[List[str]] = None,
    group: str = "Elite_Four",
    created_at_ts: Optional[int] = None,
) -> discord.Embed:
    alive_users = alive_users or []
    dead_users = dead_users or []
    ts = created_at_ts or int(time.time())

    if status == "alive":
        color = 0x2ECC71
        status_badge = "✅ VERIFIED ALIVE"
        progress_bar = "🟩🟩🟩🟩🟩🟩🟩🟩"
    elif status == "dead":
        color = 0xE74C3C
        status_badge = "💀 CONFIRMED DEAD"
        progress_bar = "🟥🟥🟥🟥🟥🟥🟥🟥"
    else:
        color = 0xF1C40F
        dead_progress = min(len(dead_users), 4)
        progress_bar = ("🟧" * (dead_progress * 2)) + ("⬛" * (8 - dead_progress * 2)) if dead_progress else "🟨🟨⬛⬛⬛⬛⬛⬛"
        status_badge = f"⏳ WAITING FOR TEST ({len(dead_users)}/4 Dead)"

    obtainer_mention = meta.get("owner_mention") or meta.get("owner_display_name") or "Unknown"
    nickname = meta.get("bot_name") or "Unknown"
    game_id = meta.get("game_id") or "UnknownID"
    packs_count = meta.get("packs_count")
    packs_text = f"{packs_count} Packs" if packs_count is not None else "? Packs"
    pack_name = meta.get("pack_name") or "DeluxeB"
    filename = meta.get("filename") or "unknown_file.xml"
    group_label = GROUP_CONFIG.get(group, {}).get("LABEL", group.replace("_", " "))

    embed = discord.Embed(
        title=f"✨ {group_label.upper()} — GOD PACK {pack_label}",
        description=(
            f"> 📦 **Expansion:** `{pack_name}`  •  🎯 **Pulls:** `{packs_text}`  •  🕒 **Found:** <t:{ts}:R>\n"
            f"> {progress_bar}  **{status_badge}**"
        ),
        color=color,
    )

    embed.add_field(name="👑 Found By", value=f"> {obtainer_mention}", inline=True)
    embed.add_field(name="🏷️ Nickname", value=f"> `{nickname}`", inline=True)
    embed.add_field(name="🎴 Rarity", value=f"> `{pack_label}`", inline=True)

    embed.add_field(
        name="📲 Friend ID (Tap to Copy)",
        value=f"```\n{game_id}\n```",
        inline=False,
    )

    alive_str = ", ".join(f"<@{u}>" for u in alive_users) if alive_users else "*No votes yet*"
    dead_str = ", ".join(f"<@{u}>" for u in dead_users) if dead_users else "*No votes yet*"
    embed.add_field(
        name="🗳️ Live Community Verification",
        value=(
            f"🟢 **Alive ({len(alive_users)}/1):** {alive_str}\n"
            f"🔴 **Dead ({len(dead_users)}/4):** {dead_str}"
        ),
        inline=False,
    )

    embed.set_footer(text=f"✨ {PRESENTATION_MODE} Showcase Mode  •  📄 {filename}")
    return embed


async def create_forum_post_with_image_bytes(
    client: discord.Client,
    group: str,
    title: str,
    image_bytes: bytes,
    filename: str = "gp_showcase.png",
) -> Optional[dict]:
    try:
        forum_id = GROUP_CONFIG[group]["FORUM_CHANNEL_ID"]
        channel = client.get_channel(forum_id) or await client.fetch_channel(forum_id)
        if not isinstance(channel, discord.ForumChannel):
            return None

        file = discord.File(io.BytesIO(image_bytes), filename=filename)
        created = await channel.create_thread(name=title[:100], content="‎", file=file)
        thread = created.thread if hasattr(created, "thread") else created
        return {"thread": thread, "jump_url": thread.jump_url}
    except Exception:
        return None


class ForumLinkView(discord.ui.View):
    def __init__(self, post_url: str, meta: dict, pack_label: str, status: str = "none"):
        super().__init__(timeout=None)
        bot_name = meta.get("bot_name") or "Unknown"
        prefix = "✅ " if status == "alive" else ("❌ " if status == "dead" else "")
        label = f"{prefix}{bot_name}".strip()

        # Botón limpio de enlace directo al post (sin lupas)
        self.add_item(discord.ui.Button(label=label[:80], style=discord.ButtonStyle.link, url=post_url, row=0))


# =========================================================
# SISTEMA DE VOTACIÓN ALIVE / DEAD (SIN LUPAS)
# =========================================================

async def load_vote_state(group: str) -> dict:
    if group not in GROUP_CONFIG:
        return {}
    data = await redis_hgetall_json(vote_state_key(group))
    cutoff = time.time() - (72 * 60 * 60)
    return {k: v for k, v in data.items() if v.get("created_at", time.time()) >= cutoff}


async def save_vote_state(group: str, data: dict) -> None:
    if group not in GROUP_CONFIG:
        return
    for vote_key, state in data.items():
        await redis_hset_json(vote_state_key(group), vote_key, state)


async def update_gp_thread_status(thread_id: int, status: str):
    try:
        thread = client.get_channel(thread_id) or await client.fetch_channel(thread_id)
        if thread is None:
            return
        clean_name = thread.name.removeprefix("✅ ").removeprefix("❌ ").strip()
        new_name = f"✅ {clean_name}" if status == "alive" else (f"❌ {clean_name}" if status == "dead" else None)
        if new_name:
            await thread.edit(name=new_name[:100])
    except Exception:
        pass


async def update_main_link_button(state: dict, status: str, meta: dict, pack_label: str):
    try:
        channel_id = state.get("link_channel_id")
        message_id = state.get("link_message_id")
        post_url = state.get("post_url")
        if not channel_id or not message_id or not post_url:
            return

        channel = client.get_channel(int(channel_id)) or await client.fetch_channel(int(channel_id))
        msg = await channel.fetch_message(int(message_id))
        await msg.edit(view=ForumLinkView(post_url, meta, pack_label, status=status))
    except Exception:
        pass


class GPVoteView(discord.ui.View):
    def __init__(self, vote_key: str, group: str):
        super().__init__(timeout=None)
        self.vote_key = vote_key
        self.group = group

        self.btn_alive = discord.ui.Button(
            label="🟢 Alive (0)", 
            style=discord.ButtonStyle.success, 
            custom_id=f"gp_alive_{vote_key}", 
            row=0
        )
        self.btn_alive.callback = self.alive_callback
        self.add_item(self.btn_alive)

        self.btn_dead = discord.ui.Button(
            label="🔴 Dead (0)", 
            style=discord.ButtonStyle.danger, 
            custom_id=f"gp_dead_{vote_key}", 
            row=0
        )
        self.btn_dead.callback = self.dead_callback
        self.add_item(self.btn_dead)

    async def alive_callback(self, interaction: discord.Interaction):
        await self.handle_vote(interaction, "alive")

    async def dead_callback(self, interaction: discord.Interaction):
        await self.handle_vote(interaction, "dead")

    async def handle_vote(self, interaction: discord.Interaction, vote_type: str):
        data = await load_vote_state(self.group)
        state = data.get(self.vote_key)
        if not state:
            return await interaction.response.send_message("❌ Vote state not found.", ephemeral=True)

        user_id = str(interaction.user.id)
        if user_id in state["alive_users"] or user_id in state["dead_users"]:
            return await interaction.response.send_message("⚠️ You already voted on this God Pack.", ephemeral=True)

        if vote_type == "alive":
            state["alive_users"].append(user_id)
        else:
            state["dead_users"].append(user_id)

        alive_count = len(state["alive_users"])
        dead_count = len(state["dead_users"])
        status = "alive" if alive_count >= 1 else ("dead" if dead_count >= 4 else state.get("status", "none"))
        state["status"] = status

        if status == "alive" and not state.get("counted_alive", False):
            await update_stats_safe(self.group, self._increment_alive)
            state["counted_alive"] = True

        data[self.vote_key] = state
        await save_vote_state(self.group, data)

        self.btn_alive.label = f"🟢 Alive ({alive_count})"
        self.btn_dead.label = f"🔴 Dead ({dead_count})"
        
        if status in ("alive", "dead"):
            self.btn_alive.disabled = True
            self.btn_dead.disabled = True
            await update_gp_thread_status(int(self.vote_key), status)
            await update_main_link_button(state, status, state.get("meta", {}), state.get("pack_label", ""))

        created_ts = int(state.get("created_at", time.time()))
        updated_embed = build_forum_info_embed(
            state.get("meta", {}),
            state.get("pack_label", "[?/5]"),
            status=status,
            alive_users=state["alive_users"],
            dead_users=state["dead_users"],
            group=self.group,
            created_at_ts=created_ts,
        )
        await interaction.response.edit_message(embed=updated_embed, view=self)

    async def _increment_alive(self, stats: dict) -> dict:
        stats["totalAlive"] += 1
        stats["daily"]["alive"] += 1
        return stats


# =========================================================
# BOT DISCORD E INICIALIZACIÓN
# =========================================================

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

async def restore_persistent_views():
    restored = 0
    for group in GROUP_CONFIG.keys():
        try:
            vote_data = await load_vote_state(group)
            for vote_key, state in vote_data.items():
                view = GPVoteView(vote_key=str(vote_key), group=group)
                alive_count = len(state.get("alive_users", []))
                dead_count = len(state.get("dead_users", []))
                
                view.btn_alive.label = f"🟢 Alive ({alive_count})"
                view.btn_dead.label = f"🔴 Dead ({dead_count})"
                
                if state.get("status") in ("alive", "dead"):
                    view.btn_alive.disabled = True
                    view.btn_dead.disabled = True
                
                client.add_view(view)
                restored += 1
        except Exception:
            pass
    logger.info("✅ Views restauradas: %s", restored)


@client.event
async def on_ready():
    logger.info(
        "✅ Bot GP Showcase (Mode=%s, Transparent=%s) conectado como %s",
        PRESENTATION_MODE,
        TRANSPARENT_BACKGROUND,
        client.user,
    )
    await restore_persistent_views()


@client.event
async def on_message(message: discord.Message):
    try:
        if message.author.id == client.user.id or message.id in PROCESSED_MESSAGES:
            return

        if not is_target_message(message):
            if message.webhook_id is not None and message.channel.id in CHANNEL_GROUP_MAP:
                await forward_invalid_message(message, "No God Pack (Faltan desencadenadores).")
            return

        image_attachments = [
            att for att in message.attachments
            if (att.content_type or "").lower().startswith("image/")
            or att.filename.lower().endswith(VALID_IMAGE_EXTENSIONS)
        ]

        if not image_attachments:
            await forward_invalid_message(message, "El mensaje coincide con el formato de GP, pero no contiene una imagen válida adjunta.")
            return

        first_att = image_attachments[0]
        raw_bytes = await first_att.read()
        source_img = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")

        PROCESSED_MESSAGES.add(message.id)
        if len(PROCESSED_MESSAGES) > 1000:
            PROCESSED_MESSAGES.clear()

        group = CHANNEL_GROUP_MAP[message.channel.id]

        # Si el adjunto viene del bot viejo en baja resolución (240x227), se publica directo sin recorte
        is_low_res = (source_img.size == (240, 227)) or (first_att.width == 240 and first_att.height == 227)

        if is_low_res:
            showcase_bytes = raw_bytes
            out_filename = first_att.filename or "gp_original.png"
        else:
            showcase_bytes, _ = await asyncio.to_thread(convert_gp_to_showcase_bytes, source_img)
            out_filename = "gp_showcase.png"

        meta = parse_heartbeat_metadata(message.content or "")
        pack_label = build_pack_label_from_meta(meta)

        await cleanup_expired_vips(group)

        owner_info = await resolve_gp_owner(client, message.content or "", group)
        friend_id = meta.get("game_id") or extract_friend_id(message.content or "")

        if group in ("Elite_Four", "Gym_Leader") and friend_id:
            rival_owner = await resolve_rival_owner_by_game_id(friend_id)
            if rival_owner:
                owner_info = rival_owner

        meta["owner_discord_id"] = owner_info.get("discord_id")
        meta["owner_display_name"] = owner_info.get("display_name")
        meta["owner_mention"] = owner_info.get("mention")
        if friend_id and not meta.get("game_id"):
            meta["game_id"] = friend_id

        if friend_id:
            await add_vip_id(friend_id, group)
        await register_user_gp(owner_info)
        await update_stats_safe(group, increment_gp_callback)

        post_title = build_post_title(meta, pack_label)
        post_data = await create_forum_post_with_image_bytes(
            client,
            group,
            post_title,
            showcase_bytes,
            filename=out_filename,
        )

        post_url = None
        now_ts = int(time.time())

        if post_data and post_data.get("thread"):
            post_thread = post_data["thread"]
            post_url = post_data["jump_url"]
            vote_key = str(post_thread.id)

            online_mentions = await get_online_mentions(group)
            mentions_text = " ".join(online_mentions) if online_mentions else "No active users"

            info_embed = build_forum_info_embed(
                meta,
                pack_label,
                status="none",
                group=group,
                created_at_ts=now_ts,
            )

            vote_data = await load_vote_state(group)
            vote_data[vote_key] = {
                "created_at": now_ts,
                "group": group,
                "owner_discord_id": owner_info.get("discord_id"),
                "friend_id": friend_id,
                "status": "none",
                "alive_users": [],
                "dead_users": [],
                "counted_alive": False,
                "link_message_id": None,
                "link_channel_id": None,
                "post_url": post_url,
                "meta": meta,
                "pack_label": pack_label,
            }
            await save_vote_state(group, vote_data)

            vote_view = GPVoteView(vote_key=vote_key, group=group)
            await post_thread.send(
                content=mentions_text,
                embed=info_embed,
                view=vote_view,
                allowed_mentions=discord.AllowedMentions(users=True),
            )

        view = ForumLinkView(post_url, meta, pack_label) if post_url else None
        main_file = discord.File(io.BytesIO(showcase_bytes), filename=out_filename)
        sent_main = await message.channel.send(file=main_file, view=view)

        # Enviar respaldo al canal de logs (GP Register)
        if LOG_CHANNEL_ID:
            try:
                log_channel = client.get_channel(LOG_CHANNEL_ID) or await client.fetch_channel(LOG_CHANNEL_ID)
                if log_channel:
                    log_files = []
                    for att in message.attachments:
                        if (att.content_type or "").lower().startswith("image/") or att.filename.lower().endswith(VALID_IMAGE_EXTENSIONS):
                            att_bytes = await att.read()
                            log_files.append(discord.File(io.BytesIO(att_bytes), filename=att.filename))

                    log_text = (
                        f"**Registro GP Processed**\n"
                        f"**Found by:** {owner_info.get('mention', 'Unknown')}\n"
                        f"**Bot:** `{meta.get('bot_name')} ({friend_id})`\n"
                        f"**Pack:** `{pack_label}`\n\n"
                        f"**Mensaje original:**\n```{message.content[:1500]}```"
                    )
                    sent_log = await log_channel.send(content=log_text, files=log_files)
                    asyncio.create_task(delete_message_later(sent_log, 172800))
            except Exception as e:
                logger.exception("Error enviando log al registro: %s", e)

        # Borrado del mensaje original
        try:
            await message.delete()
        except Exception:
            pass

        if post_data and sent_main:
            vote_key = str(post_data["thread"].id)
            vote_data = await load_vote_state(group)
            if vote_key in vote_data:
                vote_data[vote_key]["link_message_id"] = sent_main.id
                vote_data[vote_key]["link_channel_id"] = sent_main.channel.id
                await save_vote_state(group, vote_data)

    except Exception as e:
        logger.exception("🔥 Error procesando mensaje GP: %s", e)


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise RuntimeError("Falta la variable DISCORD_TOKEN en Railway")
    client.run(DISCORD_TOKEN)
