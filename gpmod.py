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
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont

# =========================================================
# SELECTOR DE MODO VISUAL ("HUD_LINEAR" o "FAN")
# =========================================================
# Usa "HUD_LINEAR" para el nuevo marco Cyber-Arcoíris en fila
# Usa "FAN" para volver al modo Abanico que guardamos antes
DISPLAY_MODE = "HUD_LINEAR"

# =========================================================
# VARIABLES DE ENTORNO DESDE RAILWAY
# =========================================================

DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN", "")
CLIENT_ID = int(os.environ.get("CLIENT_ID", "0"))
FORUM_CHANNEL_ID = int(os.environ.get("FORUM_CHANNEL_ID", "0"))
LOG_CHANNEL_ID = int(os.environ.get("LOG_CHANNEL_ID", "0"))
INVALID_LOG_CHANNEL_ID = int(os.environ.get("INVALID_LOG_CHANNEL_ID", "0"))
UPSTASH_REDIS_REST_URL = os.environ.get("UPSTASH_REDIS_REST_URL", "").rstrip("/")
UPSTASH_REDIS_REST_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")

# =========================================================
# CONFIGURACIÓN GENERAL Y CACHÉ DE CARTAS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
CARDS_CACHE_DIR = OUTPUT_DIR / "cards_cache"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CARDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("gp_hud_bot")

# Dimensiones del Modo HUD Lineal (Proporción 16:9 panorámica HD que Discord muestra gigante)
HUD_CANVAS_W = 1920
HUD_CANVAS_H = 1040

# Dimensiones del Modo Abanico (Guardado intacto)
FAN_CANVAS_W = 1480
FAN_CANVAS_H = 780
BG_DARK_COLOR = (13, 15, 32)
BG_GLOW_COLOR = (34, 42, 88)

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
    "Trainer": {"FORUM_CHANNEL_ID": 1497177430828384396, "LABEL": "TRAINER"},
    "Gym_Leader": {"FORUM_CHANNEL_ID": 1501347022773686483, "LABEL": "GYM LEADER"},
    "Elite_Four": {"FORUM_CHANNEL_ID": 1497179653276827720, "LABEL": "ELITE FOUR"},
}

group_locks = defaultdict(asyncio.Lock)
http_session: Optional[aiohttp.ClientSession] = None

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
# FUENTES Y UTILIDADES DE ESTILO
# =========================================================

def load_hud_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
        "arialbd.ttf",
        "arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def style_card_with_rounded_corners(card_img: Image.Image, target_w: int, target_h: int, radius: int = 18) -> Image.Image:
    resized = card_img.convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
    sharpened = resized.filter(ImageFilter.UnsharpMask(radius=1.2, percent=115, threshold=2)).convert("RGBA")

    mask = Image.new("L", (target_w, target_h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, target_w - 1, target_h - 1), radius=radius, fill=255)

    out = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    out.paste(sharpened, (0, 0), mask)
    return out


# =========================================================
# MOTOR GRÁFICO 1: NUEVO MODO CYBER-HUD ARCOÍRIS LINEAL
# =========================================================

def create_rainbow_gradient(width: int, height: int, alpha: int = 255) -> Image.Image:
    """Genera una textura horizontal de degradado arcoíris suave (Cian -> Violeta -> Rosa -> Dorado -> Esmeralda -> Cian)."""
    x = np.linspace(0, 1, width, dtype=np.float32)
    # Fases desplazadas para crear un espectro holográfico vibrante y continuo
    r = np.clip(128 + 127 * np.sin(2 * np.pi * (x * 1.15 + 0.00)), 30, 255).astype(np.uint8)
    g = np.clip(128 + 127 * np.sin(2 * np.pi * (x * 1.15 + 0.33)), 40, 255).astype(np.uint8)
    b = np.clip(128 + 127 * np.sin(2 * np.pi * (x * 1.15 + 0.67)), 60, 255).astype(np.uint8)

    row = np.stack([r, g, b, np.full_like(r, alpha)], axis=1)
    arr = np.broadcast_to(row[np.newaxis, :, :], (height, width, 4)).copy()
    return Image.fromarray(arr, mode="RGBA")


def build_cyber_rainbow_hud_image(
    cards: List[Image.Image],
    meta: Optional[dict] = None,
    group: str = "Elite_Four",
    pack_label: str = "[?/5]",
) -> Image.Image:
    """
    Construye el marco Cyber-HUD con acentos arcoíris holográficos, partículas de luz
    y las 4 o 5 cartas alineadas en fila con aura neón individual.
    """
    meta = meta or {}
    W, H = HUD_CANVAS_W, HUD_CANVAS_H
    num_cards = len(cards)

    # 1. Fondo espacial profundo + Nebulosa Arcoíris difuminada en el centro
    base = Image.new("RGBA", (W, H), (8, 10, 22, 255))
    rainbow_full = create_rainbow_gradient(W, H, alpha=255)

    nebula_mask = Image.new("L", (W, H), 0)
    n_draw = ImageDraw.Draw(nebula_mask)
    n_draw.ellipse((int(W * 0.06), int(H * 0.14), int(W * 0.94), int(H * 0.86)), fill=68)
    nebula_mask = nebula_mask.filter(ImageFilter.GaussianBlur(radius=110))
    base.paste(rainbow_full, (0, 0), nebula_mask)

    # 2. Cuadrícula tecnológica sutil y partículas de polvo holográfico
    grid_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    g_draw = ImageDraw.Draw(grid_layer)
    for gx in range(80, W - 80, 80):
        g_draw.line([(gx, 130), (gx, H - 130)], fill=(90, 120, 200, 11), width=1)
    for gy in range(160, H - 120, 80):
        g_draw.line([(80, gy), (W - 80, gy)], fill=(90, 120, 200, 11), width=1)

    rng = np.random.default_rng(seed=2026)
    spark_colors = [
        (0, 245, 255, 150),
        (255, 70, 220, 150),
        (255, 215, 60, 150),
        (80, 255, 170, 150),
    ]
    for _ in range(85):
        px = int(rng.integers(55, W - 55))
        py = int(rng.integers(125, H - 125))
        pr = int(rng.integers(1, 4))
        col = spark_colors[int(rng.integers(0, len(spark_colors)))]
        g_draw.ellipse((px - pr, py - pr, px + pr, py + pr), fill=col)

    base = Image.alpha_composite(base, grid_layer)

    # 3. Paneles oscuros translúcidos del HUD (Header, Centro y Footer)
    hud_dark = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    h_draw = ImageDraw.Draw(hud_dark)

    # Marco exterior general
    h_draw.rounded_rectangle((24, 22, W - 24, H - 22), radius=26, fill=(10, 13, 28, 175))

    # Trapecio central del título superior
    top_center_poly = [(470, 22), (1450, 22), (1385, 132), (535, 132)]
    h_draw.polygon(top_center_poly, fill=(14, 18, 38, 240))

    # Placas laterales superiores (Izquierda: Pack Name | Derecha: Pack Ratio)
    left_badge_poly = [(56, 52), (455, 52), (495, 118), (56, 118)]
    right_badge_poly = [(1465, 52), (W - 56, 52), (W - 56, 118), (1425, 118)]
    h_draw.polygon(left_badge_poly, fill=(18, 24, 52, 225))
    h_draw.polygon(right_badge_poly, fill=(18, 24, 52, 225))

    # Trapecio central inferior (Friend ID)
    bot_center_poly = [(580, H - 22), (1340, H - 22), (1285, H - 110), (635, H - 110)]
    h_draw.polygon(bot_center_poly, fill=(14, 18, 38, 240))

    # Placas laterales inferiores (Izquierda: Found By | Derecha: Status Pill)
    bot_left_poly = [(56, H - 102), (560, H - 102), (595, H - 40), (56, H - 40)]
    h_draw.polygon(bot_left_poly, fill=(18, 24, 52, 215))

    base = Image.alpha_composite(base, hud_dark)

    # 4. Trazado de Líneas Neón Arcoíris del HUD (con resplandor doble)
    stroke_mask = Image.new("L", (W, H), 0)
    s_draw = ImageDraw.Draw(stroke_mask)

    # Borde perimetral y biseles superiores/inferiores
    s_draw.rounded_rectangle((24, 22, W - 24, H - 22), radius=26, outline=255, width=4)
    s_draw.line([(470, 22), (535, 132), (1385, 132), (1450, 22)], fill=255, width=5)
    s_draw.polygon(left_badge_poly, outline=220, width=3)
    s_draw.polygon(right_badge_poly, outline=220, width=3)
    s_draw.line([(580, H - 22), (635, H - 110), (1285, H - 110), (1340, H - 22)], fill=255, width=4)
    s_draw.polygon(bot_left_poly, outline=200, width=2)

    # Barras de acento diagonales estilo Cyber en las 4 esquinas
    for i in range(4):
        offset = i * 26
        # Arriba izquierda
        s_draw.polygon([(95 + offset, 24), (112 + offset, 24), (98 + offset, 44), (81 + offset, 44)], fill=255)
        # Arriba derecha
        s_draw.polygon([(W - 95 - offset, 24), (W - 112 - offset, 24), (W - 98 - offset, 44), (W - 81 - offset, 44)], fill=255)
        # Abajo centro-izquierda y centro-derecha
        s_draw.polygon([(505 + offset, H - 24), (520 + offset, H - 24), (508 + offset, H - 40), (493 + offset, H - 40)], fill=255)
        s_draw.polygon([(W - 505 - offset, H - 24), (W - 520 - offset, H - 24), (W - 508 - offset, H - 40), (W - 493 - offset, H - 40)], fill=255)

    # Soportes de esquina internos (Targeting Brackets) alrededor de la fila de cartas
    bx1, by1, bx2, by2 = 64, 158, W - 64, H - 138
    blen = 42
    # Sup-Izq
    s_draw.line([(bx1, by1 + blen), (bx1, by1), (bx1 + blen, by1)], fill=210, width=3)
    # Sup-Der
    s_draw.line([(bx2 - blen, by1), (bx2, by1), (bx2, by1 + blen)], fill=210, width=3)
    # Inf-Izq
    s_draw.line([(bx1, by2 - blen), (bx1, by2), (bx1 + blen, by2)], fill=210, width=3)
    # Inf-Der
    s_draw.line([(bx2 - blen, by2), (bx2, by2), (bx2, by2 - blen)], fill=210, width=3)

    # Aplicar resplandor neón (Blur) + línea nítida encima usando el gradiente arcoíris
    glow_mask = stroke_mask.filter(ImageFilter.GaussianBlur(radius=8))
    base.paste(rainbow_full, (0, 0), glow_mask)
    base.paste(rainbow_full, (0, 0), stroke_mask)

    # 5. Renderizar las 4 o 5 cartas en Fila con Aura Arcoíris individual y marco brillante
    if num_cards == 4:
        card_w, card_h = 390, 544
        gap_x = 34
    else:
        card_w, card_h = 332, 463
        gap_x = 24

    total_row_w = (num_cards * card_w) + ((num_cards - 1) * gap_x)
    start_x = (W - total_row_w) // 2
    center_y = 515
    top_y = center_y - (card_h // 2)

    # Colores de aura individual para cada slot (Arcoíris: Rosa -> Dorado -> Verde -> Cian -> Violeta)
    slot_aura_colors = [
        (255, 75, 195),   # Slot 1: Magenta / Rosa neón
        (255, 195, 55),   # Slot 2: Dorado cálido
        (75, 255, 165),   # Slot 3: Verde esmeralda
        (0, 225, 255),    # Slot 4: Cian eléctrico
        (185, 95, 255),   # Slot 5: Violeta / Púrpura
    ]

    aura_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    a_draw = ImageDraw.Draw(aura_layer)

    styled_cards: List[Tuple[int, int, Image.Image, Tuple[int, int, int]]] = []
    for idx, card in enumerate(cards):
        cx1 = start_x + idx * (card_w + gap_x)
        cy1 = top_y
        col = slot_aura_colors[idx % len(slot_aura_colors)]

        # Halo luminoso detrás de cada carta
        a_draw.rounded_rectangle(
            (cx1 - 10, cy1 - 10, cx1 + card_w + 10, cy1 + card_h + 10),
            radius=24,
            fill=(*col, 135),
        )
        styled = style_card_with_rounded_corners(card, card_w, card_h, radius=16)
        styled_cards.append((cx1, cy1, styled, col))

    aura_layer = aura_layer.filter(ImageFilter.GaussianBlur(radius=18))
    base = Image.alpha_composite(base, aura_layer)

    # Pegar cada carta y su bisel holográfico fino
    border_draw = ImageDraw.Draw(base)
    for cx1, cy1, styled, col in styled_cards:
        base.alpha_composite(styled, (cx1, cy1))
        border_draw.rounded_rectangle(
            (cx1 - 2, cy1 - 2, cx1 + card_w + 1, cy1 + card_h + 1),
            radius=18,
            outline=(*col, 235),
            width=3,
        )

    # 6. Textos del HUD (Header y Footer igual a tu imagen de referencia)
    text_draw = ImageDraw.Draw(base)
    font_title = load_hud_font(48, bold=True)
    font_sub = load_hud_font(26, bold=True)
    font_id = load_hud_font(29, bold=True)

    group_label = GROUP_CONFIG.get(group, {}).get("LABEL", group.replace("_", " ").upper())
    title_text = f"{group_label} GOD PACK"

    # Centrar título superior con sombra cian
    tb = text_draw.textbbox((0, 0), title_text, font=font_title)
    tx = (W - (tb[2] - tb[0])) // 2
    ty = 48
    text_draw.text((tx + 2, ty + 3), title_text, font=font_title, fill=(0, 210, 255, 110))
    text_draw.text((tx, ty), title_text, font=font_title, fill=(255, 255, 255, 255))

    # Texto superior izquierdo: [XP] PACK NAME
    packs_count = meta.get("packs_count")
    packs_str = f"[{packs_count}P]" if packs_count is not None else "[1P]"
    pack_name_raw = (meta.get("pack_name") or "DELUXE PACK").upper()
    if "PACK" not in pack_name_raw:
        pack_name_raw += " PACK"
    left_top_text = f"{packs_str} {pack_name_raw}"[:24]
    text_draw.text((82, 69), left_top_text, font=font_sub, fill=(235, 240, 255, 255))

    # Texto superior derecho: 4/5 GOD PACK (o 4/4 DELUXE)
    clean_ratio = pack_label.strip("[]")
    right_top_text = f"{clean_ratio}  2-STAR GP"
    rb = text_draw.textbbox((0, 0), right_top_text, font=font_sub)
    text_draw.text((W - 82 - (rb[2] - rb[0]), 69), right_top_text, font=font_sub, fill=(255, 225, 110, 255))

    # Texto inferior izquierdo: FOUND BY: USERNAME
    owner_name = (meta.get("owner_display_name") or meta.get("obtainer_user") or "UNKNOWN").upper()
    found_text = f"FOUND BY: {owner_name}"[:28]
    text_draw.text((82, H - 86), found_text, font=font_sub, fill=(240, 245, 255, 255))

    # Texto inferior central: ID: 16 DIGITS
    game_id = meta.get("game_id") or "0000000000000000"
    id_text = f"ID: {game_id}"
    ib = text_draw.textbbox((0, 0), id_text, font=font_id)
    ix = (W - (ib[2] - ib[0])) // 2
    text_draw.text((ix, H - 84), id_text, font=font_id, fill=(255, 255, 255, 255))

    # Botón/Insignia inferior derecha: ALIVE (Estilo neón verde)
    pill_x1, pill_y1, pill_x2, pill_y2 = W - 290, H - 102, W - 64, H - 40
    border_draw.rounded_rectangle((pill_x1, pill_y1, pill_x2, pill_y2), radius=20, fill=(14, 52, 34, 230), outline=(46, 255, 138, 255), width=3)
    status_str = "ALIVE"
    sb = text_draw.textbbox((0, 0), status_str, font=font_id)
    sx = pill_x1 + ((pill_x2 - pill_x1) - (sb[2] - sb[0])) // 2
    sy = pill_y1 + ((pill_y2 - pill_y1) - (sb[3] - sb[1])) // 2 - 3
    text_draw.text((sx, sy), status_str, font=font_id, fill=(110, 255, 175, 255))

    return base.convert("RGB")


# =========================================================
# MOTOR GRÁFICO 2: MODO ABANICO (GUARDADO INTACTO)
# =========================================================

def create_studio_background(width: int, height: int) -> Image.Image:
    bg = Image.new("RGBA", (width, height), (*BG_DARK_COLOR, 255))
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow)

    cx, cy = width // 2, height // 2
    max_rx, max_ry = int(width * 0.48), int(height * 0.48)
    for step in range(6, 0, -1):
        ratio = step / 6.0
        rx, ry = int(max_rx * ratio), int(max_ry * ratio)
        alpha = int(38 * (1.0 - ratio + 0.15))
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=(*BG_GLOW_COLOR, alpha))

    glow = glow.filter(ImageFilter.GaussianBlur(radius=45))
    return Image.alpha_composite(bg, glow)


def paste_rotated_card_with_shadow(
    canvas: Image.Image,
    styled_card: Image.Image,
    angle: float,
    center_xy: Tuple[int, int],
    shadow_offset: Tuple[int, int] = (-12, 16),
    shadow_blur: int = 16,
    shadow_opacity: int = 175
) -> None:
    pad = 45
    w, h = styled_card.size
    padded_card = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    padded_card.paste(styled_card, (pad, pad))

    alpha = padded_card.split()[3]
    shadow_alpha = alpha.point(lambda p: int(p * (shadow_opacity / 255.0)))
    shadow_img = Image.new("RGBA", padded_card.size, (0, 0, 0, 0))
    shadow_img.putalpha(shadow_alpha)

    rot_card = padded_card.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    rot_shadow = shadow_img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    rot_shadow = rot_shadow.filter(ImageFilter.GaussianBlur(radius=shadow_blur))

    rw, rh = rot_card.size
    cx, cy = center_xy
    top_left_x = cx - (rw // 2)
    top_left_y = cy - (rh // 2)

    canvas.alpha_composite(rot_shadow, (top_left_x + shadow_offset[0], top_left_y + shadow_offset[1]))
    canvas.alpha_composite(rot_card, (top_left_x, top_left_y))


def build_fan_image(cards: List[Image.Image]) -> Image.Image:
    num_cards = len(cards)
    canvas = create_studio_background(FAN_CANVAS_W, FAN_CANVAS_H)

    if num_cards == 4:
        card_w, card_h = 435, 608
        angles = [12.0, 4.0, -4.0, -12.0]
        x_centers = [325, 595, 885, 1155]
        y_offsets = [38, 0, 0, 38]
        base_y = 385
    else:
        card_w, card_h = 415, 580
        angles = [15.0, 7.5, 0.0, -7.5, -15.0]
        x_centers = [265, 502, 740, 978, 1215]
        y_offsets = [54, 14, 0, 14, 54]
        base_y = 378

    for idx, card in enumerate(cards):
        styled = style_card_with_rounded_corners(card, card_w, card_h, radius=18)
        angle = angles[idx] if idx < len(angles) else 0.0
        cx = x_centers[idx] if idx < len(x_centers) else (FAN_CANVAS_W // 2)
        cy = base_y + (y_offsets[idx] if idx < len(y_offsets) else 0)
        paste_rotated_card_with_shadow(canvas, styled, angle, (cx, cy))

    return canvas.convert("RGB")


# =========================================================
# INSPECTOR INTERACTIVO DE CARTAS (ZOOM IN-GAME)
# =========================================================

def build_inspected_card_image(cards: List[Image.Image], selected_index: int) -> bytes:
    """
    Muestra la carta seleccionada gigante en el centro con el diseño actual desenfocado detrás.
    """
    if DISPLAY_MODE == "HUD_LINEAR":
        bg_base = build_cyber_rainbow_hud_image(cards).resize((FAN_CANVAS_W, FAN_CANVAS_H), Image.Resampling.LANCZOS).convert("RGBA")
    else:
        bg_base = build_fan_image(cards).convert("RGBA")

    darkened = ImageEnhance.Brightness(bg_base).enhance(0.30)
    blurred_bg = darkened.filter(ImageFilter.GaussianBlur(radius=10))

    target_h = 710
    target_w = int(round(target_h * (365.5 / 510.0)))
    selected_card = cards[selected_index]
    styled_giant = style_card_with_rounded_corners(selected_card, target_w, target_h, radius=22)

    cx, cy = FAN_CANVAS_W // 2, FAN_CANVAS_H // 2
    glow_layer = Image.new("RGBA", (FAN_CANVAS_W, FAN_CANVAS_H), (0, 0, 0, 0))
    g_draw = ImageDraw.Draw(glow_layer)
    gx1, gy1 = cx - target_w // 2 - 14, cy - target_h // 2 - 14
    gx2, gy2 = cx + target_w // 2 + 14, cy + target_h // 2 + 14
    g_draw.rounded_rectangle((gx1, gy1, gx2, gy2), radius=28, fill=(0, 235, 255, 130))
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=22))

    blurred_bg = Image.alpha_composite(blurred_bg, glow_layer)
    paste_rotated_card_with_shadow(
        blurred_bg,
        styled_giant,
        angle=0.0,
        center_xy=(cx, cy),
        shadow_offset=(0, 20),
        shadow_blur=24,
        shadow_opacity=220
    )

    out_buf = io.BytesIO()
    blurred_bg.convert("RGB").save(out_buf, format="PNG", optimize=True)
    return out_buf.getvalue()


def render_gp_showcase_bytes(
    source_img: Image.Image,
    meta: dict,
    group: str,
    pack_label: str,
) -> Tuple[bytes, List[Image.Image]]:
    """
    Genera los bytes PNG finales según DISPLAY_MODE ("HUD_LINEAR" o "FAN")
    y devuelve también la lista de cartas recortadas para el caché del Zoom.
    """
    cards = detect_and_extract_cards(source_img)
    if DISPLAY_MODE == "HUD_LINEAR":
        final_img = build_cyber_rainbow_hud_image(cards, meta=meta, group=group, pack_label=pack_label)
    else:
        final_img = build_fan_image(cards)

    out_buf = io.BytesIO()
    final_img.save(out_buf, format="PNG", optimize=True)
    return out_buf.getvalue(), cards


def save_cards_to_cache(cache_id: str, cards: List[Image.Image]) -> None:
    folder = CARDS_CACHE_DIR / str(cache_id)
    folder.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(cards):
        c.save(folder / f"card_{i}.png", format="PNG")

    now = time.time()
    for sub in CARDS_CACHE_DIR.iterdir():
        if sub.is_dir() and (now - sub.stat().st_mtime) > (72 * 3600):
            for f in sub.glob("*"):
                f.unlink(missing_ok=True)
            sub.rmdir()


def load_cards_from_cache(cache_id: str) -> Optional[List[Image.Image]]:
    folder = CARDS_CACHE_DIR / str(cache_id)
    if not folder.exists():
        return None
    files = sorted(folder.glob("card_*.png"))
    if not files:
        return None
    return [Image.open(f).convert("RGB") for f in files]


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
# DISEÑO VISUAL DEL POST DEL FORO + BOTONES DE ZOOM E HILO
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
) -> discord.Embed:
    alive_users = alive_users or []
    dead_users = dead_users or []

    if status == "alive":
        color, status_badge = 0x2ECC71, "🟢 ALIVE (Verified)"
    elif status == "dead":
        color, status_badge = 0xE74C3C, "🔴 DEAD (Expired)"
    else:
        color, status_badge = 0xF1C40F, "⏳ PENDING TEST"

    obtainer_mention = meta.get("owner_mention") or meta.get("owner_display_name") or "Unknown"
    bot_name = meta.get("bot_name") or "UnknownBot"
    game_id = meta.get("game_id") or "UnknownID"
    packs_count = meta.get("packs_count")
    packs_text = f"{packs_count}P" if packs_count is not None else "?P"
    pack_name = meta.get("pack_name") or "DeluxeB"
    filename = meta.get("filename") or "unknown_file.xml"

    embed = discord.Embed(title=f"✨ GOD PACK {pack_label} [{packs_text}] — {pack_name}", color=color)
    embed.add_field(name="👤 Found By", value=f"{obtainer_mention}", inline=True)
    embed.add_field(name="🤖 Instance", value=f"`{bot_name}`", inline=True)
    embed.add_field(name="📡 Status", value=f"**{status_badge}**", inline=True)
    embed.add_field(name="🆔 Friend ID (Tap to Copy)", value=f"```\n{game_id}\n```", inline=False)

    if alive_users or dead_users:
        alive_str = ", ".join(f"<@{u}>" for u in alive_users) if alive_users else "None"
        dead_str = ", ".join(f"<@{u}>" for u in dead_users) if dead_users else "None"
        embed.add_field(
            name="🗳️ Community Votes",
            value=f"🟢 **Alive ({len(alive_users)}):** {alive_str}\n🔴 **Dead ({len(dead_users)}):** {dead_str}",
            inline=False,
        )

    embed.set_footer(text=f"💡 Tip: Click 🔍 1-5 below to inspect any card in full HD | File: {filename}")
    return embed


async def create_forum_post_with_image_bytes(
    client: discord.Client,
    group: str,
    title: str,
    image_bytes: bytes,
    filename: str = "gp_hud.png",
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


class CardZoomButton(discord.ui.Button):
    def __init__(self, cache_id: str, card_index: int):
        super().__init__(
            label=f"{card_index + 1}",
            emoji="🔍",
            style=discord.ButtonStyle.secondary,
            custom_id=f"zoom:{cache_id}:{card_index}",
            row=1,
        )
        self.cache_id = cache_id
        self.card_index = card_index

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        cards = load_cards_from_cache(self.cache_id)
        if not cards or self.card_index >= len(cards):
            return await interaction.followup.send(
                "❌ Card inspection cache expired for this pack.",
                ephemeral=True,
            )

        zoomed_bytes = await asyncio.to_thread(build_inspected_card_image, cards, self.card_index)
        file = discord.File(io.BytesIO(zoomed_bytes), filename=f"card_{self.card_index + 1}_zoom.png")

        switch_view = discord.ui.View(timeout=180)
        for idx in range(len(cards)):
            btn = CardZoomButton(self.cache_id, idx)
            btn.row = 0
            if idx == self.card_index:
                btn.style = discord.ButtonStyle.primary
            switch_view.add_item(btn)

        await interaction.followup.send(
            content=f"🔎 **Inspecting Card #{self.card_index + 1} of {len(cards)}**",
            file=file,
            view=switch_view,
            ephemeral=True,
        )


class ForumLinkView(discord.ui.View):
    def __init__(self, post_url: str, meta: dict, pack_label: str, cache_id: str = "", num_cards: int = 5, status: str = "none"):
        super().__init__(timeout=None)
        packs = meta.get("packs_count", "?")
        bot = meta.get("bot_name", "Bot")
        game_id = meta.get("game_id", "ID")
        owner = meta.get("owner_display_name") or "unknown"
        prefix = "✅" if status == "alive" else ("❌" if status == "dead" else "")
        label = f"{prefix} {owner} | {pack_label} [{packs}P] {bot} [{game_id}]".strip()

        self.add_item(discord.ui.Button(label=label[:80], style=discord.ButtonStyle.link, url=post_url, row=0))

        if cache_id:
            for i in range(num_cards):
                self.add_item(CardZoomButton(cache_id, i))


# =========================================================
# SISTEMA DE VOTACIÓN ALIVE / DEAD + ZOOM EN FORO
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
        cache_id = state.get("cache_id", "")
        num_cards = int(state.get("num_cards", 5))
        await msg.edit(view=ForumLinkView(post_url, meta, pack_label, cache_id=cache_id, num_cards=num_cards, status=status))
    except Exception:
        pass


class GPVoteView(discord.ui.View):
    def __init__(self, vote_key: str, group: str, num_cards: int = 5):
        super().__init__(timeout=None)
        self.vote_key = vote_key
        self.group = group

        for i in range(num_cards):
            self.add_item(CardZoomButton(vote_key, i))

    @discord.ui.button(label="🟢 Alive (0)", style=discord.ButtonStyle.success, custom_id="gp_alive", row=0)
    async def alive_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.handle_vote(interaction, "alive")

    @discord.ui.button(label="🔴 Dead (0)", style=discord.ButtonStyle.danger, custom_id="gp_dead", row=0)
    async def dead_button(self, interaction: discord.Interaction, button: discord.ui.Button):
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

        for child in self.children:
            if getattr(child, "custom_id", None) == "gp_alive":
                child.label = f"🟢 Alive ({alive_count})"
                if status in ("alive", "dead"):
                    child.disabled = True
            elif getattr(child, "custom_id", None) == "gp_dead":
                child.label = f"🔴 Dead ({dead_count})"
                if status in ("alive", "dead"):
                    child.disabled = True

        if status in ("alive", "dead"):
            await update_gp_thread_status(int(self.vote_key), status)
            await update_main_link_button(state, status, state.get("meta", {}), state.get("pack_label", ""))

        updated_embed = build_forum_info_embed(
            state.get("meta", {}),
            state.get("pack_label", "[?/5]"),
            status=status,
            alive_users=state["alive_users"],
            dead_users=state["dead_users"],
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
                num_c = int(state.get("num_cards", 5))
                view = GPVoteView(vote_key=str(vote_key), group=group, num_cards=num_c)
                alive_count = len(state.get("alive_users", []))
                dead_count = len(state.get("dead_users", []))
                for child in view.children:
                    if getattr(child, "custom_id", None) == "gp_alive":
                        child.label = f"🟢 Alive ({alive_count})"
                        if state.get("status") in ("alive", "dead"):
                            child.disabled = True
                    elif getattr(child, "custom_id", None) == "gp_dead":
                        child.label = f"🔴 Dead ({dead_count})"
                        if state.get("status") in ("alive", "dead"):
                            child.disabled = True
                client.add_view(view)
                restored += 1
        except Exception:
            pass
    logger.info("✅ Views restauradas: %s", restored)


@client.event
async def on_ready():
    logger.info("✅ Bot GP (%s) conectado como %s", DISPLAY_MODE, client.user)
    await restore_persistent_views()


@client.event
async def on_message(message: discord.Message):
    try:
        if message.author.id == client.user.id or message.id in PROCESSED_MESSAGES:
            return

        if message.webhook_id is None or message.channel.id not in CHANNEL_GROUP_MAP:
            return

        image_attachments = [
            att for att in message.attachments
            if (att.content_type or "").lower().startswith("image/")
            or att.filename.lower().endswith(VALID_IMAGE_EXTENSIONS)
        ]

        if not image_attachments:
            return

        first_att = image_attachments[0]
        raw_bytes = await first_att.read()
        source_img = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")

        PROCESSED_MESSAGES.add(message.id)
        if len(PROCESSED_MESSAGES) > 1000:
            PROCESSED_MESSAGES.clear()

        group = CHANNEL_GROUP_MAP[message.channel.id]
        meta = parse_heartbeat_metadata(message.content or "")

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

        pack_label = build_pack_label_from_meta(meta)

        # Generar la imagen según DISPLAY_MODE ("HUD_LINEAR" o "FAN") con los metadatos del usuario
        showcase_bytes, extracted_cards = await asyncio.to_thread(
            render_gp_showcase_bytes,
            source_img,
            meta,
            group,
            pack_label,
        )
        num_cards = len(extracted_cards)

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
            filename="gp_showcase.png",
        )

        post_url = None
        cache_id = str(message.id)

        if post_data and post_data.get("thread"):
            post_thread = post_data["thread"]
            post_url = post_data["jump_url"]
            vote_key = str(post_thread.id)
            cache_id = vote_key

            await asyncio.to_thread(save_cards_to_cache, cache_id, extracted_cards)

            online_mentions = await get_online_mentions(group)
            mentions_text = " ".join(online_mentions) if online_mentions else "No active users"

            info_embed = build_forum_info_embed(meta, pack_label, status="none")

            vote_data = await load_vote_state(group)
            vote_data[vote_key] = {
                "created_at": time.time(),
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
                "cache_id": cache_id,
                "num_cards": num_cards,
            }
            await save_vote_state(group, vote_data)

            vote_view = GPVoteView(vote_key=vote_key, group=group, num_cards=num_cards)
            await post_thread.send(
                content=mentions_text,
                embed=info_embed,
                view=vote_view,
                allowed_mentions=discord.AllowedMentions(users=True),
            )
        else:
            await asyncio.to_thread(save_cards_to_cache, cache_id, extracted_cards)

        view = (
            ForumLinkView(post_url, meta, pack_label, cache_id=cache_id, num_cards=num_cards)
            if post_url
            else None
        )
        main_file = discord.File(io.BytesIO(showcase_bytes), filename="gp_showcase.png")
        sent_main = await message.channel.send(file=main_file, view=view)

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
