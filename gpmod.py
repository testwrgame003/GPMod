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
from PIL import Image

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
# CONFIGURACIÓN GENERAL Y DIMENSIONES LINEALES
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("gp_linear_bot")

LINEAR_HEIGHT = 540
LINEAR_WIDTH_4_CARDS = 1534
LINEAR_WIDTH_5_CARDS = 1914
LINEAR_BG_COLOR = (16, 17, 36)  # #101124

TRIGGER_PATTERNS = [
    re.compile(r"god\s*pack", re.IGNORECASE),
    re.compile(r"\[\d/[45]\]\[\d+P\]\[[^\]]+\]", re.IGNORECASE),
]

VALID_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
PROCESSED_MESSAGES = set()
INVALID_LOG_COOLDOWN = 10
INVALID_LOG_LAST_SENT = {}

CHANNEL_GROUP_MAP = {
    1564890852310913034: "Elite_Four",
    1497179450960379915: "Trainer",
    1534456693126332568: "Gym_Leader",
}

GROUP_CONFIG = {
    "Trainer": {"FORUM_CHANNEL_ID": 1497177430828384396},
    "Gym_Leader": {"FORUM_CHANNEL_ID": 1501347022773686483},
    "Elite_Four": {"FORUM_CHANNEL_ID": 1564934929048019015},
}

group_locks = defaultdict(asyncio.Lock)
http_session: Optional[aiohttp.ClientSession] = None

# =========================================================
# MOTOR DE CONVERSIÓN A FORMATO LINEAL (4 Y 5 CARTAS)
# =========================================================

def is_already_linear_gp(img: Image.Image) -> bool:
    w, h = img.size
    if h == 0:
        return False
    return (w / h) >= 2.1

def detect_and_extract_cards(source_img: Image.Image) -> List[Image.Image]:
    rgb_img = source_img.convert("RGB")
    w, h = rgb_img.size
    arr = np.array(rgb_img, dtype=np.float32)

    corner_samples = np.concatenate([
        arr[int(h * 0.85):int(h * 0.98), int(w * 0.01):int(w * 0.10)].reshape(-1, 3),
        arr[int(h * 0.85):int(h * 0.98), int(w * 0.90):int(w * 0.99)].reshape(-1, 3),
    ], axis=0)
    bg_rgb = np.median(corner_samples, axis=0)

    # Coordenadas ajustadas para evitar cortar los bordes inferiores
    top_y1, top_y2 = 0.035, 0.490
    bot_y1, bot_y2 = 0.510, 0.965

    top_x_boxes = [
        (0.015, 0.320),
        (0.345, 0.655),
        (0.675, 0.985),
    ]

    left_outer_region = arr[int(h * 0.55):int(h * 0.90), int(w * 0.18):int(w * 0.30)]
    right_outer_region = arr[int(h * 0.55):int(h * 0.90), int(w * 0.70):int(w * 0.82)]

    left_diff = float(np.mean(np.linalg.norm(left_outer_region - bg_rgb, axis=2)))
    right_diff = float(np.mean(np.linalg.norm(right_outer_region - bg_rgb, axis=2)))

    is_five_cards = (left_diff > 25.0 and right_diff > 25.0)

    if is_five_cards:
        bot_x_boxes = [
            (0.180, 0.485),
            (0.510, 0.815),
        ]
    else:
        bot_x_boxes = [
            (0.345, 0.655),
        ]

    cards: List[Image.Image] = []

    for x1_rel, x2_rel in top_x_boxes:
        box = (int(round(x1_rel * w)), int(round(top_y1 * h)), int(round(x2_rel * w)), int(round(top_y2 * h)))
        cards.append(rgb_img.crop(box))

    for x1_rel, x2_rel in bot_x_boxes:
        box = (int(round(x1_rel * w)), int(round(bot_y1 * h)), int(round(x2_rel * w)), int(round(bot_y2 * h)))
        cards.append(rgb_img.crop(box))

    return cards


def convert_gp_to_linear_bytes(source_img: Image.Image) -> Tuple[bytes, int, bool]:
    if is_already_linear_gp(source_img):
        buf = io.BytesIO()
        source_img.save(buf, format="PNG")
        w, h = source_img.size
        approx_cards = 4 if (w / max(h, 1)) < 3.15 else 5
        return buf.getvalue(), approx_cards, True

    cards = detect_and_extract_cards(source_img)
    num_cards = len(cards)

    canvas_w = LINEAR_WIDTH_4_CARDS if num_cards == 4 else LINEAR_WIDTH_5_CARDS
    canvas_h = LINEAR_HEIGHT

    margin_x = 15
    margin_y = 15
    gap_x = 14

    target_card_h = canvas_h - (2 * margin_y)
    total_gaps = gap_x * (num_cards - 1)
    target_card_w = (canvas_w - (2 * margin_x) - total_gaps) / num_cards

    canvas = Image.new("RGB", (canvas_w, canvas_h), LINEAR_BG_COLOR)

    for idx, card in enumerate(cards):
        x_pos = int(round(margin_x + idx * (target_card_w + gap_x)))
        x_next = int(round(margin_x + (idx + 1) * target_card_w + idx * gap_x))
        exact_w = x_next - x_pos

        resized_card = card.resize((exact_w, target_card_h), Image.Resampling.LANCZOS)
        canvas.paste(resized_card, (x_pos, margin_y))

    out_buf = io.BytesIO()
    canvas.save(out_buf, format="PNG", optimize=True)
    return out_buf.getvalue(), num_cards, False


# =========================================================
# HELPERS UPSTASH REDIS (SESIÓN PERSISTENTE)
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

def get_group_from_channel(channel_id: int) -> Optional[str]:
    return CHANNEL_GROUP_MAP.get(channel_id)


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


def extract_owner_discord_id_from_first_line(content: str) -> Optional[str]:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return None
    first = lines[0]

    m = re.search(r"<@?!?(\d{17,20})>", first)
    if m:
        return m.group(1)

    m = re.match(r"^@?(\d{17,20})\b", first)
    if m:
        return m.group(1)
    return None


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
        for sess_id, sess in rivals_sessions.items():
            members = (sess or {}).get("members") or []
            for m in members:
                if str(m.get("gameId") or "").strip() == game_id:
                    uid = str(m.get("discordId"))
                    name = m.get("name") or m.get("heartbeatName") or uid
                    return {
                        "discord_id": uid,
                        "display_name": name,
                        "mention": f"<@{uid}>",
                        "game_id": game_id,
                    }
    except Exception:
        pass

    try:
        rival_duos = await redis_hgetall_json("rival_duos")
        for duo_id, duo in rival_duos.items():
            online_users = (duo or {}).get("onlineUsers") or {}
            members = (duo or {}).get("members") or {}
            for uid, m_data in members.items():
                if str(m_data.get("gameId") or "").strip() == game_id and online_users.get(str(uid), False):
                    name = m_data.get("name") or m_data.get("heartbeatName") or uid
                    return {
                        "discord_id": str(uid),
                        "display_name": name,
                        "mention": f"<@{uid}>",
                        "game_id": game_id,
                    }
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
# DISEÑO VISUAL DEL POST DEL FORO (EMBED MEJORADO)
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
        embed.add_field(name="🗳️ Community Votes", value=f"🟢 **Alive ({len(alive_users)}):** {alive_str}\n🔴 **Dead ({len(dead_users)}):** {dead_str}", inline=False)

    embed.set_footer(text=f"File: {filename}")
    return embed


async def create_forum_post_with_image_bytes(
    client: discord.Client,
    group: str,
    title: str,
    image_bytes: bytes,
    filename: str = "gp_linear.png",
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
        packs = meta.get("packs_count", "?")
        bot = meta.get("bot_name", "Bot")
        game_id = meta.get("game_id", "ID")
        owner = meta.get("owner_display_name") or "unknown"
        prefix = "✅" if status == "alive" else ("❌" if status == "dead" else "")
        label = f"{prefix} {owner} | {pack_label} [{packs}P] {bot} [{game_id}]".strip()
        self.add_item(discord.ui.Button(label=label[:80], style=discord.ButtonStyle.link, url=post_url))


# =========================================================
# SISTEMA DE VOTACIÓN ALIVE / DEAD
# =========================================================

async def load_vote_state(group: str) -> dict:
    if group not in GROUP_CONFIG: return {}
    data = await redis_hgetall_json(vote_state_key(group))
    cutoff = time.time() - (72 * 60 * 60)
    return {k: v for k, v in data.items() if v.get("created_at", time.time()) >= cutoff}


async def save_vote_state(group: str, data: dict) -> None:
    if group not in GROUP_CONFIG: return
    for vote_key, state in data.items():
        await redis_hset_json(vote_state_key(group), vote_key, state)


async def update_gp_thread_status(thread_id: int, status: str):
    try:
        thread = client.get_channel(thread_id) or await client.fetch_channel(thread_id)
        if thread is None: return
        clean_name = thread.name.removeprefix("✅ ").removeprefix("❌ ").strip()
        new_name = f"✅ {clean_name}" if status == "alive" else (f"❌ {clean_name}" if status == "dead" else None)
        if new_name: await thread.edit(name=new_name[:100])
    except Exception:
        pass


async def update_main_link_button(state: dict, status: str, meta: dict, pack_label: str):
    try:
        channel_id = state.get("link_channel_id")
        message_id = state.get("link_message_id")
        post_url = state.get("post_url")
        if not channel_id or not message_id or not post_url: return

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

    @discord.ui.button(label="🟢 Alive (0)", style=discord.ButtonStyle.success, custom_id="gp_alive")
    async def alive_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.handle_vote(interaction, "alive")

    @discord.ui.button(label="🔴 Dead (0)", style=discord.ButtonStyle.danger, custom_id="gp_dead")
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

        if vote_type == "alive": state["alive_users"].append(user_id)
        else: state["dead_users"].append(user_id)

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
            if child.custom_id == "gp_alive": child.label = f"🟢 Alive ({alive_count})"
            elif child.custom_id == "gp_dead": child.label = f"🔴 Dead ({dead_count})"
        if status in ("alive", "dead"):
            await update_gp_thread_status(int(self.vote_key), status)
            await update_main_link_button(state, status, state.get("meta", {}), state.get("pack_label", ""))
            for child in self.children: child.disabled = True

        updated_embed = build_forum_info_embed(
            state.get("meta", {}), state.get("pack_label", "[?/5]"), status=status,
            alive_users=state["alive_users"], dead_users=state["dead_users"],
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
                for child in view.children:
                    if child.custom_id == "gp_alive": child.label = f"🟢 Alive ({alive_count})"
                    elif child.custom_id == "gp_dead": child.label = f"🔴 Dead ({dead_count})"
                    if state.get("status") in ("alive", "dead"): child.disabled = True
                client.add_view(view)
                restored += 1
        except Exception:
            pass

@client.event
async def on_ready():
    logger.info("✅ Bot GP Lineal conectado como %s", client.user)
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

        linear_bytes, num_cards, was_already_linear = await asyncio.to_thread(
            convert_gp_to_linear_bytes,
            source_img
        )

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

        if friend_id:
            await add_vip_id(friend_id, group)
        await register_user_gp(owner_info)
        await update_stats_safe(group, increment_gp_callback)

        post_title = build_post_title(meta, pack_label)
        post_data = await create_forum_post_with_image_bytes(
            client,
            group,
            post_title,
            linear_bytes,
            filename="gp_linear.png"
        )

        post_url = None
        if post_data and post_data.get("thread"):
            post_thread = post_data["thread"]
            post_url = post_data["jump_url"]
            vote_key = str(post_thread.id)

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
            }
            await save_vote_state(group, vote_data)

            vote_view = GPVoteView(vote_key=vote_key, group=group)
            await post_thread.send(
                content=mentions_text,
                embed=info_embed,
                view=vote_view,
                allowed_mentions=discord.AllowedMentions(users=True)
            )

        # Enviar la imagen final al canal y luego borrar el mensaje original
        view = ForumLinkView(post_url, meta, pack_label) if post_url else None
        main_file = discord.File(io.BytesIO(linear_bytes), filename="gp_linear.png")

        sent_main = await message.channel.send(file=main_file, view=view)

        # Borrado del mensaje crudo justo después de enviar la versión procesada
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
