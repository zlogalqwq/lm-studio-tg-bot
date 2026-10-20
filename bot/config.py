import logging
import os
import sys
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN: str = os.environ["TELEGRAM_TOKEN"]
LM_STUDIO_BASE_URL: str = os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY: str = os.environ.get("LM_STUDIO_API_KEY", "lm-studio")

# ── Промпты по умолчанию для изображений без подписи ──
DEFAULT_IMAGE_PROMPT_SINGLE: str = os.environ.get(
    "DEFAULT_IMAGE_PROMPT_SINGLE",
    "Опиши это изображение максимально подробно. Перечисли все объекты, "
    "людей, их действия и позы. Опиши обстановку, фон, цвета, освещение. "
    "Если на изображении есть текст, надписи, числа или код — перепиши "
    "их дословно. Если это скриншот — опиши интерфейс. Если это "
    "документ — перепиши содержимое.",
)
DEFAULT_IMAGE_PROMPT_MULTIPLE: str = os.environ.get(
    "DEFAULT_IMAGE_PROMPT_MULTIPLE",
    "На изображениях {n} картинок. Опиши каждое максимально подробно: "
    "объекты, людей, действия, обстановку, надписи. Если между ними "
    "есть связь (серия, один объект с разных ракурсов, до/после) — "
    "отметь это. Если есть текст — перепиши дословно.",
)

# ── Альбомы ──
# 1.0 с — компромисс: альбомы из 5–10 фото доезжают целиком,
# а одиночное фото не заставляет пользователя ждать.
MEDIA_GROUP_LATENCY: float = float(os.environ.get("MEDIA_GROUP_LATENCY", "1.0"))
# Жёсткий потолок: не ждём бесконечно, даже если фото продолжают приходить.
MEDIA_GROUP_MAX_WAIT: float = float(os.environ.get("MEDIA_GROUP_MAX_WAIT", "5.0"))

# ── Нативный API LM Studio (не OpenAI-совместимый) ──
LM_STUDIO_NATIVE_BASE_URL: str = os.environ.get(
    "LM_STUDIO_NATIVE_BASE_URL", "http://localhost:1234/api/v1"
)
LM_STUDIO_API_TOKEN: str = os.environ.get("LM_STUDIO_API_TOKEN", "")

DEFAULT_MODEL: Optional[str] = os.environ.get("LM_STUDIO_MODEL") or None

MAX_HISTORY_TURNS: int = int(os.environ.get("MAX_HISTORY_TURNS", "15"))
MAX_IMAGES_EXTENDED: int = int(os.environ.get("MAX_IMAGES_EXTENDED", "10"))

MAX_IMAGE_BYTES: int = int(
    os.environ.get("MAX_IMAGE_BYTES", str(20 * 1024 * 1024))
)
MAX_TOTAL_IMAGE_BYTES: int = int(
    os.environ.get("MAX_TOTAL_IMAGE_BYTES", str(20 * 1024 * 1024))
)

LM_STUDIO_TIMEOUT: float = float(os.environ.get("LM_STUDIO_TIMEOUT", "120"))
CACHE_TTL: float = 30.0
CHAT_LOG_FILE: str = os.environ.get("CHAT_LOG_FILE", "chat_log.jsonl")

PHOTO_DIR: str = os.environ.get("PHOTO_DIR", "saved_photos")

AUTH_PASSWORD: str = os.environ.get("AUTH_PASSWORD", "")
AUTH_FILE: str = os.environ.get("AUTH_FILE", "authorized_users.json")
TRACKED_FILE: str = os.environ.get("TRACKED_FILE", "tracked_messages.json")
MAX_INPUT_CHARS: int = int(os.environ.get("MAX_INPUT_CHARS", "4000"))

STREAM_EDIT_INTERVAL: float = float(os.environ.get("STREAM_EDIT_INTERVAL", "0.9"))
STREAM_PREVIEW_LIMIT: int = int(os.environ.get("STREAM_PREVIEW_LIMIT", "2500"))
STREAM_CURSOR: str = " ▍"

# ── Антиспам команд ──
COMMAND_COOLDOWN: float = float(os.environ.get("COMMAND_COOLDOWN", "10.0"))

COMMAND_COOLDOWN_EXEMPT: frozenset[str] = frozenset(
    c.strip().lower()
    for c in os.environ.get(
        "COMMAND_COOLDOWN_EXEMPT", "start,help,sudo,sudologout"
    ).split(",")
    if c.strip()
)

# ── Параметры генерации по умолчанию ──
DEFAULT_TEMPERATURE: float = float(os.environ.get("DEFAULT_TEMPERATURE", "0.9"))
DEFAULT_TOP_K: int = int(os.environ.get("DEFAULT_TOP_K", "20"))
DEFAULT_TOP_P: float = float(os.environ.get("DEFAULT_TOP_P", "0.95"))
DEFAULT_CONTEXT_LIMIT: int = int(os.environ.get("DEFAULT_CONTEXT_LIMIT", "22000"))
DEFAULT_MAX_TOKENS: int = int(os.environ.get("DEFAULT_MAX_TOKENS", "4096"))

# ── Резюме диалога ──
SUMMARY_FILE: str = os.environ.get("SUMMARY_FILE", "summaries.json")

# ── Пороги использования контекста ──
CONTEXT_WARN_RATIO: float = float(os.environ.get("CONTEXT_WARN_RATIO", "0.80"))

# ── Анимация ожидания ──
THINKING_INTERVAL: float = float(os.environ.get("THINKING_INTERVAL", "1.4"))

# ── Кастомный системный промпт ──
SYSTEM_PROMPT_FILE: str = os.environ.get("SYSTEM_PROMPT_FILE", "system_prompts.json")
MAX_SYSTEM_PROMPT_CHARS: int = int(os.environ.get("MAX_SYSTEM_PROMPT_CHARS", "4000"))

# ── Инструменты (MCP) ──
MCP_JSON_PATH: str = os.environ.get(
    "MCP_JSON_PATH", os.path.expanduser("~/.lmstudio/mcp.json")
)


def setup_logging() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        force=True,
    )


def _parse_ids(raw: str) -> frozenset[int]:
    if not raw:
        return frozenset()
    out: set[int] = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            pass
    return frozenset(out)


# ── Админ-доступ ──
ADMIN_PASSWORD: str = os.environ.get("ADMIN_PASSWORD", "")
ADMIN_SESSION_TTL: float = float(os.environ.get("ADMIN_SESSION_TTL", "3600"))
ADMIN_IDS: frozenset[int] = _parse_ids(os.environ.get("ADMIN_IDS", ""))

# ── Дампы крашей ──
CRASH_DUMP_DIR: str = os.environ.get("CRASH_DUMP_DIR", "crash_dump")