from aiogram import Bot, Dispatcher, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
import httpx

from . import config


# ── Нативный API LM Studio ──
# Нормализуем корень: гарантируем протокол и убираем хвостовой слэш.
_raw_native = (config.LM_STUDIO_NATIVE_BASE_URL or "").strip()
if not _raw_native:
    _raw_native = "http://localhost:1234/api/v1"
if not _raw_native.startswith(("http://", "https://")):
    _raw_native = "http://" + _raw_native
LM_NATIVE_ROOT: str = _raw_native.rstrip("/")

_native_headers = {"Content-Type": "application/json"}
if config.LM_STUDIO_API_TOKEN:
    _native_headers["Authorization"] = f"Bearer {config.LM_STUDIO_API_TOKEN}"

# base_url НЕ используем — пути будут абсолютными
lm_native_client = httpx.AsyncClient(
    headers=_native_headers,
    timeout=config.LM_STUDIO_TIMEOUT,
)

bot = Bot(
    token=config.TELEGRAM_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)

router = Router()
dp = Dispatcher()
dp.include_router(router)