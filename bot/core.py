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


def _create_native_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers=_native_headers,
        timeout=config.LM_STUDIO_TIMEOUT,
    )


lm_native_client: httpx.AsyncClient = _create_native_client()


def get_lm_client() -> httpx.AsyncClient:
    """Живой клиент к LM Studio.

    Если предыдущий экземпляр был закрыт (например, из-за вызова
    shutdown-хуков при падении polling с последующим перезапуском
    цикла) — создаёт новый.
    """
    global lm_native_client
    if lm_native_client is None or lm_native_client.is_closed:
        lm_native_client = _create_native_client()
    return lm_native_client


async def close_lm_client() -> None:
    """Идемпотентно закрывает клиент."""
    global lm_native_client
    client = lm_native_client
    if client is None or client.is_closed:
        return
    try:
        await client.aclose()
    except Exception:
        pass


# ── Bot / Dispatcher / Router ──
bot = Bot(
    token=config.TELEGRAM_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)

router = Router()
dp = Dispatcher()
dp.include_router(router)