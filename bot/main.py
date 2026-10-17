import asyncio
import logging
import os

from aiogram.types import ErrorEvent
from .core import bot, dp, router, close_lm_client, get_lm_client

from . import config, middlewares  # noqa: F401
from .config import setup_logging
from .crashdump import (
    CRASH_DUMP_DIR,
    dump_exception,
    install_asyncio_exception_handler,
    install_global_hooks,
)
from .handlers import commands, callbacks, messages  # noqa: F401
from .lm import fetch_models
from .storage import store


setup_logging()
install_global_hooks()

log = logging.getLogger(__name__)

# ── Идемпотентное закрытие ресурсов ──
_shutdown_lock = asyncio.Lock()


async def _shutdown_resources() -> None:
    """Закрывает все сетевые ресурсы. Идемпотентна и безопасна
    при многократном вызове из-за retry-цикла polling."""
    async with _shutdown_lock:
        log.info("Bot shutting down…")

        try:
            await bot.session.close()
        except Exception as exc:
            dump_exception(exc, context="shutdown.bot_session")

        try:
            await close_lm_client()
        except Exception as exc:
            dump_exception(exc, context="shutdown.lm_client")


@router.startup()
async def on_startup() -> None:
    log.info("Bot starting")

    # Пересоздаём HTTP-клиент, если предыдущий shutdown его закрыл,
    # но polling затем был перезапущен retry-циклом.
    try:
        get_lm_client()
    except Exception as exc:
        dump_exception(exc, context="startup.get_lm_client")
    # Папки — не критично, если не получится.
    for path, tag in ((CRASH_DUMP_DIR, "crash_dump"), (config.PHOTO_DIR, "photo")):
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as exc:
            dump_exception(exc, context=f"startup.makedirs.{tag}")

    try:
        store.set_models([], ttl=0.0)
    except Exception as exc:
        dump_exception(exc, context="startup.set_models")

    try:
        models = await fetch_models()
        if models:
            store.set_models(models)
            log.info("LM Studio models: %s", ", ".join(models))
        else:
            log.warning("LM Studio not available at startup")
    except Exception as exc:
        dump_exception(exc, context="startup.fetch_models")
        log.warning("Failed fetching models at startup: %s", exc)


@router.shutdown()
async def on_shutdown() -> None:
    try:
        await _shutdown_resources()
    except Exception as exc:
        dump_exception(exc, context="shutdown")


@dp.errors()
async def global_error_handler(event: ErrorEvent) -> bool:
    """Ловит всё, что долетело до диспетчера. Возвращает True = обработано."""
    try:
        dump_exception(
            event.exception,
            context="dispatcher_error",
            event=getattr(event, "update", None),
        )
    except Exception:
        log.exception("Failed to write crash dump for dispatcher error")
    log.error("Unhandled dispatcher error: %r", event.exception)
    return True


async def run() -> None:
    # Ставим asyncio-хук как можно раньше.
    try:
        install_asyncio_exception_handler(asyncio.get_running_loop())
    except Exception as exc:
        dump_exception(exc, context="install_asyncio_handler")

    # ── Мидлвари. Порядок как раньше; каждая сама себя защищает. ──
    dp.message.outer_middleware(middlewares.CommandLoggerMiddleware())
    dp.callback_query.outer_middleware(middlewares.CommandLoggerMiddleware())

    dp.message.outer_middleware(middlewares.CommandCooldownMiddleware())

    dp.message.outer_middleware(middlewares.AuthMiddleware())
    dp.callback_query.outer_middleware(middlewares.AuthMiddleware())

    dp.message.middleware(middlewares.MediaGroupCollector())

    try:
        await bot.delete_webhook(drop_pending_updates=True)
    except Exception as exc:
        dump_exception(exc, context="delete_webhook")
        log.warning("delete_webhook failed: %s", exc)

    # ── Retry-цикл polling: любые падения не убивают процесс. ──
    backoff = 1.0
    MAX_BACKOFF = 60.0

    while True:
        try:
            await dp.start_polling(
                bot,
                drop_pending_updates=True,
                close_bot_session=False,
            )
            # сюда попадаем только если polling завершился штатно
            break
        except (KeyboardInterrupt, SystemExit):
            raise
        except asyncio.CancelledError:
            log.info("Polling cancelled — shutting down")
            raise
        except BaseException as exc:
            # Пробуем записать дамп, но даже если не получится — идём дальше.
            try:
                dump_exception(exc, context="polling_loop")
            except Exception:
                pass
            log.exception(
                "Polling crashed; restarting in %.1fs", backoff
            )
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                raise
            backoff = min(backoff * 2, MAX_BACKOFF)
            continue

    try:
        await _shutdown_resources()
    except Exception as exc:
        dump_exception(exc, context="run.final_shutdown")