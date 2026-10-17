import asyncio
import logging
import os

import aiohttp
from aiogram.exceptions import (
    TelegramConflictError,
    TelegramNetworkError,
    TelegramRetryAfter,
    TelegramServerError,
    TelegramUnauthorizedError,
)
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

async def _reset_bot_session() -> None:
    """Пересоздаёт aiohttp-сессию бота после сетевой ошибки.

    После WinError 121 / TLS-timeout коннектор aiohttp может остаться
    в неработоспособном состоянии — все дальнейшие запросы падают
    мгновенно. Пересоздание гарантирует, что следующий retry начнётся
    с чистого коннектора.
    """
    try:
        old_session = bot.session
        try:
            await old_session.close()
        except Exception:
            log.debug("reset session: close failed", exc_info=True)
        try:
            from aiogram.client.session.aiohttp import AiohttpSession

            bot.session = AiohttpSession()
            log.info("Bot session recreated")
        except Exception as exc:
            dump_exception(exc, context="reset_bot_session.create")
    except Exception as exc:
        dump_exception(exc, context="reset_bot_session")

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

    # Tools при старте всегда выключены; Thinking — включён по умолчанию.
    # Это применяется и при перезапуске polling после падения.
    try:
        await store.set_tools_enabled(False)
        await store.set_thinking_enabled(True)
    except Exception as exc:
        dump_exception(exc, context="startup.reset_flags")
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

    # delete_webhook с ретраями: при сетевой ошибке не сдаёмся сразу,
    # иначе polling будет падать на «webhook is active».
    for attempt in range(1, 4):
        try:
            await bot.delete_webhook(drop_pending_updates=True)
            break
        except TelegramRetryAfter as exc:
            delay = float(exc.retry_after) + 1.0
            log.warning("delete_webhook flood-wait %.1fs", delay)
            await asyncio.sleep(delay)
        except (TelegramNetworkError, TelegramServerError, aiohttp.ClientError) as exc:
            log.warning(
                "delete_webhook: network error (attempt %d/3): %s", attempt, exc
            )
            if attempt < 3:
                await asyncio.sleep(min(2 ** attempt, 10))
        except Exception as exc:
            dump_exception(exc, context="delete_webhook")
            log.warning("delete_webhook failed: %s", exc)
            break

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
            break
        except (KeyboardInterrupt, SystemExit):
            raise
        except asyncio.CancelledError:
            log.info("Polling cancelled — shutting down")
            raise

        except TelegramUnauthorizedError as exc:
            # Неверный токен: ретраи бессмысленны, процесс надо гасить.
            dump_exception(exc, context="polling.unauthorized")
            log.critical(
                "Telegram отверг токен бота (401 Unauthorized). "
                "Проверьте TELEGRAM_TOKEN в .env — процесс завершён."
            )
            raise

        except TelegramConflictError:
            # Другой экземпляр бота уже поллит. Ждём, вдруг он умрёт.
            log.critical(
                "Telegram Conflict: другой экземпляр бота использует тот же "
                "токен. Остановите его. Повтор через %.1f с.", backoff
            )
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                raise
            backoff = min(backoff * 2, MAX_BACKOFF)
            continue

        except TelegramRetryAfter as exc:
            delay = float(exc.retry_after) + 1.0
            log.warning(
                "Telegram flood-wait: retry_after=%.1fs — жду %.1fs",
                exc.retry_after, delay,
            )
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                raise
            continue

        except (TelegramNetworkError, TelegramServerError, aiohttp.ClientError) as exc:
            # Транзиентные сетевые ошибки — БЕЗ crash dump (иначе мусор),
            # с пересозданием aiohttp-сессии (после WinError 121 коннектор мёртв).
            log.warning(
                "Сетевая ошибка Telegram (%s): %s — переподключаюсь через %.1f с",
                type(exc).__name__, exc, backoff,
            )
            try:
                await _reset_bot_session()
            except Exception:
                pass
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                raise
            backoff = min(backoff * 2, MAX_BACKOFF)
            continue

        except BaseException as exc:
            # Всё остальное — дампим и ретраим как раньше.
            try:
                dump_exception(exc, context="polling_loop")
            except Exception:
                pass
            log.exception(
                "Polling crashed with unexpected error; restarting in %.1f с",
                backoff,
            )
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                raise
            backoff = min(backoff * 2, MAX_BACKOFF)
            continue