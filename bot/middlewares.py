import asyncio
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message

from . import config
from .crashdump import dump_exception
from .storage import store

log = logging.getLogger(__name__)

START_RE = re.compile(r"^\s*/start(?:@\w+)?(?:\s|$)", re.IGNORECASE)


class AuthMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        if not config.AUTH_PASSWORD:
            return await handler(event, data)

        user = getattr(event, "from_user", None)
        if user is None:
            return await handler(event, data)

        try:
            is_auth = await store.is_authorized(user.id)
        except Exception as exc:
            dump_exception(exc, context="AuthMiddleware.is_authorized", event=event)
            log.exception("is_authorized crashed; failing closed")
            is_auth = False

        if is_auth:
            return await handler(event, data)

        if isinstance(event, Message):
            text = (event.text or "").strip()

            if START_RE.match(text):
                return await handler(event, data)

            if text == config.AUTH_PASSWORD:
                try:
                    await store.authorize(user.id)
                    sent = await event.answer(
                        "✅ Пароль принят. Теперь можно отправлять текст, фото или альбом."
                    )
                    await store.track_message(
                        user.id, sent.chat.id, sent.message_id
                    )
                except Exception as exc:
                    dump_exception(
                        exc, context="AuthMiddleware.authorize", event=event
                    )
                return None

            try:
                await event.answer(
                    "🔒 Бот защищён паролем.\n\n"
                    "Отправьте пароль сообщением в этот чат.\n"
                )
            except Exception as exc:
                dump_exception(
                    exc, context="AuthMiddleware.ask_password", event=event
                )
            return None

        if isinstance(event, CallbackQuery):
            if (event.data or "").lower().startswith("start"):
                return await handler(event, data)
            try:
                await event.answer("🔒 Требуется авторизация.", show_alert=True)
            except Exception as exc:
                dump_exception(
                    exc, context="AuthMiddleware.deny_callback", event=event
                )
            return None

        return None


class MediaGroupCollector(BaseMiddleware):
    """Адаптивная группировка альбомов: ждёт тишины `latency` секунд,
    но не дольше `max_wait` от первой фотографии."""

    def __init__(self, latency: Optional[float] = None) -> None:
        self.latency = (
            latency if latency is not None else config.MEDIA_GROUP_LATENCY
        )
        self.max_wait = getattr(config, "MEDIA_GROUP_MAX_WAIT", 5.0)
        self._pending: Dict[str, List[Message]] = {}
        self._last_arrival: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def __call__(self, handler: Callable, event: Message, data: Dict[str, Any]) -> Any:
        try:
            if event.from_user is not None:
                await store.track_message(
                    event.from_user.id, event.chat.id, event.message_id
                )
        except Exception as exc:
            dump_exception(exc, context="MediaGroupCollector.track", event=event)
            log.exception("track_message failed")

        mgid = event.media_group_id
        if mgid is None:
            return await handler(event, data)

        async with self._lock:
            bucket = self._pending.setdefault(mgid, [])
            bucket.append(event)
            self._last_arrival[mgid] = time.monotonic()
            is_first = len(bucket) == 1

        if not is_first:
            return None

        start = time.monotonic()
        while True:
            # FIX: проверяем max_wait ДО sleep, чтобы не вылезать за лимит.
            if time.monotonic() - start >= self.max_wait:
                break
            await asyncio.sleep(self.latency)
            async with self._lock:
                last = self._last_arrival.get(mgid, 0.0)
            if time.monotonic() - last >= self.latency:
                break

        async with self._lock:
            album = self._pending.pop(mgid, [])
            self._last_arrival.pop(mgid, None)

        album.sort(key=lambda m: m.message_id)

        log.info(
            "Media group %s collected: %d photo(s) (waited %.2fs)",
            mgid, len(album), time.monotonic() - start,
        )

        data["album"] = album
        try:
            await handler(event, data)
        except Exception:
            log.exception("Error handling media group %s", mgid)
            raise
        return None


class CommandLoggerMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        try:
            user = getattr(event, "from_user", None)
            if user is not None:
                if isinstance(event, Message):
                    text = (event.text or "").strip()
                    if text.startswith("/"):
                        cmd = text.split(maxsplit=1)[0]
                        if "@" in cmd:
                            cmd = cmd.split("@", 1)[0]
                        log.info(
                            "CMD  user=%s (@%s)  %s",
                            user.id, user.username or "—", cmd,
                        )
                elif isinstance(event, CallbackQuery):
                    log.info(
                        "CALL user=%s (@%s)  data=%s",
                        user.id, user.username or "—", event.data,
                    )
        except Exception as exc:
            dump_exception(
                exc, context="CommandLoggerMiddleware", event=event
            )
            log.exception("CommandLoggerMiddleware crashed")

        return await handler(event, data)


class CommandCooldownMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        try:
            block, left, user_id = await self._should_block(event)
        except Exception as exc:
            dump_exception(
                exc, context="CommandCooldownMiddleware.check", event=event
            )
            log.exception("Cooldown check crashed; allowing")
            block, left, user_id = False, 0.0, None

        if block and user_id is not None:
            try:
                sent = await event.answer(
                    f"⏳ Не так часто. Подожди {left:.1f} с."
                )
                await store.track_message(
                    user_id, sent.chat.id, sent.message_id
                )
            except Exception as exc:
                dump_exception(
                    exc, context="CommandCooldownMiddleware.reply", event=event
                )
            return None

        return await handler(event, data)

    async def _should_block(self, event):
        if not isinstance(event, Message):
            return False, 0.0, None

        text = (event.text or "").strip()
        if not text.startswith("/"):
            return False, 0.0, None

        user = event.from_user
        if user is None:
            return False, 0.0, None

        head = text.split(maxsplit=1)[0][1:]
        if "@" in head:
            head = head.split("@", 1)[0]
        cmd = head.lower()

        if cmd in config.COMMAND_COOLDOWN_EXEMPT:
            return False, 0.0, user.id

        left = await store.check_cooldown(
            user.id, config.COMMAND_COOLDOWN, bucket="cmd"
        )
        if left > 0:
            log.info(
                "CMD cooldown: user=%s cmd=/%s left=%.2fs",
                user.id, cmd, left,
            )
            return True, left, user.id
        return False, 0.0, user.id