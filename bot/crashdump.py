"""Централизованный дамп ошибок.

Любое необработанное исключение сериализуется в отдельный JSON-файл
внутри папки ``CRASH_DUMP_DIR`` (по умолчанию ``crash_dump/``).

Бот спроектирован так, чтобы НИКОГДА не падать: каждый слой, который
может бросить исключение, обёрнут, а глобальные хуки исключений
(sys / threading / asyncio) сохраняют диагностику сюда.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from . import config

log = logging.getLogger(__name__)

CRASH_DUMP_DIR: str = getattr(config, "CRASH_DUMP_DIR", "crash_dump")
MAX_TEXT_IN_DUMP: int = 4000


def _to_jsonable(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return "<max-depth>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return f"<bytes len={len(value)}>"
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(v, depth + 1) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v, depth + 1) for k, v in value.items()}
    try:
        return repr(value)
    except Exception:
        return "<unrepr-able>"


def _extract_user(event: Any) -> Optional[Dict[str, Any]]:
    user = getattr(event, "from_user", None)
    if user is None:
        return None
    return {
        "id": getattr(user, "id", None),
        "username": getattr(user, "username", None),
        "first_name": getattr(user, "first_name", None),
        "last_name": getattr(user, "last_name", None),
        "language_code": getattr(user, "language_code", None),
        "is_bot": getattr(user, "is_bot", None),
    }


def _extract_event(event: Any) -> Optional[Dict[str, Any]]:
    if event is None:
        return None
    info: Dict[str, Any] = {"type": type(event).__name__}

    chat = getattr(event, "chat", None)
    if chat is not None:
        info["chat"] = {
            "id": getattr(chat, "id", None),
            "type": str(getattr(chat, "type", "") or ""),
            "title": getattr(chat, "title", None),
        }

    for attr in ("message_id", "media_group_id", "id"):
        val = getattr(event, attr, None)
        if val is None or isinstance(val, (dict, list)):
            continue
        try:
            json.dumps(val)
            info[attr] = val
        except Exception:
            info[attr] = str(val)

    for attr in ("text", "caption", "data"):
        val = getattr(event, attr, None)
        if isinstance(val, str):
            info[attr] = val[:MAX_TEXT_IN_DUMP]

    msg = getattr(event, "message", None)
    if msg is not None and msg is not event:
        c = getattr(msg, "chat", None)
        if c is not None:
            info["callback_message"] = {
                "chat_id": getattr(c, "id", None),
                "message_id": getattr(msg, "message_id", None),
            }

    return info


def _format_tb(exc: BaseException) -> str:
    try:
        return "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
    except Exception:
        return "<failed to format traceback>"


def _format_chain(exc: BaseException) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    cause = getattr(exc, "__cause__", None)
    context = getattr(exc, "__context__", None)
    if cause is not None:
        items.append({
            "kind": "cause",
            "type": type(cause).__name__,
            "message": str(cause),
            "traceback": _format_tb(cause),
        })
    if context is not None and context is not cause:
        items.append({
            "kind": "context",
            "type": type(context).__name__,
            "message": str(context),
            "traceback": _format_tb(context),
        })
    return items


def dump_exception(
    exc: BaseException,
    *,
    context: Optional[str] = None,
    event: Any = None,
    extra: Optional[Dict[str, Any]] = None,
    user_id: Optional[int] = None,
) -> Optional[str]:
    """Сохранить исключение в JSON-файл. Никогда не бросает исключений.

    Возвращает путь файла при успехе, иначе ``None``.
    """
    try:
        os.makedirs(CRASH_DUMP_DIR, exist_ok=True)
    except Exception:
        log.exception("Failed to create crash dump directory %s", CRASH_DUMP_DIR)
        return None

    ts = time.time()
    stamp = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")
    filename = f"{stamp}_{uuid.uuid4().hex[:8]}.json"
    path = os.path.join(CRASH_DUMP_DIR, filename)

    user_info = _extract_user(event)
    if user_info is None and user_id is not None:
        user_info = {"id": user_id}

    try:
        record: Dict[str, Any] = {
            "timestamp": ts,
            "iso": datetime.fromtimestamp(ts).isoformat(timespec="seconds"),
            "context": context,
            "process": {
                "pid": os.getpid(),
                "cwd": os.getcwd(),
                "python": sys.version,
            },
            "exception": {
                "type": type(exc).__name__,
                "module": getattr(type(exc), "__module__", None),
                "message": str(exc),
                "repr": repr(exc),
                "traceback": _format_tb(exc),
            },
            "user": user_info,
            "event": _extract_event(event),
            "extra": _to_jsonable(extra or {}),
        }

        chain = _format_chain(exc)
        if chain:
            record["exception"]["chain"] = chain
    except Exception:
        log.exception("Failed to build crash dump record")
        return None

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
    except Exception:
        log.exception("Failed to write crash dump %s", path)
        return None

    return path


def install_global_hooks() -> None:
    """Ставит хуки sys / threading, которые пишут дампы в JSON."""
    original_hook = sys.excepthook

    def _sys_hook(exc_type, exc_value, exc_tb):
        try:
            if isinstance(exc_value, KeyboardInterrupt):
                return original_hook(exc_type, exc_value, exc_tb)
            dump_exception(exc_value, context="sys.excepthook")
        except Exception:
            pass
        try:
            original_hook(exc_type, exc_value, exc_tb)
        except Exception:
            pass

    sys.excepthook = _sys_hook

    def _thread_hook(args):  # noqa: ANN001
        try:
            if isinstance(args.exc_value, SystemExit):
                return
            dump_exception(
                args.exc_value,
                context="threading.excepthook",
                extra={"thread_name": getattr(args.thread, "name", None)},
            )
        except Exception:
            pass

    try:
        threading.excepthook = _thread_hook
    except Exception:
        log.exception("Failed to install threading excepthook")


def install_asyncio_exception_handler(loop: asyncio.AbstractEventLoop) -> None:
    """Ставит глобальный обработчик исключений на текущий event loop."""
    def _loop_handler(loop_, context):  # noqa: ARG001
        try:
            exc = context.get("exception")
            loop_ctx = {k: v for k, v in context.items() if k != "exception"}
            if isinstance(exc, BaseException):
                dump_exception(
                    exc,
                    context="asyncio.loop",
                    extra={"loop_context": _to_jsonable(loop_ctx)},
                )
            else:
                dump_exception(
                    RuntimeError(str(context.get("message") or "asyncio error")),
                    context="asyncio.loop",
                    extra={"loop_context": _to_jsonable(loop_ctx)},
                )
        except Exception:
            pass

    try:
        loop.set_exception_handler(_loop_handler)
    except Exception:
        log.exception("Failed to install asyncio loop exception handler")