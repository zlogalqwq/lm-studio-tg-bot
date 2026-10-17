import asyncio
import json
import logging
import os
import time
from typing import Any, List

from . import config
from .core import bot
from .storage import store

log = logging.getLogger(__name__)

_log_lock = asyncio.Lock()

from aiogram.types import Message
from typing import Any, List

async def finalize_and_track(
    user_id: int,
    status: Message,
    sent: List[Message],
) -> None:
    """Синхронизирует трекер с тем, что реально осталось в чате.

    * Случай 1 (`finalize` отредактировал `status`): `status` уже в `sent`,
      запись в трекере сохраняется как есть — без дублей.
    * Случай 2 (`finalize` удалил `status` и прислал новые сообщения):
      `status` снимается с трекера, новые сообщения регистрируются.
    """
    status_key = (status.chat.id, status.message_id)
    sent_keys = {(m.chat.id, m.message_id) for m in sent}

    if status_key not in sent_keys:
        await store.untrack_message(user_id, *status_key)

    for m in sent:
        if (m.chat.id, m.message_id) != status_key:
            await store.track_message(user_id, m.chat.id, m.message_id)

def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t == "text":
                parts.append(item.get("text", ""))
            elif t == "image_url":
                parts.append("[image]")
        return "\n".join(parts)
    return str(content)


async def save_exchange(user_id: int, prompt: Any, reply: str) -> None:
    record = {
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "user_id": user_id,
        "prompt": _extract_text(prompt),
        "reply": reply,
    }
    block = json.dumps(record, ensure_ascii=False, indent=2, separators=(",", ": "))
    block = block.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "")

    async with _log_lock:
        try:
            need_sep = True
            if os.path.exists(config.CHAT_LOG_FILE) and os.path.getsize(config.CHAT_LOG_FILE) > 0:
                with open(config.CHAT_LOG_FILE, "rb") as f:
                    try:
                        f.seek(-1, os.SEEK_END)
                        need_sep = f.read(1) != b"\n"
                    except OSError:
                        need_sep = False
            with open(config.CHAT_LOG_FILE, "a", encoding="utf-8") as f:
                if need_sep:
                    f.write("\n")
                f.write(block)
                f.write("\n\n")
        except Exception:
            log.exception("Failed to write chat log to %s", config.CHAT_LOG_FILE)


async def delete_tracked_messages(user_id: int) -> int:
    pairs = await store.pop_tracked_messages(user_id)
    deleted = 0
    for chat_id, msg_id in pairs:
        try:
            await bot.delete_message(chat_id, msg_id)
            deleted += 1
        except Exception:
            pass
    return deleted