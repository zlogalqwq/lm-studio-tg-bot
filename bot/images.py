import asyncio
import base64
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from aiogram.exceptions import TelegramNetworkError, TelegramRetryAfter
from aiogram.types import Message

from . import config
from .core import bot

log = logging.getLogger(__name__)

IMAGE_MIME_PREFIXES = ("image/",)

# ── Настройки ретраев при скачивании из Telegram ──
_DOWNLOAD_MAX_ATTEMPTS = 3
_DOWNLOAD_BASE_DELAY = 1.0
_DOWNLOAD_MAX_DELAY = 8.0


def _save_photo_sync(user_id: int, raw: bytes, ext: str) -> str:
    """Сохраняет файл в saved_photos/<user_id>/<timestamp>.<ext>."""
    user_dir = os.path.join(config.PHOTO_DIR, str(user_id))
    os.makedirs(user_dir, exist_ok=True)

    ts = int(time.time() * 1000)
    path = os.path.join(user_dir, f"{ts}.{ext}")
    counter = 1
    while os.path.exists(path):
        path = os.path.join(user_dir, f"{ts}_{counter}.{ext}")
        counter += 1

    with open(path, "wb") as f:
        f.write(raw)
    return path


async def _download_with_retry(file_id: str) -> bytes:
    """Скачивает файл из Telegram с ретраями при сетевых ошибках.

    Ретраит только `TelegramNetworkError` и `TelegramRetryAfter`.
    Прочие исключения (например, 400-ки) пробрасываются сразу.
    """
    last_exc: Optional[BaseException] = None

    for attempt in range(1, _DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            file = await bot.download(file_id)
            if file is None:
                raise ValueError("Telegram вернул пустой файл.")
            return file.read()
        except TelegramRetryAfter as exc:
            delay = float(exc.retry_after) + 0.5
            log.warning(
                "Download %s: retry_after=%.1fs (attempt %d/%d)",
                file_id, delay, attempt, _DOWNLOAD_MAX_ATTEMPTS,
            )
            last_exc = exc
            if attempt < _DOWNLOAD_MAX_ATTEMPTS:
                await asyncio.sleep(delay)
        except TelegramNetworkError as exc:
            delay = min(
                _DOWNLOAD_BASE_DELAY * (2 ** (attempt - 1)),
                _DOWNLOAD_MAX_DELAY,
            )
            log.warning(
                "Download %s: network error (attempt %d/%d): %s — retry in %.1fs",
                file_id, attempt, _DOWNLOAD_MAX_ATTEMPTS, exc, delay,
            )
            last_exc = exc
            if attempt < _DOWNLOAD_MAX_ATTEMPTS:
                await asyncio.sleep(delay)
        except Exception:
            # Не-сетевые ошибки (битый file_id и т.п.) нет смысла ретраить.
            log.exception("Download %s failed (non-retryable)", file_id)
            raise

    assert last_exc is not None
    log.error(
        "Download %s: exhausted %d attempts — last error: %s",
        file_id, _DOWNLOAD_MAX_ATTEMPTS, last_exc,
    )
    raise last_exc


async def _download_and_encode(file_id: str, user_id: int, ext: str = "jpg") -> str:
    raw = await _download_with_retry(file_id)

    if len(raw) > config.MAX_IMAGE_BYTES:
        raise ValueError(
            f"Изображение весит {_human_mb(len(raw))} — "
            f"это больше лимита {_human_mb(config.MAX_IMAGE_BYTES)}."
        )

    try:
        loop = asyncio.get_running_loop()
        saved_path = await loop.run_in_executor(
            None, _save_photo_sync, user_id, raw, ext
        )
        log.info("Saved photo: %s", saved_path)
    except Exception:
        log.exception("Failed to save photo for user %s", user_id)

    return base64.b64encode(raw).decode("ascii")


def _human_mb(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} МБ"


# ... остальное без изменений:
#     images_from_message, images_from_album, _check_image_limits, build_content

def images_from_message(msg: Message) -> List[Tuple[str, str, int, str]]:
    if msg.photo:
        p = msg.photo[-1]
        return [(p.file_id, "jpg", p.file_size or 0, "image/jpeg")]
    if (
        msg.document
        and msg.document.mime_type
        and msg.document.mime_type.startswith(IMAGE_MIME_PREFIXES)
    ):
        mime = msg.document.mime_type.lower()
        ext = mime.split("/")[-1]
        if ext == "jpeg":
            ext = "jpg"
        # нормализуем «официальные» MIME для популярных форматов
        if ext == "jpg":
            mime = "image/jpeg"
        return [(msg.document.file_id, ext, msg.document.file_size or 0, mime)]
    return []


def images_from_album(msgs: List[Message]) -> List[Tuple[str, str, int, str]]:
    seen: set[str] = set()
    result: List[Tuple[str, str, int, str]] = []
    for msg in msgs:
        for fid, ext, size, mime in images_from_message(msg):
            if fid not in seen:
                seen.add(fid)
                result.append((fid, ext, size, mime))
    return result

def _check_image_limits(items: List[Tuple[str, str, int, str]]) -> None:
    """Проверяет количество и суммарный вес картинок.

    Лимит на количество — MAX_IMAGES_EXTENDED (жёсткий, всегда).
    Суммарный вес набора — MAX_TOTAL_IMAGE_BYTES.
    Вес отдельного файла проверяется в _download_and_encode
    по MAX_IMAGE_BYTES.
    """
    count = len(items)
    if count == 0:
        return

    hard = config.MAX_IMAGES_EXTENDED
    cap = config.MAX_TOTAL_IMAGE_BYTES

    if count > hard:
        raise ValueError(
            f"Слишком много изображений: {count}. Максимум {hard}."
        )

    total = sum(size for _, _, size, _ in items)
    if total > cap:
        raise ValueError(
            f"{count} изображений весят {_human_mb(total)} — это больше лимита "
            f"{_human_mb(cap)}."
        )

async def build_content(
    album: Optional[List[Message]],
    single: Message,
    user_id: int,
) -> List[Dict[str, Any]]:
    if album:
        text = next(
            (m.caption or m.text for m in album if m.caption or m.text),
            "What is in these images?",
        )
    else:
        text = single.text or single.caption or "What is in this image?"

    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]

    raw_items = images_from_album(album) if album else images_from_message(single)

    # ── Проверка количества/веса ДО скачивания ──
    _check_image_limits(raw_items)

    if raw_items:
        encoded_list = await asyncio.gather(
            *(_download_and_encode(fid, user_id, ext) for fid, ext, _, _ in raw_items),
            return_exceptions=True,
        )
        for (_, _, _, mime), encoded in zip(raw_items, encoded_list):
            if isinstance(encoded, Exception):
                raise encoded
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}"},
            })
    return content