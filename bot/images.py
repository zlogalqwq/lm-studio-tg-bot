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

_DOWNLOAD_MAX_ATTEMPTS = 3
_DOWNLOAD_BASE_DELAY = 1.0
_DOWNLOAD_MAX_DELAY = 8.0

_SUPPORTED_MIMES = frozenset({
    "image/jpeg", "image/png", "image/webp",
    "image/gif", "image/bmp", "image/tiff",
})
_UNSUPPORTED_MIMES = frozenset({
    "image/heic", "image/heif", "image/heics", "image/avif",
})


def _normalize_image_mime(mime: str) -> Optional[str]:
    mime = (mime or "").lower().strip()
    if not mime:
        return None
    if mime in _UNSUPPORTED_MIMES:
        return None
    if mime == "image/jpg":
        return "image/jpeg"
    return mime


def validate_image_formats(msgs: List[Message]) -> None:
    for m in msgs:
        doc = m.document
        if doc is None or not doc.mime_type:
            continue
        if doc.mime_type.lower() in _UNSUPPORTED_MIMES:
            raise ValueError(
                f"❌ Формат <b>{doc.mime_type.lower()}</b> не поддерживается "
                f"моделью.\n\n"
                f"HEIC/HEIF/AVIF — форматы Apple и новых камер; большинство "
                f"VLM (Qwen-VL, LLaVA и др.) их не понимают.\n\n"
                f"<b>Что делать:</b>\n"
                f"• Отправьте картинку как <b>фото</b> — Telegram сам "
                f"сконвертирует в JPEG\n"
                f"• Или сконвертируйте в JPEG/PNG перед отправкой"
            )


def images_from_message(msg: Message) -> List[Tuple[str, str, int, str]]:
    if msg.photo:
        p = msg.photo[-1]
        return [(p.file_id, "jpg", p.file_size or 0, "image/jpeg")]
    doc = msg.document
    if (
        doc
        and doc.mime_type
        and doc.mime_type.lower().startswith(IMAGE_MIME_PREFIXES)
    ):
        mime = _normalize_image_mime(doc.mime_type)
        if mime is None:
            return []
        ext = mime.split("/", 1)[-1]
        if ext == "jpeg":
            ext = "jpg"
        return [(doc.file_id, ext, doc.file_size or 0, mime)]
    return []


def images_from_album(msgs: List[Message]) -> List[Tuple[str, str, int, str]]:
    seen: set[str] = set()
    result: List[Tuple[str, str, int, str]] = []
    for msg in sorted(msgs, key=lambda m: m.message_id):
        for fid, ext, size, mime in images_from_message(msg):
            if fid not in seen:
                seen.add(fid)
                result.append((fid, ext, size, mime))
    return result


def _save_photo_sync(user_id: int, raw: bytes, ext: str) -> str:
    user_dir = os.path.join(config.PHOTO_DIR, str(user_id))
    os.makedirs(user_dir, exist_ok=True)

    ts = int(time.time() * 1000)
    counter = 0
    while True:
        if counter == 0:
            path = os.path.join(user_dir, f"{ts}.{ext}")
        else:
            path = os.path.join(user_dir, f"{ts}_{counter}.{ext}")

        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            counter += 1
            if counter > 100_000:
                raise RuntimeError(
                    f"Не удалось подобрать уникальное имя файла в {user_dir}"
                )
            continue

        try:
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
        except Exception:
            try:
                os.unlink(path)
            except Exception:
                pass
            raise

        return path


async def _download_with_retry(file_id: str) -> bytes:
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
            log.exception("Download %s failed (non-retryable)", file_id)
            raise

    assert last_exc is not None
    log.error(
        "Download %s: exhausted %d attempts — last error: %s",
        file_id, _DOWNLOAD_MAX_ATTEMPTS, last_exc,
    )
    raise last_exc


def _human_mb(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} МБ"


async def _download_raw(file_id: str) -> bytes:
    raw = await _download_with_retry(file_id)
    if len(raw) > config.MAX_IMAGE_BYTES:
        raise ValueError(
            f"Изображение весит {_human_mb(len(raw))} — "
            f"это больше лимита {_human_mb(config.MAX_IMAGE_BYTES)}."
        )
    return raw


async def _save_all_best_effort(
    user_id: int,
    items: List[Tuple[str, str, int, str]],
    raw_list: List[Any],
) -> int:
    loop = asyncio.get_running_loop()
    saved = 0
    for (_, ext, _, _), raw in zip(items, raw_list):
        if not isinstance(raw, (bytes, bytearray)):
            continue
        try:
            await loop.run_in_executor(
                None, _save_photo_sync, user_id, bytes(raw), ext
            )
            saved += 1
        except Exception:
            log.exception("Failed to save photo for user %s", user_id)
    return saved


def _check_image_limits(items: List[Tuple[str, str, int, str]]) -> None:
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
    raw_items = images_from_album(album) if album else images_from_message(single)

    if album:
        user_text = next(
            (m.caption or m.text for m in album if m.caption or m.text), ""
        ).strip()
    else:
        user_text = (single.text or single.caption or "").strip()

    if not user_text and raw_items:
        if len(raw_items) == 1:
            user_text = config.DEFAULT_IMAGE_PROMPT_SINGLE
        else:
            user_text = config.DEFAULT_IMAGE_PROMPT_MULTIPLE.replace(
                "{n}", str(len(raw_items))
            )
    elif not user_text:
        user_text = "What do you want me to do with this?"

    _check_image_limits(raw_items)

    content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    if not raw_items:
        return content

    log.info(
        "build_content: user=%s album=%s → %d image(s) to download",
        user_id, bool(album), len(raw_items),
    )

    raw_list = await asyncio.gather(
        *(_download_raw(fid) for fid, _, _, _ in raw_items),
        return_exceptions=True,
    )

    failures = [
        (fid, exc)
        for (fid, _, _, _), exc in zip(raw_items, raw_list)
        if isinstance(exc, Exception)
    ]
    if failures:
        log.warning(
            "build_content: %d/%d download(s) failed: %s",
            len(failures), len(raw_items),
            [(fid, str(exc)[:120]) for fid, exc in failures],
        )
        raise failures[0][1]

    saved = await _save_all_best_effort(user_id, raw_items, raw_list)
    log.info(
        "build_content: saved %d/%d to disk for user %s",
        saved, len(raw_items), user_id,
    )

    for (_, _, _, mime), raw in zip(raw_items, raw_list):
        b64 = base64.b64encode(bytes(raw)).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })

    log.info(
        "build_content: sending %d image(s) to LM Studio (user=%s)",
        len(raw_items), user_id,
    )
    return content