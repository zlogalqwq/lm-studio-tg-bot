import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from aiogram import F
from aiogram.exceptions import TelegramNetworkError, TelegramRetryAfter
from aiogram.types import Message
import httpx

from ..core import router
from .. import config
from ..formatting import finalize_streamed
from ..images import (
    IMAGE_MIME_PREFIXES,
    build_content,
    images_from_album,
    images_from_message,
    validate_image_formats,
)
from ..keyboards import stop_keyboard, response_keyboard
from ..storage import store
from ..logging_utils import save_exchange, finalize_and_track
from ..decor import maybe_tip, render_stats
from ..lm import (
    chat_completion,
    get_model,
    sync_loaded_context_length,
    estimate_context_tokens,
)


log = logging.getLogger(__name__)

# ── Ретраи скачивания файлов из Telegram ────────────────────────────────────
_DOWNLOAD_MAX_ATTEMPTS = 3
_DOWNLOAD_BASE_DELAY = 1.0
_DOWNLOAD_MAX_DELAY = 8.0

# ── Уведомление о «занят» (чтобы не спамить одинаковыми сообщениями) ────────
_BUSY_NOTIFY_COOLDOWN = 15.0
_busy_notify_ts: Dict[int, float] = {}


def _get_prompt_text(message: Message, album: Optional[List[Message]]) -> str:
    """Возвращает текст запроса: для альбома — первую непустую подпись, иначе text/caption."""
    if album:
        return next(
            (m.caption or m.text for m in album if m.caption or m.text),
            "",
        )
    return message.text or message.caption or ""


async def _build_content_with_retry(
    album: Optional[List[Message]],
    message: Message,
    user_id: int,
) -> List[Dict[str, Any]]:
    """Скачивает фото с ретраями при сетевых ошибках Telegram.

    Ретраит только TelegramNetworkError / TelegramRetryAfter.
    Прочие исключения (битый file_id, превышение размера и т.п.)
    пробрасываются сразу — на них ретрай бессмысленен.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(1, _DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            return await build_content(album, message, user_id)
        except TelegramRetryAfter as exc:
            last_exc = exc
            delay = float(exc.retry_after) + 0.5
            log.warning(
                "build_content: retry_after=%.1fs (attempt %d/%d)",
                delay, attempt, _DOWNLOAD_MAX_ATTEMPTS,
            )
            if attempt < _DOWNLOAD_MAX_ATTEMPTS:
                await asyncio.sleep(delay)
        except TelegramNetworkError as exc:
            last_exc = exc
            delay = min(
                _DOWNLOAD_BASE_DELAY * (2 ** (attempt - 1)),
                _DOWNLOAD_MAX_DELAY,
            )
            log.warning(
                "build_content: network error (attempt %d/%d): %s — retry in %.1fs",
                attempt, _DOWNLOAD_MAX_ATTEMPTS, exc, delay,
            )
            if attempt < _DOWNLOAD_MAX_ATTEMPTS:
                await asyncio.sleep(delay)
        except Exception:
            # Не-сетевые ошибки — без ретраев.
            raise

    assert last_exc is not None
    log.error(
        "build_content: exhausted %d attempts — last error: %s",
        _DOWNLOAD_MAX_ATTEMPTS, last_exc,
    )
    raise last_exc


async def _notify_busy(message: Message, user_id: int, n_photos: int) -> None:
    """Сообщает пользователю, что запрос проигнорирован, но не чаще
    _BUSY_NOTIFY_COOLDOWN секунд (чтобы не спамить при альбомах)."""
    now = time.monotonic()
    if now - _busy_notify_ts.get(user_id, 0.0) < _BUSY_NOTIFY_COOLDOWN:
        return
    _busy_notify_ts[user_id] = now
    try:
        if n_photos:
            text = (
                "⏳ <b>Уже обрабатываю предыдущий запрос.</b>\n\n"
                f"Это сообщение (фото: {n_photos}) будет проигнорировано.\n"
                "Дождитесь ответа и отправьте снова."
            )
        else:
            text = (
                "⏳ <b>Уже обрабатываю предыдущий запрос.</b>\n\n"
                "Это сообщение будет проигнорировано.\n"
                "Дождитесь ответа и отправьте снова."
            )
        sent = await message.answer(text)
        await store.track_message(user_id, sent.chat.id, sent.message_id)
    except Exception:
        log.debug("busy notify failed", exc_info=True)


@router.message(
    (F.text & ~F.text.startswith("/")) | F.photo | F.document
)
async def handle_message(
    message: Message,
    album: Optional[List[Message]] = None,
) -> None:
    user_id = message.from_user.id

    # ── 0. Ранняя валидация форматов (HEIC/HEIF/AVIF): отсекаем ДО скачивания. ──
    try:
        validate_image_formats(album or [message])
    except ValueError as exc:
        sent = await message.answer(str(exc))
        await store.track_message(user_id, sent.chat.id, sent.message_id)
        return

    # ── 1. Длина текста ──
    prompt_text = _get_prompt_text(message, album)
    if len(prompt_text) > config.MAX_INPUT_CHARS:
        sent = await message.answer(
            f"❌ Слишком длинный запрос: <b>{len(prompt_text)}</b> символов.\n"
            f"Максимум — <b>{config.MAX_INPUT_CHARS}</b>."
        )
        await store.track_message(user_id, sent.chat.id, sent.message_id)
        return

    # ── 2. Идёт операция с моделью? ──
    if await store.is_model_op_busy():
        log.info("User %s — model op in progress, message ignored", user_id)
        sent = await message.answer(
            "⏳ Сейчас идёт операция с моделью (загрузка/выгрузка). "
            "Подождите, пока она завершится."
        )
        await store.track_message(user_id, sent.chat.id, sent.message_id)
        return

    # ── 3. Пользователь уже генерирует? — уведомляем, а не дропаем молча. ──
    if not await store.try_acquire(user_id):
        n_photos = len(album) if album else (
            1 if images_from_message(message) else 0
        )
        log.warning(
            "User %s busy — dropped incoming: album=%s photos=%d has_text=%s",
            user_id, bool(album), n_photos, bool(message.text),
        )
        await _notify_busy(message, user_id, n_photos)
        return

    stop_event = await store.start_generation(user_id)
    status: Optional[Message] = None
    try:
        status = await message.answer("⏳ Обрабатываю…", reply_markup=stop_keyboard())
        await store.track_message(user_id, status.chat.id, status.message_id)
        try:
            has_text = bool(message.text or message.caption)
            has_image = bool(images_from_message(message)) or (
                album is not None and bool(images_from_album(album))
            )
            if not has_text and not has_image:
                doc = message.document
                if doc is not None and not (
                    doc.mime_type or ""
                ).startswith(IMAGE_MIME_PREFIXES):
                    await status.delete()
                    await store.untrack_message(
                        user_id, status.chat.id, status.message_id
                    )
                    sent = await message.answer(
                        "❌ Поддерживаются только изображения и текст."
                    )
                    await store.track_message(
                        user_id, sent.chat.id, sent.message_id
                    )
                    return

            # ── Готовим контент ДО каких-либо изменений истории ──
            content = await _build_content_with_retry(album, message, user_id)

            model = await get_model()
            await sync_loaded_context_length()
            loaded_ctx = await store.get_loaded_context_length()

            settings = await store.get_settings(user_id)

            # ── Эффективный лимит: min(физический, пользовательский) ──
            user_limit = settings.context_limit
            if loaded_ctx > 0 and user_limit > 0:
                limit = min(loaded_ctx, user_limit)
                if loaded_ctx != user_limit:
                    log.info(
                        "Context limit mismatch: user=%s loaded_ctx=%s "
                        "user_limit=%s → effective=%s",
                        user_id, loaded_ctx, user_limit, limit,
                    )
            else:
                limit = loaded_ctx if loaded_ctx > 0 else user_limit

            # ── Снимок истории до добавления текущего сообщения ──
            history_for_context = await store.get_with_system(user_id)

            # ── Предупреждение о заполнении контекста ──
            effective_used = settings.last_effective_tokens
            if effective_used <= 0:
                effective_used = settings.last_prompt_tokens
            if effective_used > 0 and limit > 0:
                ratio = effective_used / limit
                log.info(
                    "context check: user=%s used=%d limit=%d ratio=%.3f",
                    user_id, effective_used, limit, ratio,
                )
                if ratio < config.CONTEXT_WARN_RATIO:
                    settings.context_warned = False
                elif not settings.context_warned:
                    settings.context_warned = True
                    pct = int(ratio * 100)
                    warn = await message.answer(
                        f"⚠️ Контекст заполнен на <b>{pct}%</b> "
                        f"({effective_used}/{limit}).\n"
                        f"Модель скоро начнёт обрезать историю. "
                        f"Начать с чистого листа: /clean"
                    )
                    await store.track_message(
                        user_id, warn.chat.id, warn.message_id
                    )

            # ── В историю ──
            await store.append_user(user_id, content)
            await store.set_last_user_input(user_id, content)


            prev_id = await store.get_response_id(user_id)
            system_prompt = await store.get_system_prompt(user_id)
            thinking_enabled = await store.is_thinking_enabled()

            reply, usage, elapsed, new_resp_id, session_reset = await chat_completion(
                message=content,
                model=model,
                status=status,
                previous_response_id=prev_id,
                stop_event=stop_event,
                settings=settings,
                system_prompt=system_prompt,
                use_tools=await store.is_tools_enabled(),
                thinking_enabled=thinking_enabled,
                history_context=history_for_context,
                loaded_context_length=loaded_ctx,
                prev_prompt_tokens=await store.get_cumulative_tokens(user_id),
            )

            if session_reset:
                await store.reset_api_state(user_id)
            if new_resp_id:
                await store.set_response_id(user_id, new_resp_id)

            if usage is not None:
                if reply:
                    await store.append_assistant(user_id, reply)
                    await save_exchange(user_id, content, reply)

                history_now = await store.get_with_system(user_id)
                estimated = estimate_context_tokens(history_now)
                # FIX: передаём estimated как floor — эвристика delta/total
                # больше не зависит от угадывания семантики API-поля.
                cumulative = await store.update_cumulative_tokens(
                    user_id, usage["prompt_tokens"], estimated_total=estimated,
                )
                effective = max(estimated, cumulative)

                await store.set_usage(
                    user_id,
                    usage["prompt_tokens"],
                    usage["completion_tokens"],
                    usage["total_tokens"],
                    effective_tokens=effective,
                )

                note = (
                    "\n\n⚠️ <i>Stateful-сессия LM Studio была сброшена "
                    "(сервер сообщил, что она недействительна). "
                    "Модель не видит предыдущий контекст. "
                    "Начать с чистого листа: /clean</i>"
                ) if session_reset else ""
                stats = (
                    render_stats(
                        usage, elapsed, settings,
                        loaded_ctx=limit, used_tokens=effective,
                    )
                    + maybe_tip()
                    + note
                )
            else:
                stats = ""

            stopped = (usage is None)
            sent = await finalize_streamed(
                status, reply, message, stopped=stopped, stats=stats
            )
            await finalize_and_track(user_id, status, sent)

        except TelegramNetworkError as exc:
            log.warning(
                "Telegram network error for user %s after retries: %s",
                user_id, exc,
            )
            try:
                await status.edit_text(
                    "📡 <b>Не удалось скачать файл из Telegram.</b>\n\n"
                    "Соединение с <code>api.telegram.org</code> "
                    "обрывается — это сетевая ошибка, а не проблема "
                    "с ботом.\n\n"
                    "Что делать:\n"
                    "• Подождать пару секунд и отправить фото ещё раз\n"
                    "• Если повторяется постоянно — проверьте "
                    "VPN/прокси и доступность Telegram из вашей сети",
                    reply_markup=response_keyboard(),
                )
            except Exception:
                pass

        except (RuntimeError, ValueError) as exc:
            try:
                await status.edit_text(
                    f"❌ {exc}",
                    reply_markup=response_keyboard(),
                )
            except Exception:
                pass
        except httpx.RequestError:
            try:
                await status.edit_text(
                    "❌ Не удалось подключиться к LM Studio.",
                    reply_markup=response_keyboard(),
                )
            except Exception:
                pass
        except Exception:
            log.exception("Processing error for user %s", user_id)
            try:
                await status.edit_text(
                    "❌ Произошла ошибка при обработке запроса.",
                    reply_markup=response_keyboard(),
                )
            except Exception:
                pass
    finally:
        await store.end_generation(user_id)
        await store.release(user_id)


# ── Неподдерживаемые типы медиа ──────────────────────────────────────────────

async def _try_delete(message: Message) -> None:
    """Удаляем сообщение пользователя. Молча логируем, если не получилось
    (нет прав, сообщение старше 48 часов и т.п.)."""
    try:
        await message.delete()
    except Exception:
        log.debug(
            "Не удалось удалить сообщение %s",
            message.message_id, exc_info=True,
        )


@router.message(F.sticker)
async def handle_sticker(message: Message) -> None:
    """Стикеры Telegram не поддерживаются — объясняем почему."""
    user_id = message.from_user.id
    sticker = message.sticker

    await _try_delete(message)

    if sticker.is_animated:
        kind = "анимированный (Lottie / <code>.tgs</code>)"
    elif sticker.is_video:
        kind = "видео-стикер (<code>.webm</code>)"
    else:
        kind = "статичный (<code>.webp</code>)"

    emoji = f" {sticker.emoji}" if sticker.emoji else ""

    sent = await message.answer(
        f"🎨 <b>Стикеры не поддерживаются.</b>{emoji}\n\n"
        f"Тип: {kind}.\n"
        f"LM Studio принимает только текст и обычные изображения "
        f"(JPG / PNG / фото Telegram). Стикер — это служебный формат "
        f"Telegram, а не картинка, и в OpenAI-совместимый API он не "
        f"передаётся.\n\n"
        f"💡 Если хочешь, чтобы модель «увидела» картинку — "
        f"отправь её как <b>фото</b> или <b>документ</b>."
    )
    await store.track_message(user_id, sent.chat.id, sent.message_id)


@router.message(F.video | F.video_note | F.animation)
async def handle_video(message: Message) -> None:
    """Видео, видеосообщения (кружки) и GIF-анимации."""
    user_id = message.from_user.id

    await _try_delete(message)

    sent = await message.answer(
        "🎬 <b>Видео не поддерживается.</b>\n\n"
        "LM Studio не умеет обрабатывать видеофайлы и анимации. "
        "Отправь текстовый запрос или изображение (фото / документ)."
    )
    await store.track_message(user_id, sent.chat.id, sent.message_id)


@router.message(F.voice | F.audio)
async def handle_audio(message: Message) -> None:
    """Голосовые и аудио."""
    user_id = message.from_user.id

    await _try_delete(message)

    sent = await message.answer(
        "🎧 <b>Аудио и голосовые не поддерживаются.</b>\n\n"
        "Отправь текстовый запрос или изображение."
    )
    await store.track_message(user_id, sent.chat.id, sent.message_id)


@router.message(F.text.startswith("/"))
async def cmd_unknown(message: Message) -> None:
    sent = await message.answer(
        "❓ Неизвестная команда. Список команд: /help"
    )
    await store.track_message(message.from_user.id, sent.chat.id, sent.message_id)