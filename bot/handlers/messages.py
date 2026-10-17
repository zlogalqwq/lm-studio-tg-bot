import logging
from typing import List, Optional

from aiogram import F
from aiogram.filters import Command
from aiogram.types import Message
import httpx

from ..core import router
from .. import config
from ..formatting import finalize_streamed
from ..images import IMAGE_MIME_PREFIXES, build_content, images_from_album, images_from_message
from ..keyboards import stop_keyboard, response_keyboard
from ..storage import store
from ..logging_utils import save_exchange, finalize_and_track
from ..decor import maybe_tip, render_stats
from ..lm import (
    chat_completion,
    get_model,
    sync_loaded_context_length,
    summarize_history,
)


log = logging.getLogger(__name__)


def _get_prompt_text(message: Message, album: Optional[List[Message]]) -> str:
    """Возвращает текст запроса: для альбома — первую непустую подпись, иначе text/caption."""
    if album:
        return next(
            (m.caption or m.text for m in album if m.caption or m.text),
            "",
        )
    return message.text or message.caption or ""

@router.message(
    (F.text & ~F.text.startswith("/")) | F.photo | F.document
)
async def handle_message(
    message: Message,
    album: Optional[List[Message]] = None,
) -> None:
    user_id = message.from_user.id
    prompt_text = _get_prompt_text(message, album)
    if len(prompt_text) > config.MAX_INPUT_CHARS:
        sent = await message.answer(
            f"❌ Слишком длинный запрос: <b>{len(prompt_text)}</b> символов.\n"
            f"Максимум — <b>{config.MAX_INPUT_CHARS}</b>."
        )
        await store.track_message(user_id, sent.chat.id, sent.message_id)
        return
    if await store.is_model_op_busy():
        log.info("User %s — model op in progress, message ignored", user_id)
        sent = await message.answer(
            "⏳ Сейчас идёт операция с моделью (загрузка/выгрузка). "
            "Подождите, пока она завершится."
        )
        await store.track_message(user_id, sent.chat.id, sent.message_id)
        return
    if not await store.try_acquire(user_id):
        log.info("User %s is busy — message ignored", user_id)
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
                if doc is not None and not (doc.mime_type or "").startswith(IMAGE_MIME_PREFIXES):
                    await status.delete()
                    sent = await message.answer("❌ Поддерживаются только изображения и текст.")
                    await store.track_message(user_id, sent.chat.id, sent.message_id)
                    return

            content = await build_content(album, message, user_id)
            await store.append_user(user_id, content)
            # Перегенерация отправит только текст — картинки уже в состоянии LM Studio.
            await store.set_last_user_input(user_id, prompt_text)

            model = await get_model()
            await sync_loaded_context_length()
            loaded_ctx = await store.get_loaded_context_length()

            settings = await store.get_settings(user_id)
            limit = loaded_ctx if loaded_ctx > 0 else settings.context_limit

            # ── Авторезюмирование при переполнении контекста ──
            # Если прошлый запрос занял почти весь контекст — сохраняем
            # резюме диалога, сбрасываем stateful-сессию и начинаем заново.
            if settings.last_prompt_tokens > 0 and limit > 0:
                ratio = settings.last_prompt_tokens / limit
                if ratio >= config.CONTEXT_AUTO_CLEAR_RATIO:
                    old_history = await store.get(user_id)
                    old_summary = await store.get_summary(user_id)
                    new_summary = await summarize_history(
                        old_history, model, old_summary
                    )
                    await store.reset_context(user_id)
                    if new_summary:
                        await store.set_summary(user_id, new_summary)
                    warn = await message.answer(
                        "🔄 <b>Контекст переполнен.</b> Я сохранил резюме "
                        "диалога и начал новый сеанс.\n"
                        "Показать: <code>/summary show</code> · "
                        "Сбросить: <code>/summary clear</code>"
                    )
                    await store.track_message(
                        user_id, warn.chat.id, warn.message_id
                    )
                elif ratio >= config.CONTEXT_WARN_RATIO:
                    pct = int(ratio * 100)
                    warn = await message.answer(
                        f"⚠️ Контекст заполнен на <b>{pct}%</b> "
                        f"({settings.last_prompt_tokens}/{limit}).\n"
                        f"Очистить: /clean · Резюмировать: /summary"
                    )
                    await store.track_message(
                        user_id, warn.chat.id, warn.message_id
                    )

            # ── Основной запрос к нативному stateful API ──
            prev_id = await store.get_response_id(user_id)
            system_prompt = await store.get_system_prompt(user_id)
            prior_summary = await store.get_summary(user_id)

            reply, usage, elapsed, new_resp_id = await chat_completion(
                message=content,
                model=model,
                status=status,
                previous_response_id=prev_id,
                stop_event=stop_event,
                settings=settings,
                system_prompt=system_prompt,
                prior_summary=prior_summary,
                use_tools=await store.is_tools_enabled(),
            )

            if new_resp_id:
                await store.set_response_id(user_id, new_resp_id)

            if usage is not None:
                await store.set_usage(
                    user_id,
                    usage["prompt_tokens"],
                    usage["completion_tokens"],
                    usage["total_tokens"],
                )
                if reply:
                    await store.append_assistant(user_id, reply)
                    await save_exchange(user_id, content, reply)
                stats = render_stats(
                    usage, elapsed, settings, loaded_ctx=loaded_ctx
                ) + maybe_tip()
            else:
                stats = ""

            stopped = (usage is None)
            sent = await finalize_streamed(
                status, reply, message, stopped=stopped, stats=stats
            )
            await finalize_and_track(user_id, status, sent)

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
        log.debug("Не удалось удалить сообщение %s", message.message_id, exc_info=True)


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