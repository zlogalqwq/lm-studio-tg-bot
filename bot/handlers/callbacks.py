import logging

from typing import Optional
from aiogram import F
from aiogram.types import CallbackQuery, Message, InlineKeyboardMarkup
import httpx
from ..logging_utils import (
    delete_tracked_messages,
    finalize_and_track,
    save_exchange,
)

from ..core import bot, router
from .. import config
from ..formatting import finalize_streamed, safe_code
from ..keyboards import (
    stop_keyboard,
    model_actions_keyboard,
    models_keyboard,
    confirm_clear_keyboard,
    response_keyboard,
)
from ..lm import (
    chat_completion,
    get_model,
    list_loaded_instances,
    load_model,
    unload_model,
    reload_model,
    list_models_native,
    run_with_progress,
    sync_loaded_context_length,
    estimate_context_tokens,
)

from ..storage import store
from ..decor import maybe_tip, render_stats

log = logging.getLogger(__name__)


# ── Существующие колбэки ──

@router.callback_query(F.data == "clean_chat")
async def cb_clean(query: CallbackQuery) -> None:
    """Показать подтверждение, не уничтожая исходный ответ."""
    sent = await query.message.answer(
        "🧹 <b>Очистить историю?</b>\n\n"
        "Все сообщения диалога и история будут удалены.\n"
        "Это действие нельзя отменить.",
        reply_markup=confirm_clear_keyboard(),
    )
    await store.track_message(query.from_user.id, sent.chat.id, sent.message_id)
    await query.answer()


@router.callback_query(F.data == "clean_confirm")
async def cb_clean_confirm(query: CallbackQuery) -> None:
    """Собственно очистка после подтверждения."""
    user_id = query.from_user.id
    chat_id = query.message.chat.id

    await query.answer("История очищена ✅", show_alert=False)

    await store.clear(user_id)
    await delete_tracked_messages(user_id)

    try:
        await query.message.delete()
    except Exception:
        pass

    sent = await bot.send_message(chat_id, "✅ История и сообщения очищены.")
    await store.track_message(user_id, sent.chat.id, sent.message_id)


@router.callback_query(F.data == "clean_cancel")
async def cb_clean_cancel(query: CallbackQuery) -> None:
    """Отмена очистки: просто убираем сообщение с подтверждением."""
    await query.answer("Отменено")
    try:
        await query.message.delete()
    except Exception:
        pass


@router.callback_query(F.data == "stop_gen")
async def cb_stop(query: CallbackQuery) -> None:
    user_id = query.from_user.id
    ok = await store.request_stop(user_id)
    if ok:
        await query.answer("⏹ Останавливаю…")
    else:
        await query.answer("Уже нечего останавливать", show_alert=False)


@router.callback_query(F.data == "regenerate")
async def cb_regenerate(query: CallbackQuery) -> None:
    user_id = query.from_user.id

    if not await store.try_acquire(user_id):
        await query.answer("⏳ Уже генерирую, подожди…", show_alert=False)
        return

    stop_event = await store.start_generation(user_id)
    status: Optional[Message] = None
    try:
        last_user_content = await store.get_last_user_input(user_id)
        if last_user_content is None:
            await query.answer("Нечего перегенерировать", show_alert=True)
            return

        # Откатываем локальную историю на один шаг (убираем прошлый assistant).
        await store.drop_last_assistant(user_id)

        # Откатываем состояние LM Studio на один шаг назад.
        await store.rollback_response_id(user_id)
        prev_id = await store.get_response_id(user_id)

        await query.answer("🔄 Генерирую заново…")

        try:
            await query.message.delete()
        except Exception:
            pass

        status = await bot.send_message(
            query.message.chat.id,
            "⏳ Генерирую заново…",
            reply_markup=stop_keyboard(),
        )
        await store.track_message(user_id, status.chat.id, status.message_id)

        try:
            model = await get_model()
            await sync_loaded_context_length()
            loaded_ctx = await store.get_loaded_context_length()
            settings = await store.get_settings(user_id)

            system_prompt = await store.get_system_prompt(user_id)
            thinking_enabled = await store.is_thinking_enabled()

            # История уже без последнего assistant (drop_last_assistant),
            # но с последним user — который мы и регенерируем. Убираем
            # его из history_context, чтобы не дублировать.
            history_for_context = await store.get_with_system(user_id)
            if history_for_context and history_for_context[-1].get("role") == "user":
                history_for_context = history_for_context[:-1]

            reply, usage, elapsed, new_resp_id, session_reset = await chat_completion(
                message=last_user_content,
                model=model,
                status=status,
                previous_response_id=prev_id,
                stop_event=stop_event,
                settings=settings,
                system_prompt=system_prompt,
                use_tools=await store.is_tools_enabled(),
                thinking_enabled=thinking_enabled,
                history_context=history_for_context,
            )

            if session_reset:
                await store.reset_api_state(user_id)
            if new_resp_id:
                await store.set_response_id(user_id, new_resp_id)

            if usage is not None:
                if reply:
                    await store.append_assistant(user_id, reply)
                    await save_exchange(user_id, last_user_content, reply)

                history_now = await store.get_with_system(user_id)
                estimated = estimate_context_tokens(history_now)
                cumulative = await store.update_cumulative_tokens(
                    user_id, usage["prompt_tokens"]
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
                    "Модель не видит предыдущий контекст.</i>"
                ) if session_reset else ""
                stats = (
                    render_stats(
                        usage, elapsed, settings,
                        loaded_ctx=loaded_ctx, used_tokens=effective,
                    )
                    + maybe_tip()
                    + note
                )
            else:
                stats = ""

            stopped = (usage is None)
            sent = await finalize_streamed(
                status, reply, query.message, stopped=stopped, stats=stats
            )
            await finalize_and_track(user_id, status, sent)

        except (RuntimeError, ValueError) as exc:
            try:
                await status.edit_text(
                    f"❌ {safe_code(exc)}",
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
            log.exception("Regenerate error for user %s", user_id)
            try:
                await status.edit_text(
                    "❌ Произошла ошибка при перегенерации.",
                    reply_markup=response_keyboard(),
                )
            except Exception:
                pass
    finally:
        await store.end_generation(user_id)
        await store.release(user_id)

    
# ── Новые колбэки: управление моделями ──

async def _models_list_view() -> tuple[str, "InlineKeyboardMarkup"]:
    """Собирает текст и клавиатуру списка моделей (общий рендер)."""
    native_models = await list_models_native()
    loaded = {inst["model_key"] for inst in await list_loaded_instances()}

    lines = ["<b>📊 Модели LM Studio</b>\n"]
    for m in native_models:
        key = m.get("key", "?")
        name = m.get("display_name", key)
        status = "🟢 загружена" if key in loaded else "⚪ не загружена"
        lines.append(f"{status} — <code>{safe_code(name)}</code>")

    kb = models_keyboard([m["key"] for m in native_models if m.get("key")])
    return "\n".join(lines), kb


@router.callback_query(F.data == "models_list")
async def cb_models_list(query: CallbackQuery) -> None:
    """Вернуться к списку моделей."""
    text, kb = await _models_list_view()
    await query.message.edit_text(text, reply_markup=kb)
    await query.answer()


@router.callback_query(F.data.startswith("model_select:"))
async def cb_model_select(query: CallbackQuery) -> None:
    """Показать действия для выбранной модели."""
    model_key = query.data.split(":", 1)[1]
    loaded = {inst["model_key"] for inst in await list_loaded_instances()}
    is_loaded = model_key in loaded

    status = "🟢 загружена" if is_loaded else "⚪ не загружена"
    await query.message.edit_text(
        f"<b>Модель:</b> <code>{safe_code(model_key)}</code>\n"
        f"<b>Статус:</b> {status}",
        reply_markup=model_actions_keyboard(model_key, is_loaded),
    )
    await query.answer()


@router.callback_query(F.data.startswith("model_load:"))
async def cb_model_load(query: CallbackQuery) -> None:
    model_key = query.data.split(":", 1)[1]

    if not await store.try_acquire_model_op():
        await query.answer("⏳ Идёт другая операция с моделью…", show_alert=True)
        return

    try:
        await query.answer("⏳ Загружаю…")
        header = f"⏳ Загружаю <code>{safe_code(model_key)}</code>"
        try:
            result = await run_with_progress(
                header, query.message, load_model(model_key)
            )
            log.info("load_model(%s) → %r", model_key, result)

            # После успешной загрузки — возвращаемся в список моделей
            text, kb = await _models_list_view()
            await query.message.edit_text(
                f"✅ <code>{safe_code(model_key)}</code> загружена.\n\n{text}",
                reply_markup=kb,
            )
        except ValueError as exc:
            log.warning("Load model rejected: %s", exc)
            text, kb = await _models_list_view()
            await query.message.edit_text(
                f"⚠️ {safe_code(exc)}\n\n{text}",
                reply_markup=kb,
            )
        except Exception as exc:
            log.exception("Load model failed")
            text, kb = await _models_list_view()
            await query.message.edit_text(
                f"❌ Ошибка загрузки: <code>{safe_code(exc)}</code>\n\n{text}",
                reply_markup=kb,
            )
    finally:
        await store.release_model_op()


@router.callback_query(F.data.startswith("model_unload:"))
async def cb_model_unload(query: CallbackQuery) -> None:
    model_key = query.data.split(":", 1)[1]

    if not await store.try_acquire_model_op():
        await query.answer("⏳ Идёт другая операция с моделью…", show_alert=True)
        return

    try:
        instances = [
            inst for inst in await list_loaded_instances()
            if inst["model_key"] == model_key
        ]
        if not instances:
            await query.answer("Модель не загружена", show_alert=True)
            return

        await query.answer("⏏ Выгружаю…")
        try:
            for inst in instances:
                await unload_model(inst["instance_id"])

            # После выгрузки — возвращаемся в список моделей
            text, kb = await _models_list_view()
            await query.message.edit_text(
                f"✅ <code>{safe_code(model_key)}</code> выгружена.\n\n{text}",
                reply_markup=kb,
            )
        except Exception as exc:
            log.exception("Unload model failed")
            text, kb = await _models_list_view()
            await query.message.edit_text(
                f"❌ Ошибка выгрузки: <code>{safe_code(exc)}</code>\n\n{text}",
                reply_markup=kb,
            )
    finally:
        await store.release_model_op()


@router.callback_query(F.data.startswith("model_reload:"))
async def cb_model_reload(query: CallbackQuery) -> None:
    model_key = query.data.split(":", 1)[1]

    if not await store.try_acquire_model_op():
        await query.answer("⏳ Идёт другая операция с моделью…", show_alert=True)
        return

    try:
        await query.answer("🔄 Перезагружаю…")
        header = f"🔄 Перезагружаю <code>{safe_code(model_key)}</code>"
        try:
            result = await run_with_progress(
                header, query.message, reload_model(model_key)
            )
            log.info("reload_model(%s) → %r", model_key, result)

            # После перезагрузки — возвращаемся в список моделей
            text, kb = await _models_list_view()
            await query.message.edit_text(
                f"✅ <code>{safe_code(model_key)}</code> перезагружена.\n\n{text}",
                reply_markup=kb,
            )
        except ValueError as exc:
            log.warning("Reload model rejected: %s", exc)
            text, kb = await _models_list_view()
            await query.message.edit_text(
                f"⚠️ {safe_code(exc)}\n\n{text}",
                reply_markup=kb,
            )
        except Exception as exc:
            log.exception("Reload model failed")
            text, kb = await _models_list_view()
            await query.message.edit_text(
                f"❌ Ошибка перезагрузки: <code>{safe_code(exc)}</code>\n\n{text}",
                reply_markup=kb,
            )
    finally:
        await store.release_model_op()