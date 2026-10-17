import logging
import html

from aiogram.filters import Command
from aiogram.types import Message
from typing import Optional
from ..core import router
from ..formatting import response_keyboard
from ..lm import (
    fetch_models,
    get_model,
    list_models_native,
    list_loaded_instances,
    load_model,
    unload_all_models,
    unload_model,
    reload_model,
    run_with_progress,
)
from ..logging_utils import delete_tracked_messages
from ..storage import store
from .. import config
from ..keyboards import model_actions_keyboard, models_keyboard, confirm_clear_keyboard
from ..formatting import safe_code
from ..decor import (
    context_bar,
    flip_coin,
    random_greeting,
    roll_dice,
    uptime_str,
)

log = logging.getLogger(__name__)


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    user_id = message.from_user.id
    await store.clear(user_id)
    await delete_tracked_messages(user_id)

    text = (
        f"<b>{random_greeting()}</b>\n\n"
        "📝 Отправьте текст, фото или альбом — LM Studio обработает запрос.\n"
        "🔄 Кнопка <b>Заново</b> — перегенерировать последний ответ.\n"
        "⏹ Кнопка <b>Остановить</b> или команда <code>/stop</code> — прервать генерацию.\n"
        "🧹 <code>/clean</code> — очистить историю диалога.\n"
        "📊 <code>/models</code> — управление моделями (список, загрузка, выгрузка).\n\n"
        "🚫 <b>Не поддерживается:</b> стикеры, видео, GIF, голосовые, аудио.\n"
        "⚠️ Пока идёт генерация, новые сообщения игнорируются.\n"
        "🔧 <code>/settings</code> — параметры генерации.\n"
        "🧠 <code>/think</code> — thinking.\n"
        "ℹ️ <code>/about</code> — о боте"
    )
    sent = await message.answer(text)
    await store.track_message(user_id, sent.chat.id, sent.message_id)


@router.message(Command("clean", "clear"))
async def cmd_clean(message: Message) -> None:
    """Запросить подтверждение очистки (для /clean и /clear)."""
    user_id = message.from_user.id
    sent = await message.answer(
        "🧹 <b>Очистить историю?</b>\n\n"
        "Все сообщения диалога и история будут удалены.\n"
        "Это действие нельзя отменить.",
        reply_markup=confirm_clear_keyboard(),
    )
    await store.track_message(user_id, sent.chat.id, sent.message_id)


@router.message(Command("stop"))
async def cmd_stop(message: Message) -> None:
    user_id = message.from_user.id
    ok = await store.request_stop(user_id)
    sent = await message.answer(
        "⏹ Останавливаю генерацию…" if ok else "Сейчас нечего останавливать."
    )
    await store.track_message(user_id, sent.chat.id, sent.message_id)


@router.message(Command("models"))
async def cmd_models(message: Message) -> None:
    """Показать все модели с возможностью управления."""
    user_id = message.from_user.id

    native_models = await list_models_native()
    if not native_models:
        sent = await message.answer("❌ LM Studio недоступен или нет моделей.")
        await store.track_message(user_id, sent.chat.id, sent.message_id)
        return

    loaded = {inst["model_key"] for inst in await list_loaded_instances()}

    lines = ["<b>📊 Модели LM Studio</b>\n"]
    for m in native_models:
        key = m.get("key", "?")
        name = m.get("display_name", key)
        status = "🟢 загружена" if key in loaded else "⚪ не загружена"
        lines.append(f"{status} — <code>{safe_code(name)}</code>")

    sent = await message.answer(
        "\n".join(lines),
        reply_markup=models_keyboard([m["key"] for m in native_models if m.get("key")]),
    )
    await store.track_message(user_id, sent.chat.id, sent.message_id)

@router.message(Command("load"))
async def cmd_load(message: Message) -> None:
    """Загрузить модель: /load <model_key> [context_length]"""
    user_id = message.from_user.id
    args = (message.text or "").split(maxsplit=2)

    if len(args) < 2:
        sent = await message.answer(
            "❌ Укажите модель: <code>/load model_key</code>\n"
            "Опционально: <code>/load model_key 8192</code>"
        )
        await store.track_message(user_id, sent.chat.id, sent.message_id)
        return

    model_key = args[1]
    ctx: int | None = None
    if len(args) >= 3:
        try:
            ctx = int(args[2])
        except ValueError:
            sent = await message.answer("❌ context_length должен быть числом.")
            await store.track_message(user_id, sent.chat.id, sent.message_id)
            return

    if not await store.try_acquire_model_op():
        sent = await message.answer("⏳ Идёт другая операция с моделью, подождите…")
        await store.track_message(user_id, sent.chat.id, sent.message_id)
        return

    try:
        sent = await message.answer(f"⏳ Загружаю <code>{safe_code(model_key)}</code>…")
        await store.track_message(user_id, sent.chat.id, sent.message_id)
        header = f"⏳ Загружаю <code>{safe_code(model_key)}</code>"
        try:
            result = await run_with_progress(
                header, sent, load_model(model_key, ctx)
            )
            await sent.edit_text(
                f"✅ Модель <code>{safe_code(model_key)}</code> загружена.\n"
                f"Instance ID: <code>{safe_code(result.get('instance_id', '?'))}</code>"
            )
        except ValueError as exc:
            log.warning("Load model rejected: %s", exc)
            await sent.edit_text(f"⚠️ {safe_code(exc)}")
        except Exception as exc:
            log.exception("Load model failed")
            await sent.edit_text(
                f"❌ Не удалось загрузить: <code>{safe_code(exc)}</code>"
            )
    finally:
        await store.release_model_op()


@router.message(Command("unload"))
async def cmd_unload(message: Message) -> None:
    """Выгрузить модель: /unload <instance_id>  или  /unload all"""
    user_id = message.from_user.id
    args = (message.text or "").split(maxsplit=1)

    if len(args) < 2:
        sent = await message.answer(
            "❌ Укажите instance_id: <code>/unload instance_id</code>\n"
            "Или выгрузить всё: <code>/unload all</code>"
        )
        await store.track_message(user_id, sent.chat.id, sent.message_id)
        return

    target = args[1].strip()

    if not await store.try_acquire_model_op():
        sent = await message.answer("⏳ Идёт другая операция с моделью, подождите…")
        await store.track_message(user_id, sent.chat.id, sent.message_id)
        return

    try:
        if target.lower() == "all":
            sent = await message.answer("⏳ Выгружаю все модели…")
            await store.track_message(user_id, sent.chat.id, sent.message_id)
            results = await unload_all_models()
            sent = await message.answer(f"✅ Выгружено моделей: {len(results)}")
        else:
            sent = await message.answer(f"⏳ Выгружаю <code>{safe_code(target)}</code>…")
            await store.track_message(user_id, sent.chat.id, sent.message_id)
            try:
                await unload_model(target)
                sent = await message.answer(
                    f"✅ Instance <code>{safe_code(target)}</code> выгружен."
                )
            except Exception as exc:
                log.exception("Unload failed")
                sent = await message.answer(
                    f"❌ Не удалось выгрузить: <code>{safe_code(exc)}</code>"
                )
        await store.track_message(user_id, sent.chat.id, sent.message_id)
    finally:
        await store.release_model_op()


@router.message(Command("reload"))
async def cmd_reload(message: Message) -> None:
    """Перезагрузить текущую модель (или указанную): /reload [model_key] [context_length]"""
    user_id = message.from_user.id
    args = (message.text or "").split(maxsplit=2)

    model_key: Optional[str] = None
    ctx: Optional[int] = None

    if len(args) >= 2:
        model_key = args[1]
        if len(args) >= 3:
            try:
                ctx = int(args[2])
            except ValueError:
                sent = await message.answer("❌ context_length должен быть числом.")
                await store.track_message(user_id, sent.chat.id, sent.message_id)
                return
    else:
        loaded = await list_loaded_instances()
        if not loaded:
            sent = await message.answer(
                "❌ Нет загруженных моделей — нечего перезагружать.\n"
                "Загрузите модель: <code>/load model_key</code>"
            )
            await store.track_message(user_id, sent.chat.id, sent.message_id)
            return
        if len(loaded) > 1:
            names = ", ".join(
                sorted({
                    (inst.get("display_name") or inst.get("model_key") or "?")
                    for inst in loaded
                })
            )
            sent = await message.answer(
                f"⚠️ Загружено несколько моделей: {names}.\n"
                f"Укажите явно: <code>/reload model_key</code>"
            )
            await store.track_message(user_id, sent.chat.id, sent.message_id)
            return
        inst = loaded[0]
        model_key = inst.get("model_key")
        ctx = inst.get("context_length")

    if not await store.try_acquire_model_op():
        sent = await message.answer("⏳ Идёт другая операция с моделью, подождите…")
        await store.track_message(user_id, sent.chat.id, sent.message_id)
        return

    try:
        sent = await message.answer(f"⏳ Перезагружаю <code>{safe_code(model_key)}</code>…")
        await store.track_message(user_id, sent.chat.id, sent.message_id)
        header = f"🔄 Перезагружаю <code>{safe_code(model_key)}</code>"
        try:
            result = await run_with_progress(
                header, sent, reload_model(model_key, ctx)
            )
            await sent.edit_text(
                f"✅ Модель <code>{safe_code(model_key)}</code> перезагружена.\n"
                f"Instance ID: <code>{safe_code(result.get('instance_id', '?'))}</code>"
            )
        except ValueError as exc:
            log.warning("Reload model rejected: %s", exc)
            await sent.edit_text(f"⚠️ {safe_code(exc)}")
        except Exception as exc:
            log.exception("Reload failed")
            await sent.edit_text(
                f"❌ Не удалось перезагрузить: <code>{safe_code(exc)}</code>"
            )
        await store.track_message(user_id, sent.chat.id, sent.message_id)
    finally:
        await store.release_model_op()


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    """Показать список команд."""
    user_id = message.from_user.id

    text = (
        "<b>📖 Список команд</b>\n\n"
        "🎮 <b>Основное</b>\n"
        "/start — начать\n"
        "/help — этот список\n"
        "/about — о боте и аптайм\n"
        "/clean — очистить историю и сообщения\n"
        "/stop — остановить генерацию\n\n"
        "🧠 <b>Модель и параметры</b>\n"
        "/models — список моделей LM Studio\n"
        "/load — загрузить модель\n"
        "/tools — включить/выключить MCP-инструменты\n"
        "/think — включить/выключить thinking (Qwen3)\n"
        "/unload — выгрузить модель (<code>/unload all</code> — всё)\n"
        "/reload — перезагрузить текущую модель\n"
        "/settings — показать/изменить параметры генерации\n"
        "/temp — температура (всем)\n"
        "/topk, /topp, /ctxlimit, /maxtokens — 🔒 только для админа\n"
        "/system — системный промпт\n"
        "/reset_settings — сбросить настройки\n\n"
        "🔐 <b>Админ</b>\n"
        "/sudo пароль — войти в админ-режим\n"
        "/sudologout — выйти\n\n"
        "🎲 <b>Пасхалки</b>\n"
        "/dice [N] — бросить N кубиков (по умолчанию 2)\n"
        "/coin — орёл или решка"
    )
    sent = await message.answer(text)
    await store.track_message(user_id, sent.chat.id, sent.message_id)

@router.message(Command("settings"))
async def cmd_settings(message: Message) -> None:
    user_id = message.from_user.id
    s = await store.get_settings(user_id)
    loaded_ctx = await store.get_loaded_context_length()

    limit = loaded_ctx if loaded_ctx > 0 else s.context_limit
    used = s.last_total_tokens

    # ── Текущее состояние переключателей ──
    thinking_on = await store.is_thinking_enabled()
    tools_on = await store.is_tools_enabled()
    thinking_str = "🧠 включён" if thinking_on else "⚡ выключен"
    tools_str = "✅ включены" if tools_on else "🔕 выключены"

    text = (
        "<b>⚙️ Настройки генерации</b>\n\n"
        f"🌡 Температура: <code>{s.temperature}</code>\n"
        f"🎯 top_k: <code>{s.top_k}</code>🔒\n"
        f"🎲 top_p: <code>{s.top_p}</code>🔒\n"
        f"📏 Лимит контекста (LM Studio): <code>{limit}</code>🔒\n"
        f"📝 Макс. токенов за 1 ответ: <code>{s.max_tokens}</code>🔒\n\n"
        f"<b>Переключатели</b>\n"
        f"🧠 Thinking: {thinking_str} — <code>/think on|off</code>\n"
        f"🛠 Инструменты (MCP): {tools_str} — <code>/tools on|off</code>\n\n"
        "<b>Изменить:</b>\n"
        "<code>/temp 1.2</code> — температура (0.0–2.0)\n"
        "<code>/topk 30</code> — top_k 🔒\n"
        "<code>/topp 0.9</code> — top_p 🔒\n"
        "<code>/ctxlimit 30000</code> — лимит контекста 🔒\n"
        "<code>/maxtokens 4096</code> — максимум токенов 🔒\n"
    )
    sent = await message.answer(text)
    await store.track_message(user_id, sent.chat.id, sent.message_id)


async def _set_numeric(message: Message, name: str, **limits) -> None:
    """Общий парсер для /temp /topk /topp /ctxlimit /maxtokens."""
    user_id = message.from_user.id
    args = (message.text or "").split(maxsplit=1)
    if len(args) < 2:
        sent = await message.answer(
    f"❌ Использование: <code>/{name} &lt;значение&gt;</code>"
)
        await store.track_message(user_id, sent.chat.id, sent.message_id)
        return

    raw = args[1].strip().replace(",", ".")
    try:
        val_float = float(raw)
    except ValueError:
        sent = await message.answer("❌ Значение должно быть числом.")
        await store.track_message(user_id, sent.chat.id, sent.message_id)
        return

    lo = limits.get("min")
    hi = limits.get("max")
    is_int = limits.get("int", False)

    if lo is not None and val_float < lo or hi is not None and val_float > hi:
        sent = await message.answer(f"❌ Диапазон: {lo} – {hi}.")
        await store.track_message(user_id, sent.chat.id, sent.message_id)
        return

    value = int(val_float) if is_int else val_float
    field = limits["field"]
    await store.update_settings(user_id, **{field: value})
    sent = await message.answer(f"✅ <b>{name}</b>: <code>{value}</code>")
    await store.track_message(user_id, sent.chat.id, sent.message_id)


@router.message(Command("temp"))
async def cmd_temp(message: Message) -> None:
    await _set_numeric(message, "temp", field="temperature", min=0.0, max=2.0)


@router.message(Command("topk"))
async def cmd_topk(message: Message) -> None:
    if not await _require_admin(message):
        return
    await _set_numeric(message, "topk", field="top_k", min=1, max=200, int=True)


@router.message(Command("topp"))
async def cmd_topp(message: Message) -> None:
    if not await _require_admin(message):
        return
    await _set_numeric(message, "topp", field="top_p", min=0.0, max=1.0)


@router.message(Command("ctxlimit"))
async def cmd_ctxlimit(message: Message) -> None:
    if not await _require_admin(message):
        return
    await _set_numeric(
        message, "ctxlimit", field="context_limit", min=1000, max=200000, int=True
    )


@router.message(Command("maxtokens"))
async def cmd_maxtokens(message: Message) -> None:
    if not await _require_admin(message):
        return
    await _set_numeric(
        message, "maxtokens", field="max_tokens", min=64, max=32768, int=True
    )


@router.message(Command("reset_settings"))
async def cmd_reset_settings(message: Message) -> None:
    user_id = message.from_user.id
    await store.reset_settings(user_id)
    sent = await message.answer("✅ Настройки сброшены к дефолтным.")
    await store.track_message(user_id, sent.chat.id, sent.message_id)

# ── Админ-доступ ─────────────────────────────────────────────────────────────

async def _require_admin(message: Message) -> bool:
    """True, если пользователь — админ (по ID или по активной sudo-сессии).

    Если ADMIN_PASSWORD не задан и ADMIN_IDS пуст — не блокируем.
    """
    user_id = message.from_user.id

    # 1. Постоянные админы из .env — без пароля.
    if user_id in config.ADMIN_IDS:
        return True

    # 2. Если ни пароль, ни список ID не настроены — не блокируем.
    if not config.ADMIN_PASSWORD:
        return True

    # 3. Активная sudo-сессия.
    if await store.is_admin(user_id):
        return True

    sent = await message.answer(
        "🔒 Эта настройка доступна только админу.\n\n"
        "Введите пароль: <code>/sudo ваш_пароль</code>"
    )
    await store.track_message(user_id, sent.chat.id, sent.message_id)
    return False


@router.message(Command("sudo"))
async def cmd_sudo(message: Message) -> None:
    user_id = message.from_user.id

    if not config.ADMIN_PASSWORD:
        sent = await message.answer(
            "⚠️ ADMIN_PASSWORD не задан в .env — команда недоступна."
        )
        await store.track_message(user_id, sent.chat.id, sent.message_id)
        return

    args = (message.text or "").split(maxsplit=1)
    if len(args) < 2:
        sent = await message.answer(
            "❌ Использование: <code>/sudo пароль</code>\n"
            "Выход из админ-режима: <code>/sudologout</code>"
        )
        await store.track_message(user_id, sent.chat.id, sent.message_id)
        return

    if args[1].strip() != config.ADMIN_PASSWORD:
        log.warning("Failed sudo attempt from user %s", user_id)
        sent = await message.answer("❌ Неверный пароль.")
        await store.track_message(user_id, sent.chat.id, sent.message_id)
        return

    await store.grant_admin(user_id, config.ADMIN_SESSION_TTL)
    mins = max(1, int(config.ADMIN_SESSION_TTL // 60))
    sent = await message.answer(
        f"✅ Админ-доступ выдан на <b>{mins} мин</b>.\n"
        "Доступно: <code>/topk</code>, <code>/topp</code>, "
        "<code>/ctxlimit</code>, <code>/maxtokens</code>\n"
        "Выйти: <code>/sudologout</code>"
    )
    await store.track_message(user_id, sent.chat.id, sent.message_id)


@router.message(Command("sudologout"))
async def cmd_sudologout(message: Message) -> None:
    user_id = message.from_user.id
    await store.revoke_admin(user_id)
    sent = await message.answer("👋 Админ-сессия завершена.")
    await store.track_message(user_id, sent.chat.id, sent.message_id)

@router.message(Command("system"))
async def cmd_system(message: Message) -> None:
    """Показать/задать/сбросить кастомный системный промпт.

    /system                — показать текущий
    /system <текст>        — задать новый (многострочный тоже поддерживается)
    /system reset          — сбросить
    """
    user_id = message.from_user.id
    parts = (message.text or "").split(maxsplit=1)
    arg = parts[1].strip() if len(parts) > 1 else ""

    if not arg:
        current = (await store.get_system_prompt(user_id)).strip()
        if current:
            body = (
                "<b>🧠 Системный промпт</b>\n\n"
                f"<pre>{html.escape(current)}</pre>\n\n"
                "<b>Изменить:</b> <code>/system &lt;текст&gt;</code>\n"
                "<b>Сбросить:</b> <code>/system reset</code>"
            )
        else:
            body = (
                "<b>🧠 Системный промпт не задан</b>\n\n"
                "Установить: <code>/system Ты — полезный ассистент.</code>\n"
                "Сбросить: <code>/system reset</code>\n\n"
                "Промпт добавляется <b>перед</b> каждым запросом к модели."
            )
        sent = await message.answer(body)
        await store.track_message(user_id, sent.chat.id, sent.message_id)
        return

    if arg.lower() in ("reset", "clear", "off", "default"):
        await store.clear_system_prompt(user_id)
        await store.reset_api_state(user_id)
        sent = await message.answer(
            "✅ Системный промпт сброшен.\n"
            "Следующее сообщение начнёт новую сессию с моделью."
        )
        await store.track_message(user_id, sent.chat.id, sent.message_id)
        return

    if len(arg) > config.MAX_SYSTEM_PROMPT_CHARS:
        sent = await message.answer(
            f"❌ Слишком длинный промпт: <b>{len(arg)}</b> символов. "
            f"Максимум — <b>{config.MAX_SYSTEM_PROMPT_CHARS}</b>."
        )
        await store.track_message(user_id, sent.chat.id, sent.message_id)
        return

    await store.set_system_prompt(user_id, arg)
    # Критично: сбрасываем previous_response_id, иначе LM Studio продолжит
    # старую цепочку и новый системный промпт до модели не дойдёт.
    await store.reset_api_state(user_id)
    sent = await message.answer(
        f"✅ Системный промпт сохранён ({len(arg)} символов).\n"
        "Со следующего сообщения начнётся новая сессия с моделью, "
        "и промпт будет применён.\n"
        "Посмотреть: <code>/system</code> · Сбросить: <code>/system reset</code>"
    )
    await store.track_message(user_id, sent.chat.id, sent.message_id)

# ── Пасхалки ─────────────────────────────────────────────────────────────────

@router.message(Command("dice", "roll"))
async def cmd_dice(message: Message) -> None:
    """Бросок кубиков: /dice [1-6]"""
    user_id = message.from_user.id
    args = (message.text or "").split(maxsplit=1)
    count = 2
    if len(args) >= 2:
        try:
            count = int(args[1])
        except ValueError:
            count = 2
    sent = await message.answer(roll_dice(count))
    await store.track_message(user_id, sent.chat.id, sent.message_id)


@router.message(Command("coin", "flip"))
async def cmd_coin(message: Message) -> None:
    """Орёл или решка."""
    user_id = message.from_user.id
    sent = await message.answer(flip_coin())
    await store.track_message(user_id, sent.chat.id, sent.message_id)


# ── О боте ───────────────────────────────────────────────────────────────────

@router.message(Command("about"))
async def cmd_about(message: Message) -> None:
    """Карточка бота: версия, аптайм, лимиты."""
    user_id = message.from_user.id

    model_display = config.DEFAULT_MODEL or "— авто —"

    text = (
        "🤖 <b>О боте</b>\n\n"
        "Телеграм-обёртка над <b>LM Studio</b>.\n"
        "Поддерживает текст, фото, альбомы и кастомные системные промпты.\n\n"
        f"⏱ <b>Аптайм:</b> <code>{uptime_str()}</code>\n"
        f"🧠 <b>Модель:</b> <code>{safe_code(model_display)}</code>\n"
        f"📚 <b>История:</b> до <b>{config.MAX_HISTORY_TURNS}</b> пар сообщений\n"
        f"🖼 <b>Картинок за раз:</b> {config.MAX_IMAGES} "
        f"(до {config.MAX_IMAGES_EXTENDED} в расширенном режиме)\n"
        f"✂️ <b>Макс. длина запроса:</b> {config.MAX_INPUT_CHARS} символов"
    )
    sent = await message.answer(text)
    await store.track_message(user_id, sent.chat.id, sent.message_id)


@router.message(Command("tools"))
async def cmd_tools(message: Message) -> None:
    """Включить/выключить использование MCP-инструментов.

    /tools          — показать статус
    /tools on       — включить
    /tools off      — выключить
    """
    user_id = message.from_user.id
    args = (message.text or "").split(maxsplit=1)
    arg = args[1].strip().lower() if len(args) > 1 else ""

    if arg in ("on", "1", "true", "yes"):
        await store.set_tools_enabled(True)
        sent = await message.answer(
            "✅ <b>Инструменты включены.</b>\n"
            "Модель будет использовать все MCP-серверы из <code>mcp.json</code>."
        )
    elif arg in ("off", "0", "false", "no"):
        await store.set_tools_enabled(False)
        sent = await message.answer(
            "🔕 <b>Инструменты выключены.</b>\n"
            "Модель будет отвечать без вызова MCP-инструментов."
        )
    else:
        enabled = await store.is_tools_enabled()
        status = "✅ включены" if enabled else "🔕 выключены"
        sent = await message.answer(
            f"<b>🛠 Инструменты (MCP):</b> {status}\n\n"
            "Использование:\n"
            "<code>/tools on</code> — включить\n"
            "<code>/tools off</code> — выключить"
        )
    await store.track_message(user_id, sent.chat.id, sent.message_id)

@router.message(Command("think"))
async def cmd_think(message: Message) -> None:
    """Управление thinking-режимом (Qwen3 / reasoning модели).

    /think          — показать статус
    /think on       — включить
    /think off      — выключить
    """
    user_id = message.from_user.id
    args = (message.text or "").split(maxsplit=1)
    arg = args[1].strip().lower() if len(args) > 1 else ""

    if arg in ("on", "1", "true", "yes"):
        await store.set_thinking_enabled(True)
        sent = await message.answer(
            "🧠 <b>Thinking включён.</b>\n"
            "Модель будет рассуждать перед ответом "
            "(медленнее, но точнее для сложных задач)."
        )
    elif arg in ("off", "0", "false", "no"):
        await store.set_thinking_enabled(False)
        sent = await message.answer(
            "⚡ <b>Thinking выключен.</b>\n"
            "Модель будет отвечать сразу, без блока размышлений "
            "(быстрее, экономит токены). Работает для Qwen3 и "
            "других reasoning-моделей через chat template."
        )
    else:
        enabled = await store.is_thinking_enabled()
        status = "🧠 включён" if enabled else "⚡ выключен"
        sent = await message.answer(
            f"<b>Thinking:</b> {status}\n\n"
            "Использование:\n"
            "<code>/think on</code> — включить\n"
            "<code>/think off</code> — выключить"
        )
    await store.track_message(user_id, sent.chat.id, sent.message_id)