from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from typing import List, Dict, Any


def response_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="🔄 Заново", callback_data="regenerate"),
        InlineKeyboardButton(text="🧹 Очистить", callback_data="clean_chat"),
    ]])


def stop_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="⏹ Остановить", callback_data="stop_gen"),
    ]])


def models_keyboard(models: List[str]) -> InlineKeyboardMarkup:
    """Клавиатура для выбора модели из списка."""
    buttons = [
        [InlineKeyboardButton(text=f"📦 {m}", callback_data=f"model_select:{m}")]
        for m in models
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def model_actions_keyboard(model_key: str, is_loaded: bool) -> InlineKeyboardMarkup:
    """Клавиатура действий с конкретной моделью."""
    rows: List[List[InlineKeyboardButton]] = []
    if is_loaded:
        rows.append([
            InlineKeyboardButton(
                text="🔄 Перезагрузить",
                callback_data=f"model_reload:{model_key}",
            ),
            InlineKeyboardButton(
                text="⏏ Выгрузить",
                callback_data=f"model_unload:{model_key}",
            ),
        ])
    else:
        rows.append([
            InlineKeyboardButton(
                text="📥 Загрузить",
                callback_data=f"model_load:{model_key}",
            ),
        ])
    rows.append([
        InlineKeyboardButton(text="◀ Назад", callback_data="models_list"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def confirm_clear_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура подтверждения очистки истории."""
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="🗑 Да, очистить", callback_data="clean_confirm"),
        InlineKeyboardButton(text="↩️ Отмена",       callback_data="clean_cancel"),
    ]])