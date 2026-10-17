"""Простые декоративные элементы: случайные фразы, красивая статистика,
полоски контекста, аптайм и мелкие пасхалки."""

from __future__ import annotations

import random
import time
from typing import Any, Dict, Optional


# ─────────────────────────── Время работы ───────────────────────────

_START_TIME: float = time.monotonic()


def uptime_str() -> str:
    """Человекочитаемый аптайм процесса бота."""
    secs = int(time.monotonic() - _START_TIME)
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    mins, s = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days} д")
    if hours:
        parts.append(f"{hours} ч")
    if mins:
        parts.append(f"{mins} мин")
    if not parts:
        parts.append(f"{s} с")
    return " ".join(parts)


# ─────────────────── Случайные фразы "думаю" ───────────────────

THINKING_PHRASES = (
    "🤔 Думаю",
    "💭 Размышляю",
    "🧠 Соображаю",
    "✨ Генерирую",
    "🌀 Обрабатываю",
    "📝 Пишу ответ",
    "🔮 Ворожею",
    "🎨 Творю",
    "🛰️ Запрашиваю модель",
    "⚙️ Считаю токены",
)


def random_thinking_phrase() -> str:
    """Случайная фраза для анимации ожидания."""
    return random.choice(THINKING_PHRASES)


# ─────────────────── Случайные приветствия ───────────────────

GREETINGS = (
    "Привет-привет! 👋",
    "Приветствую! 🤖",
    "На связи! 🛰️",
    "Готов к работе! ⚡",
    "Здравствуй! ✨",
    "Салют! 🚀",
    "Привет! Чем помочь? 💬",
)


def random_greeting() -> str:
    """Случайное приветствие для /start."""
    return random.choice(GREETINGS)


# ─────────────────── Полоска заполнения контекста ───────────────────

def context_bar(used: int, total: int, width: int = 10) -> str:
    """Визуальная полоска использования контекста.

    Пример: 🟢 <code>▰▰▰▰▱▱▱▱▱▱</code> 42%
    """
    if total <= 0:
        return ""
    ratio = max(0.0, min(1.0, used / total))
    filled = round(width * ratio)
    pct = int(ratio * 100)

    if ratio >= 0.95:
        icon = "🔴"
    elif ratio >= 0.80:
        icon = "🟡"
    else:
        icon = "🟢"

    bar = "▰" * filled + "▱" * (width - filled)
    return f"{icon} <code>{bar}</code> {pct}%"


# ─────────────────── Красивая статистика после ответа ───────────────────
def render_stats(usage, elapsed, settings=None, loaded_ctx=0, used_tokens=None):
    if settings is None:
        return ""
    if used_tokens is not None:
        used = int(used_tokens)
    else:
        used = int(usage.get("prompt_tokens", 0) or 0)
    limit = loaded_ctx or getattr(settings, "context_limit", 0) or 0
    if limit <= 0 or used <= 0:
        return ""
    return "\n\n📚 " + context_bar(used, limit)

# ─────────────────── Случайные футеры-подсказки ───────────────────

FOOTER_TIPS = (
    "💡 <i>Кнопка «Заново» — перегенерировать последний ответ</i>",
    "💡 <i>/system — задать характер и роль модели</i>",
    "💡 <i>Отправь фото — модель его «увидит»</i>",
    "💡 <i>/settings — все параметры генерации</i>",
    "💡 <i>/clean — начать диалог с чистого листа</i>",
    "💡 <i>/models — управление моделями LM Studio</i>",
    "💡 <i>/stop — прервать генерацию в любой момент</i>",
)


def maybe_tip(chance: float = 0.20) -> str:
    """С вероятностью `chance` вернуть случайную подсказку-футер."""
    if random.random() > chance:
        return ""
    return "\n\n" + random.choice(FOOTER_TIPS)


# ─────────────────── Мини-пасхалки ───────────────────

_DICE_FACES = ("⚀", "⚁", "⚂", "⚃", "⚄", "⚅")

# Приколы для случая, когда все кубики выпали одинаково
_ALL_SAME_JOKES = {
    1: [
        "😅 Все единицы! Не твой день, братан.",
        "😭 Единицы... Может, перебросишь?",
        "🥴 Один-один-один... Как скучно.",
    ],
    2: [
        "🤔 Две двойки... Зеркало?",
        "👯 Двойняшки!",
        "✌️ Двойки — это тоже результат.",
    ],
    3: [
        "😐 Три тройки. Магия числа 3?",
        "🔺 Три — священное число!",
        "🎩 Три-три-три, цирк приехал!",
    ],
    4: [
        "🍀 Четыре четвёрки! Квартет!",
        "🚗 Четыре колеса! Поехали!",
        "🏠 Четыре стены — уже дом!",
    ],
    5: [
        "✋ Пять пятёрок! Отличный бросок!",
        "🖐 Пять — это много!",
        "⭐ Пять звёзд!",
    ],
    6: [
        "🔥 ШЕСТЬ ШЕСТЁРОК! ДЬЯВОЛЬСКАЯ УДАЧА!",
        "👹 666! Сатана доволен.",
        "🎰 Джекпот! Три шестёрки и ещё три!",
    ],
}


def roll_dice(count: int = 2) -> str:
    """Бросок кубиков с эмодзи-гранями и приколом при одинаковых значениях."""
    count = max(1, min(6, int(count)))
    rolls = [random.randint(1, 6) for _ in range(count)]
    faces = " ".join(_DICE_FACES[r - 1] for r in rolls)
    total = sum(rolls)
    result = f"🎲 {faces}  →  <b>{total}</b>"

    if count > 1 and len(set(rolls)) == 1:
        val = rolls[0]
        jokes = _ALL_SAME_JOKES.get(val)
        if jokes:
            result += "\n\n" + random.choice(jokes)
        else:
            result += "\n\n😲 Все кубики одинаковые! Вот это совпадение!"

    return result


def flip_coin() -> str:
    """Орёл или решка."""
    if random.random() < 0.5:
        return "📈 <b>Орёл</b>"
    return "📉 <b>Решка</b>"