import asyncio
import html
import logging
import re
from typing import List, Optional

from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import InlineKeyboardMarkup, Message

from .keyboards import response_keyboard

log = logging.getLogger(__name__)


_PLACEHOLDER = "\x00{}\x00"
_BOLD_STASH = "\x02B{}\x02"

# Теги разметки Telegram (b, i, u, s, code, pre, a, tg-spoiler, blockquote).
# Атрибуты внутри тега не содержат < или >, поэтому [^<>]* достаточно.
_TAG_RE = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9-]*)[^<>]*>")

def safe_code(value: object, limit: int = 64) -> str:
    """Экранирует значение для вставки внутрь <code>...</code> и обрезает его."""
    if value is None:
        return "?"
    s = str(value)
    if len(s) > limit:
        s = s[: limit - 1] + "…"
    return html.escape(s, quote=False)

def _italic(s: str) -> str:
    return re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<i>\1</i>", s)

def md_to_html(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    stashed: List[str] = []

    def _stash(m: re.Match) -> str:
        lang = m.group(1) or ""
        inner = m.group(2)
        stashed.append(f'<pre><code class="language-{lang}">{inner}</code></pre>')
        return _PLACEHOLDER.format(len(stashed) - 1)

    text = re.sub(r"```(\w*)\n?(.*?)```", _stash, text, flags=re.DOTALL)
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)

    bold_items: List[str] = []

    def _stash_bold(m: re.Match) -> str:
        bold_items.append(_italic(m.group(1)))
        return _BOLD_STASH.format(len(bold_items) - 1)

    text = re.sub(r"\*\*(.+?)\*\*", _stash_bold, text)
    text = _italic(text)

    for i, inner in enumerate(bold_items):
        text = text.replace(_BOLD_STASH.format(i), f"<b>{inner}</b>")

    text = re.sub(r"^#{3,6}\s+(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)
    text = re.sub(r"^(\s*)- ", r"\1• ", text, flags=re.MULTILINE)

    for i, block in enumerate(stashed):
        text = text.replace(_PLACEHOLDER.format(i), block)
    return text

# ─────────────────────────── HTML-нарезание ───────────────────────────

def _open_tags(fragment: str) -> List[str]:
    """Стек незакрытых тегов в фрагменте."""
    stack: List[str] = []
    for m in _TAG_RE.finditer(fragment):
        closing, name = bool(m.group(1)), m.group(2).lower()
        if closing:
            if stack and stack[-1] == name:
                stack.pop()
        else:
            stack.append(name)
    return stack

def _split_html(text: str, limit: int = 4000) -> List[str]:
    """Режет HTML на части ≤ limit, никогда не разрывая тег.

    Если безопасной границы в окне нет (например, попали внутрь большого
    <pre>), режет по лимиту и балансирует теги закрытием/повторным открытием.
    """
    if len(text) <= limit:
        return [text]

    chunks: List[str] = []
    rest = text
    while len(rest) > limit:
        window = rest[:limit]

        cut = 0
        open_at_cut: List[str] = []
        stack: List[str] = []
        for m in _TAG_RE.finditer(window):
            closing, name = bool(m.group(1)), m.group(2).lower()
            if closing:
                if stack and stack[-1] == name:
                    stack.pop()
            else:
                stack.append(name)
            if not stack:
                cut = m.end()
                open_at_cut = []

        # cut==0 или слишком ранний разрез — гарантируем прогресс жёсткой резкой.
        if cut == 0 or cut < limit // 4:
            cut = limit
            open_at_cut = _open_tags(window)

        chunk = rest[:cut].rstrip()
        if open_at_cut:
            chunk += "".join(f"</{t}>" for t in reversed(open_at_cut))
        chunks.append(chunk)

        prefix = "".join(f"<{t}>" for t in open_at_cut)
        rest = (prefix + rest[cut:]).lstrip("\n")

    if rest.strip():
        chunks.append(rest)
    return chunks

# ─────────────────────────── отправка ───────────────────────────

async def _safe_answer(
    target: Message,
    text: str,
    reply_markup: Optional[InlineKeyboardMarkup] = None,
) -> Message:
    """answer() с ретраями по флуд-лимиту и откатом разметки при битых entities."""
    for _ in range(3):
        try:
            return await target.answer(text, reply_markup=reply_markup)
        except TelegramRetryAfter as exc:
            await asyncio.sleep(float(exc.retry_after) + 0.3)
        except TelegramBadRequest as exc:
            log.warning("safe_answer: bad_request: %s", exc)
            if reply_markup is None:
                raise
            reply_markup = None  # ретрай без клавиатуры
    raise RuntimeError("Не удалось отправить сообщение в Telegram")

async def send_formatted(
    target: Message,
    text: str,
    reply_markup: Optional[InlineKeyboardMarkup] = None,
    html_suffix: str = "",
) -> List[Message]:
    """Отправляет markdown-текст как HTML.

    html_suffix — готовая HTML-строка (напр. статистика), которая приклеивается
    в самый конец последнего чанка уже после конвертации.
    """
    formatted = md_to_html(text or "")
    if html_suffix:
        formatted = formatted.rstrip() + html_suffix
    if not formatted.strip():
        formatted = "⚠️ Модель вернула пустой ответ."

    chunks = _split_html(formatted, 4000)
    sent: List[Message] = []
    for i, chunk in enumerate(chunks):
        is_last = i == len(chunks) - 1
        sent.append(
            await _safe_answer(
                target, chunk, reply_markup=reply_markup if is_last else None
            )
        )
    return sent

# ─────────────────────── финализация стрима ───────────────────────

async def _edit_status(
    status: Message, formatted: str, markup: InlineKeyboardMarkup
) -> bool:
    """Пытается отредактировать сообщение-статус. True — получилось."""
    try:
        await status.edit_text(formatted, reply_markup=markup)
        return True
    except TelegramRetryAfter as exc:
        await asyncio.sleep(float(exc.retry_after) + 0.3)
        try:
            await status.edit_text(formatted, reply_markup=markup)
            return True
        except Exception as exc2:
            log.warning("finalize: retry edit_text failed: %s", exc2)
            return False
    except TelegramBadRequest as exc:
        if "not modified" in str(exc).lower():
            # Текст и разметка уже актуальны — дожимаем только клавиатуру.
            try:
                await status.edit_reply_markup(reply_markup=markup)
            except Exception:
                log.debug(
                    "finalize: markup edit after not-modified failed", exc_info=True
                )
            return True
        log.info("finalize: edit_text bad_request: %s", exc)
        return False
    except Exception:
        log.exception("finalize: edit_text unexpected error")
        return False

async def finalize_streamed(
    status: Message,
    reply: str,
    fallback_target: Message,
    stopped: bool = False,
    stats: Optional[str] = None,
) -> List[Message]:
    if not reply or not reply.strip():
        reply = "⏹ Остановлено." if stopped else "⚠️ Модель вернула пустой ответ."
        stopped = False  # маркер уже содержится в тексте

    # Хвост — уже готовая HTML-разметка.
    suffix = ""
    if stopped:
        suffix += "\n\n<i>⏹ (остановлено пользователем)</i>"
    if stats:
        suffix += stats

    formatted = md_to_html(reply).rstrip() + suffix
    if not formatted.strip():
        formatted = "⚠️ Модель вернула пустой ответ."

    markup = response_keyboard()

    # ── Случай 1: короткий ответ — правим одно сообщение ──
    if len(formatted) <= 4000:
        if await _edit_status(status, formatted, markup):
            log.info("finalize: edit ok (len=%d)", len(formatted))
            return [status]

        # Ретрай без клавиатуры: иногда ломаются именно entities кнопок.
        try:
            await status.edit_text(formatted)
            try:
                await status.edit_reply_markup(reply_markup=markup)
            except Exception:
                log.debug(
                    "finalize: markup edit after bare text", exc_info=True
                )
            log.info("finalize: bare edit ok")
            return [status]
        except TelegramRetryAfter as exc:
            await asyncio.sleep(float(exc.retry_after) + 0.3)
            try:
                await status.edit_text(formatted)
                try:
                    await status.edit_reply_markup(reply_markup=markup)
                except Exception:
                    pass
                return [status]
            except Exception as exc2:
                log.info("finalize: bare edit retry failed: %s", exc2)

    # ── Случай 2: длинный ответ — статус становится первым чанком ──
    chunks = _split_html(formatted, 4000)
    total = len(chunks)

    async def _delete_quietly() -> None:
        try:
            await status.delete()
        except Exception:
            log.debug("finalize: status delete failed", exc_info=True)

    def _with_counter(chunk: str, index: int) -> str:
        """Добавляет «(i/N)» внутрь валидного HTML (после балансировки тегов)."""
        return f"{chunk.rstrip()}\n\n<i>({index}/{total})</i>"

    # Первый чанк: переиспользуем статус, клавиатуру на нём не держим.
    first = _with_counter(chunks[0], 1) if total > 1 else chunks[0]
    sent: List[Message] = []

    if await _edit_status(status, first, None):
        sent.append(status)
    else:
        # Статус не редактируется (удалён/слишком старый) — чистим и шлём заново.
        await _delete_quietly()
        sent.append(await _safe_answer(fallback_target, first))

    # Остаток: отдельными сообщениями, клавиатура — только на последнем.
    for i, chunk in enumerate(chunks[1:], start=2):
        is_last = i == total
        body = chunk if is_last else _with_counter(chunk, i)
        sent.append(
            await _safe_answer(
                fallback_target, body, reply_markup=markup if is_last else None
            )
        )

    log.info("finalize: sent %d chunk(s), total len=%d", len(sent), len(formatted))
    return sent