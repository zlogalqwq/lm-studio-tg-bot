import httpx
import asyncio
import json
import logging
import math
import random
import time
from typing import Any, Dict, List, Optional

from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import Message

from . import config
from .core import get_lm_client, LM_NATIVE_ROOT
from .keyboards import stop_keyboard
from .storage import store

log = logging.getLogger(__name__)


EMBEDDING_PREFIX = "text-embedding-"
EMBEDDING_TYPE = "embeddings"

HIDDEN_MODEL_SUBSTRINGS = ("nomic-embed", "text-embedding-")
_CHAT_TEMPLATE_KWARGS_UNSUPPORTED: set[str] = set()

_STATE_RETRY_DELAY = 1.5
_SESSION_DEAD_MARKERS = (
    "previous_response_id",
    "response_id",
    "session",
    "conversation",
    "not found",
    "does not exist",
    "expired",
    "invalid state",
)

# Сколько токенов LM Studio тратит на служебку сверх prompt + max_tokens.
# Официальная рекомендация LM Studio: prompt + max_tokens + ~1500 headroom.
_CONTEXT_HEADROOM = 1500
_PROMPT_BUDGET_MIN = 500

_TIMEOUT_HINT = (
    "⏱ <b>LM Studio не ответил за {sec:.0f} с.</b>\n\n"
    "Скорее всего, модель ушла в <i>thinking</i> и генерирует тысячи "
    "токенов перед ответом. Reasoning-модели на медленном железе могут "
    "думать минутами.\n\n"
    "<b>Что делать:</b>\n"
    "• <code>/maxtokens 2048</code> — ограничить длину ответа\n"
    "• <code>/think off</code> — попробовать выключить thinking\n"
    "• <b>В LM Studio:</b> My Models → модель → отключить "
    "<i>Enable Thinking / Reasoning</i> (если есть)\n"
    "• Использовать модель без reasoning\n"
    "• Увеличить <code>LM_STUDIO_TIMEOUT</code> в <code>.env</code> "
    "(сейчас {sec:.0f} с)"
)


# ─────────────────────── прогресс загрузки модели ───────────────────────

def _progress_bar_text(pct: int, width: int = 20) -> str:
    pct = max(0, min(100, int(pct)))
    filled = round(width * pct / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"<code>[{bar}] {pct}%</code>"


async def _animate_load_progress(header: str, status: Message, stop: asyncio.Event) -> None:
    start = time.monotonic()
    tau = 12.0
    peak = 95
    last_pct = -1
    while not stop.is_set():
        elapsed = time.monotonic() - start
        pct = int(peak * (1 - math.exp(-elapsed / tau)))
        if pct > peak:
            pct = peak

        wait_time = 0.5
        if pct != last_pct:
            try:
                await status.edit_text(f"{header}\n{_progress_bar_text(pct)}")
            except TelegramRetryAfter as exc:
                wait_time = float(exc.retry_after) + 0.2
            except TelegramBadRequest as exc:
                log.debug("Progress edit failed: %s", exc)
            except Exception as exc:
                log.debug("Progress edit unexpected error: %s", exc)
            last_pct = pct

        try:
            await asyncio.wait_for(stop.wait(), timeout=wait_time)
        except asyncio.TimeoutError:
            pass


async def run_with_progress(header: str, status: Message, work):
    stop = asyncio.Event()
    anim_task = asyncio.create_task(_animate_load_progress(header, status, stop))
    try:
        result = await work
    finally:
        stop.set()
        try:
            await anim_task
        except Exception:
            log.debug("progress animator crashed", exc_info=True)
    try:
        await status.edit_text(f"{header}\n{_progress_bar_text(100)}")
        await asyncio.sleep(0.4)
    except Exception:
        pass
    return result


# ─────────────────────── фильтры моделей ───────────────────────

def _is_embedding_model(model: Dict[str, Any]) -> bool:
    mtype = (model.get("type") or "").lower()
    if mtype == EMBEDDING_TYPE:
        return True
    key = (model.get("key") or model.get("id") or "").lower()
    name = (model.get("display_name") or "").lower()
    return key.startswith(EMBEDDING_PREFIX) or name.startswith(EMBEDDING_PREFIX)


def _is_hidden_model(model: Dict[str, Any]) -> bool:
    if _is_embedding_model(model):
        return True
    key = (model.get("key") or model.get("id") or "").lower()
    name = (model.get("display_name") or "").lower()
    return any(s in key or s in name for s in HIDDEN_MODEL_SUBSTRINGS)


# ─────────────────────── URL endpoints (native v1) ───────────────────────

_URL_MODELS        = f"{LM_NATIVE_ROOT}/models"
_URL_MODELS_LOAD   = f"{LM_NATIVE_ROOT}/models/load"
_URL_MODELS_UNLOAD = f"{LM_NATIVE_ROOT}/models/unload"
_URL_CHAT          = f"{LM_NATIVE_ROOT}/chat"


# ─────────────────────── thinking-маркер ───────────────────────

_THINKING_MARKER_ON  = "/think"
_THINKING_MARKER_OFF = "/no_think"


def _apply_thinking_marker(payload: Dict[str, Any], thinking_enabled: bool) -> None:
    marker = _THINKING_MARKER_ON if thinking_enabled else _THINKING_MARKER_OFF
    inp = payload.get("input")

    if isinstance(inp, str):
        if not inp.rstrip().endswith(marker):
            payload["input"] = inp + " " + marker
        return

    if isinstance(inp, list):
        for part in inp:
            if isinstance(part, dict) and part.get("type") == "text":
                content = part.get("content") or ""
                if not content.rstrip().endswith(marker):
                    part["content"] = content + " " + marker
                return
        inp.append({"type": "text", "content": marker})
        return

    payload["input"] = str(inp or "") + " " + marker


async def _post_chat(
    payload: Dict[str, Any],
    model: str,
    thinking_enabled: bool = True,
) -> httpx.Response:
    if model in _CHAT_TEMPLATE_KWARGS_UNSUPPORTED:
        payload.pop("chat_template_kwargs", None)
        _apply_thinking_marker(payload, thinking_enabled)
        return await get_lm_client().post(_URL_CHAT, json=payload)

    resp = await get_lm_client().post(_URL_CHAT, json=payload)
    if (
        resp.status_code == 400
        and "chat_template_kwargs" in payload
        and "chat_template_kwargs" in (resp.text or "").lower()
    ):
        log.warning(
            "LM Studio rejected 'chat_template_kwargs' for model %s — "
            "falling back to prompt marker (%s)",
            model,
            _THINKING_MARKER_ON if thinking_enabled else _THINKING_MARKER_OFF,
        )
        _CHAT_TEMPLATE_KWARGS_UNSUPPORTED.add(model)
        payload.pop("chat_template_kwargs", None)
        _apply_thinking_marker(payload, thinking_enabled)
        resp = await get_lm_client().post(_URL_CHAT, json=payload)
    return resp


async def _send_chat_with_timeout(
    payload: Dict[str, Any],
    model: str,
    thinking_enabled: bool,
) -> httpx.Response:
    timeout = float(config.LM_STUDIO_TIMEOUT) + 5.0
    try:
        return await asyncio.wait_for(
            _post_chat(payload, model, thinking_enabled),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        log.error(
            "Chat request hard-timeout after %.1fs (model=%s). "
            "Likely a reasoning model ignoring /no_think.",
            timeout, model,
        )
        raise RuntimeError(_TIMEOUT_HINT.format(sec=config.LM_STUDIO_TIMEOUT))
    except httpx.TimeoutException as exc:
        log.error(
            "Chat request httpx-timeout after %.1fs (model=%s): %s",
            timeout, model, exc,
        )
        raise RuntimeError(_TIMEOUT_HINT.format(sec=config.LM_STUDIO_TIMEOUT))


# ─────────────────────── список моделей ───────────────────────

async def list_models_native() -> List[Dict[str, Any]]:
    try:
        resp = await get_lm_client().get(_URL_MODELS)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        log.exception("GET %s failed", _URL_MODELS)
        return []

    def _extract(obj: Any) -> Optional[List[Dict[str, Any]]]:
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for k in ("models", "data", "result", "items"):
                v = obj.get(k)
                if isinstance(v, list):
                    return v
                if isinstance(v, dict):
                    n = _extract(v)
                    if n is not None:
                        return n
        return None

    models = _extract(payload)
    if models is None:
        log.warning("Unexpected /models JSON shape: %r", payload)
        return []

    before = len(models)
    models = [m for m in models if not _is_hidden_model(m)]
    log.info("Native /models: %d entries (%d filtered)",
             len(models), before - len(models))
    return models


async def fetch_models() -> List[str]:
    models = await list_models_native()
    return [m.get("key") or m.get("id") for m in models
            if m.get("key") or m.get("id")]


async def get_model() -> str:
    if config.DEFAULT_MODEL:
        return config.DEFAULT_MODEL
    if not store.models_expired:
        cached = store.get_cached_models()
        if cached:
            return cached[0]
    models = await fetch_models()
    if not models:
        raise RuntimeError(
            "LM Studio не отвечает или нет загруженных моделей.\n"
            f"Проверьте, что сервер запущен по адресу {LM_NATIVE_ROOT}"
        )
    store.set_models(models)
    return models[0]


async def list_loaded_instances() -> List[Dict[str, Any]]:
    models = await list_models_native()
    instances: List[Dict[str, Any]] = []
    for m in models:
        for inst in (m.get("loaded_instances") or []):
            instances.append({
                "model_key": m.get("key") or m.get("id"),
                "display_name": m.get("display_name"),
                "instance_id": inst.get("id"),
                "context_length": (inst.get("config") or {}).get("context_length"),
            })
    return instances


async def sync_loaded_context_length() -> int:
    try:
        instances = await list_loaded_instances()
    except Exception:
        log.debug("sync_loaded_context_length: list failed", exc_info=True)
        return 0
    if not instances:
        return 0
    n = int(instances[0].get("context_length") or 0)
    if n > 0:
        await store.set_loaded_context_length(n)
    return n


# ─────────────────────── load / unload ───────────────────────

async def load_model(
    model_key: str,
    context_length: Optional[int] = None,
) -> Dict[str, Any]:
    loaded = await list_loaded_instances()
    if loaded:
        names = ", ".join(sorted({
            (i.get("display_name") or i.get("model_key") or "?") for i in loaded
        }))
        raise ValueError(f"Уже загружена модель: {names}. Сначала выгрузите её.")

    if context_length is None:
        context_length = config.DEFAULT_CONTEXT_LIMIT

    body: Dict[str, Any] = {
        "model": model_key,
        "context_length": context_length,
    }
    log.info("POST %s  body=%s", _URL_MODELS_LOAD, body)
    resp = await get_lm_client().post(_URL_MODELS_LOAD, json=body)
    if resp.status_code != 200:
        log.error("load_model error %s: %s", resp.status_code, resp.text[:1000])
    resp.raise_for_status()
    result = resp.json()
    try:
        n = await store.reset_all_api_states()
        if n:
            log.info("load_model: reset %d stateful session(s)", n)
    except Exception:
        log.debug("reset_all_api_states failed after load", exc_info=True)
    return result


async def unload_model(instance_id: str) -> Dict[str, Any]:
    body = {"instance_id": instance_id}
    log.info("POST %s  body=%s", _URL_MODELS_UNLOAD, body)
    resp = await get_lm_client().post(_URL_MODELS_UNLOAD, json=body)
    if resp.status_code != 200:
        log.error("unload_model error %s: %s", resp.status_code, resp.text[:1000])
    resp.raise_for_status()
    result = resp.json()
    try:
        n = await store.reset_all_api_states()
        if n:
            log.info("unload_model: reset %d stateful session(s)", n)
    except Exception:
        log.debug("reset_all_api_states failed after unload", exc_info=True)
    return result


async def unload_all_models() -> List[Dict[str, Any]]:
    instances = await list_loaded_instances()
    results: List[Dict[str, Any]] = []
    for inst in instances:
        try:
            r = await unload_model(inst["instance_id"])
            results.append(r)
        except Exception as exc:
            log.warning("Failed to unload %s: %s", inst["instance_id"], exc)
    return results


async def reload_model(
    model_key: str,
    context_length: Optional[int] = None,
) -> Dict[str, Any]:
    models = await list_models_native()
    target = next((m for m in models
                   if (m.get("key") or m.get("id")) == model_key), None)
    if target is None:
        raise ValueError(f"Модель '{model_key}' не найдена.")
    for inst in (target.get("loaded_instances") or []):
        try:
            await unload_model(inst["id"])
        except Exception as exc:
            log.warning("Failed to unload instance %s: %s", inst["id"], exc)
    return await load_model(model_key, context_length)


# ─────────────────────── оценка токенов ───────────────────────

def _estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    text_chars = 0
    image_count = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            text_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                t = part.get("type")
                if t == "text":
                    text_chars += len(part.get("text") or part.get("content") or "")
                elif t in ("image_url", "image"):
                    image_count += 1
    text_tokens = max(1, int(text_chars / 3.5))
    return text_tokens + image_count * 1500


def estimate_context_tokens(history: List[Dict[str, Any]]) -> int:
    return _estimate_tokens(history)


# ─────────────────────── native API: input helpers ───────────────────────

def _to_native_input(msg: Any) -> Any:
    if isinstance(msg, str):
        return msg
    if not isinstance(msg, list):
        return str(msg)

    native: List[Dict[str, Any]] = []
    for part in msg:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            text = part.get("text") or part.get("content") or ""
            if text:
                native.append({"type": "text", "content": text})
        elif ptype == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            if url:
                native.append({"type": "image", "data_url": url})
        elif ptype == "image":
            native.append(part)

    if not native:
        raise ValueError(
            "Пустое или нераспознанное содержимое для native API LM Studio."
        )
    return native


def _extract_text_for_history(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t == "text":
                parts.append(item.get("text") or item.get("content") or "")
            elif t in ("image_url", "image"):
                parts.append("[image]")
        return "\n".join(p for p in parts if p)
    return str(content or "")


def _build_history_input(
    history: List[Dict[str, Any]],
    current_native_input: Any,
    prompt_budget: int = 0,
) -> Any:
    """Собирает префикс из истории + текущий input.

    prompt_budget — СКОЛЬКО ТОКЕНОВ ДОСТУПНО НА ВЕСЬ ПРОМПТ
    (история + текущее сообщение), уже за вычетом max_tokens и headroom.
    Если 0 — не обрезаем.
    """
    if prompt_budget > 0 and history:
        cur_tokens = _estimate_tokens(
            [{"role": "user", "content": current_native_input}]
        )
        budget = prompt_budget - cur_tokens
        if budget < 0:
            budget = 0
        trimmed: List[Dict[str, Any]] = []
        total = 0
        for msg in reversed(history):
            if (msg.get("role") or "").lower() == "system":
                continue
            t = _estimate_tokens([msg])
            if total + t > budget:
                break
            trimmed.insert(0, msg)
            total += t
        history = trimmed

    lines: List[str] = []
    for msg in history:
        role = (msg.get("role") or "user").lower()
        if role == "system":
            continue
        text = _extract_text_for_history(msg.get("content"))
        if not text:
            continue
        label = "User" if role == "user" else ("Assistant" if role == "assistant" else role.capitalize())
        lines.append(f"{label}: {text}")

    if not lines:
        return current_native_input

    prefix = (
        "Previous conversation (for context):\n"
        + "\n\n".join(lines)
        + "\n\n---\n\nCurrent message:\n"
    )

    if isinstance(current_native_input, str):
        return prefix + current_native_input
    if isinstance(current_native_input, list):
        return [{"type": "text", "content": prefix}] + current_native_input
    return prefix + str(current_native_input)


# ─────────────────────── context-budget check ───────────────────────

def _check_context_fits(
    payload: Dict[str, Any],
    max_tokens: int,
    actual_ctx: int,
    prev_prompt_tokens: int = 0,
) -> int:
    """Проверяет, что запрос помещается в контекст LM Studio.

    Возвращает оценку размера промпта в токенах (для логов).
    Бросает RuntimeError с готовой инструкцией при переполнении —
    это лучше, чем «stall на 0.66%» в GUI LM Studio.
    """
    input_est = _estimate_tokens([{"role": "user", "content": payload["input"]}])
    sp = payload.get("system_prompt") or ""
    sp_est = _estimate_tokens([{"role": "system", "content": sp}]) if sp else 0

    if prev_prompt_tokens > 0 and "previous_response_id" in payload:
        # Stateful: сервер уже держит систему + предыдущий контекст.
        total = prev_prompt_tokens + input_est + max_tokens
    else:
        # Fresh: считаем полный объём того, что отправляем.
        total = input_est + sp_est + max_tokens

    if actual_ctx > 0 and total > actual_ctx:
        raise RuntimeError(
            f"⚠️ <b>Запрос не влезает в контекст LM Studio.</b>\n\n"
            f"Нужно: <b>~{total}</b> токенов\n"
            f"  • промпт (история + сообщение): ~{input_est}\n"
            f"  • system prompt: ~{sp_est}\n"
            f"  • ответ (max_tokens): {max_tokens}\n"
            f"Доступно в контексте: <b>{actual_ctx}</b>\n\n"
            f"<b>Что сделать:</b>\n"
            f"• <code>/maxtokens 2048</code> — уменьшить бюджет ответа\n"
            f"• <code>/clean</code> — начать новый диалог\n"
            f"• В LM Studio: Context Length → <b>{total + _CONTEXT_HEADROOM}</b> "
            f"или больше"
        )

    return input_est + sp_est


# ─────────────────────── chat_completion: NATIVE API ───────────────────────

async def chat_completion(
    message: Any,
    model: str,
    status: Message,
    previous_response_id: Optional[str] = None,
    stop_event: Optional[asyncio.Event] = None,
    settings=None,
    system_prompt: str = "",
    use_tools: bool = True,
    thinking_enabled: bool = True,
    history_context: Optional[List[Dict[str, Any]]] = None,
    loaded_context_length: int = 0,
    prev_prompt_tokens: int = 0,
) -> tuple[str, Optional[Dict[str, int]], float, Optional[str], bool]:
    """Native /api/v1/chat со stateful-режимом.

    Возвращает (reply, usage, elapsed, response_id, session_reset).

    loaded_context_length — фактический контекст из store.get_loaded_context_length().
    prev_prompt_tokens     — store.get_cumulative_tokens(user_id) — сколько
                              сервер уже накопил в stateful-сессии.
    """
    temperature = getattr(settings, "temperature", config.DEFAULT_TEMPERATURE) if settings else config.DEFAULT_TEMPERATURE
    top_k       = getattr(settings, "top_k",       config.DEFAULT_TOP_K)       if settings else config.DEFAULT_TOP_K
    top_p       = getattr(settings, "top_p",       config.DEFAULT_TOP_P)       if settings else config.DEFAULT_TOP_P
    max_tokens  = getattr(settings, "max_tokens",  config.DEFAULT_MAX_TOKENS)  if settings else config.DEFAULT_MAX_TOKENS
    user_ctx    = getattr(settings, "context_limit", config.DEFAULT_CONTEXT_LIMIT) if settings else config.DEFAULT_CONTEXT_LIMIT

    # ── фактический лимит: min(loaded, user) или что есть ──
    actual_ctx = int(loaded_context_length or 0)
    if user_ctx > 0:
        actual_ctx = min(actual_ctx, user_ctx) if actual_ctx > 0 else user_ctx

    # ── бюджет под промпт (история + текущее + system) ──
    if actual_ctx > 0:
        prompt_budget = actual_ctx - max_tokens - _CONTEXT_HEADROOM
        if prompt_budget < _PROMPT_BUDGET_MIN:
            prompt_budget = _PROMPT_BUDGET_MIN
    else:
        prompt_budget = 0  # неизвестно — не обрезаем

    # ── анимация ожидания ──
    thinking_stop = asyncio.Event()
    thinking_task = asyncio.create_task(_animate_thinking(status, thinking_stop))

    async def _stop_thinking() -> None:
        if thinking_stop.is_set():
            return
        thinking_stop.set()
        try:
            await thinking_task
        except Exception:
            log.debug("thinking animator crashed", exc_info=True)

    start_time = time.monotonic()
    native_input = _to_native_input(message)

    payload: Dict[str, Any] = {
        "model": model,
        "input": native_input,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_output_tokens": max_tokens,
        "stream": False,
    }

    if not previous_response_id and system_prompt.strip():
        payload["system_prompt"] = system_prompt.strip()

    if previous_response_id:
        payload["previous_response_id"] = previous_response_id

    if not previous_response_id and history_context:
        payload["input"] = _build_history_input(
            history_context, native_input, prompt_budget=prompt_budget,
        )
        log.info(
            "Fresh session with history: %d prior message(s) prepended, "
            "prompt_budget=%d",
            len(history_context), prompt_budget,
        )

    if not thinking_enabled and model not in _CHAT_TEMPLATE_KWARGS_UNSUPPORTED:
        payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = False

    if use_tools:
        integrations = get_mcp_integrations()
        if integrations:
            payload["integrations"] = integrations
            log.info(
                "MCP integrations enabled: %d server(s) — %s",
                len(integrations), ", ".join(integrations),
            )

    sp = system_prompt.strip()
    sp_state = (
        "none" if not sp
        else ("sent" if "system_prompt" in payload else "in-state")
    )
    log.info(
        "Native request: system_prompt=%s (%d chars), prev_id=%s, "
        "integrations=%s, thinking=%s, input_type=%s, "
        "actual_ctx=%d, max_tokens=%d, prompt_budget=%d",
        sp_state, len(sp),
        (previous_response_id[:8] + "…") if previous_response_id else None,
        len(payload.get("integrations") or []),
        "on" if thinking_enabled else "off",
        type(payload["input"]).__name__,
        actual_ctx, max_tokens, prompt_budget,
    )

    # ── ПРОВЕРКА КОНТЕКСТА ДО ОТПРАВКИ ──
    # Это ловит главную причину stall'а на 0.66%:
    # prompt + max_tokens > context_length → LM Studio виснет на префилле.
    try:
        estimated_prompt = _check_context_fits(
            payload,
            max_tokens=max_tokens,
            actual_ctx=actual_ctx,
            prev_prompt_tokens=prev_prompt_tokens,
        )
        log.info("Context check OK: prompt≈%d + max_tokens=%d <= ctx=%d",
                 estimated_prompt, max_tokens, actual_ctx)
    except RuntimeError:
        await _stop_thinking()
        raise

    log.debug("Native payload: %s", json.dumps(payload, ensure_ascii=False)[:1500])

    # ── отправка ──
    request_task: asyncio.Future = asyncio.ensure_future(
        _send_chat_with_timeout(payload, model, thinking_enabled)
    )
    stopped = False
    stop_task: Optional[asyncio.Future] = None
    try:
        if stop_event is not None:
            stop_task = asyncio.ensure_future(stop_event.wait())
            done, _ = await asyncio.wait(
                {request_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_task in done and not request_task.done():
                request_task.cancel()
                try:
                    await request_task
                except BaseException:
                    pass
                stopped = True
        else:
            await request_task
    finally:
        if stop_task is not None and not stop_task.done():
            stop_task.cancel()
            try:
                await stop_task
            except BaseException:
                pass
        await _stop_thinking()

    elapsed = time.monotonic() - start_time

    if stopped:
        log.info("Request stopped by user (elapsed=%.2fs)", elapsed)
        return "", None, elapsed, None, False

    resp = request_task.result()
    session_reset = False

    if resp.status_code != 200:
        status_code = resp.status_code
        body_lower = (resp.text or "").lower()

        if status_code in (408, 429) or status_code >= 500:
            log.warning(
                "Transient HTTP %s — retry with session intact in %.1fs",
                status_code, _STATE_RETRY_DELAY,
            )
            await asyncio.sleep(_STATE_RETRY_DELAY)
            resp = await _send_chat_with_timeout(payload, model, thinking_enabled)
            status_code = resp.status_code
            body_lower = (resp.text or "").lower()

        if previous_response_id:
            session_dead = status_code in (404, 410) or (
                status_code == 400
                and any(m in body_lower for m in _SESSION_DEAD_MARKERS)
            )
            if session_dead:
                log.warning(
                    "Stateful session is dead (HTTP %s): %s. "
                    "Resetting and retrying as fresh session. "
                    "⚠️ stateful-context lost.",
                    status_code, (resp.text or "")[:300],
                )
                payload.pop("previous_response_id", None)
                if system_prompt.strip():
                    payload["system_prompt"] = system_prompt.strip()
                if history_context:
                    payload["input"] = _build_history_input(
                        history_context, native_input,
                        prompt_budget=prompt_budget,
                    )
                # Пере-проверка после пересборки промпта.
                _check_context_fits(
                    payload,
                    max_tokens=max_tokens,
                    actual_ctx=actual_ctx,
                    prev_prompt_tokens=0,  # fresh session
                )
                resp = await _send_chat_with_timeout(payload, model, thinking_enabled)
                session_reset = True
            else:
                log.warning(
                    "Non-session HTTP %s on stateful turn — keeping session. "
                    "Body: %s", status_code, (resp.text or "")[:300],
                )

    if resp.status_code != 200:
        log.error(
            "LM Studio %s on %s\nRequest payload: %s\nResponse body: %s",
            resp.status_code,
            resp.request.url,
            json.dumps(payload, ensure_ascii=False)[:1500],
            (resp.text or "")[:2000],
        )
    resp.raise_for_status()
    data = resp.json()

    # ── парсинг output ──
    _SKIP_OUTPUT_TYPES = {
        "reasoning", "thinking", "analysis", "thought", "chain_of_thought",
    }
    output = data.get("output") or []
    reply_parts: List[str] = []
    reasoning_chars = 0
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            itype = (item.get("type") or "").lower()
            if itype in _SKIP_OUTPUT_TYPES:
                c = item.get("content") or item.get("text") or ""
                if isinstance(c, str):
                    reasoning_chars += len(c)
                continue
            if itype == "tool_call":
                tool_name = item.get("tool") or item.get("name")
                provider = item.get("provider_info") or {}
                log.info(
                    "Tool call executed: tool=%s provider=%s",
                    tool_name,
                    provider.get("plugin_id") or provider.get("server_label"),
                )
                continue
            content = item.get("content")
            if content is None:
                content = item.get("text")
            if isinstance(content, str) and content:
                reply_parts.append(content)
    reply = "".join(reply_parts)

    response_id = data.get("response_id")

    # ── usage ──
    usage_obj = data.get("usage") or data.get("stats") or {}
    log.debug(
        "Native response usage=%r stats=%r",
        data.get("usage"), data.get("stats"),
    )

    prompt_tokens = int(
        usage_obj.get("prompt_tokens")
        or usage_obj.get("input_tokens")
        or 0
    )
    completion_tokens = int(
        usage_obj.get("completion_tokens")
        or usage_obj.get("total_output_tokens")
        or usage_obj.get("output_tokens")
        or 0
    )
    total_tokens = int(
        usage_obj.get("total_tokens")
        or (prompt_tokens + completion_tokens)
    )

    if prompt_tokens or completion_tokens:
        usage = {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      total_tokens,
        }
        source = "api"
    else:
        prompt_est     = _estimate_tokens([{"role": "user", "content": message}])
        completion_est = max(1, len(reply) // 4) if reply else 0
        usage = {
            "prompt_tokens":     prompt_est,
            "completion_tokens": completion_est,
            "total_tokens":      prompt_est + completion_est,
        }
        source = "estimate"

    log.info(
        "Request done: chars=%d reasoning=%d elapsed=%.2fs usage_source=%s "
        "prompt=%d completion=%d total=%d response_id=%s session_reset=%s",
        len(reply), reasoning_chars, elapsed, source,
        usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"],
        response_id, session_reset,
    )

    # Диагностика: модель потратила всё на reasoning, но ответа нет.
    if not reply and reasoning_chars > 0:
        log.warning(
            "Model produced %d chars of reasoning but no reply. "
            "Likely hit max_output_tokens during thinking "
            "(completion_tokens=%d, max_tokens=%d).",
            reasoning_chars, completion_tokens, max_tokens,
        )
        reply = (
            "⚠️ Модель потратила все токены на reasoning "
            f"({reasoning_chars} символов размышлений) и не успела ответить.\n\n"
            f"Попробуйте:\n"
            f"• <code>/maxtokens 4096</code> — увеличить бюджет\n"
            f"• <code>/think off</code> — выключить thinking\n"
            f"• Использовать модель без reasoning"
        )

    return reply, usage, elapsed, response_id, session_reset


# ─────────────────────── анимация ожидания ───────────────────────

_THINKING_EMOJIS = (
    "🧠", "💭", "✨", "⚡", "🎯", "🎨", "🚀", "💫",
    "🌟", "🔮", "⚙️", "🛰️", "🌀", "📡", "🪄", "🔭",
)

_THINKING_VERBS = (
    "Генерирую", "Думаю", "Размышляю", "Соображаю",
    "Считаю токены", "Ворожу", "Собираю мысли", "Творю",
    "Кручу шестерёнки", "Советуюсь с моделью", "Ищу ответ", "Пишу",
)

_THINKING_TAILS = (
    "…", "… почти", "… ещё чуть-чуть", "… уже близко",
    "… без паники", "… не спеша", "… со вкусом", "… вдумчиво",
    "… на максималках", "… по классике", "… аккуратно",
    "… в фоне", "… без лишнего шума", "… и немного магии",
    "… как следует",
)

_THINKING_ACCENTS = (
    "🫧 токены летят", "🎼 подбираю рифму", "🍵 завариваю ответ",
    "🧩 складываю пазл", "🛠 подкручиваю гайки", "🎲 бросаю кубики",
    "🪄 колдую над промптом", "📚 листаю контекст",
    "🌊 синхронизируюсь с сервером", "🛰 жду пакет с ответом",
    "🐢 медленно, но верно", "🦾 держу удар",
)

_THINKING_MIN_INTERVAL = 0.9
_THINKING_MAX_INTERVAL = 1.6
_THINKING_ACCENT_CHANCE = 0.35


def _pick_tail(prev: Optional[str]) -> str:
    if prev is None or len(_THINKING_TAILS) <= 1:
        return random.choice(_THINKING_TAILS)
    tail = random.choice(_THINKING_TAILS)
    while tail == prev:
        tail = random.choice(_THINKING_TAILS)
    return tail


def _render_thinking_frame(tick: int, tail: str) -> str:
    emoji = _THINKING_EMOJIS[tick % len(_THINKING_EMOJIS)]
    verb = _THINKING_VERBS[tick % len(_THINKING_VERBS)]

    frame = f"{emoji} {verb}{tail}"
    if random.random() < _THINKING_ACCENT_CHANCE:
        frame += "\n<i>" + random.choice(_THINKING_ACCENTS) + "</i>"
    return frame


async def _animate_thinking(status: Message, stop: asyncio.Event) -> None:
    tick = 0
    prev_tail: Optional[str] = None
    while not stop.is_set():
        prev_tail = _pick_tail(prev_tail)
        frame = _render_thinking_frame(tick, prev_tail)

        wait_time = random.uniform(_THINKING_MIN_INTERVAL, _THINKING_MAX_INTERVAL)
        try:
            await status.edit_text(frame, reply_markup=stop_keyboard())
        except TelegramRetryAfter as exc:
            wait_time = float(exc.retry_after) + 0.2
        except TelegramBadRequest:
            pass
        except Exception:
            log.debug("thinking animation edit failed", exc_info=True)

        tick += 1
        try:
            await asyncio.wait_for(stop.wait(), timeout=wait_time)
        except asyncio.TimeoutError:
            pass


# ─────────────────────── MCP интеграции ───────────────────────

def get_mcp_integrations() -> List[str]:
    import json as _json
    import os as _os

    path = config.MCP_JSON_PATH
    if not path or not _os.path.exists(path):
        log.debug("MCP config not found: %s", path)
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
    except Exception:
        log.exception("Failed to read MCP config %s", path)
        return []

    servers = data.get("mcpServers") or {}
    if not isinstance(servers, dict):
        return []

    return [f"mcp/{name}" for name in servers.keys()]