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
        if pct != last_pct:
            try:
                await status.edit_text(f"{header}\n{_progress_bar_text(pct)}")
            except TelegramRetryAfter as exc:
                await asyncio.sleep(float(exc.retry_after) + 0.2)
            except TelegramBadRequest as exc:
                log.debug("Progress edit failed: %s", exc)
            except Exception as exc:
                log.debug("Progress edit unexpected error: %s", exc)
            last_pct = pct
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.5)
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


# ─────────────────────── список моделей (только native) ───────────────────────

async def list_models_native() -> List[Dict[str, Any]]:
    """GET /api/v1/models — список моделей в нативном формате.

    Возвращает список словарей с ключами: key, display_name, type,
    loaded_instances (если модель загружена) и т.п.
    """
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
            # v1: {"models": [...]}    v0: {"data": [...]}
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
    """Список ключей моделей для UI (native)."""
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


# ─────────────────────── load / unload (native v1) ───────────────────────

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

    # native v1: POST /models/load
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
    # Сбрасываем stateful-сессии: старые response_id больше не валидны.
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


# ─────────────────────── chat_completion: NATIVE API ───────────────────────

def _to_native_input(msg: Any) -> Any:
    """Приводит содержимое к формату native API /api/v1/chat.

    native:
      * строка — допустима для чисто текстового сообщения;
      * список:
          {"type": "text",  "content": "..."}
          {"type": "image", "data_url": "data:image/jpeg;base64,..."}
    """
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
            # уже в native-формате
            native.append(part)

    if not native:
        raise ValueError(
            "Пустое или нераспознанное содержимое для native API LM Studio."
        )
    return native


async def chat_completion(
    message: Any,
    model: str,
    status: Message,
    previous_response_id: Optional[str] = None,
    stop_event: Optional[asyncio.Event] = None,
    settings=None,
    system_prompt: str = "",
    prior_summary: str = "",
    use_tools: bool = True,
) -> tuple[str, Optional[Dict[str, int]], float, Optional[str]]:
    """Native /api/v1/chat со stateful-режимом.

    Возвращает (reply, usage, elapsed, response_id).
    * usage is None  →  генерация прервана пользователем.

    Особенности работы с системным промптом:
    * LM Studio держит system_prompt в stateful-сессии, но этот стейт
      НЕ гарантирован: перезагрузка модели, внутренняя обрезка контекста,
      смена response_id и т.п. — и промпт исчезает.
    * Поэтому мы отправляем его отдельным полем payload["system_prompt"]
      на КАЖДОМ ходу. Native API трактует это как override для текущей и
      последующих реплик, так что промпт «не теряется» даже на длинных
      диалогах.
    * Резюме (prior_summary) при этом подмешивается только в input при
      СТАРТЕ новой цепочки — при stateful-ходе оно уже в контексте модели,
      дублировать его значит жечь токены.

    MCP-инструменты (use_tools=True):
    * payload["integrations"] содержит список серверов из mcp.json
      в формате ["mcp/server1", "mcp/server2", ...].
    * В отличие от system_prompt, integrations применяются К КАЖДОМУ
      запросу — иначе после первого хода модель «забудет» про инструменты.
    """
    temperature = getattr(settings, "temperature", config.DEFAULT_TEMPERATURE) if settings else config.DEFAULT_TEMPERATURE
    top_k       = getattr(settings, "top_k",       config.DEFAULT_TOP_K)       if settings else config.DEFAULT_TOP_K
    top_p       = getattr(settings, "top_p",       config.DEFAULT_TOP_P)       if settings else config.DEFAULT_TOP_P
    max_tokens  = getattr(settings, "max_tokens",  config.DEFAULT_MAX_TOKENS)  if settings else config.DEFAULT_MAX_TOKENS

    log.info(
        "Request (native): model=%s temp=%s top_p=%s top_k=%s max_output_tokens=%s prev_id=%s",
        model, temperature, top_p, top_k, max_tokens,
        (previous_response_id[:8] + "…") if previous_response_id else None,
    )

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

    # ── payload в формате native ──
    native_input = _to_native_input(message)

    # Резюме прошлых сессий подмешиваем в input ТОЛЬКО при старте новой
    # цепочки. На stateful-ходе оно уже в состоянии LM Studio.
    if not previous_response_id and prior_summary.strip():
        summary_block = (
            "Контекст предыдущих сессий (резюме):\n" + prior_summary.strip()
        )
        if isinstance(native_input, str):
            native_input = f"{summary_block}\n\n{native_input}"
        elif isinstance(native_input, list):
            native_input = [
                {"type": "text", "content": summary_block}
            ] + native_input

    payload: Dict[str, Any] = {
        "model": model,
        "input": native_input,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_output_tokens": max_tokens,   # ← ключевое: НЕ max_tokens
    }

    # system_prompt отдельным полем — ТОЛЬКО при старте новой сессии.
    # На stateful-ходе LM Studio попытается вставить систему в середину
    # цепочки, и chat template модели упадёт с ошибкой
    # "System message must be at the beginning" (HTTP 500).
    if not previous_response_id and system_prompt.strip():
        payload["system_prompt"] = system_prompt.strip()

    if previous_response_id:
        payload["previous_response_id"] = previous_response_id

    # ── MCP интеграции (если включены) ──
    # ВАЖНО: integrations применяются к КАЖДОМУ запросу (и на stateful-ходе
    # тоже), иначе после первого хода модель «забудет» про инструменты.
    if use_tools:
        integrations = get_mcp_integrations()
        if integrations:
            payload["integrations"] = integrations
            log.info(
                "MCP integrations enabled: %d server(s) — %s",
                len(integrations), ", ".join(integrations),
            )
        else:
            log.debug(
                "MCP integrations enabled, but no servers found in mcp.json"
            )

    log.info(
        "Native request: system_prompt=%s (%d chars), summary=%s, "
        "prev_id=%s, integrations=%s, input_type=%s",
        "yes" if (not previous_response_id and system_prompt.strip()) else "no",
        len(system_prompt.strip()),
        "yes" if (not previous_response_id and prior_summary.strip()) else "no",
        (previous_response_id[:8] + "…") if previous_response_id else None,
        len(payload.get("integrations") or []),
        type(native_input).__name__,
    )

    log.debug(
        "Native payload: %s",
        json.dumps(payload, ensure_ascii=False)[:1500],
    )

    # ── отправка ──
    request_task: asyncio.Future = asyncio.ensure_future(
        get_lm_client().post(_URL_CHAT, json=payload)
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
        return "", None, elapsed, None

    # ── разбор ответа ──
    resp = request_task.result()

    # Stateful-сессия могла протухнуть (рестарт LM Studio, обновление модели
    # в обход нашего бота). Тогда повторяем один раз с чистого листа —
    # на этот раз system_prompt уйдёт как отдельное поле.
    if (
        resp.status_code >= 400
        and previous_response_id
        and not stopped
    ):
        log.warning(
            "Stateful turn failed (%s), retrying as fresh session",
            resp.status_code,
        )
        payload.pop("previous_response_id", None)
        if system_prompt.strip():
            payload["system_prompt"] = system_prompt.strip()
        resp = await get_lm_client().post(_URL_CHAT, json=payload)

    if resp.status_code != 200:
        # Самая важная часть — покажет, что именно не понравилось LM Studio.
        log.error(
            "LM Studio %s on %s\nRequest payload: %s\nResponse body: %s",
            resp.status_code,
            resp.request.url,
            json.dumps(payload, ensure_ascii=False)[:1500],
            (resp.text or "")[:2000],
        )
    resp.raise_for_status()
    data = resp.json()

    # native: {"output": [{"type": "message", "content": "..."}], "response_id": "..."}
    # Модели с reasoning могут вернуть дополнительные блоки:
    #   {"type": "reasoning", "content": "..."} — их нужно ОТБРАСЫВАТЬ,
    #   иначе размышления утекут в ответ пользователю.
    _SKIP_OUTPUT_TYPES = {
        "reasoning", "thinking", "analysis", "thought", "chain_of_thought",
    }
    output = data.get("output") or []
    reply_parts: List[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            itype = (item.get("type") or "").lower()
            if itype in _SKIP_OUTPUT_TYPES:
                continue
            if itype == "tool_call":
                # LM Studio уже выполнил вызов, нам нужен только финальный текст.
                # Логируем для отладки.
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
    usage_obj = data.get("usage") or data.get("stats")
    if usage_obj is not None:
        usage = {
            "prompt_tokens":     int(usage_obj.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage_obj.get("completion_tokens", 0) or 0),
            "total_tokens":      int(usage_obj.get("total_tokens", 0) or 0),
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
        "Request done: chars=%d elapsed=%.2fs usage_source=%s "
        "prompt=%d completion=%d total=%d response_id=%s",
        len(reply), elapsed, source,
        usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"],
        response_id,
    )

    if not reply:
        log.warning("LM Studio returned empty response")

    return reply, usage, elapsed, response_id


# ─────────────────────── анимация ожидания ───────────────────────

# Ротация эмодзи — это и делает анимацию «живой».
_THINKING_EMOJIS = (
    "🧠", "💭", "✨", "⚡", "🎯", "🎨", "🚀", "💫",
    "🌟", "🔮", "⚙️", "🛰️",
)
_THINKING_VERB = "Генерирую"

_THINKING_MIN_INTERVAL = 0.6
_THINKING_MAX_INTERVAL = 1.5


def _render_thinking_frame(tick: int) -> str:
    emoji = _THINKING_EMOJIS[tick % len(_THINKING_EMOJIS)]
    return f"{emoji} {_THINKING_VERB}"


async def _animate_thinking(status: Message, stop: asyncio.Event) -> None:
    """Заметная анимация ожидания: смена эмодзи + бегущие точки.

    Интервал рандомный в диапазоне 0.6–0.9 с — движение есть,
    но Telegram не заваливается edit-запросами.
    """
    tick = 0
    while not stop.is_set():
        frame = _render_thinking_frame(tick)
        try:
            await status.edit_text(frame, reply_markup=stop_keyboard())
        except TelegramRetryAfter as exc:
            await asyncio.sleep(float(exc.retry_after) + 0.2)
        except TelegramBadRequest:
            # «message is not modified» и прочее — молча пропускаем.
            pass
        except Exception:
            log.debug("thinking animation edit failed", exc_info=True)

        tick += 1
        try:
            await asyncio.wait_for(
                stop.wait(),
                timeout=random.uniform(
                    _THINKING_MIN_INTERVAL, _THINKING_MAX_INTERVAL
                ),
            )
        except asyncio.TimeoutError:
            pass


# ─────────────────────── summary ───────────────────────

def _extract_text_for_summary(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if not isinstance(p, dict):
                continue
            t = p.get("type")
            if t == "text":
                parts.append(p.get("text") or p.get("content") or "")
            elif t in ("image_url", "image"):
                parts.append("[изображение]")
        return "\n".join(x for x in parts if x)
    return str(content)


_SUMMARY_SKIP_OUTPUT_TYPES = {
    "reasoning", "thinking", "analysis", "thought", "chain_of_thought",
}


async def summarize_history(
    history: List[Dict[str, Any]],
    model: str,
    prior_summary: str = "",
) -> str:
    """Сгенерировать резюме по локальной истории (+ предыдущему резюме).

    Возвращает строку; пустая строка — если нечего суммировать или ошибка.
    """
    if not history and not prior_summary.strip():
        return ""

    blocks: List[str] = []
    if prior_summary.strip():
        blocks.append(f"[Ранее было:]\n{prior_summary.strip()}")

    for m in history:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "?").upper()
        text = _extract_text_for_summary(m.get("content"))
        if text.strip():
            blocks.append(f"{role}: {text}")

    if not blocks:
        return ""

    transcript = "\n\n".join(blocks)
    prompt = (
        f"{config.SUMMARY_PROMPT}\n\n"
        f"--- ДИАЛОГ ---\n{transcript}\n--- КОНЕЦ ---\n\n"
        f"Резюме:"
    )

    payload: Dict[str, Any] = {
        "model": model,
        "input": prompt,
        "temperature": 0.3,
        "max_output_tokens": config.SUMMARY_MAX_TOKENS,
    }

    try:
        resp = await get_lm_client().post(_URL_CHAT, json=payload)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        log.exception("summarize_history: request failed")
        return ""

    output = data.get("output") or []
    parts: List[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            itype = (item.get("type") or "").lower()
            if itype in _SUMMARY_SKIP_OUTPUT_TYPES:
                continue
            c = item.get("content")
            if c is None:
                c = item.get("text")
            if isinstance(c, str) and c:
                parts.append(c)

    summary = "".join(parts).strip()
    if not summary:
        log.warning("summarize_history: empty summary returned")
    return summary

# ─────────────────────── MCP интеграции ───────────────────────

def get_mcp_integrations() -> List[str]:
    """Возвращает список integrations для всех MCP-серверов из mcp.json.

    Формат: ["mcp/server1", "mcp/server2", ...].
    Если файл не найден или пуст — возвращает [].
    """
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

    # Ключи mcpServers — это имена серверов, например "playwright"
    return [f"mcp/{name}" for name in servers.keys()]