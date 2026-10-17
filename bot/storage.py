import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from . import config

log = logging.getLogger(__name__)


def _password_fingerprint(password: str) -> str:
    if not password:
        return ""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()[:16]


CURRENT_FINGERPRINT: str = _password_fingerprint(config.AUTH_PASSWORD)


class UserSettings:
    """Персональные настройки генерации для пользователя."""
    __slots__ = (
        "temperature", "top_k", "top_p", "context_limit", "max_tokens",
        "last_prompt_tokens", "last_completion_tokens", "last_total_tokens",
        "last_effective_tokens", "context_warned",
    )

    def __init__(self) -> None:
        self.temperature: float = config.DEFAULT_TEMPERATURE
        self.top_k: int = config.DEFAULT_TOP_K
        self.top_p: float = config.DEFAULT_TOP_P
        self.context_limit: int = config.DEFAULT_CONTEXT_LIMIT
        self.max_tokens: int = config.DEFAULT_MAX_TOKENS
        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0
        self.last_total_tokens: int = 0
        self.last_effective_tokens: int = 0
        self.context_warned: bool = False

    @property
    def usage_ratio(self) -> float:
        if self.context_limit <= 0:
            return 0.0
        return self.last_effective_tokens / self.context_limit

    def reset_usage(self) -> None:
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.last_effective_tokens = 0
        self.context_warned = False

    def reset_last_usage(self) -> None:
        self.reset_usage()


def _load_authorized() -> Set[int]:
    if not config.AUTH_PASSWORD:
        return set()
    try:
        if not os.path.exists(config.AUTH_FILE):
            return set()
        with open(config.AUTH_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            log.warning("Auth file %s has old/invalid format — ignoring", config.AUTH_FILE)
            return set()
        if str(data.get("fingerprint", "")) != CURRENT_FINGERPRINT:
            log.warning("AUTH_PASSWORD changed — users must re-authenticate")
            return set()
        return {int(x) for x in data.get("users", [])}
    except Exception:
        log.exception("Failed to load authorized users from %s", config.AUTH_FILE)
        return set()


def _save_authorized(users: Set[int]) -> None:
    try:
        payload = {"fingerprint": CURRENT_FINGERPRINT, "users": sorted(users)}
        with open(config.AUTH_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        log.exception("Failed to save authorized users to %s", config.AUTH_FILE)


def _load_system_prompts() -> Dict[int, str]:
    try:
        if not os.path.exists(config.SYSTEM_PROMPT_FILE):
            return {}
        with open(config.SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        out: Dict[int, str] = {}
        for k, v in data.items():
            try:
                text = str(v)
            except Exception:
                continue
            if text.strip():
                out[int(k)] = text
        return out
    except Exception:
        log.exception("Failed to load system prompts from %s", config.SYSTEM_PROMPT_FILE)
        return {}


def _save_system_prompts(prompts: Dict[int, str]) -> None:
    try:
        payload = {str(uid): text for uid, text in prompts.items() if text.strip()}
        with open(config.SYSTEM_PROMPT_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        log.exception("Failed to save system prompts to %s", config.SYSTEM_PROMPT_FILE)


def _load_summaries() -> Dict[int, str]:
    try:
        if not os.path.exists(config.SUMMARY_FILE):
            return {}
        with open(config.SUMMARY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {int(k): str(v) for k, v in data.items() if str(v).strip()}
    except Exception:
        log.exception("Failed to load summaries from %s", config.SUMMARY_FILE)
        return {}


def _save_summaries(summaries: Dict[int, str]) -> None:
    try:
        payload = {str(uid): text for uid, text in summaries.items() if text.strip()}
        with open(config.SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        log.exception("Failed to save summaries to %s", config.SUMMARY_FILE)


def _load_tracked() -> Dict[int, List[Tuple[int, int]]]:
    try:
        if not os.path.exists(config.TRACKED_FILE):
            return {}
        with open(config.TRACKED_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        result: Dict[int, List[Tuple[int, int]]] = {}
        for uid_str, pairs in data.items():
            try:
                uid = int(uid_str)
                result[uid] = [
                    (int(p[0]), int(p[1]))
                    for p in pairs
                    if isinstance(p, (list, tuple)) and len(p) >= 2
                ]
            except Exception:
                continue
        return result
    except Exception:
        log.exception("Failed to load tracked messages from %s", config.TRACKED_FILE)
        return {}


def _save_tracked(messages: Dict[int, List[Tuple[int, int]]]) -> None:
    try:
        payload = {
            str(uid): [[int(c), int(m)] for c, m in pairs]
            for uid, pairs in messages.items()
            if pairs
        }
        with open(config.TRACKED_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        log.exception("Failed to save tracked messages to %s", config.TRACKED_FILE)


class ChatStore:
    def __init__(self, max_turns: int) -> None:
        self._history: Dict[int, List[Dict[str, Any]]] = {}
        self._messages: Dict[int, List[Tuple[int, int]]] = _load_tracked()
        self._max_turns = max_turns
        self._lock = asyncio.Lock()
        self._cached_models: List[str] = []
        self._cache_expires: float = 0.0
        self._busy: Set[int] = set()
        self._authorized: Set[int] = _load_authorized()
        self._stop_events: Dict[int, asyncio.Event] = {}
        self._model_op_busy: bool = False
        self._settings: Dict[int, UserSettings] = {}
        self._admin_sessions: Dict[int, float] = {}
        self._last_message_ts: Dict[Tuple[str, int], float] = {}
        self._system_prompts: Dict[int, str] = _load_system_prompts()
        self._summaries: Dict[int, str] = _load_summaries()          # FIX: was missing
        self._loaded_context_length: int = 0
        self._response_ids: Dict[int, str] = {}
        self._response_id_history: Dict[int, List[str]] = {}         # replaces _prev_response_ids
        self._cumulative_prompt_tokens: Dict[int, int] = {}          # NEW
        self._last_user_input: Dict[int, Any] = {}
        # Tools выключены при старте процесса (включаются вручную
        # через /tools on). Thinking включён по умолчанию.
        # Значения из .env (TOOLS_ENABLED / DEFAULT_THINKING) намеренно
        # игнорируются.
        self._tools_enabled: bool = False
        self._thinking_enabled: bool = True

    # ── Thinking toggle ────────────────────────────────────────────────
    async def is_thinking_enabled(self) -> bool:
        async with self._lock:
            return self._thinking_enabled

    async def set_thinking_enabled(self, enabled: bool) -> None:
        async with self._lock:
            self._thinking_enabled = bool(enabled)

    # ── Tools toggle (MCP) ────────────────────────────────────────────
    async def is_tools_enabled(self) -> bool:
        async with self._lock:
            return self._tools_enabled

    async def set_tools_enabled(self, enabled: bool) -> None:
        async with self._lock:
            self._tools_enabled = bool(enabled)

    # ── Cumulative tokens (для корректного бара контекста) ────────────
    async def update_cumulative_tokens(self, user_id: int, prompt_tokens: int) -> int:
        """Учитывает API prompt_tokens с эвристикой delta vs total.

        * delta (LM Studio вернул только прирост) → складываем;
        * total (LM Studio вернул суммарный размер сессии) → заменяем.
        """
        async with self._lock:
            prev = self._cumulative_prompt_tokens.get(user_id, 0)
            pt = int(prompt_tokens or 0)
            if pt <= 0:
                return prev
            if prev == 0:
                new = pt
            elif pt < int(prev * 0.95):
                new = prev + pt
            else:
                new = pt
            self._cumulative_prompt_tokens[user_id] = new
            return new

    async def get_cumulative_tokens(self, user_id: int) -> int:
        async with self._lock:
            return self._cumulative_prompt_tokens.get(user_id, 0)

    async def reset_cumulative_tokens(self, user_id: int) -> None:
        async with self._lock:
            self._cumulative_prompt_tokens.pop(user_id, None)

    # ── Stateful-чаты: response_id stack ──────────────────────────────
    async def get_response_id(self, user_id: int) -> Optional[str]:
        async with self._lock:
            return self._response_ids.get(user_id)

    async def set_response_id(self, user_id: int, new_id: Optional[str]) -> None:
        async with self._lock:
            if new_id:
                stack = self._response_id_history.setdefault(user_id, [])
                if not stack or stack[-1] != new_id:
                    stack.append(new_id)
                self._response_ids[user_id] = new_id
            else:
                self._response_ids.pop(user_id, None)

    async def rollback_response_id(self, user_id: int) -> None:
        async with self._lock:
            stack = self._response_id_history.get(user_id, [])
            if stack:
                stack.pop()
            if stack:
                self._response_ids[user_id] = stack[-1]
            else:
                self._response_ids.pop(user_id, None)
                self._response_id_history.pop(user_id, None)

    async def reset_api_state(self, user_id: int) -> None:
        async with self._lock:
            self._response_ids.pop(user_id, None)
            self._response_id_history.pop(user_id, None)

    async def reset_all_api_states(self) -> int:
        async with self._lock:
            n = len(self._response_ids)
            self._response_ids.clear()
            self._response_id_history.clear()
            self._cumulative_prompt_tokens.clear()
        return n

    # ── Очистка ───────────────────────────────────────────────────────
    async def clear(self, user_id: int) -> None:
        async with self._lock:
            self._history.pop(user_id, None)
            self._response_ids.pop(user_id, None)
            self._response_id_history.pop(user_id, None)
            self._cumulative_prompt_tokens.pop(user_id, None)
            self._last_user_input.pop(user_id, None)
            s = self._settings.get(user_id)
            if s:
                s.reset_usage()

    async def reset_context(self, user_id: int) -> None:
        await self.clear(user_id)

    # ── Резюме ────────────────────────────────────────────────────────
    async def get_summary(self, user_id: int) -> str:
        async with self._lock:
            return self._summaries.get(user_id, "")

    async def set_summary(self, user_id: int, summary: str) -> None:
        async with self._lock:
            summary = (summary or "").strip()
            if summary:
                self._summaries[user_id] = summary
            else:
                self._summaries.pop(user_id, None)
            _save_summaries(self._summaries)

    async def clear_summary(self, user_id: int) -> None:
        async with self._lock:
            self._summaries.pop(user_id, None)
            _save_summaries(self._summaries)

    # ── Last user input ───────────────────────────────────────────────
    async def get_last_user_input(self, user_id: int) -> Optional[Any]:
        async with self._lock:
            return self._last_user_input.get(user_id)

    async def set_last_user_input(self, user_id: int, content: Any) -> None:
        async with self._lock:
            self._last_user_input[user_id] = content

    # ── Cooldown ──────────────────────────────────────────────────────
    async def check_cooldown(
        self, user_id: int, cooldown: float, bucket: str = "msg"
    ) -> float:
        if cooldown <= 0:
            return 0.0
        async with self._lock:
            key = (bucket, user_id)
            now = time.monotonic()
            last = self._last_message_ts.get(key, 0.0)
            left = cooldown - (now - last)
            if left > 0:
                return left
            self._last_message_ts[key] = now
            return 0.0

    # ── Админ-сессии ──────────────────────────────────────────────────
    async def grant_admin(self, user_id: int, ttl: float) -> None:
        async with self._lock:
            self._admin_sessions[user_id] = time.monotonic() + ttl

    async def is_admin(self, user_id: int) -> bool:
        async with self._lock:
            exp = self._admin_sessions.get(user_id)
            if exp is None:
                return False
            if time.monotonic() > exp:
                self._admin_sessions.pop(user_id, None)
                return False
            return True

    async def revoke_admin(self, user_id: int) -> None:
        async with self._lock:
            self._admin_sessions.pop(user_id, None)

    async def admin_ttl_left(self, user_id: int) -> int:
        async with self._lock:
            exp = self._admin_sessions.get(user_id)
            if exp is None:
                return 0
            left = int(exp - time.monotonic())
            if left <= 0:
                self._admin_sessions.pop(user_id, None)
                return 0
            return left

    # ── Настройки ─────────────────────────────────────────────────────
    async def get_settings(self, user_id: int) -> UserSettings:
        async with self._lock:
            return self._settings.setdefault(user_id, UserSettings())

    async def update_settings(self, user_id: int, **kwargs) -> UserSettings:
        async with self._lock:
            s = self._settings.setdefault(user_id, UserSettings())
            for k, v in kwargs.items():
                if hasattr(s, k):
                    setattr(s, k, v)
            return s

    async def set_loaded_context_length(self, n: int) -> None:
        async with self._lock:
            self._loaded_context_length = int(n) if n and n > 0 else 0

    async def get_loaded_context_length(self) -> int:
        async with self._lock:
            return self._loaded_context_length

    async def reset_settings(self, user_id: int) -> UserSettings:
        async with self._lock:
            s = UserSettings()
            self._settings[user_id] = s
            return s

    async def set_usage(
        self,
        user_id: int,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        effective_tokens: Optional[int] = None,
    ) -> None:
        async with self._lock:
            s = self._settings.setdefault(user_id, UserSettings())
            s.last_prompt_tokens = prompt_tokens
            s.last_completion_tokens = completion_tokens
            s.last_total_tokens = total_tokens
            if effective_tokens is not None:
                s.last_effective_tokens = int(effective_tokens)

    async def trim_history(self, user_id: int, keep_last: int) -> int:
        async with self._lock:
            msgs = self._history.get(user_id)
            if not msgs:
                return 0
            keep_last = max(0, keep_last)
            removed = max(0, len(msgs) - keep_last)
            self._history[user_id] = msgs[-keep_last:] if keep_last else []
            s = self._settings.get(user_id)
            if s:
                s.reset_last_usage()
            return removed

    # ── Системный промпт ──────────────────────────────────────────────
    async def get_system_prompt(self, user_id: int) -> str:
        async with self._lock:
            return self._system_prompts.get(user_id, "")

    async def set_system_prompt(self, user_id: int, prompt: str) -> None:
        async with self._lock:
            prompt = (prompt or "").strip()
            if prompt:
                self._system_prompts[user_id] = prompt
            else:
                self._system_prompts.pop(user_id, None)
            _save_system_prompts(self._system_prompts)

    async def clear_system_prompt(self, user_id: int) -> None:
        async with self._lock:
            self._system_prompts.pop(user_id, None)
            _save_system_prompts(self._system_prompts)

    async def get_with_system(self, user_id: int) -> List[Dict[str, Any]]:
        async with self._lock:
            msgs = list(self._history.setdefault(user_id, []))
            sp = (self._system_prompts.get(user_id) or "").strip()
            if sp:
                return [{"role": "system", "content": sp}, *msgs]
            return msgs

    # ── История ───────────────────────────────────────────────────────
    async def get(self, user_id: int) -> List[Dict[str, Any]]:
        async with self._lock:
            return list(self._history.setdefault(user_id, []))

    async def append_user(self, user_id: int, content: Any) -> None:
        async with self._lock:
            self._history.setdefault(user_id, []).append(
                {"role": "user", "content": content}
            )
            self._trim(user_id)

    async def append_assistant(self, user_id: int, content: str) -> None:
        async with self._lock:
            self._history.setdefault(user_id, []).append(
                {"role": "assistant", "content": content}
            )
            self._trim(user_id)

    async def drop_last_assistant(self, user_id: int) -> bool:
        async with self._lock:
            msgs = self._history.get(user_id)
            if not msgs or msgs[-1]["role"] != "assistant":
                return False
            msgs.pop()
            return True

    def _trim(self, user_id: int) -> None:
        msgs = self._history[user_id]
        if len(msgs) > self._max_turns * 2:
            first = msgs[0] if msgs and msgs[0]["role"] == "system" else None
            tail = msgs[-(self._max_turns * 2):]
            self._history[user_id] = [first, *tail] if first else tail

    # ── Трекинг сообщений ─────────────────────────────────────────────
    async def track_message(self, user_id: int, chat_id: int, message_id: int) -> None:
        async with self._lock:
            self._messages.setdefault(user_id, []).append((chat_id, message_id))
            _save_tracked(self._messages)

    async def pop_tracked_messages(self, user_id: int) -> List[Tuple[int, int]]:
        async with self._lock:
            pairs = self._messages.pop(user_id, [])
            if pairs:
                _save_tracked(self._messages)
            return pairs

    async def untrack_message(self, user_id: int, chat_id: int, message_id: int) -> None:
        async with self._lock:
            pairs = self._messages.get(user_id)
            if not pairs:
                return
            try:
                pairs.remove((chat_id, message_id))
            except ValueError:
                return
            _save_tracked(self._messages)

    # ── Кэш моделей ───────────────────────────────────────────────────
    @property
    def models_expired(self) -> bool:
        return time.monotonic() > self._cache_expires

    def get_cached_models(self) -> List[str]:
        return list(self._cached_models)

    def set_models(self, models: List[str], ttl: float = config.CACHE_TTL) -> None:
        self._cached_models = models
        self._cache_expires = time.monotonic() + ttl

    # ── Busy (per-user) ───────────────────────────────────────────────
    async def try_acquire(self, user_id: int) -> bool:
        async with self._lock:
            if user_id in self._busy:
                return False
            self._busy.add(user_id)
            return True

    async def release(self, user_id: int) -> None:
        async with self._lock:
            self._busy.discard(user_id)

    async def is_busy(self, user_id: int) -> bool:
        async with self._lock:
            return user_id in self._busy

    # ── Busy (model op, глобально) ────────────────────────────────────
    async def try_acquire_model_op(self) -> bool:
        async with self._lock:
            if self._model_op_busy:
                return False
            self._model_op_busy = True
            return True

    async def release_model_op(self) -> None:
        async with self._lock:
            self._model_op_busy = False

    async def is_model_op_busy(self) -> bool:
        async with self._lock:
            return self._model_op_busy

    # ── Авторизация ───────────────────────────────────────────────────
    async def is_authorized(self, user_id: int) -> bool:
        async with self._lock:
            return user_id in self._authorized

    async def authorize(self, user_id: int) -> None:
        async with self._lock:
            if user_id in self._authorized:
                return
            self._authorized.add(user_id)
            _save_authorized(self._authorized)
        log.info("User %s authorized", user_id)

    async def revoke(self, user_id: int) -> None:
        async with self._lock:
            if user_id in self._authorized:
                self._authorized.discard(user_id)
                _save_authorized(self._authorized)
        log.info("User %s revoked", user_id)

    # ── Остановка генерации ───────────────────────────────────────────
    async def start_generation(self, user_id: int) -> asyncio.Event:
        async with self._lock:
            ev = asyncio.Event()
            self._stop_events[user_id] = ev
            return ev

    async def end_generation(self, user_id: int) -> None:
        async with self._lock:
            self._stop_events.pop(user_id, None)

    async def request_stop(self, user_id: int) -> bool:
        async with self._lock:
            ev = self._stop_events.get(user_id)
            if ev is None:
                return False
            ev.set()
            return True


store = ChatStore(max_turns=config.MAX_HISTORY_TURNS)