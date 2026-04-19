"""OpenRouter API client (OpenAI-compatible) with hash-keyed response cache."""

import hashlib
import logging
import threading
import time

from openai import OpenAI

from .config import OpenRouterConfig

logger = logging.getLogger(__name__)


def _cache_key(system: str, user: str) -> str:
    """SHA-256 of (system || user) — compact, collision-resistant key."""
    payload = f"{system}\x00{user}".encode()
    return hashlib.sha256(payload).hexdigest()


class LLMClient:
    """Thin wrapper around OpenRouter via the OpenAI SDK.

    Cache policy: responses are cached **only for temperature=0** calls
    (deterministic NER, judge, label-check, shift-check).
    Stochastic generation (temperature > 0) is never cached.
    The cache is thread-safe via a lock.
    """

    def __init__(self, cfg: OpenRouterConfig, model_override: str = "") -> None:
        if not cfg.api_key:
            raise ValueError("OpenRouter API key is not set in config.")
        model = model_override or cfg.model
        if not model:
            raise ValueError("OpenRouter model is not set in config.")
        self._client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.timeout,
        )
        self._model = model
        self._max_retries = cfg.max_retries
        self._cache: dict[str, str] = {}
        self._cache_lock = threading.Lock()

    # ── Public API ───────────────────────────────────────────

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Single completion — cached when temperature == 0."""
        if temperature == 0.0:
            key = _cache_key(system_prompt, prompt)
            with self._cache_lock:
                if key in self._cache:
                    return self._cache[key]

        messages = _build_messages(system_prompt, prompt)
        result = self._call(messages, temperature, max_tokens)

        if temperature == 0.0:
            with self._cache_lock:
                self._cache[key] = result
        return result

    def complete_n(
        self,
        prompt: str,
        system_prompt: str = "",
        n: int = 3,
        temperature: float = 0.9,
        max_tokens: int = 2048,
    ) -> list[str]:
        """Generate *n* independent completions (separate API calls, not cached)."""
        return [
            self.complete(prompt, system_prompt, temperature, max_tokens)
            for _ in range(n)
        ]

    @property
    def cache_size(self) -> int:
        with self._cache_lock:
            return len(self._cache)

    # ── Internal ─────────────────────────────────────────────

    def _call(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                logger.warning("LLM call attempt %d/%d failed: %s",
                               attempt, self._max_retries, exc)
                if attempt == self._max_retries:
                    raise
                time.sleep(2 ** attempt)


def _build_messages(system: str, user: str) -> list[dict]:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return msgs
