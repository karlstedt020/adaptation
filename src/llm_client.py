"""OpenRouter API client (OpenAI-compatible)."""

import time
import json
import logging
from typing import Optional

from openai import OpenAI

from .config import OpenRouterConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper around OpenRouter via the OpenAI SDK."""

    def __init__(self, cfg: OpenRouterConfig):
        if not cfg.api_key:
            raise ValueError("OpenRouter API key is not set in config.")
        if not cfg.model:
            raise ValueError("OpenRouter model is not set in config.")
        self._client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.timeout,
        )
        self._model = cfg.model
        self._max_retries = cfg.max_retries

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Single completion, returns assistant text."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._call(messages, temperature, max_tokens)

    def complete_n(
        self,
        prompt: str,
        system_prompt: str = "",
        n: int = 3,
        temperature: float = 0.9,
        max_tokens: int = 2048,
    ) -> list[str]:
        """Generate *n* independent completions (separate calls)."""
        return [
            self.complete(prompt, system_prompt, temperature, max_tokens)
            for _ in range(n)
        ]

    def _call(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
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
                logger.warning("LLM call attempt %d failed: %s", attempt, exc)
                if attempt == self._max_retries:
                    raise
                time.sleep(2 ** attempt)
