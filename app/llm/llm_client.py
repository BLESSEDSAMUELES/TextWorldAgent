"""
Ollama LLM client.

Thin wrapper around the Ollama Python library for local inference.
Optimized for Gemma2:2b on CPU — persistent client, minimal context,
low temperature, short max tokens, fast retry logic.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config import AppConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for local Ollama LLM inference — optimized for CPU speed."""

    def __init__(self, config: AppConfig) -> None:
        self._model = config.llm_model
        self._temperature = config.llm_temperature
        self._max_tokens = config.llm_max_tokens
        self._max_retries = config.llm_max_retries
        self._host = config.ollama_host
        self._client: Optional[object] = None  # lazy-initialized persistent client

    def _get_client(self) -> Optional[object]:
        """Return a persistent Ollama client, creating it once on first use."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self._host)
            except ImportError:
                logger.error("ollama package not installed. Run: pip install ollama")
                return None
        return self._client

    def generate(self, prompt: str) -> Optional[str]:
        """
        Generate a response from the LLM.

        Optimizations applied:
        - Persistent client (no per-call socket reconnect)
        - Minimal num_ctx (512) — matches ~250 token world slice
        - top_k=20, top_p=0.8 — narrows sampling distribution for speed
        - num_predict capped at llm_max_tokens (default 30)
        - num_thread=4 — maximize CPU parallelism

        Args:
            prompt: The full prompt to send.

        Returns:
            The model's response text, or None on failure.
        """
        client = self._get_client()
        if client is None:
            return None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = client.generate(  # type: ignore[union-attr]
                    model=self._model,
                    prompt=prompt,
                    options={
                        "temperature": self._temperature,
                        "num_predict": self._max_tokens,
                        "num_ctx": 512,       # match world slice size; avoids huge KV cache
                        "top_k": 20,          # narrow sampling → faster decisions
                        "top_p": 0.8,
                        "repeat_penalty": 1.1,
                        "num_thread": 4,      # parallel CPU threads
                    },
                )
                # Support both Pydantic GenerateResponse and legacy dict
                if hasattr(response, "response"):
                    text = response.response.strip()
                else:
                    text = response.get("response", "").strip()  # type: ignore[union-attr]

                if text:
                    return text

                logger.warning(
                    "Empty response from LLM (attempt %d/%d)",
                    attempt, self._max_retries,
                )
            except Exception as e:
                logger.error(
                    "LLM generation failed (attempt %d/%d): %s",
                    attempt, self._max_retries, e,
                )

        return None

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            client = self._get_client()
            if client is None:
                return False
            response = client.list()  # type: ignore[union-attr]

            # Support both Pydantic model objects and older dict response types
            if hasattr(response, "models"):
                model_names = [getattr(m, "model", "") for m in response.models]
            elif isinstance(response, dict):
                model_names = [m.get("name", "") for m in response.get("models", [])]
            else:
                model_names = []
                
            return any(
                self._model in name for name in model_names
            )
        except Exception:
            return False
