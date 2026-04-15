"""Async Ollama client for LLM generation."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator

import httpx

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemma3:4b"
DEFAULT_TIMEOUT = 30.0


class OllamaGenerationError(Exception):
    """Raised when Ollama generation fails."""


class OllamaClient:
    """Async wrapper around Ollama's /api/generate endpoint."""

    def __init__(
        self,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
        self._base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream tokens from Ollama's /api/generate endpoint."""
        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                async with client.stream(
                    "POST", url, json=payload, timeout=self.timeout
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        import json

                        chunk = json.loads(line)
                        if chunk.get("done"):
                            break
                        token = chunk.get("response", "")
                        if token:
                            yield token
            except httpx.TimeoutException as exc:
                raise OllamaGenerationError(
                    f"Ollama generation timed out after {self.timeout}s"
                ) from exc
            except httpx.HTTPStatusError as exc:
                raise OllamaGenerationError(
                    f"Ollama returned {exc.response.status_code}: "
                    f"{exc.response.text[:200]}"
                ) from exc
            except httpx.HTTPError as exc:
                raise OllamaGenerationError(
                    f"Ollama connection error: {exc}"
                ) from exc

    async def generate(self, prompt: str) -> str:
        """Generate a complete response (non-streaming)."""
        tokens: list[str] = []
        async for token in self.generate_stream(prompt):
            tokens.append(token)
        return "".join(tokens)
