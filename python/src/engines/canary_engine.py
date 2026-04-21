"""HTTP-based STT engine for Canary Qwen (NVIDIA NeMo server)."""

from __future__ import annotations

import logging

import httpx

from src.engines.base import STTEngine

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60.0


class CanaryQwenEngine(STTEngine):
    """Speech-to-text via HTTP to a Canary Qwen NeMo server."""

    def __init__(self, endpoint: str = "http://localhost:8001/transcribe") -> None:
        self._endpoint = endpoint

    @property
    def name(self) -> str:
        return "Canary Qwen 2.5B"

    def transcribe(self, audio_bytes: bytes) -> str:
        try:
            response = httpx.post(
                self._endpoint,
                files={"audio": ("audio.wav", audio_bytes, "audio/wav")},
                timeout=DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            return str(data.get("text", "")).strip()
        except httpx.HTTPError as e:
            logger.error("Canary Qwen transcription failed: %s", e)
            raise RuntimeError(f"Canary Qwen transcription failed: {e}") from e

    def is_available(self) -> bool:
        try:
            response = httpx.get(
                self._endpoint.rsplit("/", 1)[0] + "/health",
                timeout=5.0,
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False
