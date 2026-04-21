"""Qwen3-ASR STT engine (in-process)."""

from __future__ import annotations

import io
import logging

from src.engines.base import STTEngine

logger = logging.getLogger(__name__)

try:
    import qwen_asr  # type: ignore[import-untyped]

    HAS_QWEN_ASR = True
except ImportError:
    HAS_QWEN_ASR = False


class Qwen3ASREngine(STTEngine):
    """Speech-to-text using Qwen3-ASR in-process."""

    def __init__(self, model_name: str = "Qwen/Qwen3-ASR-1.7B") -> None:
        self._model_name = model_name
        self._model = None

    @property
    def name(self) -> str:
        return "Qwen3-ASR"

    def _ensure_model(self) -> None:
        if self._model is None:
            if not HAS_QWEN_ASR:
                raise RuntimeError("qwen-asr is not installed")
            logger.info("Loading Qwen3-ASR model: %s", self._model_name)
            self._model = qwen_asr.load(self._model_name)

    def transcribe(self, audio_bytes: bytes) -> str:
        self._ensure_model()
        audio_file = io.BytesIO(audio_bytes)
        result = self._model.transcribe(audio_file)  # type: ignore[attr-defined]
        if isinstance(result, dict):
            return str(result.get("text", "")).strip()
        return str(result).strip()

    def is_available(self) -> bool:
        return HAS_QWEN_ASR

    def cleanup(self) -> None:
        self._model = None
