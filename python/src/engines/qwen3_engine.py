"""Qwen3-ASR STT engine (in-process)."""

from __future__ import annotations

import io
import logging
import os
from typing import Any

from src.engines.base import STTEngine

logger = logging.getLogger(__name__)

try:
    import qwen_asr  # type: ignore[import-untyped]

    HAS_QWEN_ASR = True
except ImportError:
    HAS_QWEN_ASR = False


class Qwen3ASREngine(STTEngine):
    """Speech-to-text using Qwen3-ASR in-process."""

    REQUIRED_PACKAGES = {"qwen_asr": "qwen-asr"}

    def __init__(self, model_name: str = "Qwen/Qwen3-ASR-1.7B") -> None:
        self._model_name = model_name
        self._model = None

    @property
    def name(self) -> str:
        return "Qwen3-ASR"

    def needs_download(self) -> bool:
        if not HAS_QWEN_ASR or self._model is not None:
            return False
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        repo_name = self._model_name.replace("/", "--")
        return not os.path.exists(os.path.join(cache_dir, f"models--{repo_name}"))

    def download_model(self, progress_callback: Any | None = None) -> None:
        if not HAS_QWEN_ASR:
            raise RuntimeError("qwen-asr is not installed")
        if progress_callback:
            progress_callback(f"⬇️ Downloading {self.name}...")
        logger.info("Downloading Qwen3-ASR model: %s", self._model_name)
        self._model = qwen_asr.load(self._model_name)
        if progress_callback:
            progress_callback(f"✅ {self.name} ready")

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
