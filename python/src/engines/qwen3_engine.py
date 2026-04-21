"""Qwen3-ASR STT engine (in-process)."""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any

from src.engines.base import STTEngine

logger = logging.getLogger(__name__)


def _import_qwen_asr() -> Any:
    """Dynamically import qwen_asr (works after runtime pip install)."""
    return importlib.import_module("qwen_asr")


class Qwen3ASREngine(STTEngine):
    """Speech-to-text using Qwen3-ASR in-process."""

    REQUIRED_PACKAGES = {"qwen_asr": "qwen-asr"}

    def __init__(self, model_name: str = "Qwen/Qwen3-ASR-1.7B") -> None:
        self._model_name = model_name
        self._model: Any = None

    @property
    def name(self) -> str:
        return "Qwen3-ASR"

    def needs_download(self) -> bool:
        if self._model is not None:
            return False
        if not self.is_available():
            return False
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        repo_name = self._model_name.replace("/", "--")
        return not os.path.exists(os.path.join(cache_dir, f"models--{repo_name}"))

    def download_model(self, progress_callback: Any | None = None) -> None:
        qwen_asr = _import_qwen_asr()
        if progress_callback:
            progress_callback(f"⬇️ Downloading {self.name}...")
        logger.info("Downloading Qwen3-ASR model: %s", self._model_name)
        self._model = qwen_asr.Qwen3ASRModel.from_pretrained(self._model_name)
        if progress_callback:
            progress_callback(f"✅ {self.name} ready")

    def _ensure_model(self) -> None:
        if self._model is None:
            qwen_asr = _import_qwen_asr()
            logger.info("Loading Qwen3-ASR model: %s", self._model_name)
            self._model = qwen_asr.Qwen3ASRModel.from_pretrained(self._model_name)

    def transcribe(self, audio_bytes: bytes) -> str:
        self._ensure_model()
        # Write to temp file — Qwen3ASR expects file paths
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            result = self._model.transcribe([temp_path])
            if isinstance(result, list) and len(result) > 0:
                item = result[0]
                if isinstance(item, dict):
                    return str(item.get("text", "")).strip()
                return str(item).strip()
            return str(result).strip()
        finally:
            os.unlink(temp_path)

    def is_available(self) -> bool:
        try:
            _import_qwen_asr()
            return True
        except ImportError:
            return False

    def cleanup(self) -> None:
        self._model = None
