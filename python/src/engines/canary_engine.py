"""Canary Qwen 2.5B STT engine (in-process).

Downloads and loads the model automatically on first use.
"""

from __future__ import annotations

import importlib
import logging
import os
import tempfile
from typing import Any

from src.engines.base import STTEngine

logger = logging.getLogger(__name__)


def _import_nemo_asr() -> Any:
    """Dynamically import nemo ASR models (works after runtime pip install)."""
    mod = importlib.import_module("nemo.collections.asr.models")
    return mod


class CanaryQwenEngine(STTEngine):
    """Speech-to-text using NVIDIA Canary Qwen 2.5B in-process."""

    REQUIRED_PACKAGES = {"nemo": "nemo_toolkit[asr]>=1.23"}

    def __init__(self, model_name: str = "nvidia/canary-qwen-2.5b", **kwargs: Any) -> None:
        self._model_name = model_name
        self._model: Any = None

    @property
    def name(self) -> str:
        return "Canary Qwen 2.5B"

    def needs_download(self) -> bool:
        if not self.is_available() or self._model is not None:
            return False
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        repo_name = self._model_name.replace("/", "--")
        return not os.path.exists(os.path.join(cache_dir, f"models--{repo_name}"))

    def download_model(self, progress_callback: Any | None = None) -> None:
        _import_nemo_asr()  # fail fast if not installed
        if progress_callback:
            progress_callback(f"⬇️ Downloading {self.name}... (~5GB, first time only)")
        logger.info("Downloading Canary Qwen model: %s", self._model_name)
        self._ensure_model()
        if progress_callback:
            progress_callback(f"✅ {self.name} ready")

    def _ensure_model(self) -> None:
        if self._model is None:
            mod = _import_nemo_asr()
            logger.info("Loading Canary Qwen model: %s", self._model_name)
            self._model = mod.ASRModel.from_pretrained(self._model_name)

    def transcribe(self, audio_bytes: bytes) -> str:
        self._ensure_model()

        # NeMo expects file paths, write to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            result = self._model.transcribe([temp_path])
            text = result[0] if isinstance(result, list) else str(result)
            return text.strip()
        finally:
            os.unlink(temp_path)

    def is_available(self) -> bool:
        try:
            _import_nemo_asr()
            return True
        except ImportError:
            return False

    def cleanup(self) -> None:
        self._model = None
