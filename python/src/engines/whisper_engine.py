"""Whisper STT engine using faster-whisper."""

from __future__ import annotations

import importlib
import io
import logging
import os
from typing import Any

from src.engines.base import STTEngine

logger = logging.getLogger(__name__)


def _import_faster_whisper() -> Any:
    """Dynamically import faster_whisper (works after runtime pip install)."""
    return importlib.import_module("faster_whisper")


class WhisperEngine(STTEngine):
    """Speech-to-text using faster-whisper (CTranslate2)."""

    REQUIRED_PACKAGES = {"faster_whisper": "faster-whisper>=1.0"}

    def __init__(self, model_size: str = "large-v3-turbo", device: str = "auto") -> None:
        self._model_size = model_size
        self._device = self._resolve_device(device)
        self._model: Any = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Determine the best available device."""
        if device != "auto":
            return device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    @staticmethod
    def _compute_type_for_device(device: str) -> str:
        """Pick the right compute type for the device."""
        if device == "cuda":
            return "float16"
        return "int8"

    @property
    def name(self) -> str:
        return f"Whisper ({self._model_size})"

    def needs_download(self) -> bool:
        """Check if the Whisper model needs to be downloaded."""
        if not self.is_available():
            return False
        if self._model is not None:
            return False
        # Scan cache for any repo matching this model size
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        if not os.path.isdir(cache_dir):
            return True
        suffix = f"faster-whisper-{self._model_size}"
        for entry in os.listdir(cache_dir):
            if entry.startswith("models--") and entry.endswith(suffix):
                return False
        return True

    def download_model(self, progress_callback: Any | None = None) -> None:
        """Download the Whisper model with progress updates."""
        fw = _import_faster_whisper()

        if progress_callback:
            progress_callback(f"⬇️ Downloading Whisper {self._model_size}...")

        logger.info("Downloading Whisper model: %s", self._model_size)

        # Loading the model triggers the download
        self._model = fw.WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type_for_device(self._device),
        )

        if progress_callback:
            progress_callback(f"✅ Whisper {self._model_size} ready")

        logger.info("Whisper model downloaded and loaded: %s", self._model_size)

    def _ensure_model(self) -> None:
        if self._model is None:
            fw = _import_faster_whisper()
            logger.info("Loading Whisper model: %s on %s", self._model_size, self._device)
            self._model = fw.WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type_for_device(self._device),
            )

    def transcribe(self, audio_bytes: bytes) -> str:
        self._ensure_model()
        assert self._model is not None

        audio_file = io.BytesIO(audio_bytes)
        segments, _info = self._model.transcribe(
            audio_file,
            language="en",
            beam_size=5,
            vad_filter=True,
        )
        return " ".join(segment.text.strip() for segment in segments)

    def is_available(self) -> bool:
        try:
            _import_faster_whisper()
            return True
        except ImportError:
            return False

    def cleanup(self) -> None:
        self._model = None
