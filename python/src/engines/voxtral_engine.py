"""Voxtral Transcribe 2 STT engine (in-process).

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


def _import_transformers() -> Any:
    """Dynamically import transformers (works after runtime pip install)."""
    return importlib.import_module("transformers")


def _import_torchaudio() -> Any:
    """Dynamically import torchaudio (works after runtime pip install)."""
    return importlib.import_module("torchaudio")


class VoxtralEngine(STTEngine):
    """Speech-to-text using Voxtral Transcribe 2 in-process."""

    REQUIRED_PACKAGES = {
        "transformers": "transformers>=4.40",
        "torchaudio": "torchaudio>=2.3",
    }

    def __init__(
        self, model_name: str = "mistralai/Voxtral-Mini-4B-Realtime-2602", **kwargs: Any
    ) -> None:
        self._model_name = model_name
        self._model: Any = None
        self._processor: Any = None

    @property
    def name(self) -> str:
        return "Voxtral Transcribe 2"

    def needs_download(self) -> bool:
        if not self.is_available() or self._model is not None:
            return False
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        repo_name = self._model_name.replace("/", "--")
        return not os.path.exists(os.path.join(cache_dir, f"models--{repo_name}"))

    def download_model(self, progress_callback: Any | None = None) -> None:
        _import_transformers()  # fail fast if not installed
        if progress_callback:
            progress_callback(f"⬇️ Downloading {self.name}... (~8.9GB, first time only)")
        logger.info("Downloading Voxtral model: %s", self._model_name)
        self._ensure_model()
        if progress_callback:
            progress_callback(f"✅ {self.name} ready")

    def _ensure_model(self) -> None:
        if self._model is None:
            tf = _import_transformers()
            logger.info("Loading Voxtral model: %s", self._model_name)
            self._processor = tf.AutoProcessor.from_pretrained(self._model_name)
            self._model = tf.AutoModelForSpeechSeq2Seq.from_pretrained(self._model_name)

    def transcribe(self, audio_bytes: bytes) -> str:
        self._ensure_model()

        ta = _import_torchaudio()

        # Write WAV bytes to temp file for torchaudio to load
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            waveform, sample_rate = ta.load(temp_path)
            if sample_rate != 16000:
                waveform = ta.transforms.Resample(sample_rate, 16000)(waveform)

            inputs = self._processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
            outputs = self._model.generate(**inputs, max_new_tokens=512)
            return str(self._processor.decode(outputs[0], skip_special_tokens=True)).strip()
        finally:
            os.unlink(temp_path)

    def is_available(self) -> bool:
        try:
            _import_transformers()
            _import_torchaudio()
            return True
        except ImportError:
            return False

    def cleanup(self) -> None:
        self._model = None
        self._processor = None
