"""Madlad-400 translation engine."""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any

from src.translation.base import TranslationEngine

logger = logging.getLogger(__name__)


def _import_transformers() -> Any:
    """Dynamically import transformers (works after runtime pip install)."""
    return importlib.import_module("transformers")


MADLAD_LANG_CODES: dict[str, str] = {
    "en": "en",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "it": "it",
    "pt": "pt",
    "zh": "zh",
    "ja": "ja",
    "ko": "ko",
    "ar": "ar",
    "hi": "hi",
    "ru": "ru",
    "nl": "nl",
    "pl": "pl",
    "tr": "tr",
    "vi": "vi",
    "th": "th",
    "sv": "sv",
    "da": "da",
    "no": "no",
    "fi": "fi",
    "uk": "uk",
    "cs": "cs",
    "ro": "ro",
    "hu": "hu",
}


class MadladEngine(TranslationEngine):
    """Translation using Google's Madlad-400 model."""

    REQUIRED_PACKAGES = {
        "transformers": "transformers>=4.40",
        "sentencepiece": "sentencepiece>=0.2",
    }

    def __init__(self, model_name: str = "google/madlad400-3b-mt", device: str = "auto") -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def name(self) -> str:
        return "Madlad-400"

    def needs_download(self) -> bool:
        if not self.is_available() or self._model is not None:
            return False
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        repo_name = self._model_name.replace("/", "--")
        return not os.path.exists(os.path.join(cache_dir, f"models--{repo_name}"))

    def download_model(self, progress_callback: Any | None = None) -> None:
        _import_transformers()  # fail fast if not installed
        if progress_callback:
            progress_callback(f"⬇️ Downloading {self.name}...")
        logger.info("Downloading Madlad-400 model: %s", self._model_name)
        self._ensure_model()
        if progress_callback:
            progress_callback(f"✅ {self.name} ready")

    def _ensure_model(self) -> None:
        if self._model is None:
            tf = _import_transformers()
            logger.info("Loading Madlad-400 model: %s", self._model_name)
            self._tokenizer = tf.AutoTokenizer.from_pretrained(self._model_name)
            self._model = tf.AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
            if self._device != "auto" and self._device != "cpu":
                self._model = self._model.to(self._device)

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text.strip():
            return ""

        self._ensure_model()
        tgt_code = MADLAD_LANG_CODES.get(target_lang, target_lang)

        # Madlad-400 uses "<2xx>" prefix format for target language
        prompt = f"<2{tgt_code}> {text}"
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        if self._device != "auto" and self._device != "cpu":
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        outputs = self._model.generate(**inputs, max_new_tokens=512)
        return str(self._tokenizer.decode(outputs[0], skip_special_tokens=True))

    def supported_languages(self) -> list[str]:
        return list(MADLAD_LANG_CODES.keys())

    def is_available(self) -> bool:
        try:
            _import_transformers()
            return True
        except ImportError:
            return False

    def cleanup(self) -> None:
        self._model = None
        self._tokenizer = None
