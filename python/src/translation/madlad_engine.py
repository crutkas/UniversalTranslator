"""Madlad-400 translation engine."""

from __future__ import annotations

import logging
from typing import Any

from src.translation.base import TranslationEngine

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

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

    def __init__(self, model_name: str = "google/madlad400-3b-mt", device: str = "auto") -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def name(self) -> str:
        return "Madlad-400"

    def _ensure_model(self) -> None:
        if self._model is None:
            if not HAS_TRANSFORMERS:
                raise RuntimeError("transformers is not installed")
            logger.info("Loading Madlad-400 model: %s", self._model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
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
        return HAS_TRANSFORMERS

    def cleanup(self) -> None:
        self._model = None
        self._tokenizer = None
