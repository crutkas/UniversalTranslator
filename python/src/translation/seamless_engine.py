"""SeamlessM4T v2 translation engine."""

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


SEAMLESS_LANG_CODES: dict[str, str] = {
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "de": "deu",
    "it": "ita",
    "pt": "por",
    "zh": "cmn",
    "ja": "jpn",
    "ko": "kor",
    "ar": "arb",
    "hi": "hin",
    "ru": "rus",
    "nl": "nld",
    "pl": "pol",
    "tr": "tur",
    "vi": "vie",
    "th": "tha",
    "sv": "swe",
}


class SeamlessEngine(TranslationEngine):
    """Translation using Meta's SeamlessM4T v2."""

    REQUIRED_PACKAGES = {"transformers": "transformers>=4.40"}

    def __init__(
        self,
        model_name: str = "facebook/seamless-m4t-v2-large",
        device: str = "auto",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._processor: Any = None

    @property
    def name(self) -> str:
        return "SeamlessM4T v2"

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
        logger.info("Downloading SeamlessM4T model: %s", self._model_name)
        self._ensure_model()
        if progress_callback:
            progress_callback(f"✅ {self.name} ready")

    def _ensure_model(self) -> None:
        if self._model is None:
            tf = _import_transformers()
            logger.info("Loading SeamlessM4T model: %s", self._model_name)
            self._processor = tf.AutoProcessor.from_pretrained(self._model_name)
            self._model = tf.SeamlessM4Tv2ForTextToText.from_pretrained(self._model_name)
            if self._device != "auto" and self._device != "cpu":
                self._model = self._model.to(self._device)

    def _get_seamless_code(self, lang: str) -> str:
        return SEAMLESS_LANG_CODES.get(lang, lang)

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text.strip():
            return ""

        self._ensure_model()
        src_code = self._get_seamless_code(source_lang)
        tgt_code = self._get_seamless_code(target_lang)

        inputs = self._processor(text=text, src_lang=src_code, return_tensors="pt")
        if self._device != "auto" and self._device != "cpu":
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        outputs = self._model.generate(**inputs, tgt_lang=tgt_code, max_new_tokens=512)
        return str(self._processor.decode(outputs[0], skip_special_tokens=True))

    def supported_languages(self) -> list[str]:
        return list(SEAMLESS_LANG_CODES.keys())

    def is_available(self) -> bool:
        try:
            _import_transformers()
            return True
        except ImportError:
            return False

    def cleanup(self) -> None:
        self._model = None
        self._processor = None
