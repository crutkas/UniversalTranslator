"""SeamlessM4T v2 translation engine."""

from __future__ import annotations

import logging
from typing import Any

from src.translation.base import TranslationEngine

logger = logging.getLogger(__name__)

try:
    from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText

    HAS_SEAMLESS = True
except ImportError:
    HAS_SEAMLESS = False

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

    def _ensure_model(self) -> None:
        if self._model is None:
            if not HAS_SEAMLESS:
                raise RuntimeError("transformers with SeamlessM4T support is not installed")
            logger.info("Loading SeamlessM4T model: %s", self._model_name)
            self._processor = AutoProcessor.from_pretrained(self._model_name)
            self._model = SeamlessM4Tv2ForTextToText.from_pretrained(self._model_name)
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
        return HAS_SEAMLESS

    def cleanup(self) -> None:
        self._model = None
        self._processor = None
