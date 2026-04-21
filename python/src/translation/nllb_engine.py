"""NLLB-200 translation engine."""

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

# Common NLLB language code mapping
NLLB_LANG_CODES: dict[str, str] = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "ru": "rus_Cyrl",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "tr": "tur_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "no": "nob_Latn",
    "fi": "fin_Latn",
    "uk": "ukr_Cyrl",
    "cs": "ces_Latn",
    "ro": "ron_Latn",
    "hu": "hun_Latn",
}


class NLLBEngine(TranslationEngine):
    """Translation using Meta's NLLB-200 model."""

    def __init__(self, model_name: str = "facebook/nllb-200-1.3B", device: str = "auto") -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def name(self) -> str:
        return "NLLB-200"

    def _ensure_model(self) -> None:
        if self._model is None:
            if not HAS_TRANSFORMERS:
                raise RuntimeError("transformers is not installed")
            logger.info("Loading NLLB model: %s", self._model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
            if self._device != "auto" and self._device != "cpu":
                self._model = self._model.to(self._device)

    def _get_nllb_code(self, lang: str) -> str:
        """Convert short language code to NLLB format."""
        return NLLB_LANG_CODES.get(lang, lang)

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text.strip():
            return ""

        self._ensure_model()
        src_code = self._get_nllb_code(source_lang)
        tgt_code = self._get_nllb_code(target_lang)

        self._tokenizer.src_lang = src_code
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        if self._device != "auto" and self._device != "cpu":
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        tgt_lang_id = self._tokenizer.convert_tokens_to_ids(tgt_code)
        outputs = self._model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_new_tokens=512,
        )
        return str(self._tokenizer.decode(outputs[0], skip_special_tokens=True))

    def supported_languages(self) -> list[str]:
        return list(NLLB_LANG_CODES.keys())

    def is_available(self) -> bool:
        return HAS_TRANSFORMERS

    def cleanup(self) -> None:
        self._model = None
        self._tokenizer = None
