"""NLLB-200 translation engine."""

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

    REQUIRED_PACKAGES = {
        "transformers": "transformers>=4.40",
        "sentencepiece": "sentencepiece>=0.2",
    }

    def __init__(self, model_name: str = "facebook/nllb-200-1.3B", device: str = "auto") -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def name(self) -> str:
        return "NLLB-200"

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
        logger.info("Downloading NLLB model: %s", self._model_name)
        self._ensure_model()
        if progress_callback:
            progress_callback(f"✅ {self.name} ready")

    def _ensure_model(self) -> None:
        if self._model is None:
            tf = _import_transformers()
            logger.info("Loading NLLB model: %s", self._model_name)
            self._tokenizer = tf.AutoTokenizer.from_pretrained(self._model_name)
            self._model = tf.AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
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
        try:
            _import_transformers()
            return True
        except ImportError:
            return False

    def cleanup(self) -> None:
        self._model = None
        self._tokenizer = None
