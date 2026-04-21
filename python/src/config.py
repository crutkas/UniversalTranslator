"""Configuration loader for UniversalTranslator."""

from __future__ import annotations

import json
import os
from typing import Any

DEFAULT_CONFIG_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.json"),
    os.path.join(os.path.expanduser("~"), ".universaltranslator", "config.json"),
]

DEFAULT_CONFIG: dict[str, Any] = {
    "hotkey": "ctrl+win+h",
    "default_model": "whisper",
    "models": {
        "whisper": {
            "enabled": True,
            "mode": "in-process",
            "model_size": "large-v3-turbo",
            "device": "auto",
        },
    },
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
        "format": "wav",
    },
    "translation": {
        "enabled": False,
        "target_language": "es",
        "model": "nllb-200",
        "models": {},
    },
    "ui": {
        "overlay_width": 320,
        "overlay_height": 120,
        "waveform_color": "#4CAF50",
        "processing_color": "#FF9800",
        "translating_color": "#2196F3",
        "opacity": 0.9,
    },
}


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load configuration from a JSON file.

    Args:
        path: Explicit path to config.json. If None, searches default locations.

    Returns:
        Configuration dictionary with defaults applied.
    """
    if path and os.path.exists(path):
        with open(path) as f:
            user_config = json.load(f)
        return _merge_config(DEFAULT_CONFIG, user_config)

    for default_path in DEFAULT_CONFIG_PATHS:
        if os.path.exists(default_path):
            with open(default_path) as f:
                user_config = json.load(f)
            return _merge_config(DEFAULT_CONFIG, user_config)

    return DEFAULT_CONFIG.copy()


def _merge_config(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep merge overrides into defaults."""
    result = defaults.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
    return result
