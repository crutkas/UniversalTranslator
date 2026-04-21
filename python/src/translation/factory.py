"""Factory for creating translation engines from config."""

from __future__ import annotations

from typing import Any

from src.translation.base import TranslationEngine
from src.translation.madlad_engine import MadladEngine
from src.translation.nllb_engine import NLLBEngine
from src.translation.seamless_engine import SeamlessEngine

_ENGINE_MAP: dict[str, type[TranslationEngine]] = {
    "nllb-200": NLLBEngine,  # type: ignore[type-abstract]
    "seamless-m4t": SeamlessEngine,  # type: ignore[type-abstract]
    "madlad-400": MadladEngine,  # type: ignore[type-abstract]
}


def create_translation_engine(name: str, config: dict[str, Any] | None = None) -> TranslationEngine:
    """Create a translation engine by name with optional config.

    Args:
        name: Engine identifier (nllb-200, seamless-m4t, madlad-400).
        config: Engine-specific configuration dict.

    Returns:
        Configured TranslationEngine instance.

    Raises:
        ValueError: If engine name is unknown.
    """
    if name not in _ENGINE_MAP:
        available = ", ".join(sorted(_ENGINE_MAP.keys()))
        raise ValueError(f"Unknown translation engine '{name}'. Available: {available}")

    config = config or {}

    if name == "nllb-200":
        return NLLBEngine(
            model_name=config.get("model_name", "facebook/nllb-200-1.3B"),
            device=config.get("device", "auto"),
        )
    elif name == "seamless-m4t":
        return SeamlessEngine(
            model_name=config.get("model_name", "facebook/seamless-m4t-v2-large"),
            device=config.get("device", "auto"),
        )
    elif name == "madlad-400":
        return MadladEngine(
            model_name=config.get("model_name", "google/madlad400-3b-mt"),
            device=config.get("device", "auto"),
        )

    raise ValueError(f"Unknown translation engine '{name}'")


def available_translation_engines() -> list[str]:
    """Return list of registered translation engine names."""
    return list(_ENGINE_MAP.keys())
