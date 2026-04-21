"""Factory for creating STT engines from config."""

from __future__ import annotations

from typing import Any

from src.engines.base import STTEngine
from src.engines.canary_engine import CanaryQwenEngine
from src.engines.qwen3_engine import Qwen3ASREngine
from src.engines.voxtral_engine import VoxtralEngine
from src.engines.whisper_engine import WhisperEngine

_ENGINE_MAP: dict[str, type[STTEngine]] = {
    "whisper": WhisperEngine,  # type: ignore[type-abstract]
    "canary_qwen": CanaryQwenEngine,  # type: ignore[type-abstract]
    "voxtral": VoxtralEngine,  # type: ignore[type-abstract]
    "qwen3_asr": Qwen3ASREngine,  # type: ignore[type-abstract]
}


def create_engine(name: str, config: dict[str, Any] | None = None) -> STTEngine:
    """Create an STT engine by name with optional config.

    Args:
        name: Engine identifier (whisper, canary_qwen, voxtral, qwen3_asr).
        config: Engine-specific configuration dict.

    Returns:
        Configured STTEngine instance.

    Raises:
        ValueError: If engine name is unknown.
    """
    if name not in _ENGINE_MAP:
        available = ", ".join(sorted(_ENGINE_MAP.keys()))
        raise ValueError(f"Unknown STT engine '{name}'. Available: {available}")

    config = config or {}

    if name == "whisper":
        return WhisperEngine(
            model_size=config.get("model_size", "large-v3-turbo"),
            device=config.get("device", "auto"),
        )
    elif name == "canary_qwen":
        return CanaryQwenEngine(endpoint=config.get("endpoint", "http://localhost:8001/transcribe"))
    elif name == "voxtral":
        return VoxtralEngine(endpoint=config.get("endpoint", "http://localhost:8002/transcribe"))
    elif name == "qwen3_asr":
        return Qwen3ASREngine(model_name=config.get("model_name", "Qwen/Qwen3-ASR-1.7B"))

    raise ValueError(f"Unknown STT engine '{name}'")


def available_engines() -> list[str]:
    """Return list of registered engine names."""
    return list(_ENGINE_MAP.keys())
