"""UniversalTranslator — Main entry point.

Wires together audio capture, hotkey detection, STT engine,
optional translation, overlay UI, and paste output.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any

from src.audio import AudioRecorder
from src.config import load_config
from src.engines.factory import create_engine
from src.hotkey import AppState, HotkeyManager
from src.paste import PasteManager
from src.translation.base import TranslationEngine

logger = logging.getLogger(__name__)

# Conditional imports
try:
    from PyQt6.QtCore import QObject, pyqtSignal
    from PyQt6.QtWidgets import QApplication

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


class AppController(QObject if HAS_PYQT6 else object):  # type: ignore[misc]
    """Main application controller connecting all modules."""

    if HAS_PYQT6:
        start_recording_signal = pyqtSignal()
        stop_recording_signal = pyqtSignal()

    def __init__(self, config: dict[str, Any]) -> None:
        if HAS_PYQT6:
            super().__init__()

        self._config = config
        self._recorder = AudioRecorder()
        self._paste_manager = PasteManager()

        # Create STT engine (lazy-loaded)
        model_name = config.get("default_model", "whisper")
        model_config = config.get("models", {}).get(model_name, {})
        self._stt_engine = create_engine(model_name, model_config)

        # Translation engine (lazy, only if enabled)
        self._translation_engine: TranslationEngine | None = None
        self._translation_enabled = config.get("translation", {}).get("enabled", False)
        self._target_language = config.get("translation", {}).get("target_language", "es")

        # Overlay (created after QApplication)
        self._overlay: Any = None

        # Hotkey manager
        self._hotkey = HotkeyManager(
            hotkey_str=config.get("hotkey", "ctrl+shift+space"),
            on_start=self._on_hotkey_start,
            on_stop=self._on_hotkey_stop,
        )

        if HAS_PYQT6:
            self.start_recording_signal.connect(self._handle_start_recording)
            self.stop_recording_signal.connect(self._handle_stop_recording)

    def _init_overlay(self) -> None:
        """Initialize the overlay window (must be called after QApplication)."""
        if not HAS_PYQT6:
            return

        from src.overlay import OverlayWindow

        ui_config = self._config.get("ui", {})
        self._overlay = OverlayWindow(
            width=ui_config.get("overlay_width", 320),
            height=ui_config.get("overlay_height", 120),
            waveform_color=ui_config.get("waveform_color", "#4CAF50"),
            processing_color=ui_config.get("processing_color", "#FF9800"),
            translating_color=ui_config.get("translating_color", "#2196F3"),
            opacity=ui_config.get("opacity", 0.9),
        )
        self._overlay.set_ring_buffer(self._recorder.ring_buffer)

    def _init_tray(self) -> None:
        """Initialize the system tray (must be called after QApplication)."""
        if not HAS_PYQT6:
            return

        from src.tray import SystemTray

        self._tray = SystemTray(
            config=self._config,
            on_model_change=self._on_model_change,
            on_translation_toggle=self._on_translation_toggle,
            on_translation_model_change=self._on_translation_model_change,
            on_target_language_change=self._on_target_language_change,
            on_quit=self._on_quit,
        )
        self._tray.show()

    def _on_hotkey_start(self) -> None:
        """Called from pynput thread when hotkey is pressed."""
        if HAS_PYQT6:
            self.start_recording_signal.emit()
        else:
            self._handle_start_recording()

    def _on_hotkey_stop(self) -> None:
        """Called from pynput thread when hotkey is released."""
        if HAS_PYQT6:
            self.stop_recording_signal.emit()
        else:
            self._handle_stop_recording()

    def _handle_start_recording(self) -> None:
        """Start recording — runs on Qt main thread."""
        logger.info("Recording started")
        self._paste_manager.capture_target_window()

        try:
            self._recorder.start_recording()
        except RuntimeError as e:
            logger.error("Failed to start recording: %s", e)
            self._hotkey.state = AppState.IDLE
            return

        if self._overlay:
            self._overlay.set_state_signal.emit("recording", self._stt_engine.name)
            self._overlay.show_signal.emit()

    def _handle_stop_recording(self) -> None:
        """Stop recording and start transcription — runs on Qt main thread."""
        logger.info("Recording stopped, starting transcription")

        if self._overlay:
            self._overlay.set_state_signal.emit("processing", self._stt_engine.name)

        # Run transcription in background thread
        worker = threading.Thread(target=self._transcribe_and_paste, daemon=True)
        worker.start()

    def _transcribe_and_paste(self) -> None:
        """Transcribe audio and paste result. Runs in worker thread."""
        try:
            audio = self._recorder.stop_recording()
            if len(audio) == 0:
                logger.warning("No audio captured")
                self._finish()
                return

            wav_bytes = self._recorder.get_wav_bytes(audio)
            text = self._stt_engine.transcribe(wav_bytes)

            if not text.strip():
                logger.warning("Empty transcription")
                self._finish()
                return

            logger.info("Transcribed: %s", text[:100])

            # Translation step (if enabled)
            if self._translation_enabled and self._translation_engine is not None:
                if self._overlay:
                    self._overlay.set_state_signal.emit("translating", self._target_language)

                translated = self._translation_engine.translate(text, "en", self._target_language)
                if translated:
                    if self._overlay:
                        self._overlay.update_translation_text(translated)
                    text = translated

            # Paste
            self._paste_manager.paste_text(text)
            self._finish()

        except Exception as e:
            logger.error("Transcription/translation failed: %s", e)
            self._finish()

    def _finish(self) -> None:
        """Complete the pipeline — show done state and clean up."""
        if self._overlay:
            self._overlay.set_state_signal.emit("done", "")
        self._hotkey.state = AppState.IDLE

    def _on_model_change(self, name: str) -> None:
        model_config = self._config.get("models", {}).get(name, {})
        self._stt_engine.cleanup()
        self._stt_engine = create_engine(name, model_config)
        logger.info("STT model changed to: %s", name)

    def _on_translation_toggle(self, enabled: bool) -> None:
        self._translation_enabled = enabled
        if enabled and self._translation_engine is None:
            self._load_translation_engine()
        logger.info("Translation %s", "enabled" if enabled else "disabled")

    def _on_translation_model_change(self, name: str) -> None:
        if self._translation_engine:
            self._translation_engine.cleanup()
        trans_config = self._config.get("translation", {}).get("models", {}).get(name, {})
        from src.translation.factory import create_translation_engine

        self._translation_engine = create_translation_engine(name, trans_config)
        logger.info("Translation model changed to: %s", name)

    def _on_target_language_change(self, code: str) -> None:
        self._target_language = code
        logger.info("Target language changed to: %s", code)

    def _load_translation_engine(self) -> None:
        trans_config = self._config.get("translation", {})
        model_name = trans_config.get("model", "nllb-200")
        model_config = trans_config.get("models", {}).get(model_name, {})
        from src.translation.factory import create_translation_engine

        self._translation_engine = create_translation_engine(model_name, model_config)

    def _on_quit(self) -> None:
        self._hotkey.stop()
        if self._stt_engine:
            self._stt_engine.cleanup()
        if self._translation_engine:
            self._translation_engine.cleanup()
        if HAS_PYQT6:
            QApplication.quit()

    def run(self) -> None:
        """Start the application."""
        self._hotkey.start()
        logger.info("UniversalTranslator ready. Hotkey: %s", self._config.get("hotkey"))


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config()

    if not HAS_PYQT6:
        logger.error("PyQt6 is required. Install with: pip install PyQt6")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setApplicationName("UniversalTranslator")
    app.setQuitOnLastWindowClosed(False)

    controller = AppController(config)
    controller._init_overlay()
    controller._init_tray()
    controller.run()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
