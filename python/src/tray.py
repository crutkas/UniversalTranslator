"""System tray integration with model selection and settings."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtGui import QAction, QIcon
    from PyQt6.QtWidgets import QMenu, QSystemTrayIcon

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

from src.engines.factory import available_engines  # noqa: E402
from src.translation.factory import available_translation_engines  # noqa: E402


class SystemTray:
    """System tray icon with model selection and translation controls."""

    def __init__(
        self,
        config: dict[str, Any],
        on_model_change: Callable[[str], None] | None = None,
        on_translation_toggle: Callable[[bool], None] | None = None,
        on_translation_model_change: Callable[[str], None] | None = None,
        on_target_language_change: Callable[[str], None] | None = None,
        on_quit: Callable[[], None] | None = None,
    ) -> None:
        if not HAS_PYQT6:
            raise RuntimeError("PyQt6 is required for system tray")

        self._config = config
        self._on_model_change = on_model_change
        self._on_translation_toggle = on_translation_toggle
        self._on_translation_model_change = on_translation_model_change
        self._on_target_language_change = on_target_language_change
        self._on_quit = on_quit

        self._current_model = config.get("default_model", "whisper")
        self._translation_enabled = config.get("translation", {}).get("enabled", False)
        self._translation_model = config.get("translation", {}).get("model", "nllb-200")
        self._target_language = config.get("translation", {}).get("target_language", "es")

        self._tray = QSystemTrayIcon()
        self._tray.setToolTip(self._build_tooltip())

        # Set icon
        icon_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "assets",
            "icon.png",
        )
        if os.path.exists(icon_path):
            self._tray.setIcon(QIcon(icon_path))

        self._build_menu()

    def _build_tooltip(self) -> str:
        tip = f"UniversalTranslator | STT: {self._current_model}"
        if self._translation_enabled:
            tip += f" | Translate → {self._target_language}"
        return tip

    def _build_menu(self) -> None:
        menu = QMenu()

        # STT model selection
        stt_menu = menu.addMenu("STT Model")
        assert stt_menu is not None
        stt_group: list[QAction] = []
        for engine_name in available_engines():
            action = QAction(engine_name)
            action.setCheckable(True)
            action.setChecked(engine_name == self._current_model)
            action.triggered.connect(lambda checked, name=engine_name: self._select_model(name))
            stt_menu.addAction(action)
            stt_group.append(action)
        self._stt_actions = stt_group

        menu.addSeparator()

        # Translation toggle
        self._translate_action = QAction("Enable Translation")
        self._translate_action.setCheckable(True)
        self._translate_action.setChecked(self._translation_enabled)
        self._translate_action.triggered.connect(self._toggle_translation)
        menu.addAction(self._translate_action)

        # Translation model selection
        trans_menu = menu.addMenu("Translation Model")
        assert trans_menu is not None
        trans_group: list[QAction] = []
        for engine_name in available_translation_engines():
            action = QAction(engine_name)
            action.setCheckable(True)
            action.setChecked(engine_name == self._translation_model)
            action.triggered.connect(
                lambda checked, name=engine_name: self._select_translation_model(name)
            )
            trans_menu.addAction(action)
            trans_group.append(action)
        self._trans_actions = trans_group

        # Target language selection
        lang_menu = menu.addMenu("Target Language")
        assert lang_menu is not None
        languages = {
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ar": "Arabic",
            "hi": "Hindi",
            "ru": "Russian",
            "nl": "Dutch",
        }
        lang_group: list[QAction] = []
        for code, label in languages.items():
            action = QAction(f"{label} ({code})")
            action.setCheckable(True)
            action.setChecked(code == self._target_language)
            action.triggered.connect(lambda checked, c=code: self._select_target_language(c))
            lang_menu.addAction(action)
            lang_group.append(action)
        self._lang_actions = lang_group

        menu.addSeparator()

        # Settings
        settings_action = QAction("Open Config...")
        settings_action.triggered.connect(self._open_config)
        menu.addAction(settings_action)

        menu.addSeparator()

        # Quit
        quit_action = QAction("Quit")
        quit_action.triggered.connect(self._quit)
        menu.addAction(quit_action)

        self._tray.setContextMenu(menu)

    def _select_model(self, name: str) -> None:
        self._current_model = name
        for action in self._stt_actions:
            action.setChecked(action.text() == name)
        self._tray.setToolTip(self._build_tooltip())
        if self._on_model_change:
            self._on_model_change(name)

    def _toggle_translation(self, checked: bool) -> None:
        self._translation_enabled = checked
        self._tray.setToolTip(self._build_tooltip())
        if self._on_translation_toggle:
            self._on_translation_toggle(checked)

    def _select_translation_model(self, name: str) -> None:
        self._translation_model = name
        for action in self._trans_actions:
            action.setChecked(action.text() == name)
        if self._on_translation_model_change:
            self._on_translation_model_change(name)

    def _select_target_language(self, code: str) -> None:
        self._target_language = code
        for action in self._lang_actions:
            action.setChecked(f"({code})" in action.text())
        self._tray.setToolTip(self._build_tooltip())
        if self._on_target_language_change:
            self._on_target_language_change(code)

    def _open_config(self) -> None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_path = os.path.join(base_dir, "config.json")
        if sys.platform == "win32":
            os.startfile(config_path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", config_path], check=False)
        else:
            subprocess.run(["xdg-open", config_path], check=False)

    def _quit(self) -> None:
        if self._on_quit:
            self._on_quit()

    def show(self) -> None:
        """Show the system tray icon."""
        self._tray.show()

    def hide(self) -> None:
        """Hide the system tray icon."""
        self._tray.hide()
