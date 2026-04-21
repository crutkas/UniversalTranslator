"""Paste output module.

Copies transcribed text to clipboard and simulates Ctrl+V to paste
into the currently focused application. Captures the foreground window
handle at recording start to verify focus before pasting.
"""

from __future__ import annotations

import ctypes
import logging
import time

import pyperclip

logger = logging.getLogger(__name__)

try:
    from pynput.keyboard import Controller, Key

    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False

# Win32 API for tracking foreground window
try:
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    HAS_WIN32 = True
except (AttributeError, OSError):
    HAS_WIN32 = False


def get_foreground_window() -> int:
    """Get the handle of the currently focused window."""
    if HAS_WIN32:
        return int(user32.GetForegroundWindow())
    return 0


def set_foreground_window(hwnd: int) -> bool:
    """Attempt to restore focus to a specific window."""
    if HAS_WIN32 and hwnd:
        return bool(user32.SetForegroundWindow(hwnd))
    return False


class PasteManager:
    """Manages clipboard operations and paste injection."""

    def __init__(self) -> None:
        self._keyboard = Controller() if HAS_PYNPUT else None
        self._target_hwnd: int = 0

    def capture_target_window(self) -> None:
        """Capture the foreground window handle at recording start."""
        self._target_hwnd = get_foreground_window()
        logger.debug("Captured target window: %s", self._target_hwnd)

    def paste_text(self, text: str, force_paste: bool = True) -> bool:
        """Copy text to clipboard and optionally paste via Ctrl+V.

        Args:
            text: Text to paste.
            force_paste: If True, simulate Ctrl+V. If False, clipboard-only.

        Returns:
            True if paste was attempted, False if clipboard-only.
        """
        if not text:
            return False

        # Set clipboard
        pyperclip.copy(text)

        if not force_paste or self._keyboard is None:
            logger.info("Text copied to clipboard (no paste)")
            return False

        # Verify target window is still focused
        current_hwnd = get_foreground_window()
        if self._target_hwnd and current_hwnd != self._target_hwnd:
            logger.warning(
                "Focus changed since recording started (was %s, now %s). "
                "Attempting to restore focus.",
                self._target_hwnd,
                current_hwnd,
            )
            if not set_foreground_window(self._target_hwnd):
                logger.warning("Could not restore focus. Text is in clipboard.")
                return False
            time.sleep(0.1)

        # Small delay for clipboard to be ready
        time.sleep(0.05)

        # Simulate Ctrl+V
        try:
            self._keyboard.press(Key.ctrl)
            self._keyboard.press("v")
            self._keyboard.release("v")
            self._keyboard.release(Key.ctrl)
            logger.info("Text pasted via Ctrl+V")
            return True
        except Exception as e:
            logger.error("Failed to simulate paste: %s", e)
            return False
