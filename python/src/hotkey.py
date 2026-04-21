"""Global hotkey manager using Win32 RegisterHotKey.

Uses the Windows RegisterHotKey API for reliable global hotkey detection.
Falls back to pynput on non-Windows platforms.

Interaction model:
  - Invoke hotkey (e.g. Ctrl+Alt+H) -> starts recording, shows overlay
  - Enter -> stops recording, transcribes, pastes
  - Esc -> cancels recording, dismisses overlay

State machine: IDLE -> RECORDING -> PROCESSING -> IDLE
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable
from enum import Enum, auto

logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

try:
    from pynput import keyboard as pynput_keyboard

    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False


class AppState(Enum):
    """Application state machine states."""

    IDLE = auto()
    RECORDING = auto()
    PROCESSING = auto()
    TRANSLATING = auto()


def parse_hotkey(hotkey_str: str) -> set[str]:
    """Parse a hotkey string like 'ctrl+alt+h' into a set of key names."""
    return {k.strip().lower() for k in hotkey_str.split("+")}


# Win32 modifier flags
MOD_ALT = 0x0001
MOD_CTRL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000

_WIN32_MODIFIERS: dict[str, int] = {
    "ctrl": MOD_CTRL,
    "alt": MOD_ALT,
    "shift": MOD_SHIFT,
    "win": MOD_WIN,
}

_WIN32_VK: dict[str, int] = {
    **{chr(c): c for c in range(0x41, 0x5B)},  # A-Z
    **{str(d): 0x30 + d for d in range(10)},  # 0-9
    "space": 0x20,
    "enter": 0x0D,
    "esc": 0x1B,
    "tab": 0x09,
    **{f"f{i}": 0x70 + i - 1 for i in range(1, 13)},  # F1-F12
}

HOTKEY_ID_INVOKE = 1


class HotkeyManager:
    """Manages global hotkey with invoke/dismiss semantics.

    Uses Win32 RegisterHotKey on Windows for reliable detection.
    Invoke hotkey -> starts recording.
    While recording: Enter -> confirm, Esc -> cancel.
    """

    def __init__(
        self,
        hotkey_str: str = "ctrl+win+h",
        on_start: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        self._hotkey_str = hotkey_str
        self._hotkey_keys = parse_hotkey(hotkey_str)
        self._on_start = on_start
        self._on_stop = on_stop
        self._on_cancel = on_cancel
        self._state = AppState.IDLE
        self._lock = threading.Lock()
        self._debounce_time = 0.3
        self._last_trigger_time = 0.0
        self._running = False
        self._thread: threading.Thread | None = None
        self._pynput_listener: pynput_keyboard.Listener | None = None  # type: ignore[union-attr]

        modifiers = {"ctrl", "shift", "alt", "win", "cmd"}
        non_modifiers = self._hotkey_keys - modifiers
        self._trigger_key = non_modifiers.pop() if non_modifiers else "h"
        self._modifier_keys = self._hotkey_keys & modifiers

    @property
    def state(self) -> AppState:
        return self._state

    @state.setter
    def state(self, value: AppState) -> None:
        with self._lock:
            self._state = value

    def _get_win32_modifiers(self) -> int:
        mods = MOD_NOREPEAT
        for key in self._modifier_keys:
            if key in _WIN32_MODIFIERS:
                mods |= _WIN32_MODIFIERS[key]
        return mods

    def _get_win32_vk(self) -> int:
        return _WIN32_VK.get(self._trigger_key, ord(self._trigger_key.upper()))

    def _win32_hotkey_thread(self) -> None:
        """Register hotkey and pump Windows messages."""
        import ctypes.wintypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        mods = self._get_win32_modifiers()
        vk = self._get_win32_vk()

        if not user32.RegisterHotKey(None, HOTKEY_ID_INVOKE, mods, vk):
            logger.error(
                "Failed to register hotkey %s (may be in use by another app)",
                self._hotkey_str,
            )
            return

        logger.info("Hotkey registered: %s (Enter=send, Esc=cancel)", self._hotkey_str)

        msg = ctypes.wintypes.MSG()
        while self._running:
            if user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
                if msg.message == 0x0312:  # WM_HOTKEY
                    self._handle_invoke()
            else:
                time.sleep(0.05)

        user32.UnregisterHotKey(None, HOTKEY_ID_INVOKE)

    def _handle_invoke(self) -> None:
        now = time.monotonic()
        if now - self._last_trigger_time < self._debounce_time:
            return

        with self._lock:
            if self._state != AppState.IDLE:
                return
            self._state = AppState.RECORDING
            self._last_trigger_time = now

        logger.info("Hotkey invoked - recording started")
        if self._on_start:
            self._on_start()

    def _start_recording_keys(self) -> None:
        """Listen for Enter/Esc during recording via pynput."""
        if not HAS_PYNPUT:
            return

        def on_press(key: object) -> None:
            with self._lock:
                current_state = self._state

            if current_state != AppState.RECORDING:
                return

            if isinstance(key, pynput_keyboard.Key):
                if key == pynput_keyboard.Key.enter:
                    with self._lock:
                        self._state = AppState.PROCESSING
                    logger.info("Enter pressed - confirming")
                    if self._on_stop:
                        self._on_stop()
                elif key == pynput_keyboard.Key.esc:
                    with self._lock:
                        self._state = AppState.IDLE
                    logger.info("Esc pressed - cancelling")
                    if self._on_cancel:
                        self._on_cancel()

        self._pynput_listener = pynput_keyboard.Listener(on_press=on_press)
        self._pynput_listener.daemon = True
        self._pynput_listener.start()

    def start(self) -> None:
        """Start listening for the global hotkey."""
        self._running = True

        if IS_WINDOWS:
            self._thread = threading.Thread(target=self._win32_hotkey_thread, daemon=True)
            self._thread.start()
        else:
            logger.warning("Win32 hotkey not available, falling back to pynput")

        self._start_recording_keys()

    def stop(self) -> None:
        """Stop listening for the global hotkey."""
        self._running = False
        if self._pynput_listener is not None:
            self._pynput_listener.stop()
            self._pynput_listener = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
