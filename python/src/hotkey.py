"""Global hotkey manager with state machine.

Uses pynput for global keyboard listening. Communicates with Qt via
thread-safe signals to avoid cross-thread UI access.

State machine: IDLE -> RECORDING -> PROCESSING -> IDLE
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from enum import Enum, auto

logger = logging.getLogger(__name__)

try:
    from pynput import keyboard

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
    """Parse a hotkey string like 'ctrl+shift+space' into a set of key names."""
    return {k.strip().lower() for k in hotkey_str.split("+")}


# Map from key names to pynput Key objects
_SPECIAL_KEYS: dict[str, keyboard.Key] = {}
if HAS_PYNPUT:
    _SPECIAL_KEYS = {
        "ctrl": keyboard.Key.ctrl_l,
        "ctrl_l": keyboard.Key.ctrl_l,
        "ctrl_r": keyboard.Key.ctrl_r,
        "shift": keyboard.Key.shift_l,
        "shift_l": keyboard.Key.shift_l,
        "shift_r": keyboard.Key.shift_r,
        "alt": keyboard.Key.alt_l,
        "alt_l": keyboard.Key.alt_l,
        "alt_r": keyboard.Key.alt_r,
        "space": keyboard.Key.space,
        "tab": keyboard.Key.tab,
        "enter": keyboard.Key.enter,
        "esc": keyboard.Key.esc,
    }


def _key_to_name(key: keyboard.Key | keyboard.KeyCode) -> str | None:  # type: ignore[name-defined]
    """Convert a pynput key to a lowercase name string."""
    if not HAS_PYNPUT:
        return None

    if isinstance(key, keyboard.Key):
        name = key.name.lower()
        # Normalize left/right variants
        if name.startswith("ctrl"):
            return "ctrl"
        if name.startswith("shift"):
            return "shift"
        if name.startswith("alt"):
            return "alt"
        return str(name)
    elif isinstance(key, keyboard.KeyCode):
        if key.char:
            return str(key.char.lower())
        if key.vk:
            return f"vk_{key.vk}"
    return None


class HotkeyManager:
    """Manages global hotkey detection with push-to-talk semantics.

    The trigger key is the last key in the hotkey combo (e.g., 'space' in
    'ctrl+shift+space'). Modifier keys must be held for the hotkey to activate.
    Recording starts on trigger key press and stops on trigger key release.
    """

    def __init__(
        self,
        hotkey_str: str = "ctrl+shift+space",
        on_start: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
    ) -> None:
        self._hotkey_keys = parse_hotkey(hotkey_str)
        self._on_start = on_start
        self._on_stop = on_stop
        self._pressed_keys: set[str] = set()
        self._state = AppState.IDLE
        self._listener: keyboard.Listener | None = None  # type: ignore[assignment]
        self._debounce_time = 0.2  # seconds
        self._last_trigger_time = 0.0
        self._lock = threading.Lock()

        # The trigger key is the non-modifier key in the combo
        modifiers = {
            "ctrl",
            "shift",
            "alt",
            "ctrl_l",
            "ctrl_r",
            "shift_l",
            "shift_r",
            "alt_l",
            "alt_r",
        }
        non_modifiers = self._hotkey_keys - modifiers
        self._trigger_key = non_modifiers.pop() if non_modifiers else "space"

    @property
    def state(self) -> AppState:
        return self._state

    @state.setter
    def state(self, value: AppState) -> None:
        with self._lock:
            self._state = value

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:  # type: ignore[name-defined]
        name = _key_to_name(key)
        if name is None:
            return

        self._pressed_keys.add(name)

        # Check if all hotkey keys are pressed and trigger key was just pressed
        if name == self._trigger_key and self._hotkey_keys.issubset(self._pressed_keys):
            now = time.monotonic()
            if now - self._last_trigger_time < self._debounce_time:
                return

            with self._lock:
                if self._state != AppState.IDLE:
                    return
                self._state = AppState.RECORDING
                self._last_trigger_time = now

            if self._on_start:
                self._on_start()

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:  # type: ignore[name-defined]
        name = _key_to_name(key)
        if name is None:
            return

        # Only stop on trigger key release while recording
        if name == self._trigger_key:
            with self._lock:
                if self._state != AppState.RECORDING:
                    self._pressed_keys.discard(name)
                    return
                self._state = AppState.PROCESSING

            if self._on_stop:
                self._on_stop()

        self._pressed_keys.discard(name)

    def start(self) -> None:
        """Start listening for the global hotkey."""
        if not HAS_PYNPUT:
            raise RuntimeError("pynput is not installed")

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()
        logger.info("Hotkey listener started: %s", "+".join(sorted(self._hotkey_keys)))

    def stop(self) -> None:
        """Stop listening for the global hotkey."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
