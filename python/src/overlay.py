"""Floating overlay UI with waveform visualization.

Shows a frameless, always-on-top, non-activating overlay near the cursor
with real-time audio waveform during recording and animated status indicators.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.audio import AudioRingBuffer

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import (
        QColor,
        QCursor,
        QFont,
        QGuiApplication,
        QLinearGradient,
        QPainter,
        QPainterPath,
        QPen,
    )
    from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


class OverlayState:
    """Visual states for the overlay."""

    RECORDING = "recording"
    PROCESSING = "processing"
    TRANSLATING = "translating"
    DOWNLOADING = "downloading"
    ERROR = "error"
    DONE = "done"
    HIDDEN = "hidden"


if HAS_PYQT6:

    class WaveformWidget(QWidget):
        """Draws a smooth, amplified real-time audio waveform."""

        def __init__(
            self,
            ring_buffer: AudioRingBuffer | None = None,
            color: str = "#4CAF50",
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self._ring_buffer = ring_buffer
            self._color = QColor(color)
            self._frozen_data: np.ndarray | None = None
            self._gain = 8.0  # Amplification factor
            self.setMinimumHeight(60)

        def set_ring_buffer(self, ring_buffer: AudioRingBuffer) -> None:
            self._ring_buffer = ring_buffer

        def set_color(self, color: str) -> None:
            self._color = QColor(color)

        def freeze(self) -> None:
            if self._ring_buffer is not None:
                self._frozen_data = self._ring_buffer.snapshot(self.width())

        def unfreeze(self) -> None:
            self._frozen_data = None

        def _smooth(self, data: np.ndarray, window: int = 8) -> np.ndarray:
            """Smooth waveform with a moving average for a cleaner look."""
            if len(data) < window:
                return data
            kernel = np.ones(window) / window
            smoothed = np.convolve(data, kernel, mode="same")
            return smoothed.astype(np.float32)

        def paintEvent(self, event: object) -> None:  # noqa: N802
            if not HAS_PYQT6:
                return

            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            w = self.width()
            h = self.height()
            mid_y = h / 2.0

            # Get waveform data
            if self._frozen_data is not None:
                data = self._frozen_data.copy()
            elif self._ring_buffer is not None:
                data = self._ring_buffer.snapshot(w * 2)
            else:
                data = np.zeros(w, dtype=np.float32)

            # Downsample to widget width
            if len(data) > w:
                indices = np.linspace(0, len(data) - 1, w, dtype=int)
                data = data[indices]
            elif len(data) < w:
                data = np.pad(data, (0, w - len(data)))

            # Amplify and smooth
            data = np.clip(data * self._gain, -1.0, 1.0)
            data = self._smooth(data)

            # Draw filled waveform with gradient
            path = QPainterPath()
            path.moveTo(0, mid_y)
            for x in range(w):
                y = mid_y - data[x] * mid_y * 0.85
                path.lineTo(float(x), y)
            path.lineTo(float(w - 1), mid_y)
            path.closeSubpath()

            # Gradient fill
            gradient = QLinearGradient(0, 0, 0, h)
            fill_color = QColor(self._color)
            fill_color.setAlpha(80)
            gradient.setColorAt(0.0, fill_color)
            fill_color.setAlpha(20)
            gradient.setColorAt(1.0, fill_color)
            painter.fillPath(path, gradient)

            # Draw the line on top
            pen = QPen(self._color, 2.0)
            painter.setPen(pen)
            for x in range(w - 1):
                y1 = mid_y - data[x] * mid_y * 0.85
                y2 = mid_y - data[x + 1] * mid_y * 0.85
                painter.drawLine(int(x), int(y1), int(x + 1), int(y2))

            # Draw center line (subtle)
            pen = QPen(QColor(255, 255, 255, 30), 1.0)
            painter.setPen(pen)
            painter.drawLine(0, int(mid_y), w, int(mid_y))

            painter.end()

    class SpinnerWidget(QWidget):
        """Animated spinner for processing/downloading states."""

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._angle = 0.0
            self._color = QColor("#FF9800")
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._tick)
            self._timer.setInterval(50)
            self.setFixedHeight(6)

        def set_color(self, color: str) -> None:
            self._color = QColor(color)

        def start(self) -> None:
            self._angle = 0.0
            self._timer.start()

        def stop(self) -> None:
            self._timer.stop()

        def _tick(self) -> None:
            self._angle = (self._angle + 5) % 360
            self.update()

        def paintEvent(self, event: object) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            w = self.width()
            h = self.height()

            # Draw background track
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, 20))
            painter.drawRoundedRect(0, 0, w, h, 3, 3)

            # Draw animated progress sweep
            sweep_width = w // 3
            x_pos = (self._angle / 360.0) * (w + sweep_width) - sweep_width
            gradient = QLinearGradient(x_pos, 0, x_pos + sweep_width, 0)
            transparent = QColor(self._color)
            transparent.setAlpha(0)
            gradient.setColorAt(0.0, transparent)
            gradient.setColorAt(0.5, self._color)
            gradient.setColorAt(1.0, transparent)
            painter.setBrush(gradient)
            painter.drawRoundedRect(
                int(max(0, x_pos)),
                0,
                int(min(sweep_width, w - max(0, x_pos))),
                h,
                3,
                3,
            )

            painter.end()

    class OverlayWindow(QWidget):
        """Floating overlay window that appears near the cursor."""

        show_signal = pyqtSignal()
        hide_signal = pyqtSignal()
        set_state_signal = pyqtSignal(str, str)

        def __init__(
            self,
            width: int = 360,
            height: int = 140,
            waveform_color: str = "#4CAF50",
            processing_color: str = "#FF9800",
            translating_color: str = "#2196F3",
            opacity: float = 0.95,
        ) -> None:
            super().__init__()
            self._width = width
            self._height = height
            self._waveform_color = waveform_color
            self._processing_color = processing_color
            self._translating_color = translating_color
            self._state = OverlayState.HIDDEN

            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
                | Qt.WindowType.WindowDoesNotAcceptFocus
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
            self.setFixedSize(width, height)
            self.setWindowOpacity(opacity)

            # Layout
            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 12, 16, 12)
            layout.setSpacing(6)

            # Title label
            self._title_label = QLabel("UniversalTranslator")
            self._title_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            self._title_label.setStyleSheet(
                "color: rgba(255,255,255,0.5); background: transparent;"
            )
            layout.addWidget(self._title_label)

            # Waveform
            self._waveform = WaveformWidget(color=waveform_color)
            layout.addWidget(self._waveform)

            # Spinner (hidden by default)
            self._spinner = SpinnerWidget()
            self._spinner.hide()
            layout.addWidget(self._spinner)

            # Status label
            self._status_label = QLabel("Ready")
            self._status_label.setFont(QFont("Segoe UI", 10))
            self._status_label.setStyleSheet("color: white; background: transparent;")
            self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self._status_label)

            # Detail label (for extra info like "Enter=send, Esc=cancel")
            self._detail_label = QLabel("")
            self._detail_label.setFont(QFont("Segoe UI", 8))
            self._detail_label.setStyleSheet(
                "color: rgba(255,255,255,0.4); background: transparent;"
            )
            self._detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self._detail_label)

            # Translation label (hidden by default)
            self._translation_label = QLabel("")
            self._translation_label.setFont(QFont("Segoe UI", 10))
            self._translation_label.setStyleSheet("color: #90CAF9; background: transparent;")
            self._translation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._translation_label.setWordWrap(True)
            self._translation_label.hide()
            layout.addWidget(self._translation_label)

            # Refresh timer for waveform (~30fps)
            self._refresh_timer = QTimer(self)
            self._refresh_timer.timeout.connect(self._refresh)
            self._refresh_timer.setInterval(33)

            # Done timer
            self._done_timer = QTimer(self)
            self._done_timer.setSingleShot(True)
            self._done_timer.timeout.connect(self._auto_hide)

            # Connect signals
            self.show_signal.connect(self._do_show)
            self.hide_signal.connect(self._do_hide)
            self.set_state_signal.connect(self._do_set_state)

        def set_ring_buffer(self, ring_buffer: AudioRingBuffer) -> None:
            self._waveform.set_ring_buffer(ring_buffer)

        def _position_near_cursor(self) -> None:
            cursor_pos = QCursor.pos()
            screen = QGuiApplication.screenAt(cursor_pos)
            if screen is None:
                screen = QGuiApplication.primaryScreen()

            if screen is None:
                self.move(cursor_pos.x() + 20, cursor_pos.y() + 20)
                return

            screen_geo = screen.availableGeometry()
            x = cursor_pos.x() + 20
            y = cursor_pos.y() + 20

            if x + self._width > screen_geo.right():
                x = cursor_pos.x() - self._width - 20
            if y + self._height > screen_geo.bottom():
                y = cursor_pos.y() - self._height - 20

            x = max(x, screen_geo.left())
            y = max(y, screen_geo.top())
            self.move(x, y)

        def _do_show(self) -> None:
            self._position_near_cursor()
            self.show()
            self._refresh_timer.start()

        def _do_hide(self) -> None:
            self._refresh_timer.stop()
            self._done_timer.stop()
            self._spinner.stop()
            self._spinner.hide()
            self._waveform.unfreeze()
            self._translation_label.hide()
            self.hide()
            self._state = OverlayState.HIDDEN

        def _do_set_state(self, state: str, extra_text: str) -> None:
            self._state = state

            if state == OverlayState.RECORDING:
                self._waveform.set_color(self._waveform_color)
                self._waveform.unfreeze()
                self._waveform.show()
                self._spinner.stop()
                self._spinner.hide()
                self._status_label.setText("🎤 Recording...")
                self._status_label.setStyleSheet("color: #4CAF50; background: transparent;")
                self._detail_label.setText("Enter = send  •  Esc = cancel")
                self._detail_label.show()
                self._translation_label.hide()

            elif state == OverlayState.PROCESSING:
                self._waveform.freeze()
                self._spinner.set_color(self._processing_color)
                self._spinner.show()
                self._spinner.start()
                suffix = f" using {extra_text}" if extra_text else ""
                self._status_label.setText(f"⏳ Processing{suffix}...")
                self._status_label.setStyleSheet("color: #FF9800; background: transparent;")
                self._detail_label.setText("Transcribing your speech")
                self._detail_label.show()

            elif state == OverlayState.TRANSLATING:
                self._spinner.set_color(self._translating_color)
                self._spinner.show()
                self._spinner.start()
                suffix = f" → {extra_text}" if extra_text else ""
                self._status_label.setText(f"🌐 Translating{suffix}...")
                self._status_label.setStyleSheet("color: #2196F3; background: transparent;")
                self._detail_label.setText("Converting to target language")
                self._detail_label.show()
                self._translation_label.show()

            elif state == OverlayState.DOWNLOADING:
                self._waveform.hide()
                self._spinner.set_color(self._processing_color)
                self._spinner.show()
                self._spinner.start()
                self._status_label.setText(extra_text or "⬇️ Downloading model...")
                self._status_label.setStyleSheet("color: #FF9800; background: transparent;")
                self._detail_label.setText("This is a one-time setup")
                self._detail_label.show()

            elif state == OverlayState.DONE:
                self._spinner.stop()
                self._spinner.hide()
                self._status_label.setText("✅ Done — pasted!")
                self._status_label.setStyleSheet("color: #4CAF50; background: transparent;")
                self._detail_label.hide()
                self._done_timer.start(1200)

            elif state == OverlayState.ERROR:
                self._waveform.freeze()
                self._spinner.stop()
                self._spinner.hide()
                self._status_label.setText(extra_text or "❌ Error")
                self._status_label.setStyleSheet("color: #F44336; background: transparent;")
                self._detail_label.setText("Check system tray for details")
                self._detail_label.show()
                self._done_timer.start(4000)

        def _auto_hide(self) -> None:
            self.hide_signal.emit()

        def _refresh(self) -> None:
            if self._state == OverlayState.RECORDING:
                self._waveform.update()

        def update_translation_text(self, text: str) -> None:
            self._translation_label.setText(text)
            self._translation_label.show()

        def paintEvent(self, event: object) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Background with subtle border
            painter.setBrush(QColor(32, 32, 32, 240))
            painter.setPen(QPen(QColor(255, 255, 255, 25), 1.0))
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 10, 10)
            painter.end()

else:
    # Stubs when PyQt6 is not available
    class WaveformWidget:  # type: ignore[no-redef]
        pass

    class OverlayWindow:  # type: ignore[no-redef]
        pass
