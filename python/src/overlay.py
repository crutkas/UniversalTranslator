"""Floating overlay UI with Windows Fluent Design styling.

Uses native PyQt6 with Fluent Design-inspired stylesheets:
- Segoe UI Variable font
- WinUI 3 color tokens (dark theme)
- Rounded corners, acrylic-style backdrop
- Indeterminate progress animation
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
    from PyQt6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


# WinUI 3 dark theme color tokens
FLUENT_ACCENT = "#60CDFF"
FLUENT_WARN = "#FCE100"
FLUENT_ERROR = "#FF99A4"
FLUENT_SUCCESS = "#6CCB5F"
FLUENT_TEXT = "#FFFFFF"
FLUENT_TEXT_DIM = "rgba(255,255,255,0.6)"
FLUENT_TEXT_SUBTLE = "rgba(255,255,255,0.36)"
FLUENT_SURFACE = (44, 44, 44, 235)
FLUENT_BORDER = (255, 255, 255, 12)
FLUENT_FONT = "Segoe UI Variable, Segoe UI, sans-serif"

# Fluent progress bar stylesheet
PROGRESS_STYLE = """
QProgressBar {
    background: rgba(255,255,255,0.06);
    border: none;
    border-radius: 2px;
    max-height: 3px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 transparent, stop:0.5 #60CDFF, stop:1 transparent);
    border-radius: 2px;
}
"""


class OverlayState:
    RECORDING = "recording"
    PROCESSING = "processing"
    TRANSLATING = "translating"
    DOWNLOADING = "downloading"
    ERROR = "error"
    DONE = "done"
    HIDDEN = "hidden"


if HAS_PYQT6:

    class WaveformWidget(QWidget):
        """Smooth, amplified audio waveform with Fluent accent color."""

        def __init__(
            self,
            ring_buffer: AudioRingBuffer | None = None,
            color: str = FLUENT_ACCENT,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self._ring_buffer = ring_buffer
            self._color = QColor(color)
            self._frozen_data: np.ndarray | None = None
            self._gain = 8.0
            self.setMinimumHeight(56)

        def set_ring_buffer(self, rb: AudioRingBuffer) -> None:
            self._ring_buffer = rb

        def set_color(self, color: str) -> None:
            self._color = QColor(color)

        def freeze(self) -> None:
            if self._ring_buffer is not None:
                self._frozen_data = self._ring_buffer.snapshot(self.width())

        def unfreeze(self) -> None:
            self._frozen_data = None

        def _smooth(self, data: np.ndarray, window: int = 8) -> np.ndarray:
            if len(data) < window:
                return data
            kernel = np.ones(window) / window
            return np.convolve(data, kernel, mode="same").astype(np.float32)

        def paintEvent(self, event: object) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            w, h = self.width(), self.height()
            mid = h / 2.0

            if self._frozen_data is not None:
                data = self._frozen_data.copy()
            elif self._ring_buffer is not None:
                data = self._ring_buffer.snapshot(w * 2)
            else:
                data = np.zeros(w, dtype=np.float32)

            if len(data) > w:
                data = data[np.linspace(0, len(data) - 1, w, dtype=int)]
            elif len(data) < w:
                data = np.pad(data, (0, w - len(data)))

            data = np.clip(data * self._gain, -1.0, 1.0)
            data = self._smooth(data)

            # Gradient fill
            path = QPainterPath()
            path.moveTo(0, mid)
            for x in range(w):
                path.lineTo(float(x), mid - data[x] * mid * 0.85)
            path.lineTo(float(w - 1), mid)
            path.closeSubpath()

            grad = QLinearGradient(0, 0, 0, h)
            c = QColor(self._color)
            c.setAlpha(50)
            grad.setColorAt(0.0, c)
            c.setAlpha(5)
            grad.setColorAt(1.0, c)
            painter.fillPath(path, grad)

            # Line
            painter.setPen(QPen(self._color, 2.0))
            for x in range(w - 1):
                y1 = mid - data[x] * mid * 0.85
                y2 = mid - data[x + 1] * mid * 0.85
                painter.drawLine(int(x), int(y1), int(x + 1), int(y2))

            # Center ref
            painter.setPen(QPen(QColor(255, 255, 255, 15), 1.0))
            painter.drawLine(0, int(mid), w, int(mid))
            painter.end()

    class OverlayWindow(QWidget):
        """Fluent Design overlay — styled with native Qt, no external deps."""

        show_signal = pyqtSignal()
        hide_signal = pyqtSignal()
        set_state_signal = pyqtSignal(str, str)

        def __init__(
            self,
            width: int = 380,
            height: int = 160,
            waveform_color: str = FLUENT_ACCENT,
            processing_color: str = FLUENT_WARN,
            translating_color: str = FLUENT_ACCENT,
            opacity: float = 0.96,
        ) -> None:
            super().__init__()
            self._width = width
            self._height = height
            self._state = OverlayState.HIDDEN
            self._bg_surface = QColor(*FLUENT_SURFACE)
            self._bg_border = QColor(*FLUENT_BORDER)

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

            layout = QVBoxLayout(self)
            layout.setContentsMargins(20, 14, 20, 14)
            layout.setSpacing(2)

            # Header row
            header = QHBoxLayout()
            header.setSpacing(8)
            self._title = QLabel("UNIVERSALTRANSLATOR")
            self._title.setFont(QFont(FLUENT_FONT, 8))
            self._title.setStyleSheet(
                f"color: {FLUENT_TEXT_SUBTLE}; background: transparent; letter-spacing: 1.5px;"
            )
            header.addWidget(self._title)
            header.addStretch()
            self._badge = QLabel("")
            self._badge.setFont(QFont(FLUENT_FONT, 8))
            self._badge.setStyleSheet(
                f"color: {FLUENT_ACCENT}; background: rgba(96,205,255,0.08); "
                "border-radius: 3px; padding: 1px 8px;"
            )
            header.addWidget(self._badge)
            layout.addLayout(header)

            layout.addSpacing(4)

            # Waveform
            self._waveform = WaveformWidget(color=waveform_color)
            layout.addWidget(self._waveform)

            # Progress bar
            self._progress = QProgressBar()
            self._progress.setTextVisible(False)
            self._progress.setRange(0, 0)  # Indeterminate
            self._progress.setFixedHeight(3)
            self._progress.setStyleSheet(PROGRESS_STYLE)
            self._progress.hide()
            layout.addWidget(self._progress)

            layout.addSpacing(4)

            # Status
            self._status = QLabel("Ready")
            self._status.setFont(QFont(FLUENT_FONT, 11))
            self._status.setStyleSheet(f"color: {FLUENT_TEXT}; background: transparent;")
            self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self._status)

            # Detail
            self._detail = QLabel("")
            self._detail.setFont(QFont(FLUENT_FONT, 9))
            self._detail.setStyleSheet(f"color: {FLUENT_TEXT_SUBTLE}; background: transparent;")
            self._detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self._detail)

            # Translation preview
            self._translation = QLabel("")
            self._translation.setFont(QFont(FLUENT_FONT, 10))
            self._translation.setStyleSheet(f"color: {FLUENT_ACCENT}; background: transparent;")
            self._translation.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._translation.setWordWrap(True)
            self._translation.hide()
            layout.addWidget(self._translation)

            # Timers
            self._refresh_timer = QTimer(self)
            self._refresh_timer.timeout.connect(self._refresh)
            self._refresh_timer.setInterval(33)

            self._done_timer = QTimer(self)
            self._done_timer.setSingleShot(True)
            self._done_timer.timeout.connect(self._auto_hide)

            self.show_signal.connect(self._do_show)
            self.hide_signal.connect(self._do_hide)
            self.set_state_signal.connect(self._do_set_state)

        def set_ring_buffer(self, rb: AudioRingBuffer) -> None:
            self._waveform.set_ring_buffer(rb)

        def _position_near_cursor(self) -> None:
            pos = QCursor.pos()
            screen = QGuiApplication.screenAt(pos) or QGuiApplication.primaryScreen()
            if screen is None:
                self.move(pos.x() + 20, pos.y() + 20)
                return
            geo = screen.availableGeometry()
            x = pos.x() + 20
            y = pos.y() + 20
            if x + self._width > geo.right():
                x = pos.x() - self._width - 20
            if y + self._height > geo.bottom():
                y = pos.y() - self._height - 20
            self.move(max(x, geo.left()), max(y, geo.top()))

        def _do_show(self) -> None:
            self._position_near_cursor()
            self.show()
            self._refresh_timer.start()

        def _do_hide(self) -> None:
            self._refresh_timer.stop()
            self._done_timer.stop()
            self._progress.hide()
            self._waveform.unfreeze()
            self._waveform.show()
            self._translation.hide()
            self.hide()
            self._state = OverlayState.HIDDEN

        def _do_set_state(self, state: str, extra: str) -> None:
            self._state = state

            if state == OverlayState.RECORDING:
                self._waveform.unfreeze()
                self._waveform.show()
                self._progress.hide()
                self._badge.setText(extra or "")
                self._status.setText("Recording")
                self._status.setStyleSheet(f"color: {FLUENT_ACCENT}; background: transparent;")
                self._detail.setText("Enter to send  ·  Esc to cancel")
                self._detail.show()
                self._translation.hide()

            elif state == OverlayState.PROCESSING:
                self._waveform.hide()
                self._progress.show()
                self._status.setText("Processing...")
                self._status.setStyleSheet(f"color: {FLUENT_WARN}; background: transparent;")
                self._detail.setText(f"Transcribing with {extra}" if extra else "Transcribing")
                self._detail.show()

            elif state == OverlayState.TRANSLATING:
                self._waveform.hide()
                self._progress.show()
                self._status.setText(f"Translating → {extra}" if extra else "Translating...")
                self._status.setStyleSheet(f"color: {FLUENT_ACCENT}; background: transparent;")
                self._detail.setText("Converting to target language")
                self._detail.show()
                self._translation.show()

            elif state == OverlayState.DOWNLOADING:
                self._waveform.hide()
                self._progress.show()
                self._status.setText(extra or "Downloading model...")
                self._status.setStyleSheet(f"color: {FLUENT_WARN}; background: transparent;")
                self._detail.setText("One-time setup — please wait")
                self._detail.show()

            elif state == OverlayState.DONE:
                self._waveform.hide()
                self._progress.hide()
                self._status.setText("✓ Pasted")
                self._status.setStyleSheet(f"color: {FLUENT_SUCCESS}; background: transparent;")
                self._detail.hide()
                self._done_timer.start(1200)

            elif state == OverlayState.ERROR:
                self._waveform.hide()
                self._progress.hide()
                self._status.setText(extra or "Error")
                self._status.setStyleSheet(f"color: {FLUENT_ERROR}; background: transparent;")
                self._detail.setText("Check system tray for details")
                self._detail.show()
                self._done_timer.start(4000)

        def _auto_hide(self) -> None:
            self.hide_signal.emit()

        def _refresh(self) -> None:
            if self._state == OverlayState.RECORDING:
                self._waveform.update()

        def update_translation_text(self, text: str) -> None:
            self._translation.setText(text)
            self._translation.show()

        def paintEvent(self, event: object) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(self._bg_surface)
            painter.setPen(QPen(self._bg_border, 1.0))
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 8, 8)
            painter.end()

else:

    class WaveformWidget:  # type: ignore[no-redef]
        pass

    class OverlayWindow:  # type: ignore[no-redef]
        pass
