"""Floating overlay UI using Windows Fluent Design.

Uses QFluentWidgets for native Windows look and feel with Mica/Acrylic backdrop,
Fluent typography, and animated progress indicators.
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
        QGuiApplication,
        QLinearGradient,
        QPainter,
        QPainterPath,
        QPen,
    )
    from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

try:
    from qfluentwidgets import (
        BodyLabel,
        CaptionLabel,
        IndeterminateProgressBar,
        Theme,
        setTheme,
    )

    HAS_FLUENT = True
except ImportError:
    HAS_FLUENT = False


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
            color: str = "#60CDFF",
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self._ring_buffer = ring_buffer
            self._color = QColor(color)
            self._frozen_data: np.ndarray | None = None
            self._gain = 8.0
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
            if len(data) < window:
                return data
            kernel = np.ones(window) / window
            return np.convolve(data, kernel, mode="same").astype(np.float32)

        def paintEvent(self, event: object) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            w = self.width()
            h = self.height()
            mid_y = h / 2.0

            if self._frozen_data is not None:
                data = self._frozen_data.copy()
            elif self._ring_buffer is not None:
                data = self._ring_buffer.snapshot(w * 2)
            else:
                data = np.zeros(w, dtype=np.float32)

            if len(data) > w:
                indices = np.linspace(0, len(data) - 1, w, dtype=int)
                data = data[indices]
            elif len(data) < w:
                data = np.pad(data, (0, w - len(data)))

            data = np.clip(data * self._gain, -1.0, 1.0)
            data = self._smooth(data)

            # Filled area under curve
            path = QPainterPath()
            path.moveTo(0, mid_y)
            for x in range(w):
                y = mid_y - data[x] * mid_y * 0.85
                path.lineTo(float(x), y)
            path.lineTo(float(w - 1), mid_y)
            path.closeSubpath()

            gradient = QLinearGradient(0, 0, 0, h)
            fill_color = QColor(self._color)
            fill_color.setAlpha(60)
            gradient.setColorAt(0.0, fill_color)
            fill_color.setAlpha(10)
            gradient.setColorAt(1.0, fill_color)
            painter.fillPath(path, gradient)

            # Line
            pen = QPen(self._color, 2.0)
            painter.setPen(pen)
            for x in range(w - 1):
                y1 = mid_y - data[x] * mid_y * 0.85
                y2 = mid_y - data[x + 1] * mid_y * 0.85
                painter.drawLine(int(x), int(y1), int(x + 1), int(y2))

            # Center reference line
            pen = QPen(QColor(255, 255, 255, 20), 1.0)
            painter.setPen(pen)
            painter.drawLine(0, int(mid_y), w, int(mid_y))

            painter.end()

    class OverlayWindow(QWidget):
        """Fluent Design floating overlay."""

        show_signal = pyqtSignal()
        hide_signal = pyqtSignal()
        set_state_signal = pyqtSignal(str, str)

        # Fluent Design colors (dark theme)
        ACCENT = "#60CDFF"
        ACCENT_WARN = "#FCE100"
        ACCENT_ERROR = "#FF99A4"
        ACCENT_SUCCESS = "#6CCB5F"
        TEXT_PRIMARY = "#FFFFFF"
        TEXT_SECONDARY = "rgba(255,255,255,0.6)"
        TEXT_TERTIARY = "rgba(255,255,255,0.4)"

        def __init__(
            self,
            width: int = 380,
            height: int = 160,
            waveform_color: str = "#60CDFF",
            processing_color: str = "#FCE100",
            translating_color: str = "#60CDFF",
            opacity: float = 0.96,
        ) -> None:
            super().__init__()
            self._bg_surface = QColor(44, 44, 44, 240)
            self._bg_border = QColor(255, 255, 255, 15)
            self._width = width
            self._height = height
            self._waveform_color = waveform_color
            self._processing_color = processing_color
            self._translating_color = translating_color
            self._state = OverlayState.HIDDEN

            if HAS_FLUENT:
                setTheme(Theme.DARK)

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

            # Main layout
            layout = QVBoxLayout(self)
            layout.setContentsMargins(20, 16, 20, 16)
            layout.setSpacing(4)

            # Title row
            title_row = QHBoxLayout()
            title_row.setSpacing(8)
            if HAS_FLUENT:
                self._title_label = CaptionLabel("UNIVERSALTRANSLATOR")
            else:
                self._title_label = QWidget()  # type: ignore[assignment]
            self._title_label.setStyleSheet(  # type: ignore[union-attr]
                f"color: {self.TEXT_TERTIARY}; background: transparent; letter-spacing: 1px;"
            )
            title_row.addWidget(self._title_label)  # type: ignore[arg-type]
            title_row.addStretch()

            # Model badge
            if HAS_FLUENT:
                self._model_badge = CaptionLabel("")
            else:
                self._model_badge = QWidget()  # type: ignore[assignment]
            self._model_badge.setStyleSheet(  # type: ignore[union-attr]
                f"color: {self.ACCENT}; background: rgba(96,205,255,0.1); "
                "border-radius: 4px; padding: 2px 8px;"
            )
            title_row.addWidget(self._model_badge)  # type: ignore[arg-type]
            layout.addLayout(title_row)

            layout.addSpacing(4)

            # Waveform
            self._waveform = WaveformWidget(color=waveform_color)
            layout.addWidget(self._waveform)

            # Progress bar (Fluent indeterminate)
            if HAS_FLUENT:
                self._progress = IndeterminateProgressBar(self)
            else:
                self._progress = QWidget()  # type: ignore[assignment]
            self._progress.setFixedHeight(4)  # type: ignore[union-attr]
            self._progress.hide()  # type: ignore[union-attr]
            layout.addWidget(self._progress)  # type: ignore[arg-type]

            # Status label
            if HAS_FLUENT:
                self._status_label = BodyLabel("Ready")
            else:
                self._status_label = QWidget()  # type: ignore[assignment]
            self._status_label.setStyleSheet(  # type: ignore[union-attr]
                f"color: {self.TEXT_PRIMARY}; background: transparent;"
            )
            self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # type: ignore[union-attr]
            layout.addWidget(self._status_label)  # type: ignore[arg-type]

            # Detail label
            if HAS_FLUENT:
                self._detail_label = CaptionLabel("")
            else:
                self._detail_label = QWidget()  # type: ignore[assignment]
            self._detail_label.setStyleSheet(  # type: ignore[union-attr]
                f"color: {self.TEXT_TERTIARY}; background: transparent;"
            )
            self._detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # type: ignore[union-attr]
            layout.addWidget(self._detail_label)  # type: ignore[arg-type]

            # Translation label
            if HAS_FLUENT:
                self._translation_label = BodyLabel("")
            else:
                self._translation_label = QWidget()  # type: ignore[assignment]
            self._translation_label.setStyleSheet(  # type: ignore[union-attr]
                f"color: {self.ACCENT}; background: transparent;"
            )
            self._translation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # type: ignore[union-attr]
            self._translation_label.setWordWrap(True)  # type: ignore[union-attr]
            self._translation_label.hide()  # type: ignore[union-attr]
            layout.addWidget(self._translation_label)  # type: ignore[arg-type]

            # Waveform refresh timer (~30fps)
            self._refresh_timer = QTimer(self)
            self._refresh_timer.timeout.connect(self._refresh)
            self._refresh_timer.setInterval(33)

            # Auto-hide timer
            self._done_timer = QTimer(self)
            self._done_timer.setSingleShot(True)
            self._done_timer.timeout.connect(self._auto_hide)

            # Signals
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

            geo = screen.availableGeometry()
            x = cursor_pos.x() + 20
            y = cursor_pos.y() + 20
            if x + self._width > geo.right():
                x = cursor_pos.x() - self._width - 20
            if y + self._height > geo.bottom():
                y = cursor_pos.y() - self._height - 20
            self.move(max(x, geo.left()), max(y, geo.top()))

        def _do_show(self) -> None:
            self._position_near_cursor()
            self.show()
            self._refresh_timer.start()

        def _do_hide(self) -> None:
            self._refresh_timer.stop()
            self._done_timer.stop()
            self._progress.hide()  # type: ignore[union-attr]
            self._waveform.unfreeze()
            self._waveform.show()
            self._translation_label.hide()  # type: ignore[union-attr]
            self.hide()
            self._state = OverlayState.HIDDEN

        def _do_set_state(self, state: str, extra_text: str) -> None:
            self._state = state

            if state == OverlayState.RECORDING:
                self._waveform.set_color(self._waveform_color)
                self._waveform.unfreeze()
                self._waveform.show()
                self._progress.hide()  # type: ignore[union-attr]
                self._status_label.setText("Recording")  # type: ignore[union-attr]
                self._status_label.setStyleSheet(  # type: ignore[union-attr]
                    f"color: {self.ACCENT}; background: transparent;"
                )
                self._model_badge.setText(extra_text if extra_text else "")  # type: ignore[union-attr]
                self._detail_label.setText("Enter to send  ·  Esc to cancel")  # type: ignore[union-attr]
                self._detail_label.show()  # type: ignore[union-attr]
                self._translation_label.hide()  # type: ignore[union-attr]

            elif state == OverlayState.PROCESSING:
                self._waveform.hide()
                self._progress.show()  # type: ignore[union-attr]
                if HAS_FLUENT:
                    self._progress.start()  # type: ignore[union-attr]
                self._status_label.setText("Processing...")  # type: ignore[union-attr]
                self._status_label.setStyleSheet(  # type: ignore[union-attr]
                    f"color: {self.ACCENT_WARN}; background: transparent;"
                )
                self._detail_label.setText(  # type: ignore[union-attr]
                    f"Transcribing with {extra_text}" if extra_text else "Transcribing"
                )

            elif state == OverlayState.TRANSLATING:
                self._waveform.hide()
                self._progress.show()  # type: ignore[union-attr]
                if HAS_FLUENT:
                    self._progress.start()  # type: ignore[union-attr]
                self._status_label.setText(  # type: ignore[union-attr]
                    f"Translating → {extra_text}" if extra_text else "Translating..."
                )
                self._status_label.setStyleSheet(  # type: ignore[union-attr]
                    f"color: {self.ACCENT}; background: transparent;"
                )
                self._detail_label.setText("Converting to target language")  # type: ignore[union-attr]
                self._translation_label.show()  # type: ignore[union-attr]

            elif state == OverlayState.DOWNLOADING:
                self._waveform.hide()
                self._progress.show()  # type: ignore[union-attr]
                if HAS_FLUENT:
                    self._progress.start()  # type: ignore[union-attr]
                self._status_label.setText(extra_text or "Downloading model...")  # type: ignore[union-attr]
                self._status_label.setStyleSheet(  # type: ignore[union-attr]
                    f"color: {self.ACCENT_WARN}; background: transparent;"
                )
                self._detail_label.setText("One-time setup — please wait")  # type: ignore[union-attr]
                self._detail_label.show()  # type: ignore[union-attr]

            elif state == OverlayState.DONE:
                self._waveform.hide()
                self._progress.hide()  # type: ignore[union-attr]
                if HAS_FLUENT:
                    self._progress.stop()  # type: ignore[union-attr]
                self._status_label.setText("✓ Pasted")  # type: ignore[union-attr]
                self._status_label.setStyleSheet(  # type: ignore[union-attr]
                    f"color: {self.ACCENT_SUCCESS}; background: transparent;"
                )
                self._detail_label.hide()  # type: ignore[union-attr]
                self._done_timer.start(1200)

            elif state == OverlayState.ERROR:
                self._waveform.hide()
                self._progress.hide()  # type: ignore[union-attr]
                if HAS_FLUENT:
                    self._progress.stop()  # type: ignore[union-attr]
                self._status_label.setText(extra_text or "Error")  # type: ignore[union-attr]
                self._status_label.setStyleSheet(  # type: ignore[union-attr]
                    f"color: {self.ACCENT_ERROR}; background: transparent;"
                )
                self._detail_label.setText("Check system tray for details")  # type: ignore[union-attr]
                self._detail_label.show()  # type: ignore[union-attr]
                self._done_timer.start(4000)

        def _auto_hide(self) -> None:
            self.hide_signal.emit()

        def _refresh(self) -> None:
            if self._state == OverlayState.RECORDING:
                self._waveform.update()

        def update_translation_text(self, text: str) -> None:
            self._translation_label.setText(text)  # type: ignore[union-attr]
            self._translation_label.show()  # type: ignore[union-attr]

        def paintEvent(self, event: object) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Fluent dark surface with subtle border
            painter.setBrush(self._bg_surface)
            painter.setPen(QPen(self._bg_border, 1.0))
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 8, 8)
            painter.end()

else:

    class WaveformWidget:  # type: ignore[no-redef]
        pass

    class OverlayWindow:  # type: ignore[no-redef]
        pass
