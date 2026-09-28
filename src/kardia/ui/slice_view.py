from __future__ import annotations
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSlider, QVBoxLayout, QWidget
from PySide6.QtCore import Qt
from ..io.volume import Volume

class SliceView(QWidget):

    def __init__(self, parent: QWidget | None=None) -> None:
        super().__init__(parent)
        self._volume: Volume | None = None
        self._labels: np.ndarray | None = None
        self.image_label = QLabel('No image loaded')
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(180)
        self.image_label.setStyleSheet('background: #111; color: #ccc;')
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slice_changed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image_label)
        layout.addWidget(self.slider)

    def set_volume(self, volume: Volume | None, labels: np.ndarray | None=None) -> None:
        self._volume = volume
        self._labels = labels
        if volume is None or volume.array.size == 0:
            self.slider.setEnabled(False)
            self.image_label.setText('No image loaded')
            return
        depth = volume.array.shape[0]
        self.slider.setRange(0, max(depth - 1, 0))
        self.slider.setValue(depth // 2)
        self.slider.setEnabled(True)
        self._render_slice(self.slider.value())

    def set_labels(self, labels: np.ndarray | None) -> None:
        self._labels = labels
        if self._volume is not None:
            self._render_slice(self.slider.value())

    def _on_slice_changed(self, index: int) -> None:
        self._render_slice(index)

    def _render_slice(self, index: int) -> None:
        if self._volume is None:
            return
        frame = np.asarray(self._volume.array[index], dtype=np.float32)
        vis = _normalize_u8(frame)
        if self._labels is not None and self._labels.shape == self._volume.array.shape:
            vis = _overlay_labels(vis, self._labels[index])
        qimage = _gray_to_qimage(vis)
        pixmap = QPixmap.fromImage(qimage).scaled(self.image_label.size() if self.image_label.size().width() > 0 else qimage.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._volume is not None and self.slider.isEnabled():
            self._render_slice(self.slider.value())

def _normalize_u8(frame: np.ndarray) -> np.ndarray:
    (lo, hi) = np.percentile(frame, (1, 99))
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((frame - lo) / (hi - lo), 0, 1)
    return (scaled * 255).astype(np.uint8)

def _overlay_labels(gray: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rgb = np.stack([gray, gray, gray], axis=-1)
    palette = {1: np.array([220, 50, 60], dtype=np.uint8), 2: np.array([240, 190, 80], dtype=np.uint8), 3: np.array([60, 110, 220], dtype=np.uint8)}
    for (label, color) in palette.items():
        mask = labels == label
        if not np.any(mask):
            continue
        rgb[mask] = (0.55 * rgb[mask] + 0.45 * color).astype(np.uint8)
    return rgb

def _gray_to_qimage(array: np.ndarray) -> QImage:
    if array.ndim == 2:
        (height, width) = array.shape
        bytes_per_line = width
        fmt = QImage.Format.Format_Grayscale8
        buffer = np.ascontiguousarray(array)
    else:
        (height, width, _) = array.shape
        buffer = np.ascontiguousarray(array)
        bytes_per_line = 3 * width
        fmt = QImage.Format.Format_RGB888
    image = QImage(buffer.data, width, height, bytes_per_line, fmt)
    return image.copy()