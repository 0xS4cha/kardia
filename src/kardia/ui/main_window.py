from __future__ import annotations
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QSplitter, QStatusBar, QToolBar, QVBoxLayout, QWidget
from pyvistaqt import QtInteractor
from ..config import load_config
from ..io.demo import make_demo_volume
from ..io.loaders import load_volume
from ..io.volume import Volume
from ..model.labels import structures_from_config
from ..pipeline import CardiacPipeline
from ..visualization.scene import add_heart_meshes, apply_scene_style
from .slice_view import SliceView

class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Kardia — Cardiac MRI')
        self.resize(1280, 800)
        self.config = load_config()
        self.pipeline = CardiacPipeline(self.config)
        self.volume: Volume | None = None
        self.actor_names: dict[int, str] = {}
        self.structure_checks: dict[int, QCheckBox] = {}
        self._build_ui()
        apply_scene_style(self.plotter, self.config)
        self.statusBar().showMessage('Open an MRI or launch demo mode.')

    def _build_ui(self) -> None:
        toolbar = QToolBar('Actions')
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        open_nifti = QPushButton('Open NIfTI')
        open_dicom = QPushButton('Open DICOM folder')
        demo = QPushButton('Demo Mode')
        reconstruct = QPushButton('Start Reconstruction')
        open_nifti.clicked.connect(self.open_nifti)
        open_dicom.clicked.connect(self.open_dicom)
        demo.clicked.connect(self.load_demo)
        reconstruct.clicked.connect(self.reconstruct)
        for widget in (open_nifti, open_dicom, demo, reconstruct):
            toolbar.addWidget(widget)
        side = QWidget()
        side_layout = QVBoxLayout(side)
        self.info_label = QLabel('No volume.')
        self.info_label.setWordWrap(True)
        side_layout.addWidget(self.info_label)
        side_layout.addWidget(QLabel('Structures'))
        self.structure_panel = QVBoxLayout()
        side_layout.addLayout(self.structure_panel)
        side_layout.addStretch(1)
        self._populate_structure_checks()
        self.plotter = QtInteractor(self)
        self.slice_view = SliceView()
        viewer = getattr(self.plotter, 'interactor', self.plotter)
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(viewer)
        right.addWidget(self.slice_view)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 1)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(side)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 1000])
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addWidget(splitter)
        self.setCentralWidget(container)
        self.setStatusBar(QStatusBar())

    def _populate_structure_checks(self) -> None:
        structures = structures_from_config(self.config)
        for (label, structure) in structures.items():
            if label == 0:
                continue
            box = QCheckBox(structure.name)
            box.setChecked(structure.visible)
            box.stateChanged.connect(lambda _state, lbl=label: self._on_visibility_changed(lbl))
            self.structure_checks[label] = box
            self.structure_panel.addWidget(box)

    def open_nifti(self) -> None:
        (path, _) = QFileDialog.getOpenFileName(self, 'Open NIfTI MRI', str(Path.cwd()), 'Volumes (*.nii *.nii.gz *.nrrd *.mha *.mhd);;All files (*)')
        if path:
            self._load_path(Path(path))

    def open_dicom(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, 'Open DICOM folder', str(Path.cwd()))
        if folder:
            self._load_path(Path(folder))

    def load_demo(self) -> None:
        self.volume = make_demo_volume()
        self.slice_view.set_volume(self.volume)
        self._update_info('Synthetic Volume (Demo)')
        self.statusBar().showMessage('Demo mode loaded. Start 3D reconstruction.')

    def _load_path(self, path: Path) -> None:
        try:
            self.volume = load_volume(path)
        except Exception as exc:
            QMessageBox.critical(self, 'Cannot read', str(exc))
            return
        self.slice_view.set_volume(self.volume)
        self._update_info(str(path))
        self.statusBar().showMessage(f'Volume loaded: {path}')

    def reconstruct(self) -> None:
        if self.volume is None:
            QMessageBox.information(self, 'Kardia', 'Load an MRI or demo mode first.')
            return
        try:
            result = self.pipeline.reconstruct(self.volume)
        except Exception as exc:
            QMessageBox.critical(self, 'Reconstruction failed', str(exc))
            return
        visibility = {label: box.isChecked() for (label, box) in self.structure_checks.items()}
        self.actor_names = add_heart_meshes(self.plotter, result.meshes, self.config, visibility)
        self.plotter.update()
        self.slice_view.set_labels(result.labels)
        mode = 'geometric demo' if result.used_demo_segmentation else 'neural network'
        self.statusBar().showMessage(f'3D reconstruction complete ({mode}).')

    def _on_visibility_changed(self, label: int) -> None:
        name = self.actor_names.get(label)
        if not name or name not in self.plotter.actors:
            return
        visible = self.structure_checks[label].isChecked()
        actor = self.plotter.actors.get(name)
        if actor is not None:
            actor.SetVisibility(visible)
            if hasattr(actor, 'visibility'):
                actor.visibility = visible
        self.plotter.update()

    def _update_info(self, source: str) -> None:
        if self.volume is None:
            self.info_label.setText('No volume.')
            return
        shape = ' × '.join((str(v) for v in self.volume.shape))
        spacing = ', '.join((f'{v:.2f}' for v in self.volume.spacing))
        self.info_label.setText(f'{source}\nSize: {shape}\nSpacing: {spacing} mm')