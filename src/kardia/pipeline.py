from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pyvista as pv
from .build.mesh import meshes_from_segmentation
from .inference.segment import Segmenter
from .io.volume import Volume

@dataclass
class Reconstruction:
    volume: Volume
    labels: np.ndarray
    meshes: dict[int, pv.PolyData]
    used_demo_segmentation: bool

class CardiacPipeline:

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.segmenter = Segmenter(config)

    def reconstruct(self, volume: Volume) -> Reconstruction:
        labels = self.segmenter.predict(volume)
        meshes = meshes_from_segmentation(labels, volume.spacing, self.config)
        return Reconstruction(volume=volume, labels=labels, meshes=meshes, used_demo_segmentation=self.segmenter.uses_demo)