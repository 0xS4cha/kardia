from __future__ import annotations
from typing import Any
import numpy as np
import pyvista as pv
from ..model.labels import structures_from_config

def meshes_from_segmentation(labels: np.ndarray, spacing: tuple[float, float, float], config: dict[str, Any]) -> dict[int, pv.PolyData]:
    structures = structures_from_config(config)
    volume = np.ascontiguousarray(np.transpose(labels, (2, 1, 0)))
    grid = pv.ImageData()
    grid.dimensions = np.array(volume.shape) + 1
    grid.spacing = spacing
    grid.origin = (0.0, 0.0, 0.0)
    grid.cell_data['label'] = volume.ravel(order='F')
    meshes: dict[int, pv.PolyData] = {}
    for (label, structure) in structures.items():
        if label == 0:
            continue
        surface = _extract_surface(grid, label)
        if surface.n_points == 0:
            continue
        meshes[label] = surface
    return meshes

def _extract_surface(grid: pv.ImageData, label: int) -> pv.PolyData:
    mask = (grid.cell_data['label'] == label).astype(np.uint8)
    labeled = grid.copy()
    labeled.cell_data['mask'] = mask
    surface = labeled.cell_data_to_point_data().contour([0.5], scalars='mask')
    if surface.n_points == 0:
        return surface
    try:
        return surface.smooth_taubin(n_iter=10, pass_band=0.1)
    except Exception:
        return surface