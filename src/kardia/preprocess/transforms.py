from __future__ import annotations
from typing import Any
import numpy as np
import torch
from monai.transforms import Compose, EnsureType, Resize, ScaleIntensityRange
from ..io.volume import Volume

def build_preprocess(config: dict[str, Any]) -> Compose:
    spatial = tuple((int(v) for v in config['model']['spatial_size']))
    prep = config.get('preprocess', {})
    return Compose([ScaleIntensityRange(a_min=float(prep.get('a_min', 0.0)), a_max=float(prep.get('a_max', 2000.0)), b_min=float(prep.get('b_min', 0.0)), b_max=float(prep.get('b_max', 1.0)), clip=True), Resize(spatial_size=spatial, mode='trilinear'), EnsureType(dtype=torch.float32)])

def volume_to_tensor(volume: Volume, config: dict[str, Any]) -> torch.Tensor:
    array = np.asarray(volume.array, dtype=np.float32)
    image = np.expand_dims(array, axis=0)
    transformed = build_preprocess(config)(image)
    if not torch.is_tensor(transformed):
        transformed = torch.as_tensor(transformed)
    return transformed.unsqueeze(0)