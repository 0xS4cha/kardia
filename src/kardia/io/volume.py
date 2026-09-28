from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import SimpleITK as sitk

@dataclass
class Volume:
    array: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]
    path: Path | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple((int(v) for v in self.array.shape))

    def to_sitk(self) -> sitk.Image:
        image = sitk.GetImageFromArray(self.array)
        image.SetSpacing(self.spacing)
        image.SetOrigin(self.origin)
        image.SetDirection(self.direction)
        return image

    @classmethod
    def from_sitk(cls, image: sitk.Image, path: Path | None=None) -> Volume:
        array = sitk.GetArrayFromImage(image)
        return cls(array=np.ascontiguousarray(array), spacing=tuple((float(v) for v in image.GetSpacing())), origin=tuple((float(v) for v in image.GetOrigin())), direction=tuple((float(v) for v in image.GetDirection())), path=path)