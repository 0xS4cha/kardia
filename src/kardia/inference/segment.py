from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import SimpleITK as sitk
import torch
from ..io.demo import make_demo_segmentation
from ..io.volume import Volume
from ..model.unet import build_unet
from ..preprocess.transforms import volume_to_tensor

class Segmenter:

    def __init__(self, config: dict[str, Any], device: str | None=None) -> None:
        self.config = config
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = build_unet(config).to(self.device)
        self.model.eval()
        self.checkpoint_path = self._resolve_checkpoint()
        self.uses_demo = True
        if self.checkpoint_path is not None:
            payload = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            state = payload['model'] if isinstance(payload, dict) and 'model' in payload else payload
            self.model.load_state_dict(state, strict=True)
            self.uses_demo = False

    def _resolve_checkpoint(self) -> Path | None:
        raw = self.config.get('paths', {}).get('default_checkpoint')
        if not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path if path.is_file() else None

    @torch.inference_mode()
    def predict(self, volume: Volume) -> np.ndarray:
        if self.uses_demo:
            return make_demo_segmentation(volume)
        tensor = volume_to_tensor(volume, self.config).to(self.device)
        logits = self.model(tensor)
        pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        if pred.shape != volume.array.shape:
            pred = _resample_labels(pred, volume.array.shape)
        return pred

def _resample_labels(labels: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    source = sitk.GetImageFromArray(labels.astype(np.uint8))
    reference = sitk.Image(int(target_shape[2]), int(target_shape[1]), int(target_shape[0]), sitk.sitkUInt8)
    (sx, sy, sz) = source.GetSpacing()
    (src_z, src_y, src_x) = labels.shape
    (tgt_z, tgt_y, tgt_x) = target_shape
    source.SetSpacing((sx * src_x / tgt_x, sy * src_y / tgt_y, sz * src_z / tgt_z))
    reference.SetSpacing(source.GetSpacing())
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampled = resampler.Execute(source)
    return sitk.GetArrayFromImage(resampled).astype(np.uint8)