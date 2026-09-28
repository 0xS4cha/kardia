from __future__ import annotations
import numpy as np
from .volume import Volume

def make_demo_volume(shape: tuple[int, int, int]=(64, 128, 128), spacing: tuple[float, float, float]=(1.5, 1.25, 1.25)) -> Volume:
    (depth, height, width) = shape
    (zz, yy, xx) = np.ogrid[:depth, :height, :width]
    (z0, y0, x0) = (depth / 2, height / 2, width / 2)
    lv = _organic_shape(zz, yy, xx, z0, y0, x0, 0.28 * depth, 0.22 * height, 0.2 * width, taper=0.4)
    myo = _organic_shape(zz, yy, xx, z0, y0, x0, 0.38 * depth, 0.32 * height, 0.3 * width, taper=0.4, bump_freq=6.0, bump_amp=0.08)
    rv = _organic_shape(zz, yy, xx, z0, y0 + 0.12 * height, x0 + 0.18 * width, 0.26 * depth, 0.12 * height, 0.18 * width, taper=0.2, bump_freq=4.0, bump_amp=0.05)
    intensity = np.zeros(shape, dtype=np.float32)
    intensity[myo] = 420.0
    intensity[lv] = 780.0
    intensity[rv] = 740.0
    noise = np.random.default_rng(7).normal(0, 25, size=shape).astype(np.float32)
    intensity = np.clip(intensity + noise + 80.0, 0, None)
    return Volume(array=intensity, spacing=spacing, origin=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), path=None)

def make_demo_segmentation(volume: Volume) -> np.ndarray:
    (depth, height, width) = volume.array.shape
    (zz, yy, xx) = np.ogrid[:depth, :height, :width]
    (z0, y0, x0) = (depth / 2, height / 2, width / 2)
    lv = _organic_shape(zz, yy, xx, z0, y0, x0, 0.28 * depth, 0.22 * height, 0.2 * width, taper=0.4)
    myo_outer = _organic_shape(zz, yy, xx, z0, y0, x0, 0.38 * depth, 0.32 * height, 0.3 * width, taper=0.4, bump_freq=6.0, bump_amp=0.08)
    rv = _organic_shape(zz, yy, xx, z0, y0 + 0.12 * height, x0 + 0.18 * width, 0.26 * depth, 0.12 * height, 0.18 * width, taper=0.2, bump_freq=4.0, bump_amp=0.05)
    labels = np.zeros(volume.array.shape, dtype=np.uint8)
    labels[myo_outer] = 2
    labels[lv] = 1
    labels[rv] = 3
    return labels

def _organic_shape(zz: np.ndarray, yy: np.ndarray, xx: np.ndarray, z0: float, y0: float, x0: float, rz: float, ry: float, rx: float, taper: float=0.0, bump_freq: float=0.0, bump_amp: float=0.0) -> np.ndarray:
    dz = (zz - z0) / max(rz, 1.0)
    dy = (yy - y0) / max(ry, 1.0)
    dx = (xx - x0) / max(rx, 1.0)
    
    taper_factor = 1.0 + taper * dz
    taper_factor = np.clip(taper_factor, 0.2, 2.0)
    
    r = np.sqrt((dx / taper_factor)**2 + (dy / taper_factor)**2 + dz**2)
    
    if bump_amp > 0:
        theta = np.arctan2(dy, dx)
        phi = np.arcsin(np.clip(dz / (r + 1e-6), -1.0, 1.0))
        noise = np.sin(bump_freq * theta) * np.cos(bump_freq * phi)
        threshold = 1.0 + bump_amp * noise
    else:
        threshold = 1.0
        
    return r <= threshold