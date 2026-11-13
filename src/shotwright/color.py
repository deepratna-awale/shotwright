from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)
_WHITE = np.array([0.95047, 1.0, 1.08883])
_EPS = 216 / 24389
_KAPPA = 24389 / 27


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _RGB_TO_XYZ.T / _WHITE
    f = np.where(xyz > _EPS, np.cbrt(xyz), (_KAPPA * xyz + 16) / 116)
    lab = np.empty_like(f)
    lab[..., 0] = 116 * f[..., 1] - 16
    lab[..., 1] = 500 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200 * (f[..., 1] - f[..., 2])
    return lab


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], axis=-1)
    xyz = np.where(f**3 > _EPS, f**3, (116 * f - 16) / _KAPPA) * _WHITE
    linear = np.clip(xyz @ _XYZ_TO_RGB.T, 0, 1)
    c = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)
    return (np.clip(c, 0, 1) * 255).round().astype(np.uint8)


@dataclass(frozen=True)
class LabStats:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def of(cls, frames: np.ndarray, stride: int = 4) -> LabStats:
        arr = np.asarray(frames)
        sample = arr[..., ::stride, ::stride, :].reshape(-1, 3)
        lab = rgb_to_lab(sample)
        return cls(lab.mean(axis=0), lab.std(axis=0))


def transfer(lab: np.ndarray, source: LabStats, target: LabStats, max_gain: float = 2.0):
    gain = np.clip(target.std / np.maximum(source.std, 1e-3), 1 / max_gain, max_gain)
    return (lab - source.mean) * gain + target.mean


def match_to(
    frames: np.ndarray, source: LabStats, target: LabStats, ramp: float | None = None
) -> np.ndarray:
    """Shift every frame so that `source` statistics become `target`.

    With `ramp`, the shift grows linearly from 0 on the first frame to `ramp` on the last.
    """
    out = np.empty_like(frames)
    n = len(frames)
    for i, frame in enumerate(frames):
        lab = rgb_to_lab(frame)
        moved = transfer(lab, source, target)
        if ramp is not None:
            w = ramp * (i / (n - 1) if n > 1 else 1.0)
            moved = lab + (moved - lab) * w
        out[i] = lab_to_rgb(moved)
    return out
