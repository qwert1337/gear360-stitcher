"""Minimal Double Sphere camera model (Usenko et al., "The Double Sphere
Camera Model") -- just the projection the stitcher needs, with no torch or
autocalib dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np


class ParamsDoubleSphere(NamedTuple):
    fx: float
    fy: float
    cx: float
    cy: float
    alpha: float
    Xi: float


class DoubleSphere:
    def __init__(self, params: ParamsDoubleSphere):
        self.params = params

    def world2cam(self, XYZ):
        """Project

        Args:
            XYZ: shape [height, width, 3]
        Returns:
            uv: float32 [height, width, 2] pixel coordinates
            valid: bool [1, 1, height, width]
        """
        p = self.params
        h, w = XYZ.shape[0:2]
        x, y, z = XYZ.reshape(-1, 3).T
        d1 = np.sqrt(x**2 + y**2 + z**2)
        d2 = np.sqrt(x**2 + y**2 + (p.Xi * d1 + z) ** 2)
        z_fe = p.alpha * d2 + (1 - p.alpha) * (p.Xi * d1 + z)

        u = p.fx * x / z_fe + p.cx
        v = p.fy * y / z_fe + p.cy
        uv = np.stack((u, v), axis=1).reshape((h, w, 2))

        if p.alpha > 0.5:
            w1 = (1 - p.alpha) / p.alpha
        else:
            w1 = p.alpha / (1 - p.alpha)
        w2 = (w1 + p.Xi) / np.sqrt(2 * w1 * p.Xi + p.Xi**2 + 1)
        valid = (z > -w2 * d1).reshape((1, 1, h, w))

        return uv.astype("float32"), valid


def load_calibration(path: Path) -> tuple[ParamsDoubleSphere, tuple[int, int]]:
    """Read a lens calibration.json.

    Format::

        {"resolution": [width, height],
         "params_ds": {"fx": ..., "fy": ..., "cx": ..., "cy": ...,
                       "alpha": ..., "Xi": ...}}

    Returns the Double Sphere parameters and the (width, height) the
    calibration was fit at.
    """
    with open(path) as f:
        calib = json.load(f)
    params = ParamsDoubleSphere(**{k: float(calib["params_ds"][k]) for k in ParamsDoubleSphere._fields})
    width, height = calib["resolution"]
    return params, (int(width), int(height))
