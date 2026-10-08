"""Dual-fisheye -> equirectangular stitching for the Samsung Gear 360.

Loads per-lens Double Sphere calibration + vignette models, projects both
lenses onto a shared equirectangular grid, color-matches the rear lens onto
the front lens, and blends them across the two seams (either a seam-carved
mask or a hard column cut) using Laplacian-pyramid multiband blending.
"""

from __future__ import annotations

import concurrent.futures
import functools
import json
import logging
import os
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .camera import DoubleSphere, load_calibration

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Vignette correction
# --------------------------------------------------------------------------


@dataclass
class VignetteModel:
    """Fitted vignette geometry (image circle) and radial falloff polynomial.

    See autocalib's auto_calib_gear360.py's fit_vignette_circle/fit_vignette_polynomial
    steps, which produce the two JSON files this is loaded from.
    """

    cx: float
    cy: float
    r: float
    coeffs: np.ndarray
    powers: np.ndarray

    @classmethod
    def load(cls, calib_dir: Path) -> VignetteModel:
        circle = json.loads((calib_dir / "vignette_circle.json").read_text())
        poly = json.loads((calib_dir / "vignette_polynomial.json").read_text())
        return cls(
            cx=circle["cx"],
            cy=circle["cy"],
            r=circle["r"],
            coeffs=np.asarray(poly["coeffs"]),
            powers=np.asarray(poly["powers"]),
        )

    def _eval(self, rho: np.ndarray) -> np.ndarray:
        rho = np.asarray(rho)
        return (rho[..., None] ** self.powers) @ self.coeffs

    def falloff_map(self, shape: tuple[int, int]) -> np.ndarray:
        """The fitted radial luminance falloff over an image of `shape`.
        Depends only on the vignette model and the image size, so callers
        that process many frames of the same size (video) should compute
        this once and reuse it instead of calling `correct` per frame."""
        yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
        rho = np.hypot(xx - self.cx, yy - self.cy) / self.r
        return self._eval(np.clip(rho, 0, 1)).astype(np.float32)

    def gain_map(self, shape: tuple[int, int]) -> np.ndarray:
        """`1 / falloff_map(shape)`, replicated to three channels — the
        form `apply_vignette_gain` needs to correct a whole BGR frame in
        one pass. Frame-invariant like falloff_map, so build it once per
        shape and reuse it (see Gear360Stitcher._prepare_for_shapes).

        Three identical channels rather than one broadcast plane because
        cv2's arithmetic requires matching channel counts, and the cv2
        path is what makes the per-frame correction multi-threaded; the
        redundancy costs one extra float32 frame of cache, which video
        buys back many times over (see apply_vignette_gain).
        """
        return cv2.merge([np.reciprocal(self.falloff_map(shape))] * 3)

    def correct(
        self, image: np.ndarray, falloff: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Undo the fitted radial luminance falloff so front/back exposures
        match better for stitching. `image` is float BGR in [0, 1].

        Straightforward and allocating, for one-off use. The per-frame path
        uses apply_vignette_gain instead."""
        if falloff is None:
            falloff = self.falloff_map(image.shape[:2])
        corrected = image / falloff[..., None]
        np.clip(
            corrected, 0, 1, out=corrected
        )  # clip in place: skip a 2nd full-frame allocation
        return corrected

    def disk_mask(self, shape: tuple[int, int], shrink: float = 0.95) -> np.ndarray:
        """A filled disk covering the trustworthy interior of the image
        circle, shrunk by `shrink` to stay clear of the noisy vignette edge."""
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.circle(
            mask,
            (round(self.cx), round(self.cy)),
            round(self.r * shrink),
            1,
            -1,
        )
        return mask


# Resident bytes ONE of Gear360Stitcher's per-frame scratch caches may
# occupy before it falls back to allocating as it goes. Two are budgeted
# against it independently — the vignette gain maps and buffers, which
# scale with the INPUT frame, and the equirectangular remap output
# buffers, which scale with the OUTPUT panorama — so a stitcher holds at
# most twice this, which is what _estimate_worker_memory_mb charges.
#
# Both caches trade memory for per-frame time, so the trade is worth it
# exactly when there are many frames. 256MB draws that line where the Gear
# 360 itself does: its video half-frames want ~180MB and its photo
# half-frames ~730MB, so video takes the fast paths and a one-frame photo
# does not. Raise it if you are stitching long sequences of frames larger
# than the camera's own video.
_FRAME_SCRATCH_BUDGET_BYTES = 256_000_000


def _vignette_fast_path_bytes(
    front_shape: tuple[int, int], back_shape: tuple[int, int]
) -> int:
    """What the fast path would cost resident: per lens, a 3-channel
    float32 gain map plus a 3-channel float32 output buffer."""
    return 2 * 3 * 4 * (front_shape[0] * front_shape[1] + back_shape[0] * back_shape[1])


def apply_vignette_gain(
    image: np.ndarray, gain: np.ndarray, out: Optional[np.ndarray] = None
) -> np.ndarray:
    """Undo one lens' vignette falloff, as float32 BGR in [0, 1].

    Same result as `VignetteModel.correct` (to within one float32 ULP,
    verified), but built for the per-frame video path, where this used to
    be one of the more expensive stages. Three things make it ~4x cheaper:

    - It multiplies by a precomputed reciprocal (`VignetteModel.gain_map`)
      instead of dividing by the falloff every frame. Division has several
      times the throughput cost of multiplication.
    - It goes through cv2, which is SIMD-vectorized and multi-threaded,
      rather than numpy, which is single-threaded here.
    - `out` lets the caller hand in a persistent buffer, so a video's
      steady state stops allocating (and first-touching) a fresh
      full-frame float32 array per lens per frame.

    `image` may be the raw uint8 frame straight from cv2.VideoCapture, in
    which case the /255 conversion to [0, 1] is folded into the same pass
    — for video that removes a whole separate full-frame conversion, and
    it is why the video path passes stitch() an unconverted frame.

    Only the upper clip of `correct`'s [0, 1] is applied: `gain` is >= 1
    everywhere and neither a uint8 nor an in-range float32 frame can be
    negative, so the lower clip has nothing to do. THRESH_TRUNC does it,
    over an (H, W*3) single-channel reshape — free, because the result of
    cv2.multiply is contiguous, so that reshape is a view and the
    truncation lands in `out` itself. Guarded rather than assumed: a
    reshape that had to copy would clip the copy and silently return
    unclipped data.
    """
    scale = 1.0 / 255.0 if image.dtype == np.uint8 else 1.0
    out = cv2.multiply(image, gain, dst=out, scale=scale, dtype=cv2.CV_32F)
    if out.flags["C_CONTIGUOUS"]:
        flat = out.reshape(out.shape[0], -1)
        cv2.threshold(flat, 1.0, 0.0, cv2.THRESH_TRUNC, dst=flat)
    else:
        np.minimum(out, 1.0, out=out)
    return out


# --------------------------------------------------------------------------
# Camera loading
# --------------------------------------------------------------------------


@dataclass
class FisheyeCamera:
    """One Gear 360 lens: its Double Sphere model plus vignette correction."""

    ds: DoubleSphere
    vignette: VignetteModel
    calib_resolution: tuple[int, int]  # (width, height) the calibration was fit at

    @classmethod
    def load(cls, calib_dir: Path) -> FisheyeCamera:
        params, calib_resolution = load_calibration(calib_dir / "calibration.json")
        ds = DoubleSphere(params)
        vignette = VignetteModel.load(calib_dir)
        return cls(ds=ds, vignette=vignette, calib_resolution=calib_resolution)

    def scaled_to(self, shape: tuple[int, int]) -> FisheyeCamera:
        """Rescale the pixel-space calibration (intrinsics + vignette
        geometry) to an actual frame of `shape` (height, width), when it
        differs from the resolution the calibration was fit at. The Gear
        360 records video at a lower resolution than its photos (e.g. a
        1920x1920 half-frame vs. calibration fit at 3872x3872) — using
        photo-resolution fx/fy/cx/cy/vignette-circle unscaled against a
        smaller frame places the whole valid image circle outside the
        frame's actual pixel bounds, remapping to solid black almost
        everywhere. alpha/Xi (Double Sphere) and the vignette polynomial's
        coefficients are already scale-invariant and carry over unchanged.
        """
        height, width = shape
        calib_width, calib_height = self.calib_resolution
        scale_x = width / calib_width
        scale_y = height / calib_height
        if abs(scale_x - scale_y) > 0.01 * max(scale_x, scale_y):
            logger.warning(
                "Non-uniform scale between calibration resolution %s and "
                "frame shape %s (x=%.4f, y=%.4f)",
                self.calib_resolution,
                (width, height),
                scale_x,
                scale_y,
            )
        if scale_x == 1.0 and scale_y == 1.0:
            return self

        p = self.ds.params
        scaled_ds = DoubleSphere(
            p._replace(
                fx=p.fx * scale_x,
                cx=p.cx * scale_x,
                fy=p.fy * scale_y,
                cy=p.cy * scale_y,
            )
        )
        scaled_vignette = VignetteModel(
            cx=self.vignette.cx * scale_x,
            cy=self.vignette.cy * scale_y,
            r=self.vignette.r * scale_x,
            coeffs=self.vignette.coeffs,
            powers=self.vignette.powers,
        )
        return FisheyeCamera(scaled_ds, scaled_vignette, (width, height))


def load_extrinsics(path: Path, baseline_scale: float) -> np.ndarray:
    """Load the back-camera-relative-to-front-camera 3x4 pose and give its
    translation a metric length.

    The translation in R_t.txt is UNIT-NORM by construction -- see
    autocalib's auto_calib_gear360.py `baseline = 1.0  # t_rel stays unit-norm`, which
    writes t_rel / |t_rel| * baseline. The whole reconstruction there is
    scale-free (a two-view relative pose fixes the baseline's direction but
    never its length), so the file carries a direction and nothing else.

    `baseline_scale` therefore IS the front-back lens separation in metres,
    applied here. It is not a unit conversion and has nothing to do with the
    calibration board: an earlier version of this docstring called it
    "calibration-board units ... 0.05 for a 50mm checkerboard square", which
    was wrong and survived only because the Gear 360's ~50mm lens separation
    and the 50mm board square happen to be the same number.
    """
    R_t = np.eye(4)
    R_t[:3] = np.loadtxt(path)
    R_t[:3, -1] *= baseline_scale
    return R_t


def split_dual_fisheye(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split one side-by-side dual-fisheye frame into (front, back)."""
    half = image.shape[1] // 2
    return image[:, half:], image[:, :half]


# --------------------------------------------------------------------------
# Equirectangular projection
# --------------------------------------------------------------------------


def equirectangular_directions(width: int, height: int) -> np.ndarray:
    """Unit view direction for every pixel of a `height` x `width`
    equirectangular grid, shape [height, width, 3].

    float32: this and its transformed copy (see transform_directions) are
    held for the life of a Gear360Stitcher, and each is ~200MB in float64
    at the default 4096x2048 — the single biggest fixed cost per worker
    process when running video stitching in parallel (see
    stitch_video_parallel), so halving it is worth the trivial precision
    loss (still far finer than a pixel) for a unit direction vector.
    """
    lam, phi = np.meshgrid(
        np.linspace(-np.pi, np.pi, width, dtype=np.float32),
        np.linspace(np.pi / 2, -np.pi / 2, height, dtype=np.float32),
    )
    z = np.cos(phi) * np.cos(lam)
    x = np.cos(phi) * np.sin(lam)
    y = -np.sin(phi)
    return np.stack([x, y, z], axis=-1)


def transform_directions(
    directions: np.ndarray, R_t: np.ndarray, depth: float
) -> np.ndarray:
    """Re-express directions (assumed at `depth`) in another camera's frame."""
    scaled = depth * directions
    homogeneous = np.concatenate(
        [scaled, np.ones(scaled.shape[:-1] + (1,), dtype=scaled.dtype)], axis=-1
    )
    return (homogeneous @ R_t.T)[..., :3].astype(directions.dtype)


# --------------------------------------------------------------------------
# Auto-leveling
# --------------------------------------------------------------------------
#
# The equirectangular grid from equirectangular_directions() is built
# directly in the front camera's own frame, so the panorama's "up" is
# whatever tilt the camera happened to be held/mounted at. Leveling means
# rotating that grid, once, before it's used for anything else downstream
# (see Gear360Stitcher._build_geometry) -- everything is parameterized by
# the unit gravity-down vector `g`, expressed in front-camera coordinates
# (g = (0, 1, 0) for a perfectly level camera, matching the convention that
# row 0 of the grid is +Y_world = up and the image center is +Z = forward).


def gravity_from_roll_pitch(roll_deg: float, pitch_deg: float) -> np.ndarray:
    """Unit gravity-down vector in front-camera coordinates for a given
    roll (rotation about the optical axis, matching image-plane rotation)
    and pitch (positive = the optical axis points above the true horizon).
    Inverse of roll_pitch_from_gravity."""
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    return np.array(
        [
            -np.sin(roll) * np.cos(pitch),
            np.cos(roll) * np.cos(pitch),
            -np.sin(pitch),
        ]
    )


def roll_pitch_from_gravity(g: np.ndarray) -> tuple[float, float]:
    """Inverse of gravity_from_roll_pitch, in degrees."""
    roll = np.degrees(np.arctan2(-g[0], g[1]))
    pitch = np.degrees(-np.arcsin(np.clip(g[2], -1.0, 1.0)))
    return float(roll), float(pitch)


def rotation_from_gravity(g: np.ndarray) -> np.ndarray:
    """The shortest-arc rotation R with `R @ (0, 1, 0) == g`: maps the
    leveled frame's "down" axis onto the actual measured gravity-down
    direction in front-camera coordinates, so that
    `directions_front = grid_leveled @ R.T` re-expresses a perfectly level
    equirectangular grid in the camera's own (tilted) frame — see
    Gear360Stitcher._build_geometry.

    Rotation about g itself (yaw) is left undetermined by construction:
    there's no way to observe it from tilt alone, and it doesn't matter —
    the panorama's yaw origin is arbitrary regardless of leveling.
    """
    g = np.asarray(g, dtype=np.float64)
    g = g / np.linalg.norm(g)
    e_y = np.array([0.0, 1.0, 0.0])
    c = float(np.dot(e_y, g))
    if c < -0.99:
        logger.warning(
            "Cannot construct a leveling rotation for a near-upside-down "
            "camera (gravity . up = %.3f); leaving the panorama unleveled",
            c,
        )
        return np.eye(3, dtype=np.float32)
    v = np.cross(e_y, g)
    K = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )
    R = np.eye(3) + K + K @ K / (1.0 + c)
    return R.astype(np.float32)


def patch_directions(
    fov_deg: float, size: int, projection: str = "rectilinear"
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Unit view directions (front-camera frame), for a `size`x`size` patch
    covering `fov_deg` of the front lens' central field of view — the least
    fisheye-distorted part of the image, and the region a line-based
    orientation estimator (see the module docstring above) samples from.

    Rectilinear (gnomonic) is the default: every straight 3D line maps to
    an exactly straight 2D line here, which is what makes backprojecting a
    detected line segment to its 3D interpretation-plane normal exact
    algebra rather than an approximation (see segment_plane_normals).
    Cylindrical is offered as an alternative, but note it does NOT share
    that property — true-vertical lines only stay straight when the camera
    is already level, i.e. exactly the case this estimator can't assume.

    Returns (directions, f, (cx, cy)): `f` and the principal point are the
    patch's own "intrinsics", needed by segment_plane_normals to backproject
    a detected segment's pixel coordinates to a 3D ray.
    """
    if projection not in ("rectilinear", "cylindrical"):
        raise ValueError(
            f"projection must be 'rectilinear' or 'cylindrical', got {projection!r}"
        )
    f = (size / 2) / np.tan(np.radians(fov_deg) / 2)
    cx = cy = size / 2.0
    u, v = np.meshgrid(
        np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32)
    )
    if projection == "rectilinear":
        dirs = np.stack([(u - cx) / f, (v - cy) / f, np.ones_like(u)], axis=-1)
    else:
        lam = (u - cx) / f
        t = (v - cy) / f
        dirs = np.stack([np.sin(lam), t, np.cos(lam)], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs.astype(np.float32), float(f), (cx, cy)


def render_center_patch(
    front_image: np.ndarray, front_cam: FisheyeCamera, dirs: np.ndarray
) -> np.ndarray:
    """Warp `front_image` (float32 BGR in [0, 1], vignette-uncorrected —
    negligible falloff this close to the lens center) into the patch grid
    `dirs` (see patch_directions), returning a uint8 grayscale image ready
    for line detection."""
    uv, valid = front_cam.ds.world2cam(dirs)
    invalid = np.logical_not(valid).squeeze()
    if invalid.any():
        logger.warning(
            "%d/%d center-patch pixels fall outside the fisheye's valid "
            "domain; consider a smaller --level-fov",
            invalid.sum(),
            invalid.size,
        )
    patch = remap_to_equirectangular(front_image, uv, invalid)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return np.clip(gray * 255, 0, 255).astype(np.uint8)


# Line-detection / classification thresholds for the estimator below.
# Module-level constants rather than StitchConfig fields: these are
# detector internals a user shouldn't need to tune, unlike --level-fov or
# --level-max-tilt. See the vertical/horizon docstrings for what each
# guards against; the angle gates and the horizon-agreement gate are the
# ones most likely to need adjustment after testing on real footage.
_LEVEL_MIN_SEGMENT_LENGTH_FRAC = 0.02
_LEVEL_VERTICAL_ANGLE_DEG = 35.0
_LEVEL_HORIZON_ANGLE_DEG = 25.0
_LEVEL_HORIZON_MIN_LENGTH_FRAC = 0.5
_LEVEL_RANSAC_INLIER_DEG = 1.5
_LEVEL_MIN_VERTICAL_SEGMENTS = 8
_LEVEL_MIN_INLIERS = 6
_LEVEL_MIN_INLIER_LENGTH_FRAC = 0.3
_LEVEL_MIN_EIGENVALUE_RATIO = 10.0
_LEVEL_MAX_FIT_RESIDUAL_DEG = 1.0
_LEVEL_HORIZON_AGREEMENT_DEG = 2.0


@dataclass
class LevelEstimate:
    """Diagnostic report from one orientation-estimation attempt (see
    estimate_level_rotation) — always returned, even on failure, so
    callers can log exactly why leveling wasn't applied."""

    ok: bool
    mode: str  # "vertical", "horizon", or "none" (no reliable cue found)
    reason: str
    g: Optional[np.ndarray] = None
    roll_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    tilt_deg: Optional[float] = None
    n_segments: int = 0
    n_inliers: int = 0
    residual_deg: Optional[float] = None


def detect_line_segments(patch: np.ndarray) -> np.ndarray:
    """Straight line segments in a grayscale patch, as an (N, 4) array of
    (x1, y1, x2, y2) endpoints. Prefers LSD (sub-pixel endpoints, no
    per-scene threshold tuning needed); falls back to Canny + HoughLinesP
    if this OpenCV build lacks it (LSD was pulled from OpenCV for patent
    reasons for a few 4.x releases)."""
    try:
        detector = cv2.createLineSegmentDetector(0)
        lines = detector.detect(patch)[0]
    except (cv2.error, AttributeError) as exc:
        logger.debug("LSD unavailable (%s); falling back to Canny + HoughLinesP", exc)
        size = patch.shape[0]
        edges = cv2.Canny(patch, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=max(1, round(0.05 * size)),
            minLineLength=0.10 * size,
            maxLineGap=0.01 * size,
        )
    if lines is None:
        return np.empty((0, 4), dtype=np.float32)
    return lines.reshape(-1, 4).astype(np.float32)


def segment_plane_normals(
    segments: np.ndarray, f: float, center: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """For each segment (rectilinear patch pixel endpoints), the unit
    normal of the 3D plane through the optical center and the 3D line it's
    the image of (its "interpretation plane"), plus its pixel length as a
    reliability weight. Exact for a rectilinear patch — see patch_directions
    and the module docstring above rotation_from_gravity."""
    if len(segments) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)
    cx, cy = center
    x1, y1, x2, y2 = segments[:, 0], segments[:, 1], segments[:, 2], segments[:, 3]
    ones = np.ones_like(x1)
    r1 = np.stack([(x1 - cx) / f, (y1 - cy) / f, ones], axis=-1)
    r2 = np.stack([(x2 - cx) / f, (y2 - cy) / f, ones], axis=-1)
    n = np.cross(r1, r2)
    norm = np.linalg.norm(n, axis=-1)
    lengths = np.hypot(x2 - x1, y2 - y1)
    keep = norm > 1e-9
    return n[keep] / norm[keep, None], lengths[keep]


def classify_segments(
    segments: np.ndarray, patch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """(vertical_candidate, horizon_candidate) boolean masks for each
    segment, from its 2D orientation/length in the patch alone — see
    estimate_gravity_vertical/estimate_gravity_horizon for how each is
    actually used, and the module docstring for why they're kept separate."""
    if len(segments) == 0:
        empty = np.empty((0,), dtype=bool)
        return empty, empty
    x1, y1, x2, y2 = segments[:, 0], segments[:, 1], segments[:, 2], segments[:, 3]
    du, dv = x2 - x1, y2 - y1
    length = np.hypot(du, dv)
    long_enough = length >= _LEVEL_MIN_SEGMENT_LENGTH_FRAC * patch_size
    angle_from_vertical = np.degrees(np.arctan2(np.abs(du), np.abs(dv) + 1e-9))
    angle_from_horizontal = 90.0 - angle_from_vertical
    vertical = long_enough & (angle_from_vertical < _LEVEL_VERTICAL_ANGLE_DEG)
    horizon = (
        long_enough
        & (angle_from_horizontal < _LEVEL_HORIZON_ANGLE_DEG)
        & (length >= _LEVEL_HORIZON_MIN_LENGTH_FRAC * patch_size)
    )
    return vertical, horizon


def estimate_gravity_vertical(
    normals: np.ndarray,
    lengths: np.ndarray,
    max_tilt_deg: float,
    rng: Optional[np.random.Generator] = None,
) -> LevelEstimate:
    """Recover the gravity-down direction g (front-camera coords) from the
    interpretation-plane normals of segments classified as "vertical":
    every true-vertical 3D line has direction g, so its normal n satisfies
    n . g = 0 — g is the vector best satisfying that for all of them at
    once, i.e. the eigenvector of the weighted second-moment matrix
    Σ w_i n_i n_iᵀ with the SMALLEST eigenvalue (standard vertical
    vanishing-point estimation). RANSAC over normal pairs first, to keep
    non-vertical outliers that slipped through the 2D angle gate from
    corrupting the fit.
    """
    n_segments = len(normals)
    if n_segments < _LEVEL_MIN_VERTICAL_SEGMENTS:
        return LevelEstimate(
            ok=False,
            mode="vertical",
            n_segments=n_segments,
            reason=f"only {n_segments} vertical-candidate segments "
            f"(need >= {_LEVEL_MIN_VERTICAL_SEGMENTS})",
        )

    rng = rng or np.random.default_rng(0)
    best_g, best_score, best_inliers = None, -1.0, None
    n_pairs = n_segments * (n_segments - 1) // 2
    total_length = lengths.sum()
    for _ in range(min(300, n_pairs)):
        i, j = rng.choice(n_segments, size=2, replace=False)
        cross = np.cross(normals[i], normals[j])
        norm = np.linalg.norm(cross)
        if norm < 1e-6:
            continue
        g = cross / norm
        if g[1] < 0:
            g = -g
        residual = np.degrees(np.arcsin(np.clip(np.abs(normals @ g), 0.0, 1.0)))
        inliers = residual < _LEVEL_RANSAC_INLIER_DEG
        score = lengths[inliers].sum()
        if score > best_score:
            best_g, best_score, best_inliers = g, score, inliers

    n_inliers = 0 if best_inliers is None else int(best_inliers.sum())
    if (
        best_g is None
        or n_inliers < _LEVEL_MIN_INLIERS
        or best_score < _LEVEL_MIN_INLIER_LENGTH_FRAC * total_length
    ):
        return LevelEstimate(
            ok=False,
            mode="vertical",
            n_segments=n_segments,
            n_inliers=n_inliers,
            reason=f"RANSAC found no consistent vertical family ({n_inliers}/{n_segments} inliers)",
        )

    inlier_normals = normals[best_inliers]
    inlier_lengths = lengths[best_inliers]
    M = (inlier_normals * inlier_lengths[:, None]).T @ inlier_normals
    eigvals, eigvecs = np.linalg.eigh(M)  # ascending order
    g = eigvecs[:, 0]
    if g[1] < 0:
        g = -g

    # How well-determined g actually is: if the inlier normals are all
    # nearly parallel (one edge detected many times, say), the second
    # eigenvalue collapses towards the first and g spins freely in between
    # them — this ratio is a direct, principled measure of that, free from
    # the eigen-solve itself.
    if (
        eigvals[0] <= 0
        or eigvals[1] / max(eigvals[0], 1e-12) < _LEVEL_MIN_EIGENVALUE_RATIO
    ):
        return LevelEstimate(
            ok=False,
            mode="vertical",
            n_segments=n_segments,
            n_inliers=n_inliers,
            reason="ill-conditioned solve (inlier normals too close to parallel; "
            "g is underdetermined, likely just one edge detected repeatedly)",
        )

    residuals = np.degrees(np.arcsin(np.clip(np.abs(inlier_normals @ g), 0.0, 1.0)))
    rms = float(np.sqrt(np.mean(residuals**2)))
    if rms > _LEVEL_MAX_FIT_RESIDUAL_DEG:
        return LevelEstimate(
            ok=False,
            mode="vertical",
            n_segments=n_segments,
            n_inliers=n_inliers,
            residual_deg=rms,
            reason=f"poor fit (RMS residual {rms:.2f}° > {_LEVEL_MAX_FIT_RESIDUAL_DEG}°)",
        )

    tilt = float(np.degrees(np.arccos(np.clip(g[1], -1.0, 1.0))))
    if tilt > max_tilt_deg:
        return LevelEstimate(
            ok=False,
            mode="vertical",
            n_segments=n_segments,
            n_inliers=n_inliers,
            residual_deg=rms,
            tilt_deg=tilt,
            reason=f"implausible tilt {tilt:.1f}° > --level-max-tilt {max_tilt_deg}° "
            "(likely locked onto the wrong line family, e.g. a picture frame)",
        )

    roll, pitch = roll_pitch_from_gravity(g)
    return LevelEstimate(
        ok=True,
        mode="vertical",
        g=g,
        roll_deg=roll,
        pitch_deg=pitch,
        tilt_deg=tilt,
        n_segments=n_segments,
        n_inliers=n_inliers,
        residual_deg=rms,
        reason=f"{n_inliers}/{n_segments} vertical segments, residual {rms:.2f}°",
    )


def estimate_gravity_horizon(
    segments: np.ndarray,
    horizon_mask: np.ndarray,
    f: float,
    center: tuple[float, float],
    max_tilt_deg: float,
) -> LevelEstimate:
    """Recover g from a detected horizon line: its interpretation plane IS
    the horizontal plane through the optical center, so g = its normal
    directly (sign resolved to point down) — no vanishing-point solve
    needed, one line is enough.

    Important caveat (there's no way to make this fully safe without scene
    semantics): a horizontal 3D line that ISN'T at the camera's own height
    — a roofline, a table edge, a balcony rail — has a DIFFERENT
    interpretation plane than the true horizon, and using it is wrong in
    both roll AND pitch, not just pitch. The only cheap defense here is
    requiring the top-2 longest candidates to agree: unrelated
    architectural horizontals at different heights generally won't, while
    a real horizon and, say, a shoreline reflection will.
    """
    candidates = segments[horizon_mask]
    normals, lengths = segment_plane_normals(candidates, f, center)
    if len(normals) == 0:
        return LevelEstimate(
            ok=False, mode="horizon", reason="no horizon-candidate segments"
        )

    order = np.argsort(-lengths)
    normals = normals[order]
    # A horizon plane's normal is sign-ambiguous; resolve consistently so
    # "agreement" below compares like with like.
    signed = np.where(normals[:, 1:2] < 0, -normals, normals)

    if len(signed) < 2:
        return LevelEstimate(
            ok=False,
            mode="horizon",
            n_segments=len(candidates),
            reason="only one horizon candidate; can't check it's actually the horizon "
            "(vs. an architectural horizontal at some other height)",
        )

    agreement = float(
        np.degrees(np.arccos(np.clip(np.dot(signed[0], signed[1]), -1.0, 1.0)))
    )
    if agreement > _LEVEL_HORIZON_AGREEMENT_DEG:
        return LevelEstimate(
            ok=False,
            mode="horizon",
            n_segments=len(candidates),
            reason=f"top-2 horizon candidates disagree by {agreement:.1f}° "
            f"(> {_LEVEL_HORIZON_AGREEMENT_DEG}°) — looks like architecture, not a horizon",
        )

    g = signed[0]
    tilt = float(np.degrees(np.arccos(np.clip(g[1], -1.0, 1.0))))
    if tilt > max_tilt_deg:
        return LevelEstimate(
            ok=False,
            mode="horizon",
            n_segments=len(candidates),
            tilt_deg=tilt,
            reason=f"implausible tilt {tilt:.1f}° > --level-max-tilt {max_tilt_deg}°",
        )

    roll, pitch = roll_pitch_from_gravity(g)
    return LevelEstimate(
        ok=True,
        mode="horizon",
        g=g,
        roll_deg=roll,
        pitch_deg=pitch,
        tilt_deg=tilt,
        n_segments=len(candidates),
        n_inliers=2,
        reason=f"horizon line, top-2 candidates agree to {agreement:.2f}°",
    )


def estimate_level_rotation(
    front_cam: FisheyeCamera, front_image: np.ndarray, config: StitchConfig
) -> tuple[np.ndarray, LevelEstimate]:
    """Estimate the camera's roll/pitch from one frame's image content and
    return the corresponding leveling rotation (rotation_from_gravity) plus
    a diagnostic report. Falls back to identity (no leveling) on any
    failure and never raises — this is cosmetic and must never be able to
    break a stitch. `config.auto_level` selects the cue: "vertical",
    "horizon", or "auto" (try vertical, fall back to horizon)."""
    if config.level_projection != "rectilinear":
        reason = (
            "detection needs an exact line backprojection, only implemented for "
            f"level_projection='rectilinear' (got {config.level_projection!r})"
        )
        logger.warning("Auto-level: %s; leaving the panorama unleveled", reason)
        return np.eye(3, dtype=np.float32), LevelEstimate(
            ok=False, mode="none", reason=reason
        )

    dirs, f, center = patch_directions(
        config.level_fov_deg, config.level_patch_size, config.level_projection
    )
    patch = render_center_patch(front_image, front_cam, dirs)
    segments = detect_line_segments(patch)
    vertical_mask, horizon_mask = classify_segments(segments, config.level_patch_size)

    estimate = LevelEstimate(ok=False, mode="none", reason="no cue attempted")
    if config.auto_level in ("vertical", "auto"):
        normals, lengths = segment_plane_normals(segments[vertical_mask], f, center)
        estimate = estimate_gravity_vertical(
            normals, lengths, config.level_max_tilt_deg
        )
    if not estimate.ok and config.auto_level in ("horizon", "auto"):
        estimate = estimate_gravity_horizon(
            segments, horizon_mask, f, center, config.level_max_tilt_deg
        )

    if not estimate.ok:
        logger.warning(
            "Auto-level: %s; leaving the panorama unleveled", estimate.reason
        )
        return np.eye(3, dtype=np.float32), estimate

    logger.info(
        "Auto-level: roll=%+.2f° pitch=%+.2f° (tilt %.1f°) via %s cue — %s",
        estimate.roll_deg,
        estimate.pitch_deg,
        estimate.tilt_deg,
        estimate.mode,
        estimate.reason,
    )
    return rotation_from_gravity(estimate.g), estimate


# Video rig tilt is fixed per capture (see Gear360Stitcher.stitch), but a
# single frame's line detection can be thrown off by transient occlusion or
# a momentarily unlucky line configuration — this is the threshold at which
# estimate_level_rotation_from_video warns that its sampled frames disagree
# enough that the fixed-tilt assumption itself might be wrong (the rig
# moved during capture), not just that any one frame was noisy.
_LEVEL_VIDEO_SPREAD_WARN_DEG = 5.0


def estimate_level_rotation_from_video(
    video_path: Path,
    front_cam: FisheyeCamera,
    config: StitchConfig,
    sample_frames: int = 5,
) -> tuple[np.ndarray, LevelEstimate]:
    """Like estimate_level_rotation, but averages over `sample_frames`
    frames spread evenly through the video instead of trusting a single
    one — necessary because video bakes the estimate in once for the whole
    clip (re-estimating per frame would invalidate everything
    Gear360Stitcher._prepare_for_shapes caches, and would give
    stitch_video_parallel's chunks visibly different leveling at their
    boundaries), so a single unlucky frame has nothing to average away.

    Fuses per-frame estimates with a plain mean of roll and pitch, not a
    single combined vanishing-point-style solve over all frames' segments:
    for the small angles this deals with, roll and pitch compose and
    average linearly to good approximation, and a plain mean is far easier
    to reason about and debug than weighting per-frame gravity vectors.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        logger.warning(
            "Auto-level: could not determine a frame count for %s; "
            "estimating from the first frame only",
            video_path,
        )
        total_frames = 1
    n_samples = max(1, min(sample_frames, total_frames))
    frame_indices = np.linspace(0, total_frames - 1, n_samples, dtype=int)

    per_frame: list[LevelEstimate] = []
    try:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            front_im, _ = split_dual_fisheye(frame.astype(np.float32) / 255.0)
            cam = front_cam.scaled_to(front_im.shape[:2])
            _, estimate = estimate_level_rotation(cam, front_im, config)
            per_frame.append(estimate)
    finally:
        cap.release()

    ok_estimates = [e for e in per_frame if e.ok]
    if not ok_estimates:
        reason = f"no reliable estimate in any of {len(per_frame)} sampled frames"
        logger.warning("Auto-level: %s; leaving the panorama unleveled", reason)
        return np.eye(3, dtype=np.float32), LevelEstimate(
            ok=False, mode="none", reason=reason
        )

    rolls = np.array([e.roll_deg for e in ok_estimates])
    pitches = np.array([e.pitch_deg for e in ok_estimates])
    spread = float(max(rolls.max() - rolls.min(), pitches.max() - pitches.min()))
    if spread > _LEVEL_VIDEO_SPREAD_WARN_DEG:
        logger.warning(
            "Auto-level: sampled frames disagree by up to %.1f° (roll %.1f to %.1f°, "
            "pitch %.1f to %.1f°) — the rig may have moved during capture; using the "
            "average anyway as one fixed correction for the whole clip",
            spread,
            rolls.min(),
            rolls.max(),
            pitches.min(),
            pitches.max(),
        )

    roll, pitch = float(rolls.mean()), float(pitches.mean())
    g = gravity_from_roll_pitch(roll, pitch)
    tilt = float(np.degrees(np.arccos(np.clip(g[1], -1.0, 1.0))))
    modes = {e.mode for e in ok_estimates}
    combined = LevelEstimate(
        ok=True,
        mode=modes.pop() if len(modes) == 1 else "mixed",
        g=g,
        roll_deg=roll,
        pitch_deg=pitch,
        tilt_deg=tilt,
        n_segments=sum(e.n_segments for e in ok_estimates),
        n_inliers=sum(e.n_inliers for e in ok_estimates),
        reason=f"averaged {len(ok_estimates)}/{len(per_frame)} sampled frames",
    )
    logger.info(
        "Auto-level: roll=%+.2f° pitch=%+.2f° (tilt %.1f°), averaged over %d/%d sampled frames",
        roll,
        pitch,
        tilt,
        len(ok_estimates),
        len(per_frame),
    )
    return rotation_from_gravity(g), combined


# Where bake_invalid_directions sends pixels the camera model rejects. Any
# coordinate far outside the source frame does; cv2.remap's default
# BORDER_CONSTANT then fills them with 0, which is what the explicit mask
# write used to do. Kept well inside int16 so the map can still be
# converted to cv2's fixed-point form if that is ever wanted.
_REMAP_INVALID_COORD = -32000.0


def bake_invalid_directions(uv: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    """`uv` with every pixel the camera model marks invalid (behind the
    camera, etc) redirected off the edge of the source frame, so remapping
    zeroes them as a side effect of the border rule instead of needing a
    separate full-frame masked write afterwards.

    `invalid` is frame-invariant, so this is done once per shape (see
    Gear360Stitcher._prepare_for_shapes) rather than every frame. Bit-exact
    against masking after the fact: both paths write exactly 0."""
    baked = uv.copy()
    baked[invalid] = _REMAP_INVALID_COORD
    return baked


def remap_to_equirectangular(
    image: np.ndarray,
    uv: np.ndarray,
    invalid: Optional[np.ndarray] = None,
    dst: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Warp `image` into the equirectangular grid defined by `uv`, zeroing
    directions the camera model marks invalid (behind the camera, etc).

    Pass `invalid` (`np.logical_not(valid).squeeze()`) for a one-off warp,
    or omit it when `uv` has already been through bake_invalid_directions —
    which is what the per-frame path does, since `invalid` never changes
    and masking afterwards costs a full extra pass over the output.

    `dst` reuses a caller-owned output buffer. Worth threading through for
    video: the panorama is tens of megabytes, and allocating (and
    first-touching) a fresh one per lens per frame costs more than the
    resampling itself."""
    warped = cv2.remap(image, uv, None, cv2.INTER_LINEAR, dst=dst)
    if invalid is not None:
        warped[invalid] = 0
    return warped


def _odd(value: float) -> int:
    value = max(3, round(value))
    return value if value % 2 == 1 else value + 1


# --------------------------------------------------------------------------
# Anti-aliasing
# --------------------------------------------------------------------------
#
# Each lens' calibrated resolution (~3888px across the fisheye circle) is
# angularly much denser than the panorama needs (output_width px for the
# full 360°, of which one lens only ever fills about half) — one output
# pixel step skips over multiple source pixels almost everywhere, not just
# near a lens' own distorted edge (that's the opposite of where the
# distortion is worst; it's driven by the resolution mismatch, not lens
# geometry). cv2.remap's INTER_LINEAR only interpolates the 4 nearest source
# pixels at each sample point — it does not average away the source pixels
# a downsampled output pixel's footprint actually covers, so this aliases:
# fine texture and sensor grain fold into visible low-frequency noise
# instead of being properly blurred out. (cv2.remap also accepts
# cv2.INTER_AREA without erroring, but verified empirically — not just by
# reading the docs — that it produces bit-identical output to INTER_LINEAR;
# OpenCV's real area-averaging resampler is only wired up in cv2.resize.)
# StitchConfig.output_width's default is chosen so this ratio is ~1 at each
# lens' own optical axis (see the module's derivation notes); the functions
# below handle the rest of the (still downsampled, especially near the
# seams) footprint by pre-blurring the source before remapping it, sized to
# how much each output pixel is actually downsampling by.


def local_sampling_scale(uv: np.ndarray) -> np.ndarray:
    """How many source pixels (per axis) map into one output-pixel step, at
    every pixel of an equirectangular grid — from the local Jacobian of
    `uv` (a camera model's world2cam output) with respect to the output
    grid. 1.0 means the source and output are equally dense there; >1 means
    the source is being downsampled and is at risk of the aliasing
    described above without a matching pre-filter; <1 means it's being
    (safely) upsampled.

    Uses the Jacobian's determinant (area scale factor, then square-rooted
    to a linear one) rather than comparing axes separately, so it's exact
    regardless of how the projection locally rotates or shears — relevant
    here since a leveled panorama's iso-columns/rows in output space don't
    generally line up with a fisheye's own radial/tangential axes.
    """
    u, v = uv[..., 0], uv[..., 1]
    du_dcol, du_drow = np.gradient(u, axis=1), np.gradient(u, axis=0)
    dv_dcol, dv_drow = np.gradient(v, axis=1), np.gradient(v, axis=0)
    area_scale = np.abs(du_dcol * dv_drow - du_drow * dv_dcol)
    return np.sqrt(np.maximum(area_scale, 0.0)).astype(np.float32)


# Below this sigma a Gaussian's first off-centre tap is under float32's
# epsilon relative to the centre one (exp(-1/(2*sigma**2)) < 1e-7), so the
# discrete kernel _odd() can build for it IS the identity — and since _odd
# floors at 3, asking cv2 for it would still cost a full-frame pass to copy
# the image unchanged. Doubles as antialias_ladder's floor: a rung this
# blurry is indistinguishable from the unblurred one, so there is no point
# building anything below it.
_MIN_EFFECTIVE_BLUR_SIGMA = 0.176


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < _MIN_EFFECTIVE_BLUR_SIGMA:
        return image
    ksize = _odd(sigma * 3)
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def required_blur_sigma(scale_map: np.ndarray) -> np.ndarray:
    """The Gaussian sigma (in SOURCE pixels) each output pixel needs before
    remapping, from its local sampling scale s (local_sampling_scale).

    sigma = sqrt(s**2 - 1) / 2, i.e. the blur that has to be ADDED to what
    the source already carries. A source pixel is not a point sample; it
    already integrates roughly one pixel of the scene, and Gaussian widths
    add in quadrature, so reaching an effective width of s from a starting
    width of 1 costs sqrt(s**2 - 1), not s and not s - 1. It is zero at
    s <= 1 — where the panorama is no coarser than the source, nothing is
    being discarded and any blur is pure loss — which matters because this
    pipeline's default resolutions put each lens' optical axis at exactly
    s ~ 1, with only the seam regions reaching s ~ 2.

    Calibrated, not assumed. Measured against a 2x-supersampled render of
    the same frame (the operational definition of alias-free) over the
    seam region, RMS error against that reference was 0.92/255 with no
    pre-filter at all, 0.90 for a naive s - 1, 0.87 for a plain s/2, and
    0.81 here — and the excess high-frequency energy that aliasing shows
    up as fell from +47% (unfiltered) to +13%. s/2 drives that excess to
    zero but scores worse overall: it over-blurs, trading real detail for
    the last of the aliasing."""
    return np.sqrt(np.maximum(scale_map**2 - 1.0, 0.0)) / 2.0


def antialias_ladder(sigma_max: float) -> tuple[float, ...]:
    """Blur sigmas for the pre-filter's mip stack, spanning what this
    geometry actually needs: `sigma_max` and repeated halvings of it, down
    to where a Gaussian stops doing anything (_MIN_EFFECTIVE_BLUR_SIGMA),
    plus an unblurred rung at 0.

    Derived rather than hardcoded, which is what keeps the stack small
    where it can be. At the native output widths this pipeline defaults to,
    the whole required range is 0..0.9 and this returns four rungs; the
    old fixed (0, 1, 2, 4, 8) ladder built five, of which the top two could
    not draw any weight at all and the rest bracketed the range so coarsely
    that a pixel needing sigma 0.49 was served 65% of a sigma=1.0 blur.
    Ratio 2 between rungs is what the interpolation below is accurate at;
    measured against a true per-pixel Gaussian, halving the ratio again
    does not improve it, and doubling it doubles the error. Rung count
    therefore grows as log2 of the downsampling, so an aggressive --width
    costs more stack, which is exactly when it is needed.

    A geometry that never downsamples gives sigma_max = 0 and a single
    rung, i.e. no pre-filtering at all — correct, and free."""
    rungs: list[float] = []
    sigma = float(sigma_max)
    while sigma >= _MIN_EFFECTIVE_BLUR_SIGMA:
        rungs.append(sigma)
        sigma /= 2.0
    return (0.0,) + tuple(reversed(rungs))


def antialias_mip_level(
    scale_map: np.ndarray, ladder: Sequence[float]
) -> np.ndarray:
    """Each output pixel's position within `ladder`, as a continuous rung
    index for apply_anti_alias_prefilter's blend.

    Within a rung interval the position is linear in VARIANCE, not in
    sigma: convolving two Gaussians adds their variances, so that is the
    axis along which blending two pre-blurred images approximates a blur
    in between. Measured against a true per-pixel Gaussian over a real
    frame, it halves the error of interpolating on sigma.

    Frame-invariant (scale_map is fixed for a shape), so this is built once
    per shape rather than per frame."""
    sigma_required = required_blur_sigma(scale_map)
    rungs = np.asarray(ladder, dtype=np.float32)
    if rungs.size < 2:
        return np.zeros_like(sigma_required)
    variance = rungs**2
    lower = np.clip(np.searchsorted(rungs, sigma_required, side="right") - 1, 0, rungs.size - 2)
    span = variance[lower + 1] - variance[lower]
    frac = np.clip((sigma_required**2 - variance[lower]) / span, 0.0, 1.0)
    return (lower + frac).astype(np.float32)


def antialias_blend_weights(
    level: np.ndarray, ladder: Sequence[float]
) -> list[np.ndarray]:
    """The per-rung mixing weights apply_anti_alias_prefilter's telescoping
    lerp needs, one (H, W, 1) float32 map for each rung above the first.

    Split out and cached because `level` never changes for a given shape,
    so neither do these — and computing them inside the per-frame loop cost
    a clip plus a broadcast over the whole panorama, per rung, per lens,
    per frame."""
    return [
        np.clip(level - (i - 1), 0.0, 1.0)[..., None].astype(np.float32)
        for i in range(1, len(ladder))
    ]


def apply_anti_alias_prefilter(
    source: np.ndarray,
    uv_baked: np.ndarray,
    ladder: Sequence[float],
    weights: Sequence[np.ndarray],
    dst: Optional[np.ndarray] = None,
) -> np.ndarray:
    """remap_to_equirectangular, but pre-filtering `source` first so the
    downsampling described above the module docstring gets properly
    averaged away instead of aliasing.

    The pre-filter each output pixel needs varies across the panorama —
    none at all at a lens' optical axis, the most near the seams — so this
    is the standard mip approach: build `ladder`'s progressively blurrier
    versions of `source`, remap each through the same grid, and blend them
    per output pixel by that pixel's own `level` (antialias_mip_level).
    That is correct everywhere rather than one compromise for the whole
    frame, which is what a single global blur can never be: at this
    pipeline's own default resolutions, 62% of pixels need essentially no
    blur while the seam regions need the most, so any one sigma either
    softens the sharpest part of the lens or leaves the seams aliasing.

    Both `ladder` (antialias_ladder) and `weights` (antialias_blend_weights,
    one (H, W, 1) map per rung above the first) are frame-invariant, so
    Gear360Stitcher derives them once per shape. A single-rung ladder means
    this geometry never downsamples anywhere and nothing needs filtering,
    which collapses to a plain remap — and is also how "off" is expressed.

    `dst` is a reusable output buffer, used only on that single-rung path;
    the blend below builds its own result.
    """
    if len(ladder) < 2:
        return remap_to_equirectangular(source, uv_baked, dst=dst)

    # Telescoping lerp: for a pixel whose level lies in [k, k+1] every rung
    # below k has already been fully replaced, rung k+1 mixes in by the
    # fractional part, and everything above draws zero weight. Written as
    # result += frac * (rung - result) and evaluated in place against one
    # scratch array, rather than the arithmetically identical
    # (1 - frac) * result + frac * rung, which allocates four
    # panorama-sized temporaries per rung and made this blend cost more
    # than all the blurring and remapping it is combining.
    result = remap_to_equirectangular(_gaussian_blur(source, ladder[0]), uv_baked)
    if result.base is not None or result is source:
        result = result.copy()  # never accumulate into a caller's buffer
    scratch = np.empty_like(result)
    for i in range(1, len(ladder)):
        rung = remap_to_equirectangular(_gaussian_blur(source, ladder[i]), uv_baked)
        np.subtract(rung, result, out=scratch)
        np.multiply(scratch, weights[i - 1], out=scratch)
        np.add(result, scratch, out=result)
    return result


# --------------------------------------------------------------------------
# Color matching
# --------------------------------------------------------------------------


def _fit_affine_shrunk(s: np.ndarray, t: np.ndarray) -> tuple[float, float, float]:
    """OLS-fit t ~ a*s + b, shrunk towards the identity map (a=1, b=0) in
    proportion to how much of it is actually explained (R²). Without this,
    momentarily decorrelated samples (see fit_color_correction) still get
    an unconstrained "best" fit: minimizing squared error collapses the
    gain and uses the offset to hit the target mean, which doesn't fail
    loudly but instead flattens the source towards one washed-out value.
    Returns (a, b, r2)."""
    s_mean, t_mean = s.mean(), t.mean()
    var_s, var_t = s.var(), t.var()
    cov_st = np.mean((s - s_mean) * (t - t_mean))

    if var_s < 1e-8 or var_t < 1e-8:
        a_ols, r2 = 1.0, 0.0
    else:
        a_ols = cov_st / var_s
        r2 = np.clip(cov_st**2 / (var_s * var_t), 0.0, 1.0)
    b_ols = t_mean - a_ols * s_mean

    return r2 * a_ols + (1 - r2) * 1.0, r2 * b_ols + (1 - r2) * 0.0, r2


# ITU-R BT.601 luma weights, in BGR order (OpenCV's native channel order),
# for the luminance-agreement gate in fit_color_correction.
_LUMA_WEIGHTS_BGR = np.array([0.114, 0.587, 0.299])

# Below this many surviving pixels, the luminance-filtered fit is too noisy
# to trust; fall back to fitting the whole (unfiltered) region instead.
_MIN_AGREEMENT_PIXELS = 200


def _masked_blur_samples(
    image: np.ndarray,
    mask_f3: np.ndarray,
    ksize: tuple[int, int],
    mask: np.ndarray,
    weight_at_mask: np.ndarray,
) -> np.ndarray:
    """The mask-aware box blur of `image`, returned as an (N, 3) array of
    just the values under `mask`.

    Mask-aware because a plain cv2.blur over the region's bounding area
    would pull each valid pixel's blurred value towards whatever's outside
    the mask (typically black, from beyond a lens' own calibrated vignette
    circle, or content from the other lens' non-overlapping territory),
    biasing exactly the pixels fit_color_correction is trying to compare.
    The standard fix is blur(image*mask) / blur(mask), and that is what
    this computes — only with everything depending on the mask alone
    (`mask_f3`, and `weight_at_mask`, the divisor pre-gathered) supplied by
    ColorMatchRegion, since the mask is fixed for the life of a stitcher
    and re-deriving those per frame was most of the cost here.

    Dividing after the gather rather than before is what makes the divisor
    one-dimensional: the quotient is only ever read under the mask, so
    computing it across the whole window spent three quarters of those
    divides on values nothing looks at. Same operands per surviving pixel,
    so the result is unchanged bit for bit."""
    weighted = cv2.blur(cv2.multiply(image, mask_f3), ksize)
    return weighted[mask] / weight_at_mask[:, None]


@dataclass
class ColorMatchRegion:
    """The overlap region fit_color_correction samples, pre-chewed into the
    parts that don't change from frame to frame.

    The region (build_overlap_region) is the two lenses' overlapping
    coverage: two narrow vertical bands flanking the seams, ~6% of the
    panorama's pixels in ~30% of its columns. Blurring whole frames to read
    that 6% was the single most expensive stage of a video frame, so the
    work is cropped to one padded window per band.

    The crop is exact, not an approximation. A box blur reads at most
    `ksize // 2` pixels away, so padding each band's bounding box by that
    much means every pixel actually sampled — all of which are inside the
    region, hence at least the pad away from its own window's edge — sees
    precisely the neighborhood it would have seen in the full frame. Where
    a window is clamped to the frame's own border the pixels there were
    already reading cv2's border extrapolation, and still do, from the same
    border. Verified against the whole-frame version: identical fitted
    coefficients, to the bit.
    """

    ksize: tuple[int, int]
    windows: list[tuple[int, int, int, int]]  # (y0, y1, x0, x1) per band
    masks: list[np.ndarray]  # bool region, cropped to each window
    masks_f3: list[np.ndarray]  # the same as 3-channel float32, for cv2.multiply
    weights_at_mask: list[np.ndarray]  # blur(mask) under the mask — the divisor

    @classmethod
    def build(cls, region: np.ndarray, blur_ksize: int) -> ColorMatchRegion:
        ksize = (_odd(blur_ksize), _odd(blur_ksize))
        pad = ksize[0] // 2
        height, width = region.shape
        windows = []
        for x0, x1 in _column_runs(region.any(axis=0)):
            rows = np.where(region[:, x0 : x1 + 1].any(axis=1))[0]
            windows.append(
                (
                    max(int(rows[0]) - pad, 0),
                    min(int(rows[-1]) + pad + 1, height),
                    max(x0 - pad, 0),
                    min(x1 + pad + 1, width),
                )
            )
        # Degenerate regions (one spanning the whole frame, or so many
        # scattered bands that the padded windows overlap into more work
        # than the frame itself) get the whole frame as a single window:
        # same result, and never slower than what this replaces.
        area = sum((y1 - y0) * (x1 - x0) for y0, y1, x0, x1 in windows)
        if not windows or area >= region.size:
            windows = [(0, height, 0, width)]

        masks, masks_f3, weights_at_mask = [], [], []
        for y0, y1, x0, x1 in windows:
            mask = region[y0:y1, x0:x1]
            mask_f = mask.astype(np.float32)
            masks.append(mask)
            masks_f3.append(cv2.merge([mask_f] * 3))
            weights_at_mask.append(np.maximum(cv2.blur(mask_f, ksize), 1e-6)[mask])
        return cls(
            ksize=ksize,
            windows=windows,
            masks=masks,
            masks_f3=masks_f3,
            weights_at_mask=weights_at_mask,
        )

    def sample(self, image: np.ndarray) -> np.ndarray:
        """`image`'s masked-blurred values at every region pixel, as an
        (N, 3) float64 array. Bands are concatenated in column order; the
        fit downstream is a set statistic, so the order only has to be the
        same for both images, which it is."""
        return np.concatenate(
            [
                _masked_blur_samples(
                    image[y0:y1, x0:x1], mask_f3, self.ksize, mask, weight
                )
                for (y0, y1, x0, x1), mask, mask_f3, weight in zip(
                    self.windows, self.masks, self.masks_f3, self.weights_at_mask
                )
            ]
        ).astype(np.float64)


def _column_runs(occupied: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (first, last) index pairs for each run of consecutive True
    values — the disjoint column bands of the overlap region."""
    (columns,) = np.nonzero(occupied)
    if columns.size == 0:
        return []
    breaks = np.nonzero(np.diff(columns) > 1)[0]
    starts = np.concatenate([columns[:1], columns[breaks + 1]])
    ends = np.concatenate([columns[breaks], columns[-1:]])
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def fit_color_correction(
    source: np.ndarray,
    target: np.ndarray,
    region: ColorMatchRegion,
    luminance_threshold: float,
) -> np.ndarray:
    """Fit a per-channel affine map (a, b) pulling `source` towards
    `target`'s colors over `region`, using low-pass filtered samples (see
    _masked_blur_samples) so small misregistration between the two lenses
    dominate the fit. Returns shape (3, 2): one (a, b) row per channel.

    `region` is a ColorMatchRegion over the real overlap mask
    (build_overlap_region, from the two lenses' calibrated, transformed
    vignette circles); `source`/`target` are the full equirectangular
    frames, which it crops to the bands it actually samples. The mask stays
    per-pixel throughout — the crop is only about where the blur runs, and
    every pixel outside the curved region is still excluded from both the
    blur's average and the fit.

    Before fitting, pixels are dropped where the two lenses' *luminance*
    disagrees by more than `luminance_threshold` — the overlap region still
    admits plenty of pixels that aren't actually the same real-world point
    (near the poles, and wherever close-range parallax shifts it between
    the two lenses), and a strong luminance mismatch is the cheapest
    available sign of that. The same luminance-based mask is used for all
    three channels, since it's a per-pixel judgment of whether the two
    lenses are looking at the same thing, not a per-channel one. The
    remaining fit is still R²-shrunk towards the identity map (see
    _fit_affine_shrunk) as a second line of defense against whatever
    mismatch the threshold doesn't catch.
    """
    blurred_source = region.sample(source)
    blurred_target = region.sample(target)

    luma_source = blurred_source @ _LUMA_WEIGHTS_BGR
    luma_target = blurred_target @ _LUMA_WEIGHTS_BGR
    agree = np.abs(luma_target - luma_source) <= luminance_threshold
    if agree.sum() < _MIN_AGREEMENT_PIXELS:
        agree = slice(None)

    coeffs = np.empty((3, 2), dtype=np.float32)
    for c in range(3):
        a, b, _ = _fit_affine_shrunk(blurred_source[agree, c], blurred_target[agree, c])
        coeffs[c] = (a, b)
    return coeffs


def apply_color_correction(source: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Apply the per-channel affine map from `fit_color_correction`."""
    # np.asarray(..., dtype=float32) is a no-op view when source is already
    # float32 (always, in this pipeline); .astype() alone would copy either
    # way, so chaining both on a full-frame array was a silent double copy.
    corrected = np.asarray(source, dtype=np.float32).copy()
    for c in range(3):
        a, b = coeffs[c]
        corrected[..., c] = a * corrected[..., c] + b
    np.clip(corrected, 0, 1, out=corrected)
    return corrected


# --------------------------------------------------------------------------
# Blend masks
# --------------------------------------------------------------------------


def build_simple_mask(width: int, height: int) -> np.ndarray:
    """Hard column cut: the front lens covers the central half of the
    equirectangular image, the back lens covers the two wrap-around edges."""
    mask = np.ones((height, width), dtype=np.float32)
    mask[:, width // 4 : 3 * width // 4] = 0.0
    return mask


def _seam_ownership_masks(
    front_valid: np.ndarray,
    back_valid: np.ndarray,
    cx: int,
    front_is_right: bool,
    contest_half: int,
    anchor_half: int,
    pole_rows: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Build the (front, back) candidate masks cv2's seam finder needs for
    one seam crop: each camera's real valid footprint inside a contested
    strip of `2*contest_half` columns around `cx`, surrounded by an
    `anchor_half`-wide margin on each side where ownership is pinned to the
    nominal front/back split. That margin gives the graph cut genuine
    exclusive-ownership anchors to cut a boundary between — handing it a
    fully-overlapping mask everywhere (no anchors) makes the split
    degenerate (all-or-nothing). Pole rows are pinned too: the equirectangular
    projection smears the zenith/nadir across every column, so both cameras'
    "valid" footprints spuriously cover the whole crop there.
    """
    crop_half = contest_half + anchor_half
    l, r = cx - crop_half, cx + crop_half
    height = front_valid.shape[0]

    col_idx = np.arange(2 * crop_half)
    front_default_row = (
        (col_idx >= crop_half) if front_is_right else (col_idx < crop_half)
    )
    front_default_row = front_default_row.astype(np.uint8) * 255
    back_default_row = 255 - front_default_row

    front_mask = np.broadcast_to(front_default_row, (height, 2 * crop_half)).copy()
    back_mask = np.broadcast_to(back_default_row, (height, 2 * crop_half)).copy()

    contested = np.abs(col_idx - crop_half) <= contest_half
    front_mask[:, contested] = front_valid[:, l:r][:, contested].astype(np.uint8) * 255
    back_mask[:, contested] = back_valid[:, l:r][:, contested].astype(np.uint8) * 255

    if pole_rows:
        front_mask[:pole_rows] = front_default_row
        front_mask[height - pole_rows :] = front_default_row
        back_mask[:pole_rows] = back_default_row
        back_mask[height - pole_rows :] = back_default_row

    return front_mask, back_mask, l, r


def _make_seam_finder(algorithm: str):
    if algorithm == "graphcut":
        # OpenCV Stitcher's own default: a true 2D min-cut, not restricted to
        # one column per row. ~3x the cost of "dp" below on our band crops.
        return cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
    if algorithm == "dp":
        # Per-row dynamic programming with proper backtracking (unlike the
        # hand-rolled version this replaced, which took an independent
        # per-row argmin and could jump between unrelated paths). Cheaper
        # than graphcut; a reasonable default for video where speed matters.
        return cv2.detail_DpSeamFinder("COLOR_GRAD")
    raise ValueError(f"unknown seam algorithm {algorithm!r}")


def build_seam_mask(
    front: np.ndarray,
    back: np.ndarray,
    front_valid: np.ndarray,
    back_valid: np.ndarray,
    seam_band_frac: float,
    pole_crop_frac: float,
    algorithm: str = "graphcut",
    base_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Find a minimum-cost cut (color + gradient discontinuity, via one of
    OpenCV's stitching-pipeline seam finders) within a band around each of
    the two front/back boundaries, and build a mask that keeps the front
    lens between the two seams. `base_mask` reuses an already-built
    build_simple_mask (frame-invariant — see Gear360Stitcher.simple_mask)
    instead of building an identical one from scratch on every call."""
    height, width = front.shape[:2]
    contest_half = max(1, round(width * seam_band_frac / 2))
    pole_rows = round(height * pole_crop_frac)
    finder = _make_seam_finder(algorithm)

    mask = (
        base_mask.copy() if base_mask is not None else build_simple_mask(width, height)
    )
    for cx, front_is_right in ((width // 4, True), (3 * width // 4, False)):
        front_mask, back_mask, l, r = _seam_ownership_masks(
            front_valid,
            back_valid,
            cx,
            front_is_right,
            contest_half,
            contest_half,
            pole_rows,
        )
        front_crop = np.ascontiguousarray(front[:, l:r])
        back_crop = np.ascontiguousarray(back[:, l:r])
        result = finder.find(
            [back_crop, front_crop], [(0, 0), (0, 0)], [back_mask, front_mask]
        )
        mask[:, l:r] = (result[0].get() > 127).astype(np.float32)
    return mask


def build_overlap_region(
    front_circle: np.ndarray,
    back_circle: np.ndarray,
    pole_crop_frac: float,
    color_match_row_frac: float,
) -> np.ndarray:
    """Pixels where both lenses have valid (non-vignette-edge) coverage,
    away from the poles, restricted to the upper `color_match_row_frac` of
    the image (the lower half tends to show the tripod/rig at close range,
    where the two lenses disagree far more than a global affine can fix)."""
    height = front_circle.shape[0]
    overlap = np.logical_and(front_circle, back_circle)
    pole_crop = round(height * pole_crop_frac)
    if pole_crop:
        overlap[:pole_crop] = False
        overlap[-pole_crop:] = False
    overlap[round(height * color_match_row_frac) :] = False
    return overlap


# --------------------------------------------------------------------------
# Multiband (Laplacian pyramid) blending
# --------------------------------------------------------------------------


def build_gaussian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    pyramid = [image.astype(np.float32)]
    for _ in range(1, levels):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def build_laplacian_pyramid(gaussian_pyramid: list[np.ndarray]) -> list[np.ndarray]:
    pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        up = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
        pyramid.append(cv2.subtract(gaussian_pyramid[i], up))
    pyramid.append(gaussian_pyramid[-1])
    return pyramid


def blend_pyramids(
    laplacian_a: list[np.ndarray],
    laplacian_b: list[np.ndarray],
    gaussian_mask: list[np.ndarray],
) -> list[np.ndarray]:
    blended = []
    for lap_a, lap_b, mask in zip(laplacian_a, laplacian_b, gaussian_mask):
        if lap_a.ndim == 3 and mask.ndim == 2:
            mask = cv2.merge([mask, mask, mask])
        blended.append(lap_a * mask + lap_b * (1.0 - mask))
    return blended


def reconstruct_image(blended_pyramid: list[np.ndarray]) -> np.ndarray:
    current = blended_pyramid[-1]
    for level in reversed(blended_pyramid[:-1]):
        size = (level.shape[1], level.shape[0])
        current = cv2.add(cv2.pyrUp(current, dstsize=size), level)
    return np.clip(current, 0, 1)


def multiband_blend(
    image_a: np.ndarray, image_b: np.ndarray, mask: np.ndarray, levels: int
) -> np.ndarray:
    """Blend `image_a` (where mask==1) and `image_b` (where mask==0) across
    scales so the seam doesn't show as a hard edge or a soft ghost."""
    lap_a = build_laplacian_pyramid(build_gaussian_pyramid(image_a, levels))
    lap_b = build_laplacian_pyramid(build_gaussian_pyramid(image_b, levels))
    gp_mask = build_gaussian_pyramid(mask, levels)
    return reconstruct_image(blend_pyramids(lap_a, lap_b, gp_mask))


def _floor_to_multiple(value: int, multiple: int) -> int:
    return (value // multiple) * multiple


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def blend_crop_windows(
    width: int, blend_levels: int, mask_mode: str, seam_band_frac: float
) -> list[tuple[int, int]]:
    """Column ranges around the two front/back boundaries within which the
    blend mask can be non-flat. Both build_simple_mask and build_seam_mask
    are hard 0/1 splits at width//4 and 3*width//4 that only wiggle (seam
    mode) within a band of `2*contest_half` columns around each — the
    center never moves, so these ranges depend only on the output
    resolution and config, never on frame content. Padded generously for
    the multiband pyramid to fully settle back to flat before each
    window's own edge, and snapped to a multiple of 2**blend_levels so a
    crop can be pyrDown/pyrUp'd cleanly on its own.

    Outside these ranges the mask is exactly 0 or 1, so multiband-blending
    there reduces (up to floating point) to a plain copy of one source
    image — see composite_panorama, which is why this is worth carving out
    at all: blending the *whole* panorama over and over for a mask that's
    flat almost everywhere wastes the bulk of the work.
    """
    multiple = 2**blend_levels
    contest_half = (
        max(1, round(width * seam_band_frac / 2)) if mask_mode == "seam" else 0
    )
    half_span = 2 * contest_half + 4 * multiple
    windows = []
    for cx in (width // 4, 3 * width // 4):
        l = max(_floor_to_multiple(cx - half_span, multiple), 0)
        r = min(_ceil_to_multiple(cx + half_span, multiple), width)
        windows.append((l, r))
    return windows


def composite_panorama(
    front: np.ndarray,
    back: np.ndarray,
    mask: np.ndarray,
    blend_levels: int,
    crop_windows: list[tuple[int, int]],
    front_has_data: Optional[np.ndarray] = None,
    back_has_data: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Equivalent to `multiband_blend(back, front, mask, blend_levels)` over
    the whole image, but only actually blends inside `crop_windows` and
    plain-copies elsewhere, where the mask is already known to be flat.

    `front_has_data`/`back_has_data` (each 1.0 where that lens actually has
    real, trustworthy pixels and 0.0 where it doesn't, e.g. Gear360Stitcher's
    per-frame circle validity — a plain boolean mask also works, as a hard
    cut) are optional but should always be passed once the panorama can be
    tilted (see StitchConfig.level_roll_deg/level_pitch_deg): multiband
    blending smooths the mask transition over a neighborhood, and if the
    *true* front/back boundary has drifted so far from the nominal seam that
    one side of a crop window has no real data at all, blending would mix in
    that invalid (zero/black) content right at the edge instead of falling
    back cleanly to the lens that does have data — this reapplies that
    fallback after each crop's blend.

    Passing a *feathered* float mask (see Gear360Stitcher's
    front_has_data_soft/back_has_data_soft) instead of a hard boolean one
    eases that fallback in over a narrow margin rather than snapping to it
    at a single pixel: a hard cut is invisible when it falls back to a lens
    with a matching color, but auto-leveling can push it somewhere the two
    lenses' independently-corrected colors genuinely differ (see the module
    docstring above rotation_from_gravity), where a one-pixel snap between
    them is a visible step. Feathering can't fix that color difference, only
    spread it over a few pixels instead of one.
    """
    # np.where already returns float32 here since front/back are (always,
    # in this pipeline); copy=False makes the dtype cast a no-op instead of
    # a second full-panorama-sized copy in the common case.
    result = np.where(mask[..., None] >= 0.5, back, front).astype(
        np.float32, copy=False
    )
    for l, r in crop_windows:
        blended = multiband_blend(
            back[:, l:r], front[:, l:r], mask[:, l:r], blend_levels
        )
        if front_has_data is not None:
            w = front_has_data[:, l:r].astype(np.float32)[..., None]
            blended = w * blended + (1 - w) * back[:, l:r]
        if back_has_data is not None:
            w = back_has_data[:, l:r].astype(np.float32)[..., None]
            blended = w * blended + (1 - w) * front[:, l:r]
        result[:, l:r] = blended
    return result


def _pad_wrap_x(img: np.ndarray, pad: int) -> np.ndarray:
    """Pad the equirectangular panorama's left/right edges with the OPPOSITE
    edge's own pixels (they're the same yaw seam, not a real boundary) so a
    filter's window near x=0 or x=W-1 sees genuine neighboring content
    instead of an artificial edge. Only x wraps — top/bottom (y) are real
    poles, not a seam, and keep OpenCV's default border handling."""
    return np.concatenate([img[:, -pad:], img, img[:, :pad]], axis=1)


def _unpad_x(img: np.ndarray, pad: int) -> np.ndarray:
    return img[:, pad:-pad]


# CLAHE's tile grid for enhance_panorama, across the whole panorama: four
# tiles horizontally is one per lens-territory quarter, and the pair is kept
# here because the wrap-padding below has to reproduce the same tile width.
# See enhance_panorama's docstring for why this is coarse, and for why the
# grid must span the whole panorama rather than each lens separately.
_CLAHE_TILES_X = 4
_CLAHE_TILES_Y = 4


def enhance_panorama(
    panorama: np.ndarray,
    denoise_strength: float,
    contrast_clip: float,
    sharpen_amount: float,
) -> np.ndarray:
    """Denoise, locally boost contrast, and sharpen the final composited
    panorama. Runs once, after blending/color-correction, on the finished
    image — no stage here ever sees the two lenses separately, so none can
    reintroduce a seam the blend has already removed (see the CLAHE note
    below, which is where that went wrong once).

    Each stage is skippable via its own <= 0 value. Order matters: denoise
    first (bilateral, so it smooths flat noisy regions like walls/ceilings
    while preserving edges) and sharpen last (so it isn't just re-amplifying
    the noise the first step removed); CLAHE runs on L only, between the
    two, since it changes local contrast, not detail.

    CLAHE's tile grid is deliberately coarse (4x4 across the whole
    panorama, i.e. one tile per lens-territory quarter) and its clip low:
    this photo's real sensor noise survives the bilateral pass as a faint
    per-pixel residual that's invisible on its own, but a finer grid or
    higher clip renormalizes each small tile's histogram enough to turn
    that residual into visible blotches on flat surfaces (ceilings, walls)
    — i.e. CLAHE can put noise back in rather than only lifting contrast.

    All three passes wrap x, because x=0 and x=W-1 are the SAME yaw seam
    (the 360 wrap-around), not unrelated edges: plain OpenCV border
    handling (reflect) would fabricate each side's "outside" content from
    its own interior instead of the other side's real pixels. Denoise and
    sharpen use _pad_wrap_x/_unpad_x, whose reach is a few tens of pixels;
    CLAHE gets the same treatment at tile scale — one extra tile of
    wrapped padding on each side (_CLAHE_TILES_X + 2 tiles over a
    1.5x-wide image, so every tile keeps its original W/4 width), which
    puts the padded grid's tile centers at the true wrapped neighbors of
    the real edge tiles.

    Do NOT split CLAHE per lens. An earlier version ran it on the front
    and back territories separately, out of a concern that a tile
    straddling a front/back boundary would interpolate across genuinely
    different-lens content. By this point in the pipeline that premise is
    wrong: color matching and multiband blending have already made the
    panorama continuous at both boundaries (measured: 0.4/255 of
    column-to-column change there before post-processing). What the split
    did instead was hand the two territories independent equalization
    curves that meet at a hard column cut, turning that continuous image
    into a 14/255 step at x=W/4 and 3W/4 — which the sharpen pass then
    amplified to 22/255, by far the most visible artifact in the finished
    panorama. Running CLAHE across the whole (wrap-padded) image leaves
    0.9/255 there, and the broad tone gradient the split was trying to
    prevent is exactly what local contrast enhancement is supposed to do.
    """
    img8 = np.clip(panorama * 255.0, 0, 255).astype(np.uint8)
    w = img8.shape[1]

    if denoise_strength > 0:
        pad = 32  # comfortably past bilateralFilter's own (d=9, sigmaSpace=9) reach
        img8 = _unpad_x(
            cv2.bilateralFilter(
                _pad_wrap_x(img8, pad), d=9, sigmaColor=denoise_strength, sigmaSpace=9
            ),
            pad,
        )

    if contrast_clip > 0:
        # One extra tile of wrapped padding on each side, so the tile
        # whose curve gets interpolated into the pixels near x=0 is the
        # real content from x=W-1, and vice versa, instead of OpenCV
        # clamping to the edge tile's own curve across the 360 wrap. The
        # pad is exactly one tile wide, so every tile in the padded grid
        # still spans w/_CLAHE_TILES_X columns and the noise-blotch
        # trade-off above is unchanged.
        tile_w = w // _CLAHE_TILES_X
        l, a, b = cv2.split(cv2.cvtColor(img8, cv2.COLOR_BGR2LAB))
        clahe = cv2.createCLAHE(
            clipLimit=contrast_clip,
            tileGridSize=(_CLAHE_TILES_X + 2, _CLAHE_TILES_Y),
        )
        l = _unpad_x(clahe.apply(_pad_wrap_x(l, tile_w)), tile_w)
        img8 = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    if sharpen_amount > 0:
        pad = 16  # comfortably past the sharpen blur's own sigmaX=2.0 reach
        blurred = _unpad_x(
            cv2.GaussianBlur(_pad_wrap_x(img8, pad), (0, 0), sigmaX=2.0), pad
        )
        img8 = cv2.addWeighted(img8, 1 + sharpen_amount, blurred, -sharpen_amount, 0)

    return img8.astype(np.float32) / 255.0


# --------------------------------------------------------------------------
# Stitcher
# --------------------------------------------------------------------------


def _round_down_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, (value // multiple) * multiple)


@dataclass
class StitchConfig:
    # 6592 rather than a round "6K" number: derived, not chosen for looks —
    # each lens' own optical axis (its least-distorted, highest quality
    # point) is downsampled ~1.6x at the old 4096 default (measured via
    # local_sampling_scale on the calibrated rig), and that ratio scales as
    # 1/output_width, so 4096*1.6 ≈ 6554 is where a lens' own best pixels
    # stop being thrown away before anti-aliasing even gets involved (see
    # the module docstring above local_sampling_scale) — rounded up to 6592,
    # a multiple of 64 so output_height (half of this) is still a multiple
    # of 2**blend_levels=32 too, avoiding __post_init__'s rounding-down
    # warning. The rest of each lens' footprint is still downsampled more
    # than this (worst near the seams, ~2.9x at 4096, i.e. ~1.8x still
    # here) — anti_alias's per-pixel prefilter is what actually covers
    # that, not this number by itself. Recompute if the calibration (lens
    # FOV/resolution) changes meaningfully.
    output_width: int = 6592
    output_height: Optional[int] = None  # defaults to output_width // 2
    mask_mode: str = "seam"  # "seam" or "simple"
    blend_levels: int = 5
    # The front-back lens separation in METRES (the extrinsics file's
    # translation is unit-norm, so this sets its length outright -- see
    # load_extrinsics). Measured at 60mm +-10mm on the physical rig.
    baseline_scale: float = 0.06
    # Assumed scene distance in metres, used with baseline_scale to reproject
    # the back lens; only the baseline_scale/depth RATIO affects the output.
    #
    # Note that 1.0 is NOT a measured room distance -- it is the value that
    # empirically minimised seam artifacts, and it does so even for content
    # metres away, which no genuine depth can do (parallax correction scales
    # as 1/distance, so one setting cannot suit near and far content at
    # once). That scene-independence is the signature of a constant angular
    # offset between the lenses -- a residual error in the lenses' fitted
    # Double Sphere models -- rather than of parallax. This knob absorbs it
    # only by coincidence, so a value that minimises seam artifacts here is
    # not a scene distance and should not be read as one.
    depth: float = 1.0
    seam_band_frac: float = 0.02
    pole_crop_frac: float = 0.05
    color_match_row_frac: float = 0.8
    # fit_color_correction: pixels whose blurred front/back luminance
    # disagrees by more than this much (0-1 scale, same units as the float
    # image) are dropped before fitting. Two >=180deg-FOV lenses' nominal
    # "overlap" (both have vignette-circle coverage) includes plenty of
    # content that isn't actually the same real-world point -- near the
    # poles especially, and wherever close-range parallax shifts it between
    # the two lenses -- and letting that content pull the fit was observed
    # to bias it into a contrast-crushing, black-lifting map.
    color_match_luminance_threshold: float = 0.3
    vignette_shrink: float = 0.95
    # See apply_anti_alias_prefilter. "adaptive" pre-filters each lens per output pixel before remapping
    # (see apply_anti_alias_prefilter); "off" skips it. There used to be a
    # third, "fixed", applying one global blur sized from the median
    # sampling scale -- removed because it could not do the job at either
    # of this pipeline's default resolutions: the median there is ~1.06, so
    # the sigma it chose rounded to nothing and it was an exact synonym for
    # "off", while any sigma large enough to help the seams would have
    # softened the 62% of the frame that needs no blur at all.
    anti_alias: str = "adaptive"
    # Width (as a fraction of output_width) of the Gaussian feather applied
    # to the front/back coverage-validity fallback in composite_panorama,
    # instead of switching lenses at a single hard-edged pixel. Only matters
    # once auto-leveling can move that coverage boundary away from the
    # nominal seam, into territory where the two lenses' independently
    # color-corrected content can genuinely differ (see the module docstring
    # above rotation_from_gravity) — feathering can't remove that residual
    # color difference, only spread it over a narrow margin instead of
    # snapping to it in one pixel.
    validity_feather_frac: float = 0.005
    seam_algorithm: str = "graphcut"  # "graphcut" or "dp"
    # Recompute the seam from scratch every `seam_interval`-th stitched frame
    # (1 = every frame); in between, the mask actually used for blending
    # keeps easing towards the latest computed seam (see seam_smoothing)
    # rather than holding it rigidly, so raising this trades seam freshness
    # for speed without introducing a visible pop when it does update.
    seam_interval: int = 1
    # Exponential-smoothing factor applied every frame to ease the blend
    # mask towards the latest raw seam instead of snapping to it: 1.0 = no
    # smoothing (snap immediately, the natural choice for a single image);
    # smaller values trade responsiveness for less frame-to-frame seam
    # jitter in video, where a subject crossing the seam band can otherwise
    # flip the minimum-cost cut from one side to the other between frames.
    seam_smoothing: float = 0.3
    # Same idea as seam_smoothing, applied to the fitted color-correction
    # (a, b) coefficients instead of the blend mask: the affine fit is cheap
    # to redo every frame (unlike the seam), but its inputs are just as
    # content-dependent, so it can pump/flicker across frames without this.
    color_match_smoothing: float = 0.3
    # Manual auto-leveling override: rotate the panorama so its "up" matches
    # true gravity-up rather than however the camera was actually held/
    # mounted. Setting either enables leveling and skips detection entirely
    # (automatic detection from image content is a separate, later feature).
    level_roll_deg: Optional[float] = None
    level_pitch_deg: Optional[float] = None
    # Central-FOV patch used for line-based orientation estimation (not yet
    # wired up — currently only consumed by the --level-preview diagnostic).
    level_fov_deg: float = 60.0
    level_patch_size: int = 640
    level_projection: str = "rectilinear"  # or "cylindrical"
    # Automatic orientation detection from image content (estimate_level_rotation).
    # Ignored when level_roll_deg/level_pitch_deg are set — manual always wins.
    auto_level: str = "off"  # "off", "vertical", "horizon", or "auto" (try both)
    level_max_tilt_deg: float = 35.0
    # Final-panorama post-processing (see enhance_panorama): runs once on the
    # finished, already-blended image, so it's independent of anti_alias/
    # mask_mode/color-match above. Each strength is individually skippable at
    # <= 0. Costs one bilateral filter, one CLAHE pass, and one Gaussian blur
    # over the full panorama — fine for a photo, likely too slow per-frame
    # for video (see cli.py's post_process wiring).
    post_process: bool = True
    post_process_denoise: float = 50.0  # bilateral filter sigmaColor, 0-255 scale
    post_process_contrast: float = 1.0  # CLAHE clip limit on L* (see enhance_panorama)
    post_process_sharpen: float = 0.6  # unsharp mask amount

    def __post_init__(self):
        if not (0.0 < self.color_match_luminance_threshold <= 1.0):
            raise ValueError(
                "color_match_luminance_threshold must be in (0, 1], got "
                f"{self.color_match_luminance_threshold!r}"
            )
        if self.mask_mode not in ("seam", "simple"):
            raise ValueError(
                f"mask_mode must be 'seam' or 'simple', got {self.mask_mode!r}"
            )
        if self.anti_alias == "fixed":
            # Removed (see the field above); accepted as a synonym for what
            # it actually did at any resolution near native, so existing
            # commands and saved configs don't break on an error.
            logger.warning(
                "anti_alias='fixed' has been removed and is being treated as 'off'; "
                "it applied one global blur sized from the median sampling scale, "
                "which at native output widths rounded to no blur at all. Use "
                "'adaptive' for a correct per-pixel pre-filter."
            )
            self.anti_alias = "off"
        if self.anti_alias not in ("off", "adaptive"):
            raise ValueError(
                f"anti_alias must be 'off' or 'adaptive', got {self.anti_alias!r}"
            )
        if self.seam_algorithm not in ("graphcut", "dp"):
            raise ValueError(
                f"seam_algorithm must be 'graphcut' or 'dp', got {self.seam_algorithm!r}"
            )
        if not (0.0 < self.seam_smoothing <= 1.0):
            raise ValueError(
                f"seam_smoothing must be in (0, 1], got {self.seam_smoothing!r}"
            )
        if self.seam_interval < 1:
            raise ValueError(f"seam_interval must be >= 1, got {self.seam_interval!r}")
        if not (0.0 < self.color_match_smoothing <= 1.0):
            raise ValueError(
                f"color_match_smoothing must be in (0, 1], got {self.color_match_smoothing!r}"
            )
        if self.level_projection not in ("rectilinear", "cylindrical"):
            raise ValueError(
                f"level_projection must be 'rectilinear' or 'cylindrical', "
                f"got {self.level_projection!r}"
            )
        if not (0.0 < self.level_fov_deg < 120.0):
            raise ValueError(
                f"level_fov_deg must be in (0, 120), got {self.level_fov_deg!r}"
            )
        if self.level_patch_size < 64:
            raise ValueError(
                f"level_patch_size must be >= 64, got {self.level_patch_size!r}"
            )
        if self.auto_level not in ("off", "vertical", "horizon", "auto"):
            raise ValueError(
                f"auto_level must be 'off', 'vertical', 'horizon', or 'auto', "
                f"got {self.auto_level!r}"
            )
        if self.output_height is None:
            self.output_height = self.output_width // 2

        # Multiband blending halves the resolution `blend_levels` times, so
        # both dimensions must divide evenly or pyrUp/pyrDown round-trips
        # come back the wrong size.
        multiple = 2**self.blend_levels
        width = _round_down_to_multiple(self.output_width, multiple)
        height = _round_down_to_multiple(self.output_height, multiple)
        if (width, height) != (self.output_width, self.output_height):
            logger.warning(
                "Rounding output resolution from %dx%d down to %dx%d "
                "(must be a multiple of 2**blend_levels=%d)",
                self.output_width,
                self.output_height,
                width,
                height,
                multiple,
            )
            self.output_width, self.output_height = width, height


@dataclass
class VideoStitchConfig(StitchConfig):
    """StitchConfig with the defaults that should differ for video.

    Everything StitchConfig chooses is right for a single photo, which
    pays each cost exactly once. Video pays every one of them per frame,
    thousands of times, so a handful of them are worth re-deciding — and
    NONE of the re-decisions here lowers the output resolution, which is
    the one lever that would actually cost detail.

    Measured on a real Gear 360 clip (3840x1920 dual-fisheye) at this
    class's own output_width, on a 16-core machine, per frame:

        StitchConfig's defaults, applied to video      1202 ms
        ... with seam_interval=5 alone                   468 ms   2.6x
        this class's defaults                           309 ms   3.9x
        everything except the seam search                263 ms

    i.e. the seam search goes from 78% of the frame to 15% of it, and
    what remains is the remap-and-blend that actually builds the pixels.

    Decode (12 ms) and mp4v encode (26 ms) are noise next to those, which
    is why no codec or I/O default is overridden here.
    """

    # Gear 360 video frames are captured at roughly half the photo's
    # native per-lens resolution (measured: 3840x1920 dual-fisheye vs.
    # 7776x3888 for a photo) — halving StitchConfig.output_width follows
    # the same "don't throw away resolvable pixels, don't pay for pixels
    # that aren't there" reasoning as that default's own derivation (see
    # its comment), and was confirmed empirically: a real video frame
    # stitched at 2048/2688/3264/4096 stops getting visibly sharper at
    # 3264, matching this value. Rounded to a multiple of 64, like the
    # photo default, so output_height (half of this) is still a multiple
    # of 2**blend_levels=32.
    #
    # This costs ~1.6GB of resident memory per parallel worker (measured
    # peak RSS; see _estimate_worker_memory_mb), which is what caps
    # _auto_worker_count on a memory-constrained machine. That trade —
    # fewer workers, full per-frame quality — is deliberate, and the
    # reason none of the speed work below touches the resolution.
    output_width: int = 3264

    # A photo pays for the anti-alias pre-filter once, so it takes the
    # correct per-pixel one; video pays per frame, and at 3264 wide that
    # is +496 ms/frame (measured), roughly doubling the cost of
    # everything else in the frame put together.
    #
    # What video gives up: the seam regions (~20% of the frame, where the
    # fisheye is downsampled up to ~2x) keep about 47% more
    # high-frequency energy than a supersampled reference, and in motion
    # that excess is what shimmers along the seams rather than sitting
    # still. See required_blur_sigma for the measurements. Pass
    # --anti-alias adaptive to buy it back, especially for a moving
    # camera.
    #
    # Note this is NOT the split that used to live in
    # cli.py. That one gave video "fixed", a since-removed
    # mode whose one global blur was sized from the median sampling scale
    # — which at any width near native rounded to no blur at all, so
    # video was silently getting "off" while appearing to be filtered.
    # This makes that explicit rather than restoring it: the cost is now
    # a deliberate choice against a real alternative, not an accident of
    # a no-op default.
    anti_alias: str = "off"

    # Same reasoning: one bilateral filter, one CLAHE pass and one
    # Gaussian blur over the full panorama are fine as a one-off for a
    # photo, but +133 ms/frame (measured) for video.
    post_process: bool = False

    # Re-run the seam search every 5th frame instead of every frame.
    #
    # This is the single biggest speed default here, because the seam
    # search dominates the frame: 939 of the 1202 ms above, i.e. 78%, is
    # the graphcut. Amortising it over 5 frames takes the frame to 468 ms
    # for no loss in what the search itself computes.
    #
    # It cannot introduce a visible pop, because the mask actually used
    # for blending keeps easing towards the latest computed seam every
    # frame in between (see seam_smoothing) rather than holding it
    # rigidly. Verified rather than assumed: against a
    # seam-every-frame reference over 9 consecutive frames, no pixel
    # differed by more than 2/255 (mean |difference| 0.002/255).
    #
    # 5 is chosen against seam_smoothing = 0.3: the mask is ~97% of the
    # way to a new seam after 5 frames (1 - 0.7**5), so each recomputed
    # seam has essentially landed before the next one arrives. Raising
    # this much further starts leaving the mask chasing a stale cut when
    # the scene moves.
    seam_interval: int = 5

    # Per-row dynamic programming instead of graphcut's true 2D min-cut,
    # taking the frame from 468 ms to 309 ms — on top of seam_interval
    # above, since the two multiply: this is the difference between
    # re-running a cheap search every 5th frame and an expensive one.
    #
    # On quiet content the two are indistinguishable (measured over 9
    # consecutive frames: max difference 0.5/255, mean 0.002/255). Where
    # they can differ is the case a seam finder exists for — a subject
    # crossing the seam band — because a per-row DP picks each row's cut
    # independently of its neighbours and can wander where the graphcut
    # would hold a straight line through the object. Two things make that
    # affordable here and not for a photo: seam_smoothing eases the mask
    # over several frames, so a single bad row-path is attenuated rather
    # than shown, and no one inspects one video frame the way they
    # inspect a still. Pass --seam-algorithm graphcut for the stricter
    # search if a moving subject does smear at a seam.
    seam_algorithm: str = "dp"

    # Pipeline-level defaults for stitch_video_parallel. Intentionally
    # NOT dataclass fields (no annotation) — they parameterise how the
    # video is driven across processes, not how one frame is stitched, so
    # they must not ride along in the StitchConfig every worker receives
    # and the stitcher ignores. They live here so that everything worth
    # re-deciding for video is readable in one place.
    #
    # 0 workers = decide from the machine (see _auto_worker_count). One
    # frame's cv2 work already uses all cores internally but leaves most
    # of them idle most of the time — a single-process run of this
    # workload measured 217% CPU on a 16-core machine — so parallelism
    # across frames, not within a frame, is the main wall-clock lever.
    DEFAULT_WORKERS = 0

    # Frames each worker replays before its chunk so its temporal
    # smoothing has converged (see _stitch_video_chunk). 15 rather than
    # 30: both smoothing factors are 0.3, so the residual after n frames
    # is 0.7**n — under 1% by frame 13. The old 30 was ~2x more than the
    # smoothing needs, and it is charged once per worker, so it grows
    # with parallelism exactly when it is least affordable.
    DEFAULT_WARMUP_FRAMES = 15


class Gear360Stitcher:
    """Stitches one or many dual-fisheye frames from a fixed rig.

    Everything that depends only on calibration and the output resolution
    (the direction grid, each camera's world2cam projection, the vignette
    falloff maps, the overlap/circle masks) is computed once — either here
    or lazily on the first frame, whichever needs the input frame's shape —
    and reused across calls to `stitch`, which is what makes processing a
    video frame-by-frame practical.
    """

    def __init__(
        self,
        front: FisheyeCamera,
        back: FisheyeCamera,
        R_t: np.ndarray,
        config: StitchConfig,
    ):
        self.front = front
        self.back = back
        self.R_t = R_t
        self.config = config

        self.level_estimate: Optional[LevelEstimate] = None
        if config.level_roll_deg is not None or config.level_pitch_deg is not None:
            g = gravity_from_roll_pitch(
                config.level_roll_deg or 0.0, config.level_pitch_deg or 0.0
            )
            self.set_level_rotation(g)
        elif config.auto_level != "off":
            # Detection needs actual image content, which isn't known until
            # the first frame arrives — stitch() estimates it there (once;
            # see StitchConfig.auto_level) and calls set_level_rotation,
            # which finishes what this constructor would otherwise do here.
            self.level_rotation = None
            self.level_tilt_deg = None
        else:
            self.set_level_rotation(np.array([0.0, 1.0, 0.0]))

        # Everything below depends on the shape of the *input* fisheye
        # frame: world2cam needs intrinsics rescaled to match it (see
        # FisheyeCamera.scaled_to — the Gear 360 records video at a lower
        # resolution than its photos), and the vignette circle/falloff are
        # defined in that image's own pixel coordinates. None of it is known
        # until the first frame arrives, so _prepare_for_shapes fills it in
        # lazily and reuses it for every later frame of the same shape.
        self._prepared_shapes: Optional[tuple[tuple[int, int], tuple[int, int]]] = None
        self.uv_front: Optional[np.ndarray] = None
        self.valid_front: Optional[np.ndarray] = None
        self.invalid_front: Optional[np.ndarray] = None
        self.uv_back: Optional[np.ndarray] = None
        self.valid_back: Optional[np.ndarray] = None
        self.invalid_back: Optional[np.ndarray] = None
        # Exactly one of the (gain, buffer) pair and the falloff map is
        # populated per lens — see _prepare_for_shapes.
        self.front_gain: Optional[np.ndarray] = None
        self.back_gain: Optional[np.ndarray] = None
        self._front_corrected: Optional[np.ndarray] = None
        self._back_corrected: Optional[np.ndarray] = None
        self.front_falloff: Optional[np.ndarray] = None
        self.back_falloff: Optional[np.ndarray] = None
        self.overlap_region: Optional[np.ndarray] = None
        self.color_match_region: Optional[ColorMatchRegion] = None
        self.uv_front_baked: Optional[np.ndarray] = None
        self.uv_back_baked: Optional[np.ndarray] = None
        self.front_ladder: tuple[float, ...] = (0.0,)
        self.back_ladder: tuple[float, ...] = (0.0,)
        self.front_blend_weights: list[np.ndarray] = []
        self.back_blend_weights: list[np.ndarray] = []
        self._equi_front_buf: Optional[np.ndarray] = None
        self._equi_back_buf: Optional[np.ndarray] = None
        self.front_has_data: Optional[np.ndarray] = None
        self.back_has_data: Optional[np.ndarray] = None
        self.front_has_data_soft: Optional[np.ndarray] = None
        self.back_has_data_soft: Optional[np.ndarray] = None

        # Temporal state for the low-pass-filtered seam mask and color
        # correction (video only; each just equals the fresh value on the
        # first of a single stitch() call).
        self._raw_seam_mask: Optional[np.ndarray] = None
        self._smoothed_seam_mask: Optional[np.ndarray] = None
        self._smoothed_color_coeffs: Optional[np.ndarray] = None
        self._frame_index = 0

    def set_level_rotation(self, g: np.ndarray) -> None:
        """Adopt `g` (unit gravity-down, front-camera coords) as the
        panorama's leveling correction and (re)build the geometry that
        depends on it. Called from __init__ for level_roll/pitch_deg and
        the "off" case (both known before any frame is seen), and from
        stitch() once auto_level's image-content estimate is available.

        Public so video callers can set this explicitly before the first
        stitch() — see estimate_level_rotation_from_video: video needs the
        SAME rotation applied for every frame regardless of how many
        frames were sampled to estimate it, and stitch_video_parallel's
        workers each build their own Gear360Stitcher, so leaving it to
        each one's own lazy first-frame estimate would give slightly
        different (and independently noisy) leveling per --workers chunk.
        """
        self.level_rotation = rotation_from_gravity(g)
        # How far the front/back seam (a fixed plane in front-camera
        # coordinates) has rotated away from the output grid's static
        # width//4 / 3*width//4 columns — 0 for pure roll (its axis IS the
        # seam plane's normal, see the module docstring above
        # rotation_from_gravity) but grows with any pitch component, up to
        # 90° within `level_tilt_deg` of a pole. Widens the seam search band
        # and pole crop below so leveling doesn't uncover the black wedges
        # of front/back-less pixels that a fixed-column assumption produces
        # once the seam actually moves.
        self.level_tilt_deg = float(np.degrees(np.arccos(np.clip(g[1], -1.0, 1.0))))
        self._build_geometry()

    def _build_geometry(self) -> None:
        """(Re)builds everything that depends only on config + calibration +
        `self.level_rotation`, never on frame content: the direction grids,
        the blend crop windows, and the base simple mask. Called once from
        __init__; a future auto-leveling estimator that determines
        `level_rotation` from image content would call this again once the
        rotation is known, after resetting `_prepared_shapes` so
        _prepare_for_shapes recomputes uv/valid/falloff/overlap against the
        new (rotated) `directions_front`.
        """
        config = self.config
        width, height = config.output_width, config.output_height
        # A perfectly level grid, rotated into the front camera's own
        # (possibly tilted) frame — identity leaves this unchanged, so
        # leveling costs nothing when it's off.
        directions_level = equirectangular_directions(width, height)
        self.directions_front = directions_level @ self.level_rotation.T
        self.directions_back = transform_directions(
            self.directions_front, self.R_t, config.depth
        )

        # The front/back seam and the poles both need extra margin once the
        # panorama is tilted (see level_tilt_deg above). seam_band_frac
        # covers both sides of each nominal column, hence the factor of 2;
        # pole_crop_frac is a fraction of height on each end. margin_deg
        # covers estimation slack on top of the tilt itself.
        margin_deg = 2.0
        self.effective_seam_band_frac = max(
            config.seam_band_frac, 2 * (self.level_tilt_deg + margin_deg) / 360
        )
        self.effective_pole_crop_frac = max(
            config.pole_crop_frac, (self.level_tilt_deg + margin_deg) / 180
        )
        if self.effective_seam_band_frac > config.seam_band_frac:
            logger.info(
                "Leveling tilt %.1f°: widening seam band to %.1f%% of width and "
                "pole crop to %.1f%% of height (from %.1f%% / %.1f%%) to cover "
                "the tilted front/back boundary",
                self.level_tilt_deg,
                self.effective_seam_band_frac * 100,
                self.effective_pole_crop_frac * 100,
                config.seam_band_frac * 100,
                config.pole_crop_frac * 100,
            )

        self.blend_crop_windows = blend_crop_windows(
            width, config.blend_levels, config.mask_mode, self.effective_seam_band_frac
        )
        # The "ideal" hard column cut, ignoring where the lenses actually
        # have data — frame-invariant (config-only), so built once here.
        # _prepare_for_shapes clamps this by per-pixel validity once the
        # input shape is known (self.simple_mask), and build_seam_mask
        # starts its search from that clamped version so it doesn't spend
        # its band on a region where one camera has no data at all.
        self._ideal_simple_mask = build_simple_mask(width, height)

    @classmethod
    def from_calibration(
        cls,
        calib_front_dir: Path,
        calib_back_dir: Path,
        extrinsics_path: Path,
        config: StitchConfig,
    ) -> Gear360Stitcher:
        front = FisheyeCamera.load(calib_front_dir)
        back = FisheyeCamera.load(calib_back_dir)
        R_t = load_extrinsics(extrinsics_path, config.baseline_scale)
        return cls(front, back, R_t, config)

    def _prepare_for_shapes(
        self, front_shape: tuple[int, int], back_shape: tuple[int, int]
    ) -> None:
        if self._prepared_shapes == (front_shape, back_shape):
            return
        if self._prepared_shapes is not None:
            logger.warning(
                "Input frame shape changed (%s/%s -> %s/%s); recomputing "
                "calibration, vignette and overlap caches",
                *self._prepared_shapes,
                front_shape,
                back_shape,
            )
        front_cam = self.front.scaled_to(front_shape)
        back_cam = self.back.scaled_to(back_shape)

        self.uv_front, self.valid_front = front_cam.ds.world2cam(self.directions_front)
        self.uv_back, self.valid_back = back_cam.ds.world2cam(self.directions_back)
        self.invalid_front = np.logical_not(self.valid_front).squeeze()
        self.invalid_back = np.logical_not(self.valid_back).squeeze()
        # How much each lens is being downsampled by remap_to_equirectangular
        # (see the module docstring above local_sampling_scale) — depends
        # only on uv_front/uv_back, so it's cached here alongside them
        # rather than recomputed by apply_anti_alias_prefilter every frame.
        self.front_scale_map = local_sampling_scale(self.uv_front)
        self.back_scale_map = local_sampling_scale(self.uv_back)

        # Everything the per-frame projection needs that depends only on
        # the calibration and these shapes, rather than being re-derived
        # every frame: the invalid directions folded into the sampling map,
        # and the anti-alias stack below.
        self.uv_front_baked = bake_invalid_directions(self.uv_front, self.invalid_front)
        self.uv_back_baked = bake_invalid_directions(self.uv_back, self.invalid_back)
        # The anti-alias mip stack, sized to what this geometry actually
        # needs (antialias_ladder) and indexed per output pixel
        # (antialias_mip_level). Only the valid domain sets the ceiling:
        # invalid pixels remap to 0 from every rung alike, so whatever
        # sigma they would ask for is irrelevant and would only inflate the
        # stack. "off" is expressed as a one-rung ladder, which
        # apply_anti_alias_prefilter short-circuits to a plain remap.
        if self.config.anti_alias == "off":
            self.front_ladder = self.back_ladder = (0.0,)
        else:
            self.front_ladder = antialias_ladder(
                float(required_blur_sigma(self.front_scale_map)[~self.invalid_front].max())
            )
            self.back_ladder = antialias_ladder(
                float(required_blur_sigma(self.back_scale_map)[~self.invalid_back].max())
            )
        self.front_blend_weights = antialias_blend_weights(
            antialias_mip_level(self.front_scale_map, self.front_ladder),
            self.front_ladder,
        )
        self.back_blend_weights = antialias_blend_weights(
            antialias_mip_level(self.back_scale_map, self.back_ladder), self.back_ladder
        )
        logger.debug(
            "anti-alias ladders: front %s, back %s",
            [round(x, 3) for x in self.front_ladder],
            [round(x, 3) for x in self.back_ladder],
        )

        # Reusable remap outputs, for the same reason as the vignette
        # buffers below: a panorama is tens of megabytes, and allocating
        # two per frame costs more than the resampling. Only for the modes
        # that remap once per lens — "adaptive" builds a whole mip stack
        # and has nothing to reuse — and only within the scratch budget,
        # which at photo resolutions this deliberately exceeds.
        equi_bytes = 2 * 3 * 4 * self.config.output_width * self.config.output_height
        single_rung = len(self.front_ladder) < 2 and len(self.back_ladder) < 2
        if single_rung and equi_bytes <= _FRAME_SCRATCH_BUDGET_BYTES:
            shape = (self.config.output_height, self.config.output_width, 3)
            self._equi_front_buf = np.empty(shape, dtype=np.float32)
            self._equi_back_buf = np.empty(shape, dtype=np.float32)
        else:
            self._equi_front_buf = self._equi_back_buf = None

        # Vignette correction, in one of two forms holding the same model:
        #
        #   fast  — a 3-channel reciprocal plus a persistent output buffer,
        #           applied by one multi-threaded cv2 multiply per lens per
        #           frame (apply_vignette_gain). ~4x cheaper per frame, and
        #           it absorbs the caller's uint8 -> [0, 1] conversion too.
        #   plain — the 1-channel falloff map, divided out per frame by
        #           VignetteModel.correct, allocating as it goes.
        #
        # The fast form is what video wants, but it is resident where the
        # plain one is mostly transient, and the difference scales with the
        # INPUT frame, which is not the resolution anything else here is
        # budgeted against: the Gear 360's photos are far larger than its
        # video (3888x3888 half-frames vs 1920x1920), so the same code is
        # ~180MB for a video pair and ~730MB for a photo pair. Spending
        # that to save 44ms per frame is obviously right across thousands
        # of video frames and obviously wrong for the one frame of a photo,
        # which is all _FRAME_SCRATCH_BUDGET_BYTES encodes. Photos
        # take the plain path and are unaffected by any of this.
        fast_bytes = _vignette_fast_path_bytes(front_shape, back_shape)
        if fast_bytes <= _FRAME_SCRATCH_BUDGET_BYTES:
            self.front_gain = front_cam.vignette.gain_map(front_shape)
            self.back_gain = back_cam.vignette.gain_map(back_shape)
            self._front_corrected = np.empty((*front_shape, 3), dtype=np.float32)
            self._back_corrected = np.empty((*back_shape, 3), dtype=np.float32)
            self.front_falloff = self.back_falloff = None
        else:
            logger.debug(
                "vignette fast path needs %.0f MB for %dx%d/%dx%d half-frames, "
                "over the %.0f MB budget; using the per-frame divide instead",
                fast_bytes / 1e6,
                front_shape[1],
                front_shape[0],
                back_shape[1],
                back_shape[0],
                _FRAME_SCRATCH_BUDGET_BYTES / 1e6,
            )
            self.front_gain = self.back_gain = None
            self._front_corrected = self._back_corrected = None
            self.front_falloff = front_cam.vignette.falloff_map(front_shape)
            self.back_falloff = back_cam.vignette.falloff_map(back_shape)

        front_circle_disk = front_cam.vignette.disk_mask(
            front_shape, self.config.vignette_shrink
        )
        back_circle_disk = back_cam.vignette.disk_mask(
            back_shape, self.config.vignette_shrink
        )
        circle_front = remap_to_equirectangular(
            front_circle_disk, self.uv_front, self.invalid_front
        )
        circle_back = remap_to_equirectangular(
            back_circle_disk, self.uv_back, self.invalid_back
        )
        self.overlap_region = build_overlap_region(
            circle_front,
            circle_back,
            self.effective_pole_crop_frac,
            self.config.color_match_row_frac,
        )
        # Frame-invariant, so the crop windows and the mask's own blur are
        # derived once here rather than per frame — see ColorMatchRegion.
        self.color_match_region = ColorMatchRegion.build(
            self.overlap_region, blur_ksize=self.config.output_width // 80
        )

        # Clamp the base mask by where each lens actually has data: once
        # tilted (level_tilt_deg > 0), the nominal seam columns no longer
        # match the true front/back boundary, and near the poles that drift
        # can exceed 90° — without this, composite_panorama would render
        # solid black wherever the (fixed-column) mask picks a camera that
        # has no pixels there at all, instead of falling back to whichever
        # lens actually does. Cached (not just used inline) because
        # composite_panorama's crop-window blend needs the same validity to
        # avoid mixing invalid/black data into the blend near the true
        # boundary — see its docstring.
        self.front_has_data = circle_front.astype(bool)
        self.back_has_data = circle_back.astype(bool)
        # Feathered (0..1) versions of the same validity, for
        # composite_panorama's coverage fallback: a plain boolean mask
        # switches lenses at a single hard-edged pixel, which is only
        # invisible as long as that boundary stays inside the well
        # color-matched seam band. Auto-leveling can push it elsewhere (see
        # StitchConfig.validity_feather_frac); the Gaussian blur turns that
        # snap into a narrow gradient instead.
        feather_px = max(
            1, round(self.config.output_width * self.config.validity_feather_frac)
        )
        ksize = (_odd(feather_px), _odd(feather_px))
        self.front_has_data_soft = cv2.GaussianBlur(
            self.front_has_data.astype(np.float32), ksize, 0
        )
        self.back_has_data_soft = cv2.GaussianBlur(
            self.back_has_data.astype(np.float32), ksize, 0
        )
        self.simple_mask = self._ideal_simple_mask.copy()
        self.simple_mask[~self.front_has_data] = (
            1.0  # back owns where front has no data
        )
        self.simple_mask[~self.back_has_data] = 0.0  # front owns where back has no data

        self._prepared_shapes = (front_shape, back_shape)

    def _blend_mask(self, equi_front: np.ndarray, equi_back: np.ndarray) -> np.ndarray:
        if self.config.mask_mode != "seam":
            return self.simple_mask

        due = self._raw_seam_mask is None or (
            self._frame_index % self.config.seam_interval == 0
        )
        if due:
            self._raw_seam_mask = build_seam_mask(
                equi_front,
                equi_back,
                self.valid_front.squeeze(),
                self.valid_back.squeeze(),
                self.effective_seam_band_frac,
                self.effective_pole_crop_frac,
                algorithm=self.config.seam_algorithm,
                base_mask=self.simple_mask,
            )

        if self._smoothed_seam_mask is None:
            self._smoothed_seam_mask = self._raw_seam_mask.copy()
        else:
            alpha = self.config.seam_smoothing
            self._smoothed_seam_mask = (
                alpha * self._raw_seam_mask + (1 - alpha) * self._smoothed_seam_mask
            )
        return self._smoothed_seam_mask

    def _color_correction(
        self, equi_back: np.ndarray, equi_front: np.ndarray
    ) -> np.ndarray:
        """Color-correct `equi_back` towards `equi_front`: fit one per-
        channel affine map over the overlap region (see build_overlap_region
        — the two lenses' calibrated, transformed vignette circles) and
        apply it to the whole back frame. fit_color_correction low-pass
        filters both sides before comparing them and drops pixels whose
        luminance disagrees too much (see color_match_luminance_threshold)."""
        raw_coeffs = fit_color_correction(
            equi_back,
            equi_front,
            self.color_match_region,
            luminance_threshold=self.config.color_match_luminance_threshold,
        )
        if self._smoothed_color_coeffs is None:
            self._smoothed_color_coeffs = raw_coeffs
        else:
            alpha = self.config.color_match_smoothing
            self._smoothed_color_coeffs = (
                alpha * raw_coeffs + (1 - alpha) * self._smoothed_color_coeffs
            )
        return apply_color_correction(equi_back, self._smoothed_color_coeffs)

    @staticmethod
    def _correct_vignette(
        lens_image: np.ndarray,
        camera: FisheyeCamera,
        gain: Optional[np.ndarray],
        buffer: Optional[np.ndarray],
        falloff: Optional[np.ndarray],
    ) -> np.ndarray:
        """One lens' vignette correction, by whichever of the two forms
        _prepare_for_shapes decided this frame size could afford."""
        if gain is not None:
            return apply_vignette_gain(lens_image, gain, buffer)
        if lens_image.dtype == np.uint8:
            lens_image = lens_image.astype(np.float32) / 255.0
        return camera.vignette.correct(lens_image, falloff)

    def _render_lenses(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Both lenses projected onto the shared equirectangular grid, after
        vignette correction and anti-alias prefiltering but BEFORE color
        matching and blending.

        The two returned arrays may be reused scratch buffers (see
        _prepare_for_shapes), so they are only valid until the next call:
        read them, or copy what you need to keep. The caller consumes them
        within one frame, and the one value that outlives the frame --
        stitch()'s color-corrected back -- is a fresh array out of
        apply_color_correction rather than one of these.
        """
        front_im, back_im = split_dual_fisheye(image)
        if self.level_rotation is None:
            # First frame, auto_level != "off" and no manual override — see
            # __init__. Estimates once and freezes for the rest of this
            # stitcher's life, same as R_t: re-estimating per frame would
            # invalidate everything _prepare_for_shapes caches, and would
            # give stitch_video_parallel's chunks different leveling.
            front_cam_native = self.front.scaled_to(front_im.shape[:2])
            # render_center_patch wants float BGR in [0, 1]; `image` may be
            # a raw uint8 frame (see stitch). Converting here rather than up
            # front costs nothing: this branch runs at most once per
            # stitcher, and never at all for video, where
            # _resolve_level_config_for_video has already baked the
            # rotation into the config as an explicit roll/pitch.
            front_level_im = (
                front_im.astype(np.float32) / 255.0
                if front_im.dtype == np.uint8
                else front_im
            )
            _, self.level_estimate = estimate_level_rotation(
                front_cam_native, front_level_im, self.config
            )
            g = (
                self.level_estimate.g
                if self.level_estimate.ok
                else np.array([0.0, 1.0, 0.0])
            )
            self.set_level_rotation(g)
        self._prepare_for_shapes(front_im.shape[:2], back_im.shape[:2])

        front_im = self._correct_vignette(
            front_im, self.front, self.front_gain, self._front_corrected,
            self.front_falloff,
        )
        back_im = self._correct_vignette(
            back_im, self.back, self.back_gain, self._back_corrected,
            self.back_falloff,
        )

        equi_front = apply_anti_alias_prefilter(
            front_im,
            self.uv_front_baked,
            self.front_ladder,
            self.front_blend_weights,
            dst=self._equi_front_buf,
        )
        equi_back = apply_anti_alias_prefilter(
            back_im,
            self.uv_back_baked,
            self.back_ladder,
            self.back_blend_weights,
            dst=self._equi_back_buf,
        )
        return equi_front, equi_back

    def stitch(self, image: np.ndarray) -> np.ndarray:
        """`image` is a BGR dual-fisheye frame — either float32 in [0, 1]
        or the raw uint8 frame as cv2 read it. Returns a float32 BGR
        equirectangular panorama in [0, 1].

        Prefer passing uint8 when you have it (video does): the vignette
        correction has to touch every pixel anyway, so it absorbs the /255
        conversion for free instead of the caller paying for a separate
        full-frame pass — see apply_vignette_gain.

        Frames from the same rig are expected to share resolution; calling
        this repeatedly (video) is what amortizes the one-time setup work
        and lets the seam mask and color correction be temporally smoothed
        across calls."""
        equi_front, equi_back = self._render_lenses(image)

        equi_back = self._color_correction(equi_back, equi_front)

        mask = self._blend_mask(equi_front, equi_back)
        panorama = composite_panorama(
            equi_front,
            equi_back,
            mask,
            self.config.blend_levels,
            self.blend_crop_windows,
            self.front_has_data_soft,
            self.back_has_data_soft,
        )
        if self.config.post_process:
            panorama = enhance_panorama(
                panorama,
                self.config.post_process_denoise,
                self.config.post_process_contrast,
                self.config.post_process_sharpen,
            )
        self._frame_index += 1
        return panorama


def load_image(path: Path) -> np.ndarray:
    """Read a dual-fisheye frame as float32 BGR in [0, 1]."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image.astype(np.float32) / 255.0


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.clip(image * 255, 0, 255).astype(np.uint8))


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _has_audio_stream(path: Path) -> bool:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


@functools.lru_cache(maxsize=None)
def _find_ffmpeg(encoder: str) -> str:
    """First ffmpeg on PATH that has `encoder`. Not necessarily the first ffmpeg:
    a conda env's build may lack libx264 while the system one has it."""
    seen = set()
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        exe = shutil.which("ffmpeg", path=directory)
        if exe is None or os.path.realpath(exe) in seen:
            continue
        seen.add(os.path.realpath(exe))
        try:
            out = subprocess.run(
                [exe, "-hide_banner", "-encoders"], capture_output=True, text=True
            ).stdout
        except OSError:
            continue
        if f" {encoder} " in out:
            return exe
    raise RuntimeError(
        f"no ffmpeg with the {encoder!r} encoder on PATH (see --codec); "
        "video output is encoded by piping frames into ffmpeg"
    )


def _mux_audio(silent_video: Path, audio_source: Path, output_path: Path) -> None:
    """Copy `silent_video`'s (video-only) stream and `audio_source`'s audio
    stream into `output_path`, without re-encoding either."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(silent_video), "-i", str(audio_source),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy",
            "-shortest", str(output_path),
        ],
        check=True,
    )


class _FfmpegVideoWriter:
    """Encodes uint8 BGR frames by piping them raw into an ffmpeg process, so
    each frame is compressed exactly once, straight to the final codec.

    (cv2.VideoWriter is no use here: it can only write MPEG-4 Part 2 at a
    profile/level that phones and VLC refuse at this resolution, and no
    audio — which would force a second, lossy re-encode afterwards.)
    With `audio_source`, its audio stream is copied into the output as well."""

    def __init__(
        self, path: Path, fps: float, width: int, height: int, codec: str = "libx264",
        crf: int = 18, threads: int = 0, audio_source: Optional[Path] = None,
    ):
        cmd = [
            _find_ffmpeg(codec), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-r", repr(fps), "-i", "-",
        ]
        if audio_source is not None:
            cmd += ["-i", str(audio_source), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
        cmd += ["-c:v", codec, "-crf", str(crf), "-pix_fmt", "yuv420p"]
        if codec == "libx264":
            cmd += ["-preset", "medium"]
        if "265" in codec:
            cmd += ["-tag:v", "hvc1"]  # the tag Apple/Android players expect for HEVC
        if threads > 0:
            cmd += ["-threads", str(threads)]
        if audio_source is not None:
            cmd.append("-shortest")
        # +faststart puts moov first: players can start before the download ends.
        cmd += ["-movflags", "+faststart", str(path)]
        self._cmd = cmd
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).data)
        except BrokenPipeError:
            raise RuntimeError(
                f"ffmpeg exited early ({self._proc.wait()}): {' '.join(self._cmd)}"
            ) from None

    def release(self) -> None:
        self._proc.stdin.close()
        if self._proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed ({self._proc.returncode}): {' '.join(self._cmd)}")


def stitch_video(
    stitcher: Gear360Stitcher,
    input_path: Path,
    output_path: Path,
    codec: str = "libx264",
    crf: int = 18,
    max_frames: Optional[int] = None,
    progress_interval: int = 30,
    copy_audio: bool = True,
) -> int:
    """Stitch every frame of a dual-fisheye video into an equirectangular
    video, reusing `stitcher`'s cached geometry/vignette/overlap across
    frames. Returns the number of frames written.

    Frames are piped into ffmpeg (see _FfmpegVideoWriter), which encodes them
    once and, when `copy_audio` is set, copies `input_path`'s audio stream into
    the same file."""
    audio_source = None
    if copy_audio:
        if _has_audio_stream(input_path):
            audio_source = input_path
        else:
            logger.info("%s has no audio stream; nothing to copy", input_path)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    frame_index = 0
    start_time = time.time()
    try:
        while max_frames is None or frame_index < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            panorama = stitcher.stitch(frame)  # uint8: see stitch()
            out_frame = np.clip(panorama * 255, 0, 255).astype(np.uint8)

            if writer is None:
                h, w = out_frame.shape[:2]
                writer = _FfmpegVideoWriter(
                    output_path, fps, w, h, codec, crf, audio_source=audio_source
                )
            writer.write(out_frame)
            frame_index += 1

            if progress_interval and frame_index % progress_interval == 0:
                elapsed = time.time() - start_time
                logger.info(
                    "frame %d/%s (%.2fs/frame)",
                    frame_index,
                    total_frames if total_frames else "?",
                    elapsed / frame_index,
                )
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    return frame_index


def _concat_videos(segment_paths: list[Path], output_path: Path) -> None:
    """Concatenate video files with no re-encoding (they must already share
    a codec/resolution/fps — true of segments this module writes itself)."""
    concat_list = output_path.with_name(f".{output_path.stem}.concat.txt")
    concat_list.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in segment_paths) + "\n"
    )
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c",
                "copy",
                str(output_path),
            ],
            check=True,
        )
    finally:
        concat_list.unlink(missing_ok=True)


def _stitch_video_chunk(
    input_path: str,
    calib_front_dir: str,
    calib_back_dir: str,
    extrinsics_path: str,
    config: StitchConfig,
    codec: str,
    crf: int,
    start_frame: int,
    end_frame: int,
    warmup_frames: int,
    output_path: str,
    chunk_index: int,
    progress_interval: int = 30,
    cv2_threads: int = 1,
) -> int:
    """Worker entry point for stitch_video_parallel: stitches frames
    [start_frame, end_frame) of `input_path` into its own output segment at
    `output_path`. Replays `warmup_frames` frames immediately before
    start_frame without writing them, purely so this chunk's own temporal
    seam/color-match smoothing (see StitchConfig.seam_smoothing /
    color_match_smoothing) has already converged by the time its real
    output begins — otherwise the mask/color-match would visibly snap
    towards its steady state right at the seam between two chunks, instead
    of the gradual easing seen everywhere else in the video.

    Must be a plain module-level function (not a closure or method) so
    ProcessPoolExecutor can pickle it to send to the worker process; the
    stitcher itself is built fresh here rather than passed in for the same
    reason (its cv2 objects aren't guaranteed picklable).

    `cv2_threads` caps OpenCV's own internal thread pool for this process.
    Each worker process otherwise defaults to spinning up one cv2 thread
    per core, so N worker processes oversubscribe the machine N-fold —
    measured on real footage, that made 3 worker processes come out no
    faster than 1, purely from context-switch/cache-thrashing overhead
    across ~3x too many threads. cv2's own threading only bought ~14% on
    this workload's array sizes anyway (measured separately), so capping
    it hard in favor of process-level parallelism is the right trade.
    """
    cv2.setNumThreads(max(1, cv2_threads))
    stitcher = Gear360Stitcher.from_calibration(
        Path(calib_front_dir), Path(calib_back_dir), Path(extrinsics_path), config
    )
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    warmup_start = max(0, start_frame - warmup_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)

    writer = None
    written = 0
    try:
        for frame_idx in range(warmup_start, end_frame):
            ok, frame = cap.read()
            if not ok:
                break
            panorama = stitcher.stitch(frame)  # uint8: see stitch()
            if frame_idx < start_frame:
                continue  # warm-up only: let temporal smoothing converge, don't write

            out_frame = np.clip(panorama * 255, 0, 255).astype(np.uint8)
            if writer is None:
                h, w = out_frame.shape[:2]
                writer = _FfmpegVideoWriter(
                    Path(output_path), fps, w, h, codec, crf, threads=cv2_threads
                )
            writer.write(out_frame)
            written += 1
            if progress_interval and written % progress_interval == 0:
                logger.info(
                    "chunk %d: %d/%d frames",
                    chunk_index,
                    written,
                    end_frame - start_frame,
                )
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    return written


def _available_memory_mb() -> Optional[float]:
    """Best-effort available system memory in MB, or None if it can't be
    determined (non-Linux, unusual /proc, etc) — callers should skip
    memory-based capping in that case rather than guess."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _estimate_worker_memory_mb(config: StitchConfig) -> float:
    """Rough estimate of one stitch_video_parallel worker's peak RSS: each
    worker independently holds its own full direction grid / world2cam /
    vignette / overlap cache plus per-frame working arrays (see
    Gear360Stitcher), so this scales the ~1.9GB measured at 4096x2048 by
    output pixel count, plus a fixed baseline that doesn't shrink with
    resolution. 4096x2048 is just the reference point that measurement was
    taken at, not either current default (see StitchConfig.output_width and
    the CLI's DEFAULT_VIDEO_WIDTH, both of which have since
    moved independently of it). Deliberately conservative: underestimating
    risks the OOM this exists to prevent, which is far worse than starting
    with fewer workers than the machine could actually have handled."""
    baseline_mb = 300.0
    measured_mb_at_reference = 1900.0
    reference_pixels = 4096 * 2048
    pixels = config.output_width * config.output_height
    # The two per-frame scratch caches (see _prepare_for_shapes): the
    # vignette one scales with the INPUT frame, so it cannot ride on the
    # output-pixel term above, and while the remap buffers do scale with
    # the output, they are capped rather than proportional. Both are
    # charged at their full budget — a hard cap on what either can
    # allocate, and the conservative direction.
    scratch_mb = 2 * _FRAME_SCRATCH_BUDGET_BYTES / 1e6
    return (
        baseline_mb
        + scratch_mb
        + (measured_mb_at_reference - baseline_mb) * (pixels / reference_pixels)
    )


def _auto_worker_count(config: StitchConfig) -> int:
    """How many chunk workers this machine should run: one per core,
    capped by what memory allows (see _estimate_worker_memory_mb).

    The cap is the binding constraint at video resolution, not the core
    count — a worker measured ~1.6GB resident at VideoStitchConfig's
    output_width, so a 16GB machine lands around 5-6 workers however many
    cores it has. stitch_video_parallel applies the same memory cap again
    to whatever it is handed, so an explicit --workers stays safe too;
    this only decides what to ask for when nothing was asked.
    """
    cores = os.cpu_count() or 1
    available_mb = _available_memory_mb()
    if available_mb is None:
        # No reading (non-Linux, unusual /proc): stitch_video_parallel's
        # own cap can't protect us either, since it reads the same number,
        # so don't fan out to the core count on a machine whose memory we
        # cannot see — a worker costs ~1.6GB at video resolution, and 16
        # of those would OOM anything short of a workstation. 4 keeps most
        # of the win (the memory-safe count lands at 3-6 on the machines
        # we can measure) with no risk of an unseen OOM. An explicit
        # --workers still overrides this.
        return min(cores, 4)
    per_worker_mb = _estimate_worker_memory_mb(config)
    return max(1, min(cores, int((available_mb * 0.8) // per_worker_mb)))


def _resolve_level_config_for_video(
    input_path: Path,
    calib_front_dir: Path,
    config: StitchConfig,
    sample_frames: int = 5,
) -> StitchConfig:
    """If `config.auto_level` requests detection and no manual roll/pitch
    override is already set, estimate the leveling rotation once (from
    several sampled frames — see estimate_level_rotation_from_video) and
    bake the result into the config as an explicit level_roll_deg/
    level_pitch_deg (auto_level="off"). Otherwise returns `config` as-is.

    This is what lets stitch_video and every stitch_video_parallel worker
    apply exactly the same rotation without passing anything beyond
    StitchConfig itself: each worker builds its own Gear360Stitcher, so
    leaving auto_level enabled would have each one independently detect
    from only its own chunk's first (post-warmup) frame — inconsistent
    leveling at every chunk boundary, instead of one decision, made from
    the best evidence available, applied uniformly to the whole video."""
    if config.level_roll_deg is not None or config.level_pitch_deg is not None:
        return config
    if config.auto_level == "off":
        return config
    front_cam = FisheyeCamera.load(calib_front_dir)
    _, estimate = estimate_level_rotation_from_video(
        input_path, front_cam, config, sample_frames
    )
    if estimate.ok:
        return dataclass_replace(
            config,
            level_roll_deg=estimate.roll_deg,
            level_pitch_deg=estimate.pitch_deg,
            auto_level="off",
        )
    # Already logged why; force auto_level="off" so workers don't each
    # redundantly repeat (and separately log) the same failed detection.
    return dataclass_replace(config, auto_level="off")


def stitch_video_parallel(
    input_path: Path,
    output_path: Path,
    calib_front_dir: Path,
    calib_back_dir: Path,
    extrinsics_path: Path,
    config: StitchConfig,
    workers: int = VideoStitchConfig.DEFAULT_WORKERS,
    codec: str = "libx264",
    crf: int = 18,
    max_frames: Optional[int] = None,
    warmup_frames: int = VideoStitchConfig.DEFAULT_WARMUP_FRAMES,
    copy_audio: bool = True,
    progress_interval: int = 30,
    level_sample_frames: int = 5,
) -> int:
    """Stitch a video using `workers` processes, each handling its own
    contiguous chunk of frames (see _stitch_video_chunk) — the low-hanging
    parallelism here is across frames, not within one frame's cv2 calls
    (those already use all available cores internally via OpenCV's own
    thread pool, but that alone leaves most cores idle most of the time
    since only one frame's moderate-sized arrays are ever in flight).
    Segments are concatenated with no re-encoding and the original audio
    remuxed in via ffmpeg, so both require ffmpeg — this falls back to the
    single-process `stitch_video` if it's unavailable or workers == 1.

    `workers=0` (the default) sizes the pool to the machine via
    _auto_worker_count."""
    config = _resolve_level_config_for_video(
        input_path, calib_front_dir, config, level_sample_frames
    )
    if workers <= 0:
        workers = _auto_worker_count(config)
        logger.info(
            "Using %d worker process(es) (auto: %d cores, ~%.0fMB estimated per "
            "worker at %dx%d); pass --workers to override",
            workers,
            os.cpu_count() or 1,
            _estimate_worker_memory_mb(config),
            config.output_width,
            config.output_height,
        )
    if workers <= 1 or not _ffmpeg_available():
        if workers > 1:
            logger.warning(
                "ffmpeg/ffprobe not found on PATH; falling back to a single process"
            )
        stitcher = Gear360Stitcher.from_calibration(
            calib_front_dir, calib_back_dir, extrinsics_path, config
        )
        return stitch_video(
            stitcher,
            input_path,
            output_path,
            codec=codec,
            crf=crf,
            max_frames=max_frames,
            copy_audio=copy_audio,
            progress_interval=progress_interval,
        )

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames <= 0:
        raise RuntimeError(
            f"Could not determine a frame count for {input_path}; can't "
            "split it into chunks for parallel processing (try --workers 1)"
        )
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    workers = max(1, min(workers, total_frames))

    available_mb = _available_memory_mb()
    if available_mb is not None:
        per_worker_mb = _estimate_worker_memory_mb(config)
        # Leave headroom for the main process, OS, and anything else running.
        safe_workers = max(1, int((available_mb * 0.8) // per_worker_mb))
        if safe_workers < workers:
            logger.warning(
                "Reducing --workers from %d to %d: each worker holds its own full "
                "geometry/vignette cache plus per-frame arrays, estimated ~%.0fMB "
                "at this resolution, against ~%.0fMB available memory. A lower "
                "--width leaves room for more workers.",
                workers,
                safe_workers,
                per_worker_mb,
                available_mb,
            )
            workers = safe_workers

    boundaries = np.linspace(0, total_frames, workers + 1, dtype=int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    segment_paths = [
        output_path.with_name(f".{output_path.stem}.part{i}{output_path.suffix}")
        for i in range(workers)
    ]

    mux_audio = copy_audio and _has_audio_stream(input_path)
    concat_target = (
        output_path.with_name(f".{output_path.stem}.silent{output_path.suffix}")
        if mux_audio
        else output_path
    )

    cv2_threads = max(1, (os.cpu_count() or workers) // workers)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _stitch_video_chunk,
                str(input_path),
                str(calib_front_dir),
                str(calib_back_dir),
                str(extrinsics_path),
                config,
                codec,
                crf,
                int(boundaries[i]),
                int(boundaries[i + 1]),
                warmup_frames,
                str(segment_paths[i]),
                i,
                progress_interval,
                cv2_threads,
            )
            for i in range(workers)
        ]
        counts = [f.result() for f in futures]

    try:
        _concat_videos(segment_paths, concat_target)
    finally:
        for p in segment_paths:
            p.unlink(missing_ok=True)

    if mux_audio:
        try:
            _mux_audio(concat_target, input_path, output_path)
        finally:
            concat_target.unlink(missing_ok=True)

    return sum(counts)
