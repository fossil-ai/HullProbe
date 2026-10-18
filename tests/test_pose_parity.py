"""Parity guard: hullprobe's pose must match the reference's exactly.

`pca_from_mask` is the single source of alignment. If it drifts from the reference,
every hull/superstructure swap lands in a different frame than the bow/stern
work and cross-repo comparisons become meaningless. These tests pin it down.
"""

import os
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from hullprobe.pose import (
    longitudinal_coord,
    pca_from_mask,
    transverse_coord,
)

REFERENCE_SRC = Path(
    os.environ.get("REFERENCE_POSE_SRC", "___absent___")
)


def _golden_pca(mask):
    """Independent transcription of the reference pca_from_mask (the reference)."""
    ys, xs = np.where(mask)
    if len(xs) < 60:
        return None
    points = np.column_stack((xs, ys)).astype(np.float64)
    centroid = points.mean(axis=0)
    covariance = np.cov(points - centroid, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    axis = vectors[:, order[0]]
    if axis[0] < 0:
        axis = -axis
    angle = math.degrees(math.atan2(axis[1], axis[0]))
    aspect = math.sqrt(max(values[0], 1e-9) / max(values[1], 1e-9))
    return axis, centroid, float(angle), float(aspect)


def _random_vessel_masks(count=40, size=180, seed=0):
    rng = np.random.default_rng(seed)
    masks = []
    yy, xx = np.indices((size, size))
    for _ in range(count):
        cx, cy = rng.uniform(60, 120, size=2)
        a, b = rng.uniform(40, 80), rng.uniform(8, 22)  # elongated
        angle = rng.uniform(-40, 40)
        th = math.radians(angle)
        xr = (xx - cx) * math.cos(th) + (yy - cy) * math.sin(th)
        yr = -(xx - cx) * math.sin(th) + (yy - cy) * math.cos(th)
        mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
        if mask.sum() >= 60:
            masks.append(mask)
    return masks


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pca_matches_golden(seed):
    for mask in _random_vessel_masks(seed=seed):
        axis, centroid, angle, aspect = pca_from_mask(mask)
        gaxis, gcentroid, gangle, gaspect = _golden_pca(mask)
        np.testing.assert_array_equal(axis, gaxis)
        np.testing.assert_array_equal(centroid, gcentroid)
        assert angle == gangle
        assert aspect == gaspect


def test_pca_matches_reference_source_if_available():
    """Cross-check against the actual reference file when it's importable."""
    if not REFERENCE_SRC.exists():
        pytest.skip("reference source not present")
    spec = importlib.util.spec_from_file_location("_reference_pose", REFERENCE_SRC)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # pulls transformers via segmenter import
    except Exception as exc:  # heavy optional deps unavailable
        pytest.skip(f"cannot import reference source: {exc}")
    for mask in _random_vessel_masks(seed=7):
        ours = pca_from_mask(mask)
        theirs = module.pca_from_mask(mask)
        np.testing.assert_array_equal(ours[0], theirs[0])
        np.testing.assert_array_equal(ours[1], theirs[1])
        assert ours[2] == theirs[2]
        assert ours[3] == theirs[3]


def test_longitudinal_matches_regions_formula():
    """The projection must equal the reference regions.complete_region_masks:
    L = (x-cx)cos(theta) + (y-cy)sin(theta)."""
    xs = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    ys = np.array([5.0, 15.0, 25.0], dtype=np.float32)
    angle, cx, cy = 12.5, 15.0, 12.0
    theta = math.radians(angle)
    expected = (xs - cx) * math.cos(theta) + (ys - cy) * math.sin(theta)
    np.testing.assert_allclose(longitudinal_coord(xs, ys, angle, cx, cy), expected)


def test_transverse_is_perpendicular_to_longitudinal():
    """Longitudinal and transverse axes are orthogonal at every pixel."""
    rng = np.random.default_rng(3)
    xs = rng.uniform(0, 200, size=50).astype(np.float32)
    ys = rng.uniform(0, 200, size=50).astype(np.float32)
    angle, cx, cy = -18.0, 90.0, 95.0
    lon = longitudinal_coord(xs, ys, angle, cx, cy)
    tra = transverse_coord(xs, ys, angle, cx, cy)
    # Centered coords projected on orthonormal axes -> zero correlation direction:
    # reconstruct centered vector length is preserved (isometry).
    centered_sq = (xs - cx) ** 2 + (ys - cy) ** 2
    np.testing.assert_allclose(lon ** 2 + tra ** 2, centered_sq, rtol=1e-5)


def test_horizontal_vessel_axes_reduce_to_image_axes():
    """At angle 0, longitudinal = x-cx and transverse = y-cy."""
    xs = np.array([0.0, 50.0, 100.0], dtype=np.float32)
    ys = np.array([0.0, 30.0, 60.0], dtype=np.float32)
    cx, cy = 50.0, 30.0
    np.testing.assert_allclose(longitudinal_coord(xs, ys, 0.0, cx, cy), xs - cx, atol=1e-6)
    np.testing.assert_allclose(transverse_coord(xs, ys, 0.0, cx, cy), ys - cy, atol=1e-6)
