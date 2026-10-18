"""Vessel pose estimation and canonical-frame PCA.

`pca_from_mask` and `longitudinal_split` are ported **verbatim** from a reference maritime pose implementation so that hullprobe registers every vessel into *exactly* the
same canonical hull frame used for the bow / center / stern decomposition. The two implementations
must agree bit-for-bit on `pca_angle_deg`, centroid, and the longitudinal
projection; `tests/test_pose_parity.py` guards this. Do not "improve" these
functions — alignment parity depends on them being unchanged.

The one addition is the **transverse** coordinate (perpendicular to the PCA
major axis). The major axis is the longitudinal bow<->stern direction; its
perpendicular is the below-deck (hull) <-> above-deck (superstructure)
direction. Same pose, new cut — this is the axis the hull/superstructure swap
operates on.
"""

from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------------------- #
# VERBATIM from the reference implementation build_complete_hull_regions.py — do not modify.
# --------------------------------------------------------------------------- #
def pca_from_mask(mask):
    """Return (axis, centroid, angle_deg, aspect) for a boolean vessel mask."""
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
    # Fix the eigenvector sign: low projection is image-left/front and high is
    # image-right/rear, matching the existing region convention.
    if axis[0] < 0:
        axis = -axis
    angle = math.degrees(math.atan2(axis[1], axis[0]))
    aspect = math.sqrt(max(values[0], 1e-9) / max(values[1], 1e-9))
    return axis, centroid, float(angle), float(aspect)


def longitudinal_split(mask, axis, centroid):
    """Exactly partition a vessel mask into (front, center, rear) thirds along
    the PCA major axis. Verbatim from the reference implementation `split_complete`."""
    ys, xs = np.where(mask)
    projection = (xs - centroid[0]) * axis[0] + (ys - centroid[1]) * axis[1]
    t0, t1 = float(projection.min()), float(projection.max())
    if t1 <= t0:
        return None
    q1 = t0 + (t1 - t0) / 3.0
    q2 = t0 + 2.0 * (t1 - t0) / 3.0
    assignments = (projection > q1).astype(np.uint8) + (projection > q2).astype(np.uint8)
    masks = []
    for region_index in range(3):
        region = np.zeros(mask.shape, dtype=bool)
        keep = assignments == region_index
        region[ys[keep], xs[keep]] = True
        masks.append(region)
    if any(region.sum() < 20 for region in masks):
        return None
    union = masks[0] | masks[1] | masks[2]
    overlap = sum(int(region.sum()) for region in masks) - int(union.sum())
    if not np.array_equal(union, mask) or overlap != 0:
        raise AssertionError("longitudinal regions must exactly partition the vessel")
    return masks, {"t0": t0, "q1": q1, "q2": q2, "t1": t1, "tail_percentiles": [0, 100]}


# --------------------------------------------------------------------------- #
# Canonical-frame coordinates from a stored record's pca_angle_deg / centroid.
# The longitudinal formula is identical to the reference implementation
# regions.complete_region_masks; the transverse formula is its perpendicular.
# --------------------------------------------------------------------------- #
def longitudinal_coord(xs, ys, angle_deg, centroid_x, centroid_y):
    """Signed distance along the bow<->stern (PCA major) axis.

    Identical to the projection used in the reference regions.complete_region_masks:
        L = (x - cx) cos(theta) + (y - cy) sin(theta)
    """
    theta = math.radians(float(angle_deg))
    return (
        (xs - float(centroid_x)) * math.cos(theta)
        + (ys - float(centroid_y)) * math.sin(theta)
    )


def transverse_coord(xs, ys, angle_deg, centroid_x, centroid_y):
    """Signed distance along the hull<->superstructure axis (perpendicular to
    the PCA major axis):
        T = -(x - cx) sin(theta) + (y - cy) cos(theta)

    Image y points down, so for a near-horizontal side view T increases toward
    the waterline/hull and decreases toward the sky/superstructure. The deck
    line is a threshold on T (T <= deck -> above-deck, T > deck -> hull); that
    threshold is the quantity the hull/superstructure segmenter must supply.
    """
    theta = math.radians(float(angle_deg))
    return (
        -(xs - float(centroid_x)) * math.sin(theta)
        + (ys - float(centroid_y)) * math.cos(theta)
    )
