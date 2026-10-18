"""Synthetic vessel fixtures: a hull rectangle with a smaller deckhouse on top,
saved to disk with a hullprobe-style pose record so swap/probe/augment logic can
be exercised on CPU without the segmentation stack."""

import numpy as np
import pytest
from PIL import Image

from hullprobe.pose import pca_from_mask


def _rect(mask, y0, y1, x0, x1):
    mask[y0:y1, x0:x1] = True


def make_vessel(tmp_path, name, *, hull_x=(20, 180), hull_y=(60, 92),
                super_x=(80, 130), super_y=(40, 60), hue=(180, 120, 90),
                super_hue=(230, 230, 230), size=(200, 130)):
    """Write a synthetic vessel image + full-alpha crop, return a pose record.

    Horizontal vessel (PCA angle ~0). The deck line sits at the hull/super
    boundary; `deck_transverse` is set so a transverse split separates them.
    """
    width, height = size
    rgb = np.full((height, width, 3), (30, 90, 140), dtype=np.uint8)  # sea/sky bg
    hull = np.zeros((height, width), dtype=bool)
    superstructure = np.zeros((height, width), dtype=bool)
    _rect(hull, hull_y[0], hull_y[1], hull_x[0], hull_x[1])
    _rect(superstructure, super_y[0], super_y[1], super_x[0], super_x[1])
    rgb[hull] = hue
    rgb[superstructure] = super_hue
    vessel = hull | superstructure

    image_path = tmp_path / f"{name}.png"
    Image.fromarray(rgb, "RGB").save(image_path)

    ys, xs = np.where(vessel)
    box = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
    rgba = np.dstack([rgb, (vessel * 255).astype(np.uint8)])
    crop = Image.fromarray(rgba, "RGBA").crop(tuple(box))
    full_path = tmp_path / f"{name}_full.png"
    crop.save(full_path)

    _axis, centroid, angle, aspect = pca_from_mask(vessel)
    cy = float(centroid[1])
    # Deck line at the hull top (boundary between super and hull), as a
    # transverse threshold T = y - cy at angle 0.
    deck = float(hull_y[0]) - cy
    return {
        "image_path": str(image_path),
        "full_vessel_path": str(full_path),
        "full_vessel_box_xyxy": box,
        "pca_angle_deg": angle,
        "pca_centroid_x": float(centroid[0]),
        "pca_centroid_y": cy,
        "pca_aspect": aspect,
        "deck_transverse": deck,
        "label": name,
    }


@pytest.fixture
def vessel_a(tmp_path):
    return make_vessel(tmp_path, "a", hue=(180, 120, 90), super_hue=(240, 240, 240))


@pytest.fixture
def vessel_b(tmp_path):
    # Different length + colors so swaps are visibly distinct from A.
    return make_vessel(
        tmp_path, "b", hull_x=(30, 150), super_x=(60, 100),
        hue=(90, 150, 90), super_hue=(250, 210, 120),
    )
