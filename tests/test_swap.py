"""Geometric correctness of the hull/superstructure swap operator."""

import numpy as np
from PIL import Image

from hullprobe.canonical import hull_superstructure_masks
from hullprobe.swap import (
    compose_swap,
    hull_swap,
    similarity_to_canonical,
    superstructure_swap,
)


def _arr(image):
    return np.asarray(image.convert("RGB"), dtype=np.int16)


def test_identity_self_swap_reproduces_image(vessel_a):
    """Swapping a component with the same vessel returns ~the original."""
    original = _arr(Image.open(vessel_a["image_path"]))
    for component in ("superstructure", "hull"):
        out, info = compose_swap(vessel_a, vessel_a, component)
        diff = np.abs(_arr(out) - original)
        # Only resampling error in the swapped-in region; retained is exact.
        assert diff.mean() < 3.0, (component, diff.mean())
        assert info["hole_area"] == 0


def test_retained_component_is_pixel_exact(vessel_a, vessel_b):
    """The recipient's kept component is never modified."""
    recip = _arr(Image.open(vessel_a["image_path"]))
    hull_r, super_r = hull_superstructure_masks(vessel_a)

    out_super, _ = superstructure_swap(vessel_a, vessel_b)  # keep recip hull
    np.testing.assert_array_equal(_arr(out_super)[hull_r], recip[hull_r])

    out_hull, _ = hull_swap(vessel_a, vessel_b)  # keep recip superstructure
    np.testing.assert_array_equal(_arr(out_hull)[super_r], recip[super_r])


def test_cross_donor_changes_swapped_region(vessel_a, vessel_b):
    """A different donor actually changes the swapped-in component."""
    recip = _arr(Image.open(vessel_a["image_path"]))
    _hull_r, super_r = hull_superstructure_masks(vessel_a)
    out, info = superstructure_swap(vessel_a, vessel_b)
    assert info["donor_paint_area"] > 0
    # Something above the deck differs from the recipient's own superstructure.
    changed = (np.abs(_arr(out) - recip).sum(axis=-1) > 20)
    assert changed.sum() > 0


def test_similarity_maps_extent_to_canonical(vessel_a):
    """The similarity map sends the vessel's longitudinal span to [0, CANON_W]."""
    from hullprobe.swap import CANON_W, frame_params

    S = similarity_to_canonical(vessel_a)
    angle, cx, cy, t0, t1, _deck = frame_params(vessel_a)
    # Two points at the longitudinal extremes on the axis (angle ~ 0 => along x).
    import math

    theta = math.radians(angle)
    p0 = np.array([cx + t0 * math.cos(theta), cy + t0 * math.sin(theta), 1.0])
    p1 = np.array([cx + t1 * math.cos(theta), cy + t1 * math.sin(theta), 1.0])
    x0 = (S @ p0)[0]
    x1 = (S @ p1)[0]
    assert abs(x0 - 0.0) < 1e-6
    assert abs(x1 - CANON_W) < 1e-6


def test_swap_is_deterministic(vessel_a, vessel_b):
    out1, _ = superstructure_swap(vessel_a, vessel_b)
    out2, _ = superstructure_swap(vessel_a, vessel_b)
    np.testing.assert_array_equal(_arr(out1), _arr(out2))
