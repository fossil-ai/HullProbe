"""Label-preserving augmentation invariants."""

import numpy as np
from PIL import Image

from hullprobe.augment import LabelPreservingSwapAug, degrade_hull, hull_randomize
from hullprobe.canonical import hull_superstructure_masks


def _arr(image):
    return np.asarray(image.convert("RGB"), dtype=np.int16)


def test_hull_randomize_preserves_superstructure(vessel_a, vessel_b):
    """Hull swap keeps the recipient superstructure (its label cue) pixel-exact
    and actually changes the hull."""
    recip = _arr(Image.open(vessel_a["image_path"]))
    hull_r, super_r = hull_superstructure_masks(vessel_a)
    out = hull_randomize(vessel_a, vessel_b)
    np.testing.assert_array_equal(_arr(out)[super_r], recip[super_r])
    # The hull region is drawn from the donor / background, so it changes.
    assert (np.abs(_arr(out)[hull_r] - recip[hull_r]).sum(axis=-1) > 10).any()


def test_degrade_hull_keeps_superstructure(vessel_a):
    recip = _arr(Image.open(vessel_a["image_path"]))
    _hull_r, super_r = hull_superstructure_masks(vessel_a)
    for mode in ("blur", "blackout", "noise"):
        out = degrade_hull(vessel_a, mode=mode)
        np.testing.assert_array_equal(_arr(out)[super_r], recip[super_r])


def test_aug_p_zero_returns_original(vessel_a, vessel_b):
    aug = LabelPreservingSwapAug(donor_pool=[vessel_b], p=0.0)
    original = _arr(Image.open(vessel_a["image_path"]))
    np.testing.assert_array_equal(_arr(aug(vessel_a)), original)


def test_aug_p_one_swaps_hull(vessel_a, vessel_b):
    aug = LabelPreservingSwapAug(donor_pool=[vessel_b], p=1.0)
    _hull_r, super_r = hull_superstructure_masks(vessel_a)
    original = _arr(Image.open(vessel_a["image_path"]))
    out = _arr(aug(vessel_a))
    # Superstructure (label) preserved; something changed overall.
    np.testing.assert_array_equal(out[super_r], original[super_r])
    assert (np.abs(out - original).sum(axis=-1) > 10).any()
