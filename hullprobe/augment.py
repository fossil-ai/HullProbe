"""Label-preserving hull randomization.

If the superstructure defines the class (the probe's hypothesis), then swapping
the *hull* keeps the label. That makes hull-swap a nuisance-factor randomization
augmentation: it injects diversity while staying on-manifold *and* keeping a
correct hard label — unlike CutMix/MixUp, which mix labels. Training a
recognizer on hull-randomized vessels pushes it off the (spurious) hull and onto
the (causal) superstructure.

    hull_randomize(recip, donor)   -> recip superstructure + donor hull   (label = recip)

`degrade_hull` is the paired evaluation stress: blur/black-out the hull so the
invariance the augmentation buys can actually be measured.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageFilter

from .canonical import hull_superstructure_masks
from .swap import hull_swap, superstructure_swap


def hull_randomize(recip_rec, donor_rec, **kwargs):
    """Recip superstructure + donor hull. Label-preserving for any donor class."""
    image, _info = hull_swap(recip_rec, donor_rec, **kwargs)
    return image


def superstructure_randomize_same_class(recip_rec, donor_rec, **kwargs):
    """Recip hull + donor superstructure. Label-preserving ONLY when the donor
    shares the recipient's class — an extra on-class diversity source."""
    image, _info = superstructure_swap(recip_rec, donor_rec, **kwargs)
    return image


def degrade_hull(recip_rec, mode="blur", radius=14):
    """Return the image with the hull region degraded and the superstructure
    intact — the predeclared hull-invariance stress test."""
    image = Image.open(recip_rec["image_path"]).convert("RGB")
    hull, _superstructure = hull_superstructure_masks(recip_rec)
    arr = np.asarray(image).copy()
    if mode == "blackout":
        arr[hull] = 0
    elif mode == "blur":
        blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius)))
        arr[hull] = blurred[hull]
    elif mode == "noise":
        rng = np.random.default_rng(0)
        arr[hull] = rng.integers(0, 256, size=(int(hull.sum()), 3), dtype=np.uint8)
    else:
        raise ValueError(f"unknown degrade mode: {mode}")
    return Image.fromarray(arr, "RGB")


class LabelPreservingSwapAug:
    """Callable augmentation: with probability `p`, replace the recipient's hull
    with a random donor's hull (label preserved). Optionally also draw same-class
    superstructure swaps for extra on-class diversity.

    `donor_pool` is a flat list of pose records (any class). `same_class_pool` is
    a dict {class -> [records]} used only for the same-class superstructure route.
    """

    def __init__(self, donor_pool, same_class_pool=None, label_key="class",
                 p=0.5, p_same_class_super=0.0, seed=42):
        self.donor_pool = list(donor_pool)
        self.same_class_pool = same_class_pool or {}
        self.label_key = label_key
        self.p = float(p)
        self.p_same_class_super = float(p_same_class_super)
        self.rng = random.Random(seed)

    def _original(self, rec):
        return Image.open(rec["image_path"]).convert("RGB")

    def __call__(self, rec):
        roll = self.rng.random()
        try:
            if self.donor_pool and roll < self.p:
                return hull_randomize(rec, self.rng.choice(self.donor_pool))
            if self.p_same_class_super and roll < self.p + self.p_same_class_super:
                same = self.same_class_pool.get(rec.get(self.label_key), [])
                same = [r for r in same if r is not rec]
                if same:
                    return superstructure_randomize_same_class(rec, self.rng.choice(same))
        except Exception:
            # Never let a compositing failure kill a training step.
            return self._original(rec)
        return self._original(rec)
