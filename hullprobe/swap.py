"""Hull / superstructure swap operator.

Both vessels are registered with the *same* PCA pose used everywhere else. Each
vessel gets a similarity map into a common canonical frame:

    Xc = s (L - t0)                      L, T = longitudinal / transverse coords
    Yc = DECK_ROW + s (T - deck)         s   = CANON_W / longitudinal_extent

so vessels are derotated, length-normalized, and deck-aligned. Composing the
donor's forward map with the recipient's inverse map,

    M = S_recip^{-1} @ S_donor

places a donor component into the recipient's frame: scaled by the length ratio
(recip/donor) and attached at the recipient's deck line. The retained recipient
component is kept pixel-exact; the removed recipient component is background-
filled where the donor doesn't cover it.

The compositor is deliberately simple (no learned harmonizer yet); the causal
probe cancels its seam/fill artifacts with same-class controls, so photorealism
is not required for the probe to be valid. A harmonizer is the drop-in upgrade
for the augmentation path.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .canonical import full_vessel_alpha, hull_superstructure_masks
from .constants import ALIGN_IMG_W, ALIGN_IMG_H
from .pose import longitudinal_coord, transverse_coord

CANON_W = float(ALIGN_IMG_W)
DECK_ROW = float(ALIGN_IMG_H) / 2.0

RETAINED_OF = {"superstructure": "hull", "hull": "superstructure"}


def frame_params(rec, deck=None):
    """(angle, cx, cy, t0, t1, deck) for a vessel record."""
    full_alpha, width, height = full_vessel_alpha(rec)
    ys, xs = np.where(full_alpha > 12)
    if len(xs) < 60:
        raise ValueError("vessel alpha too small for a swap")
    angle = float(rec["pca_angle_deg"])
    cx, cy = float(rec["pca_centroid_x"]), float(rec["pca_centroid_y"])
    lon = longitudinal_coord(xs.astype(np.float64), ys.astype(np.float64), angle, cx, cy)
    t0, t1 = float(lon.min()), float(lon.max())
    if t1 <= t0:
        raise ValueError("non-positive longitudinal extent")
    if deck is None:
        deck = float(rec.get("deck_transverse", 0.0))
    return angle, cx, cy, t0, t1, float(deck)


def similarity_to_canonical(rec, deck=None):
    """3x3 matrix mapping this vessel's image coords -> common canonical frame."""
    angle, cx, cy, t0, t1, deck = frame_params(rec, deck=deck)
    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    s = CANON_W / (t1 - t0)
    c1 = -s * (cos_t * cx + sin_t * cy) - s * t0
    c2 = s * (sin_t * cx - cos_t * cy) + DECK_ROW - s * deck
    return np.array([
        [s * cos_t,  s * sin_t, c1],
        [-s * sin_t, s * cos_t, c2],
        [0.0,        0.0,       1.0],
    ], dtype=np.float64)


def _affine_coeffs(matrix):
    """PIL AFFINE data (a,b,c,d,e,f) from a 3x3 output->input matrix."""
    return (matrix[0, 0], matrix[0, 1], matrix[0, 2],
            matrix[1, 0], matrix[1, 1], matrix[1, 2])


def _component_rgba(rec, component, deck=None, superstructure_mask=None, region_map=None):
    """Donor component as an RGBA image in the donor's own image frame."""
    hull, superstructure = hull_superstructure_masks(
        rec, deck=deck, superstructure_mask=superstructure_mask, region_map=region_map
    )
    mask = superstructure if component == "superstructure" else hull
    rgb = np.asarray(Image.open(rec["image_path"]).convert("RGB"))
    rgba = np.dstack([rgb, (mask * 255).astype(np.uint8)])
    return Image.fromarray(rgba, "RGBA"), mask


def _background_fill(rgb, vessel_mask):
    """Fill the vessel region with a background estimate (cv2 inpaint if present,
    else an iterative nearest-neighbor fill)."""
    try:
        import cv2

        return cv2.inpaint(
            rgb, (vessel_mask.astype(np.uint8) * 255), 3, cv2.INPAINT_TELEA
        )
    except Exception:
        out = rgb.astype(np.float32).copy()
        hole = vessel_mask.copy()
        for _ in range(64):
            if not hole.any():
                break
            known = ~hole
            acc = np.zeros_like(out)
            cnt = np.zeros(hole.shape, dtype=np.float32)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                shifted = np.roll(known, (dy, dx), axis=(0, 1))
                vals = np.roll(out, (dy, dx), axis=(0, 1))
                take = shifted & hole
                acc[take] += vals[take]
                cnt[take] += 1.0
            fillable = (cnt > 0) & hole
            out[fillable] = acc[fillable] / cnt[fillable, None]
            hole &= ~fillable
        return np.clip(out, 0, 255).astype(np.uint8)


def compose_swap(recip_rec, donor_rec, donor_component,
                 deck_recip=None, deck_donor=None,
                 super_mask_recip=None, super_mask_donor=None,
                 region_map_recip=None, region_map_donor=None):
    """Composite where `donor_component` comes from the donor and the other
    component is the recipient's own. Returns (PIL RGB, info).

    A trained segmenter's `region_map_recip` / `region_map_donor` take priority
    over any deck threshold (see canonical.hull_superstructure_masks)."""
    if donor_component not in RETAINED_OF:
        raise ValueError(f"donor_component must be hull|superstructure, got {donor_component}")
    retained_component = RETAINED_OF[donor_component]

    recip_rgb = np.asarray(Image.open(recip_rec["image_path"]).convert("RGB"))
    height, width = recip_rgb.shape[:2]
    hull_r, super_r = hull_superstructure_masks(
        recip_rec, deck=deck_recip, superstructure_mask=super_mask_recip,
        region_map=region_map_recip,
    )
    retained_mask = hull_r if retained_component == "hull" else super_r
    removed_mask = super_r if retained_component == "hull" else hull_r
    vessel_r = hull_r | super_r

    # Donor component warped into the recipient frame.
    donor_rgba, _donor_mask = _component_rgba(
        donor_rec, donor_component, deck=deck_donor, superstructure_mask=super_mask_donor,
        region_map=region_map_donor,
    )
    S_recip = similarity_to_canonical(recip_rec, deck=deck_recip)
    S_donor = similarity_to_canonical(donor_rec, deck=deck_donor)
    recip_to_donor = np.linalg.inv(S_donor) @ S_recip  # output(recip)->input(donor)
    warped = donor_rgba.transform(
        (width, height), Image.AFFINE, _affine_coeffs(recip_to_donor),
        resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0),
    )
    warped_arr = np.asarray(warped)
    donor_rgb_w = warped_arr[..., :3]
    donor_paintable = (warped_arr[..., 3] > 128) & ~retained_mask

    result = recip_rgb.copy()
    hole = removed_mask & ~donor_paintable
    if hole.any():
        bg = _background_fill(recip_rgb, vessel_r)
        result[hole] = bg[hole]
    result[donor_paintable] = donor_rgb_w[donor_paintable]

    info = {
        "donor_component": donor_component,
        "retained_component": retained_component,
        "donor_paint_area": int(donor_paintable.sum()),
        "hole_area": int(hole.sum()),
        "retained_area": int(retained_mask.sum()),
    }
    return Image.fromarray(result, "RGB"), info


def superstructure_swap(recip_rec, donor_rec, **kwargs):
    """Recipient hull + donor superstructure."""
    return compose_swap(recip_rec, donor_rec, "superstructure", **kwargs)


def hull_swap(recip_rec, donor_rec, **kwargs):
    """Recipient superstructure + donor hull."""
    return compose_swap(recip_rec, donor_rec, "hull", **kwargs)
