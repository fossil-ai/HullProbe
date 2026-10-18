"""2x3 anatomical part-lattice operator.

Two orthogonal, complete decompositions of a vessel, both already registered in
the same canonical PCA frame:

  vertical   (segmenter):  hull / superstructure          -- hull_superstructure_masks
  horizontal (pose):       fwd / mid / aft (PCA thirds)    -- longitudinal_split

Their product is 6 cells  P = {hull,sup} x {fwd,mid,aft}.  A composite is a
Boolean vector z in {0,1}^6 : z_p = 1 -> replace cell p with a donor's cell p
(registered by the same similarity map used by swap.compose_swap), z_p = 0 ->
keep the recipient's own cell.  z = 0 is the original vessel; z = (all hull)
reproduces swap.hull_swap.

This is the estimable primitive behind the part-lattice Fourier / influence /
semantic-L0 analysis: g(z) = E_donor f_y(compose_lattice(x, d, z)) is a function
on a 64-vertex cube, evaluable exactly.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
from PIL import Image

from .canonical import hull_superstructure_masks
from .pose import longitudinal_split
from .swap import (similarity_to_canonical, _affine_coeffs, _background_fill)

VERT = ("hull", "sup")
LONG = ("fwd", "mid", "aft")
CELLS = tuple((v, l) for v in VERT for l in LONG)   # canonical order, len 6


def _axis_centroid(rec):
    theta = math.radians(float(rec["pca_angle_deg"]))
    axis = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    centroid = np.array([float(rec["pca_centroid_x"]), float(rec["pca_centroid_y"])])
    return axis, centroid


def cell_masks(rec, region_map=None, deck=None, superstructure_mask=None):
    """OrderedDict {(vert,long): bool HxW} for the 6 cells, plus the vessel mask.
    Cells exactly partition the vessel (empty cells allowed, e.g. sup-fwd)."""
    hull, sup = hull_superstructure_masks(
        rec, deck=deck, superstructure_mask=superstructure_mask, region_map=region_map)
    vessel = hull | sup
    axis, centroid = _axis_centroid(rec)
    split = longitudinal_split(vessel, axis, centroid)
    if split is None:
        raise ValueError("longitudinal split failed (vessel too small/degenerate)")
    thirds, _meta = split                       # [fwd, mid, aft] bool masks
    vmask = {"hull": hull, "sup": sup}
    cells = OrderedDict()
    for v in VERT:
        for i, l in enumerate(LONG):
            cells[(v, l)] = vmask[v] & thirds[i]
    return cells, vessel


def compose_lattice(recip_rec, donor_rec, z,
                    region_map_recip=None, region_map_donor=None,
                    deck_recip=None, deck_donor=None,
                    recip_cells=None, recip_vessel=None):
    """Composite for Boolean assignment z (dict cell->bool or 6-vector aligned to
    CELLS). Flipped cells come from `donor_rec`; the rest are the recipient's.
    Returns a PIL RGB image. Pass recip_cells/recip_vessel to reuse a cache."""
    if not isinstance(z, dict):
        z = {c: bool(zi) for c, zi in zip(CELLS, z)}

    recip_rgb = np.asarray(Image.open(recip_rec["image_path"]).convert("RGB"))
    h, w = recip_rgb.shape[:2]
    if recip_cells is None:
        recip_cells, recip_vessel = cell_masks(recip_rec, region_map=region_map_recip,
                                               deck=deck_recip)

    flipped = [c for c in CELLS if z.get(c, False)]
    if not flipped:
        return Image.fromarray(recip_rgb, "RGB")

    retained = np.zeros((h, w), bool)
    removed = np.zeros((h, w), bool)
    for c in CELLS:
        (removed if z.get(c, False) else retained)[recip_cells[c]] = True

    # donor: union of the flipped cells, warped into the recipient frame
    donor_cells, _ = cell_masks(donor_rec, region_map=region_map_donor, deck=deck_donor)
    donor_alpha = np.zeros((h, w), bool)  # placeholder shape; rebuild in donor frame
    donor_rgb = np.asarray(Image.open(donor_rec["image_path"]).convert("RGB"))
    dh, dw = donor_rgb.shape[:2]
    dalpha = np.zeros((dh, dw), bool)
    for c in flipped:
        dalpha |= donor_cells[c]
    donor_rgba = Image.fromarray(np.dstack([donor_rgb, (dalpha * 255).astype(np.uint8)]), "RGBA")

    S_recip = similarity_to_canonical(recip_rec, deck=deck_recip)
    S_donor = similarity_to_canonical(donor_rec, deck=deck_donor)
    recip_to_donor = np.linalg.inv(S_donor) @ S_recip
    warped = donor_rgba.transform((w, h), Image.AFFINE, _affine_coeffs(recip_to_donor),
                                  resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0))
    warped = np.asarray(warped)
    donor_rgb_w = warped[..., :3]
    donor_paintable = (warped[..., 3] > 128) & ~retained

    result = recip_rgb.copy()
    hole = removed & ~donor_paintable
    if hole.any():
        bg = _background_fill(recip_rgb, recip_vessel)
        result[hole] = bg[hole]
    result[donor_paintable] = donor_rgb_w[donor_paintable]
    return Image.fromarray(result, "RGB")
