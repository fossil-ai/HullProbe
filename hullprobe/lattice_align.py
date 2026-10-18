"""Per-cell target-conditioned registration for the part lattice.

The base operator (swap.similarity_to_canonical + exact stencil) fixes silhouette
but fills each recipient cell with a GLOBALLY-warped donor -> blurry, because the
donor pixels at the recipient cell location are not the donor's corresponding cell.

Here each donor cell is fit to the recipient's cell independently: a canonical-frame
affine matches the two cells' longitudinal extent, area (determinant), and centroid
(a general form of the old target-graft coarse fit), so the donor's ACTUAL bow/hull
texture lands in the recipient's bow/hull cell. Composition is in the recipient's
image frame (natural orientation) with exact stencil to the recipient cell + inpaint
of any residual.
"""
from __future__ import annotations
import numpy as np
import cv2
from PIL import Image

from .swap import similarity_to_canonical, _affine_coeffs
from .constants import ALIGN_IMG_W, ALIGN_IMG_H
from .lattice import cell_masks, CELLS

CW, CH = int(ALIGN_IMG_W), int(ALIGN_IMG_H)


def _mask_to_canon(mask, S):
    coeffs = _affine_coeffs(np.linalg.inv(S))          # canonical(out) -> image(in)
    m = Image.fromarray((mask * 255).astype(np.uint8)).transform(
        (CW, CH), Image.AFFINE, coeffs, resample=Image.NEAREST)
    return np.asarray(m) > 128


def _geom(mask_canon):
    ys, xs = np.nonzero(mask_canon)
    if xs.size < 10:
        return None
    return float(xs.mean()), float(ys.mean()), float(xs.min()), float(xs.max()), float(xs.size)


def _fit_cell(src, tgt):
    """Canonical affine mapping donor cell (src) onto recipient cell (tgt)."""
    scx, scy, sx0, sx1, sarea = src
    tcx, tcy, tx0, tx1, tarea = tgt
    a = min(2.5, max(0.35, (tx1 - tx0 + 1.0) / max(1.0, sx1 - sx0 + 1.0)))
    det = min(4.0, max(0.20, tarea / max(sarea, 1e-6)))
    d = min(2.5, max(0.35, det / a))
    return np.array([[a, 0.0, tcx - a * scx], [0.0, d, tcy - d * scy], [0.0, 0.0, 1.0]])


def precompute_aligned(donor_rec, recip_rec, recip_region=None, donor_region=None):
    """Per-cell donor RGB warped into recipient frame + validity, one fit per cell.
    Returns (recip_rgb, recip_cells, [donor_rgb_c]*6, [donor_valid_c]*6)."""
    recip_rgb = np.asarray(Image.open(recip_rec["image_path"]).convert("RGB"))
    donor_rgb = np.asarray(Image.open(donor_rec["image_path"]).convert("RGB"))
    h, w = recip_rgb.shape[:2]
    rc, _ = cell_masks(recip_rec, region_map=recip_region)
    dc, _ = cell_masks(donor_rec, region_map=donor_region)
    S_r = similarity_to_canonical(recip_rec)
    S_d = similarity_to_canonical(donor_rec)
    donor_rgb_c, donor_valid_c = [], []
    for c in CELLS:
        rg = _geom(_mask_to_canon(rc[c], S_r))
        dg = _geom(_mask_to_canon(dc[c], S_d))
        if rg is None or dg is None:
            donor_rgb_c.append(None); donor_valid_c.append(None); continue
        Cp = _fit_cell(dg, rg)                          # donor-canon cell -> recip-canon cell
        M = np.linalg.inv(S_d) @ np.linalg.inv(Cp) @ S_r  # recip image -> donor image
        coeffs = _affine_coeffs(M)
        rgb_w = np.asarray(Image.fromarray(donor_rgb).transform(
            (w, h), Image.AFFINE, coeffs, resample=Image.BILINEAR))
        val = np.asarray(Image.fromarray((dc[c] * 255).astype(np.uint8)).transform(
            (w, h), Image.AFFINE, coeffs, resample=Image.NEAREST)) > 128
        donor_rgb_c.append(rgb_w); donor_valid_c.append(val)
    return recip_rgb, rc, donor_rgb_c, donor_valid_c


def assemble_all_aligned(recip_rgb, recip_cells, donor_rgb_c, donor_valid_c):
    """64 composites; swapped cell = recipient cell support filled with its per-cell
    fitted donor texture, residual inpainted (silhouette = recipient's)."""
    rc = [recip_cells[c] for c in CELLS]
    out = []
    for z in range(64):
        res = recip_rgb.copy()
        resid = np.zeros(recip_rgb.shape[:2], bool)
        for i in range(6):
            if not (z >> i & 1):
                continue
            if donor_rgb_c[i] is None:
                resid |= rc[i]; continue
            have = rc[i] & donor_valid_c[i]
            res[have] = donor_rgb_c[i][have]
            resid |= rc[i] & ~donor_valid_c[i]
        if resid.any():
            res = cv2.inpaint(res, (resid * 255).astype(np.uint8), 3, cv2.INPAINT_TELEA)
        out.append(res)
    return out


def compose_aligned(recip_rec, donor_rec, z, recip_region=None, donor_region=None):
    recip_rgb, rc, drgb, dval = precompute_aligned(donor_rec, recip_rec, recip_region, donor_region)
    return Image.fromarray(assemble_all_aligned(recip_rgb, rc, drgb, dval)[z])
