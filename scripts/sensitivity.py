"""Sensitivity analysis: is the corrected hull-share robust to the audit's own
choices -- donor selection, donor count, segmentation boundary -- and does the
same-class artifact correction hold when composites are deliberately made worse?

For each variant we recompute the artifact-corrected hull-share on the same recipients.
Stability across variants = the conclusion is not an artifact of these choices; in
particular, if RANDOM (shape-mismatched) donors -- which produce far worse composites and
a larger raw artifact floor -- still yield a similar CORRECTED hull-share, the same-class
control is doing its job.

Writes experiments/sensitivity.json.
"""
from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

from hullprobe.lattice import cell_masks, CELLS
from hullprobe.swap import _background_fill
from scripts.part_lattice_spectrum import (
    test_pool, rmap, warp_donor_cells, assemble_all, classify, load_model, compatible_cross)
from scripts.part_lattice_debias_audit import spectrum_from_A


def perturb_rmap(rm, k):
    """Shift the hull/superstructure boundary by |k| px (k>0 grows superstructure into
    hull, k<0 shrinks it) -- a proxy for segmentation error."""
    if k == 0:
        return rm
    vessel = rm > 0
    sup = (rm == 2).astype(np.uint8)
    ker = np.ones((2 * abs(k) + 1, 2 * abs(k) + 1), np.uint8)
    sup = cv2.dilate(sup, ker) if k > 0 else cv2.erode(sup, ker)
    out = np.where(vessel, 1, 0).astype(rm.dtype)
    out[(sup > 0) & vessel] = 2
    return out


def aspect(rec, _c={}):
    k = rec["image_path"]
    if k not in _c:
        try:
            _, v = cell_masks(rec, region_map=rmap(rec)); ys, xs = np.nonzero(v)
            _c[k] = (xs.max() - xs.min() + 1) / max(1, ys.max() - ys.min() + 1)
        except Exception:
            _c[k] = None
    return _c[k]


def hull_share(model, by_c, all_v, device, mode="compatible", K=4, kpx=0, per_class=3, seed=42):
    rng = random.Random(seed)
    recips = []
    for c, lst in by_c.items():
        lst = list(lst); rng.shuffle(lst); recips += lst[:per_class]
    cross_infs, same_infs = [], []      # pool influence vectors across recipients
    for r in recips:
        try:
            y = r["label"]
            rm = perturb_rmap(rmap(r), kpx)
            recip_rgb = np.asarray(Image.open(r["image_path"]).convert("RGB"))
            rc, vessel = cell_masks(r, region_map=rm)
            bg = _background_fill(recip_rgb, vessel)
            if mode == "compatible":
                cross = compatible_cross(r, all_v, K, rng)
                spool = sorted([d for d in by_c[y] if d["image_path"] != r["image_path"]],
                               key=lambda d: abs((aspect(d) or 9) - (aspect(r) or 0)))[:K]
            else:  # random donors (and random same-class controls to match severity)
                cross = rng.sample([d for d in all_v if d["label"] != y], K)
                sp = [d for d in by_c[y] if d["image_path"] != r["image_path"]]
                spool = rng.sample(sp, min(K, len(sp)))
            if not spool:
                continue

            def A_of(donors):
                A = np.zeros((len(donors), 64), np.int8)
                for di, d in enumerate(donors):
                    drgb, dval = warp_donor_cells(d, r)
                    imgs = assemble_all(recip_rgb, rc, bg, drgb, dval)
                    A[di] = (classify(model, imgs, device) == y).astype(np.int8)
                return A
            Ax = A_of(cross)
            if Ax[:, 0].mean() < 0.5:
                continue
            As = A_of(spool)
            cross_infs.append(spectrum_from_A(Ax)[0])
            same_infs.append(spectrum_from_A(As)[0])
        except Exception:
            continue
    C = np.array(cross_infs); Smat = np.array(same_infs)
    # bootstrap over recipients for a CI on the POOLED corrected hull-share
    rs = np.random.RandomState(0); boot = []
    for _ in range(2000):
        idx = rs.randint(0, len(C), len(C))
        cc = C[idx].mean(0) - Smat[idx].mean(0)
        boot.append(cc[:3].sum() / cc.sum())
    boot = np.array(boot)
    corr = C.mean(0) - Smat.mean(0)
    raw = C.mean(0)
    return {"hull_share": float(corr[:3].sum() / corr.sum()),
            "hull_share_raw": float(raw[:3].sum() / raw.sum()),
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "n": len(C)}


def main():
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    try:
        import torch
        if device == "cuda" and torch.cuda.mem_get_info()[0] < 2.5e9:
            device = "cpu"
    except Exception:
        pass
    model = load_model(device)
    by_c = test_pool(); all_v = [r for lst in by_c.values() for r in lst]
    variants = {
        "reference (compatible, K=4)": dict(mode="compatible", K=4, kpx=0),
        "random donors (K=4)":          dict(mode="random", K=4, kpx=0),
        "compatible, K=2":              dict(mode="compatible", K=2, kpx=0),
        "compatible, K=8":              dict(mode="compatible", K=8, kpx=0),
        "seg. eroded (-3px)":           dict(mode="compatible", K=4, kpx=-3),
        "seg. dilated (+3px)":          dict(mode="compatible", K=4, kpx=3),
    }
    res = {}
    for name, kw in variants.items():
        r = hull_share(model, by_c, all_v, device, **kw)
        res[name] = r
        print(f"{name:32s} corr {r['hull_share']:.3f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}]  "
              f"raw {r['hull_share_raw']:.3f}  (n={r['n']})", flush=True)
    Path("experiments/sensitivity.json").write_text(json.dumps(res, indent=1))
    print("wrote experiments/sensitivity.json")


if __name__ == "__main__":
    main()
