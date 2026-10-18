"""Step 5: run the part-lattice on SeaShips (6-class) to test whether the MARVEL findings
REPLICATE on a second dataset: artifact-corrected hull-share, low-order additivity
(deg<=2 variance), and semantic-L0. Reuses the exact operator/estimator by pointing the
shared helpers at SeaShips paths + model. Writes experiments/seaships_lattice.json."""
from __future__ import annotations
import json, random
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
from torchvision.models import convnext_tiny

import scripts.part_lattice_spectrum as pls          # shared operator/estimator
from scripts.part_lattice_debias_audit import spectrum_from_A
from hullprobe.lattice import cell_masks, CELLS
from hullprobe.swap import _background_fill

pls.PRED = Path("data/seaships/predictions")           # redirect masks to SeaShips
CKPT = "experiments/seaships_clf/best.pt"


def load_seaships_model(device, n_classes):
    m = convnext_tiny(num_classes=n_classes)
    st = torch.load(CKPT, map_location="cpu"); st = st.get("model", st.get("state_dict", st))
    m.load_state_dict({k.replace("module.", ""): v for k, v in st.items()}, strict=False)
    return m.to(device).eval()


def main(per_class=8, n_cross=4, n_same=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if device == "cuda" and torch.cuda.mem_get_info()[0] < 2.5e9:
            device = "cpu"
    except Exception:
        pass
    recs = json.loads(Path("data/seaships/pose.json").read_text())["records"]
    n_classes = 1 + max(r["label"] for r in recs)
    model = load_seaships_model(device, n_classes)
    by_c = defaultdict(list)
    for r in recs:
        if r["source_split"] == "test":
            by_c[r["label"]].append(r)
    all_v = [r for lst in by_c.values() for r in lst]
    rng = random.Random(42)
    recips = []
    for c, lst in by_c.items():
        lst = list(lst); rng.shuffle(lst); recips += lst[:per_class]
    print(f"SeaShips: {n_classes} classes, {len(recips)} recipients", flush=True)

    cross_inf, same_inf, gvecs, rstars = [], [], [], []
    for ri, r in enumerate(recips):
        try:
            y = r["label"]
            recip = np.asarray(Image.open(r["image_path"]).convert("RGB"))
            rc, vessel = cell_masks(r, region_map=pls.rmap(r))
            bg = _background_fill(recip, vessel)
            cross = pls.compatible_cross(r, all_v, n_cross, rng)
            spool = [d for d in by_c[y] if d["image_path"] != r["image_path"]]
            same = rng.sample(spool, min(n_same, len(spool))) if spool else []
            if not same:
                continue

            def A_of(dz):
                A = np.zeros((len(dz), 64), np.int8)
                for di, d in enumerate(dz):
                    drgb, dval = pls.warp_donor_cells(d, r)
                    A[di] = (pls.classify(model, pls.assemble_all(recip, rc, bg, drgb, dval), device) == y)
                return A
            Ax = A_of(cross)
            if Ax[:, 0].mean() < 0.5:
                continue
            As = A_of(same)
            cross_inf.append(spectrum_from_A(Ax)[0]); same_inf.append(spectrum_from_A(As)[0])
            g = Ax.mean(0); coeff = (pls.H64 @ g) / 64.0; power = coeff ** 2
            W = np.array([power[pls.POP == d].sum() for d in range(7)])
            gvecs.append(W)
            flipped = (Ax == 0).any(0); rr = 7
            for z in range(1, 64):
                if flipped[z]:
                    rr = min(rr, int(pls.POP[z]))
            rstars.append(rr)
            if (ri + 1) % 15 == 0:
                print(f"  {ri+1}/{len(recips)}", flush=True)
        except Exception:
            continue

    C = np.array(cross_inf).mean(0); S = np.array(same_inf).mean(0); corr = C - S
    hull_share = float(corr[:3].sum() / corr.sum())
    W = np.array(gvecs).mean(0); Wpos = W[1:].sum() + 1e-12
    deg1, deg2 = float(W[1] / Wpos), float((W[1] + W[2]) / Wpos)
    rs = np.array(rstars)
    res = {"n_classes": int(n_classes), "n_recip": len(cross_inf),
           "hull_share": hull_share, "deg1": deg1, "deg_le2": deg2,
           "semL0_median": float(np.median(rs)), "semL0_frac_r1": float((rs == 1).mean()),
           "corr_influence": corr.round(4).tolist(), "cells": [f"{v}-{l}" for v, l in CELLS]}
    print("\n SEASHIPS LATTICE (2nd-dataset replication)")
    print(f"hull-share {hull_share:.3f} | deg<=2 {deg2:.3f} | semL0 median {res['semL0_median']:.0f} "
          f"frac r=1 {res['semL0_frac_r1']:.3f} | n={res['n_recip']}")
    Path("experiments/seaships_lattice.json").write_text(json.dumps(res, indent=1))
    print("wrote experiments/seaships_lattice.json")


if __name__ == "__main__":
    main()
