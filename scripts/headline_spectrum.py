"""Headline part-lattice spectrum (paper Secs. 5.1-5.2) with per-recipient records and CIs.

Runs the audit of scripts/part_lattice_spectrum.py on the 277-recipient attribution pool
(per_class=14 over the 26 MARVEL-B classes -> 364 candidates, of which the 277 the classifier gets
right are kept; K=4 cross-class + 2 same-class donors; seed 42) and additionally stores, per kept
recipient, its label, cross-class and same-class donor identities, per-cell cross/same influence,
Walsh degree fractions, and semantic-L0 radius. It reports 95% bootstrap CIs (2000 resamples over
recipients) for the pooled corrected hull-share, W1, W<=2 and Pr[r=1].

The donor identities it records are published in protocol/attribution_pool_n277.json.

Usage (repo root):
  python -m scripts.headline_spectrum --out experiments/headline_spectrum_n277.json
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from hullprobe.lattice import cell_masks, CELLS
from hullprobe.swap import _background_fill
from scripts.part_lattice_spectrum import (
    test_pool, rmap, load_model, compatible_cross, warp_donor_cells, assemble_all, classify,
    H64, POP)


def stem(rec):
    return Path(rec["image_path"]).stem


def summarize(C, S, Wd, R, n_boot=2000, seed=0):
    """Pooled corrected hull-share (ratio of means), W1, W<=2, Pr[r=1], each with a recipient
    bootstrap CI."""
    def stats(idx):
        corr = C[idx].mean(0) - S[idx].mean(0)
        return np.array([corr[:3].sum() / corr.sum(), Wd[idx, 0].mean(),
                         Wd[idx, :2].sum(1).mean(), (R[idx] == 1).mean()])
    n = len(C); point = stats(np.arange(n))
    rs = np.random.RandomState(seed)
    boot = np.array([stats(rs.randint(0, n, n)) for _ in range(n_boot)])
    lo, hi = np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0)
    names = ["hull_share", "W1", "W_le2", "frac_r1"]
    return {k: {"value": float(point[i]), "ci": [float(lo[i]), float(hi[i])]}
            for i, k in enumerate(names)}


def main(per_class, n_cross, n_same, seed, out):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.mem_get_info()[0] < 2.2e9:
        device = "cpu"
    print("device:", device, flush=True)
    rng = random.Random(seed)
    by_c = test_pool()
    model = load_model(device)

    recips = []
    for c, lst in by_c.items():
        lst = list(lst); rng.shuffle(lst); recips += lst[:per_class]
    all_v = [r for lst in by_c.values() for r in lst]

    rows, skipped = [], {"misclassified": 0, "error": 0}
    t0 = time.time()
    for ri, r in enumerate(recips):
        try:
            y = r["label"]
            recip_rgb = np.asarray(Image.open(r["image_path"]).convert("RGB"))
            rc, vessel = cell_masks(r, region_map=rmap(r))
            if classify(model, [recip_rgb], device)[0] != y:
                skipped["misclassified"] += 1
                continue
            bg = _background_fill(recip_rgb, vessel)
            cross_ds = compatible_cross(r, all_v, n_cross, rng)
            same_pool = [d for d in by_c[y] if d["image_path"] != r["image_path"]]
            same_ds = [rng.choice(same_pool) for _ in range(n_same)] if same_pool else []

            def A_of(donors):
                A = np.zeros((len(donors), 64), dtype=np.int8)
                for di, d in enumerate(donors):
                    drgb_w, daw = warp_donor_cells(d, r)
                    imgs = assemble_all(recip_rgb, rc, bg, drgb_w, daw)
                    A[di] = (classify(model, imgs, device) == y).astype(np.int8)
                return A

            def influence(A):
                return np.array([np.mean(A != A[:, np.arange(64) ^ (1 << p)]) for p in range(6)])

            Ax = A_of(cross_ds)
            As = A_of(same_ds) if same_ds else None
            g = Ax.mean(0); coeff = (H64 @ g) / 64.0; power = coeff ** 2
            W = np.array([power[POP == d].sum() for d in range(7)]); Wpos = W[1:].sum() + 1e-12
            flipped_any = (Ax == 0).any(0)
            r_star = 7
            for z in range(1, 64):
                if flipped_any[z]:
                    r_star = min(r_star, int(POP[z]))
            rows.append({
                "recipient": stem(r), "label": int(y),
                "cross_donors": [stem(d) for d in cross_ds], "same_donors": [stem(d) for d in same_ds],
                "inf_cross": influence(Ax).tolist(),
                "inf_same": influence(As).tolist() if As is not None else None,
                "W_by_degree": (W[1:] / Wpos).tolist(), "r": r_star,
            })
            if (ri + 1) % 10 == 0:
                print(f"  {ri+1}/{len(recips)} kept={len(rows)}  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:  # mirror the original's skip, but count it
            skipped["error"] += 1
            print(f"  skip {stem(r)}: {type(e).__name__}: {e}", flush=True)
            continue

    rows_s = [x for x in rows if x["inf_same"] is not None]
    C = np.array([x["inf_cross"] for x in rows_s]); S = np.array([x["inf_same"] for x in rows_s])
    Wd = np.array([x["W_by_degree"] for x in rows]); R = np.array([x["r"] for x in rows])
    # the original averages cross over all kept and same over those with controls; report both
    corr_orig = np.array([x["inf_cross"] for x in rows]).mean(0) - S.mean(0)
    res = {
        "config": dict(per_class=per_class, n_cross=n_cross, n_same=n_same, seed=seed,
                       n_candidates=len(recips)),
        "n_recipients": len(rows), "skipped": skipped,
        "cell_order": [f"{v}-{l}" for (v, l) in CELLS],
        "influence_cross_mean": np.array([x["inf_cross"] for x in rows]).mean(0).round(4).tolist(),
        "influence_same_mean": S.mean(0).round(4).tolist(),
        "influence_corrected_mean": corr_orig.round(4).tolist(),
        "variance_by_degree_mean": Wd.mean(0).round(4).tolist(),
        "semantic_L0_hist": {int(k): int((R == k).sum()) for k in sorted(set(R.tolist()))},
        "semantic_L0_median": float(np.median(R)),
        "bootstrap": summarize(C, S, Wd[[rows.index(x) for x in rows_s]], R[[rows.index(x) for x in rows_s]]),
        "records": rows,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print("corrected influence:", res["influence_corrected_mean"])
    print("variance by degree:", res["variance_by_degree_mean"])
    print("semantic-L0 hist:", res["semantic_L0_hist"])
    for k, v in res["bootstrap"].items():
        print(f"{k:10s} {v['value']:.4f}  [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]")
    print(f"n={len(rows)} skipped={skipped}  wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_class", type=int, default=14)
    ap.add_argument("--n_cross", type=int, default=4)
    ap.add_argument("--n_same", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="experiments/headline_spectrum_n277.json")
    a = ap.parse_args()
    main(a.per_class, a.n_cross, a.n_same, a.seed, a.out)
