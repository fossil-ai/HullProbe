"""De-biasing audit + ABSOLUTE influence (answers the ratio-vs-absolute review point).

Same pool/logic as part_lattice_debias_multiseed.py (per_class=4, cross-class cube),
but also records, per (arm, seed), the mean ABSOLUTE hull influence, superstructure
influence, and TOTAL influence across recipients -- not just the hull-share ratio.
This lets us check whether an unchanged hull-share hides a proportional drop in
absolute influence. Reproduces the Table-2 hull-share as a sanity check.

Writes experiments/debias_absinf.json.
"""
from __future__ import annotations
import json, random
from pathlib import Path

import numpy as np
from PIL import Image

from hullprobe.lattice import cell_masks
from hullprobe.swap import _background_fill
from scripts.part_lattice_spectrum import (
    test_pool, rmap, warp_donor_cells, assemble_all, compatible_cross)
from scripts.part_lattice_debias_audit import load, preds_all, spectrum_from_A

SEEDS = [42, 43, 44]
ARMS = ["baseline", "cutmix", "bgswap", "hullswap", "jtt", "lff", "afr"]


def main(per_class=4, n_cross=4, seed=42):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    rng = random.Random(seed)
    by_c = test_pool()

    models = {}
    for a in ARMS:
        for s in SEEDS:
            p = Path(f"experiments/arms_seed{s}/{a}.pt")
            if p.exists():
                models[f"{a}@{s}"] = load(str(p), device)
    print("models:", len(models), flush=True)

    recips = []
    for c, lst in by_c.items():
        lst = list(lst); rng.shuffle(lst); recips += lst[:per_class]
    all_v = [r for lst in by_c.values() for r in lst]

    # per key -> {ri: (hull_abs, sup_abs, tot_abs, hull_share)}
    rec = {k: {} for k in models}
    for ri, r in enumerate(recips):
        try:
            y = r["label"]
            recip_rgb = np.asarray(Image.open(r["image_path"]).convert("RGB"))
            rc, vessel = cell_masks(r, region_map=rmap(r))
            bg = _background_fill(recip_rgb, vessel)
            cross_ds = compatible_cross(r, all_v, n_cross, rng)
            imgs, slices = [], []
            for d in cross_ds:
                drgb_w, daw = warp_donor_cells(d, r)
                comp = assemble_all(recip_rgb, rc, bg, drgb_w, daw)
                slices.append((len(imgs), len(imgs) + 64)); imgs.extend(comp)
            P = preds_all(models, imgs, device)
            for k in models:
                A = np.zeros((n_cross, 64), np.int8)
                for di, (s0, e0) in enumerate(slices):
                    A[di] = (P[k][s0:e0] == y).astype(np.int8)
                if A[:, 0].mean() < 0.5:      # arm gets clean recipient wrong
                    continue
                inf, j1, j2, rstar = spectrum_from_A(A)
                hull_abs = float(inf[:3].sum()); sup_abs = float(inf[3:].sum())
                tot_abs = float(inf.sum())
                rec[k][ri] = (hull_abs, sup_abs, tot_abs,
                              hull_abs / (tot_abs + 1e-9))
            if (ri + 1) % 20 == 0:
                print(f"  {ri+1}/{len(recips)}", flush=True)
        except Exception as e:
            print("skip", ri, repr(e), flush=True)
            continue

    def arm_seed(a, s):
        return rec.get(f"{a}@{s}", {})

    out = {"per_class": per_class, "n_cross": n_cross, "arms": {}}
    base = {s: arm_seed("baseline", s) for s in SEEDS}
    hdr = f"{'arm':10s} {'hull_abs':>9s} {'sup_abs':>9s} {'tot_abs':>9s} {'hull_share':>11s} {'Dhull_abs':>10s} {'Dtot_abs':>9s}"
    print("\n===== ABSOLUTE INFLUENCE DE-BIASING AUDIT (mean over seeds) =====")
    print(hdr)
    for a in ARMS:
        hA, sA, tA, sh, dH, dT, ns = [], [], [], [], [], [], []
        for s in SEEDS:
            d = arm_seed(a, s)
            if not d:
                continue
            vals = np.array([d[i][:4] for i in d])  # (n,4)
            hA.append(vals[:, 0].mean()); sA.append(vals[:, 1].mean())
            tA.append(vals[:, 2].mean()); sh.append(vals[:, 3].mean())
            ns.append(len(d))
            b = base[s]; shared = set(d) & set(b)
            if shared:
                dH.append(np.mean([d[i][0] - b[i][0] for i in shared]))
                dT.append(np.mean([d[i][2] - b[i][2] for i in shared]))
        f = lambda x: (float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0)
        hm, hs = f(hA); sm, ss = f(sA); tm, ts = f(tA); shm, shs = f(sh)
        dhm, dhs = f(dH) if dH else (0.0, 0.0)
        dtm, dts = f(dT) if dT else (0.0, 0.0)
        out["arms"][a] = {
            "hull_abs_mean": hm, "hull_abs_std": hs, "sup_abs_mean": sm, "sup_abs_std": ss,
            "tot_abs_mean": tm, "tot_abs_std": ts, "hull_share_mean": shm, "hull_share_std": shs,
            "delta_hull_abs_mean": dhm, "delta_hull_abs_std": dhs,
            "delta_tot_abs_mean": dtm, "delta_tot_abs_std": dts,
            "n_per_seed": ns,
        }
        print(f"{a:10s} {hm:.3f}±{hs:.3f} {sm:.3f}±{ss:.3f} {tm:.3f}±{ts:.3f} "
              f"{shm:.3f}±{shs:.3f}   {('---' if a=='baseline' else f'{dhm:+.3f}'):>10s} "
              f"{('---' if a=='baseline' else f'{dtm:+.3f}'):>9s}")

    Path("experiments/debias_absinf.json").write_text(json.dumps(out, indent=1))
    print("\nwrote experiments/debias_absinf.json")


if __name__ == "__main__":
    main()
