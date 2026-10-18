"""Multi-seed causal part-lattice de-biasing audit (Study 1, final).

Runs the causal hull-influence audit on all 5 arms x 3 seeds (JTT included),
reusing the model-independent composite cube. Reports per-arm hull-share and
single-cell fragility with ACROSS-SEED mean+-std (the multi-seed CI reviewers
want), plus the paired delta vs baseline aggregated over seeds.

Reads experiments/arms_seed{42,43,44}/{arm}.pt. Writes
experiments/part_lattice_debias_multiseed.json.
"""
from __future__ import annotations
import json, random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

from hullprobe.lattice import cell_masks, CELLS
from hullprobe.swap import _background_fill
from scripts.part_lattice_spectrum import (
    test_pool, rmap, warp_donor_cells, assemble_all, compatible_cross)
from scripts.part_lattice_debias_audit import load, preds_all, spectrum_from_A

SEEDS = [42, 43, 44]
ARMS = ["baseline", "cutmix", "bgswap", "hullswap", "jtt", "lff", "afr"]


def main(per_class=4, n_cross=4, seed=42):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if device == "cuda" and torch.cuda.mem_get_info()[0] < 3.0e9:
            device = "cpu"
    except Exception:
        pass
    print("device:", device, flush=True)
    rng = random.Random(seed)
    by_c = test_pool()

    # 15 models keyed arm@seed
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

    # per (arm,seed): list of per-recipient hull-share; and r* for fragility
    share = {k: {} for k in models}       # key -> {ri: hull_share}
    frac = {k: [] for k in models}        # key -> list of (r==1)
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
                share[k][ri] = float(inf[:3].sum() / (inf.sum() + 1e-9))
                frac[k].append(int(rstar == 1))
            if (ri + 1) % 20 == 0:
                print(f"  {ri+1}/{len(recips)}", flush=True)
        except Exception:
            continue

    # aggregate: per-arm across-seed mean+-std of (hull-share, frac_r1, paired delta vs baseline)
    def arm_seed_share(a, s):
        d = share.get(f"{a}@{s}", {})
        return d
    res = {"arms": {}}
    print("\n MULTI-SEED CAUSAL DE-BIASING AUDIT")
    print(f"{'arm':10s} {'hull_share(mean+-std over seeds)':32s} {'frac_r1':14s} {'Δ vs base':16s}")
    base_share = {s: arm_seed_share("baseline", s) for s in SEEDS}
    for a in ARMS:
        seed_shares, seed_fracs, seed_deltas = [], [], []
        for s in SEEDS:
            d = arm_seed_share(a, s)
            if d:
                seed_shares.append(np.mean(list(d.values())))
            fk = frac.get(f"{a}@{s}", [])
            if fk:
                seed_fracs.append(np.mean(fk))
            # paired delta vs baseline over shared recipients this seed
            b = base_share[s]
            shared = set(d) & set(b)
            if shared:
                seed_deltas.append(np.mean([d[i] - b[i] for i in shared]))
        ss = np.array(seed_shares); ff = np.array(seed_fracs); dd = np.array(seed_deltas)
        res["arms"][a] = {
            "hull_share_mean": float(ss.mean()), "hull_share_std": float(ss.std(ddof=1)) if len(ss) > 1 else 0.0,
            "hull_share_per_seed": ss.round(4).tolist(),
            "frac_r1_mean": float(ff.mean()) if len(ff) else None,
            "delta_vs_base_mean": float(dd.mean()) if len(dd) else 0.0,
            "delta_vs_base_std": float(dd.std(ddof=1)) if len(dd) > 1 else 0.0,
            "delta_per_seed": dd.round(4).tolist(),
        }
        r = res["arms"][a]
        dtxt = "---" if a == "baseline" else f"{r['delta_vs_base_mean']:+.3f}±{r['delta_vs_base_std']:.3f}"
        print(f"{a:10s} {r['hull_share_mean']:.3f} ± {r['hull_share_std']:.3f}  "
              f"(seeds {r['hull_share_per_seed']}) {r['frac_r1_mean']:.3f}         {dtxt}")

    Path("experiments/part_lattice_debias_multiseed.json").write_text(json.dumps(res, indent=1))
    print("wrote experiments/part_lattice_debias_multiseed.json")


if __name__ == "__main__":
    main()
