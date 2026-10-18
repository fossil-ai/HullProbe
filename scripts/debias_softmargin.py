"""Fixed-recipient, soft-margin de-biasing audit (addresses the correct-only
recipient-conditioning review point).

Unlike the hard-flip audit, this scores every arm on the SAME fixed recipient set
(no correct-only filter) using a continuous response: g_soft(z) = P(true class)
for each swap config. Soft cell influence is the mean |g_soft(z)-g_soft(z^e_p)|,
defined even when the model misclassifies the original. Reports soft hull-share and
its paired change vs ERM on the identical recipient population.

Writes experiments/debias_softmargin.json.
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
from scripts.part_lattice_debias_audit import load, tf

SEEDS = [42, 43, 44]
ARMS = ["baseline", "cutmix", "bgswap", "hullswap", "jtt", "lff", "afr"]
TOGGLE = np.array([[z ^ (1 << p) for z in range(64)] for p in range(6)])  # (6,64)


def probs_all(models, imgs, device, batch=64):
    import torch
    xs = torch.stack([tf(Image.fromarray(a)) for a in imgs])
    out = {a: [] for a in models}
    with torch.inference_mode():
        for i in range(0, len(xs), batch):
            xb = xs[i:i + batch].to(device)
            for a, m in models.items():
                out[a].append(torch.softmax(m(xb), 1).cpu().numpy())
    return {a: np.concatenate(v) for a, v in out.items()}


def soft_influence(G):  # G: (n_donor, 64) true-class prob
    return np.array([np.mean(np.abs(G - G[:, TOGGLE[p]])) for p in range(6)])


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

    # per key -> {ri: (hull_soft, tot_soft, share_soft)}  -- NO correct-only filter
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
            Pr = probs_all(models, imgs, device)
            for k in models:
                G = np.zeros((n_cross, 64), np.float32)
                for di, (s0, e0) in enumerate(slices):
                    G[di] = Pr[k][s0:e0, y]
                inf = soft_influence(G)
                hull = float(inf[:3].sum()); tot = float(inf.sum())
                rec[k][ri] = (hull, tot, hull / (tot + 1e-9))
            if (ri + 1) % 20 == 0:
                print(f"  {ri+1}/{len(recips)}", flush=True)
        except Exception as e:
            print("skip", ri, repr(e), flush=True)
            continue

    def arm_seed(a, s):
        return rec.get(f"{a}@{s}", {})

    out = {"per_class": per_class, "n_cross": n_cross, "note": "soft margin, fixed recipients, no correct-only filter", "arms": {}}
    base = {s: arm_seed("baseline", s) for s in SEEDS}
    print("\n===== FIXED-RECIPIENT SOFT-MARGIN AUDIT (mean over seeds) =====")
    print(f"{'arm':10s} {'hull_soft':>10s} {'tot_soft':>9s} {'share_soft':>11s} {'Dshare':>15s} {'n(fixed)':>9s}")
    for a in ARMS:
        H, T, S, dS, ns = [], [], [], [], []
        for s in SEEDS:
            d = arm_seed(a, s)
            if not d:
                continue
            v = np.array([d[i] for i in d])
            H.append(v[:, 0].mean()); T.append(v[:, 1].mean()); S.append(v[:, 2].mean())
            ns.append(len(d))
            b = base[s]; shared = set(d) & set(b)
            if shared:
                dS.append(np.mean([d[i][2] - b[i][2] for i in shared]))
        f = lambda x: (float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0)
        hm, hs = f(H); tm, ts = f(T); sm, ss = f(S); dm, ds = f(dS) if dS else (0.0, 0.0)
        out["arms"][a] = {"hull_soft_mean": hm, "tot_soft_mean": tm,
                          "share_soft_mean": sm, "share_soft_std": ss,
                          "delta_share_mean": dm, "delta_share_std": ds, "n_per_seed": ns}
        dtxt = "---" if a == "baseline" else f"{dm:+.3f}±{ds:.3f}"
        print(f"{a:10s} {hm:>10.3f} {tm:>9.3f} {sm:>7.3f}±{ss:.3f} {dtxt:>15s} {str(ns):>9s}")

    Path("experiments/debias_softmargin.json").write_text(json.dumps(out, indent=1))
    print("\nwrote experiments/debias_softmargin.json")


if __name__ == "__main__":
    main()
