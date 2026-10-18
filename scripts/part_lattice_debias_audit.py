"""De-biasing audit on the part lattice (the spine).

Composites are model-independent, so we build the 64-vertex x donor cube ONCE per
recipient and classify it under every trained arm. For each arm we get its causal
part-influence vector (artifact-corrected) + hull-share + junta + semantic-L0.

The question: augmentations sold as spurious-robustness interventions (hull-swap,
bg-swap, CutMix) -- do they actually move the model's CAUSAL reliance off the
spurious hull, or do their headline robustness numbers move while the causal mass
stays put? bg-swap is the adversarial prediction: it should INCREASE hull reliance.

Writes experiments/part_lattice_debias_audit.json.
"""
from __future__ import annotations
import json, random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models import convnext_tiny

from hullprobe.lattice import cell_masks, CELLS
from hullprobe.swap import _background_fill
from hullprobe.segmenter import load_region_png
from scripts.part_lattice_spectrum import (
    test_pool, rmap, warp_donor_cells, assemble_all, H64, POP, tf)

ARMS = {
    "baseline": "experiments/robustness_soft/baseline.pt",
    "cutmix": "experiments/robustness_soft/cutmix.pt",
    "hullswap": "experiments/robustness_soft/hullswap.pt",
    "hullswap_soft": "experiments/robustness_soft/hullswap_soft.pt",
    "bgswap": "experiments/robustness/bgswap.pt",
}


def load(path, device):
    m = convnext_tiny(num_classes=26)
    st = torch.load(path, map_location="cpu"); st = st.get("model", st.get("state_dict", st))
    m.load_state_dict({k.replace("module.", ""): v for k, v in st.items()}, strict=False)
    return m.to(device).eval()


@torch.inference_mode()
def preds_all(models, imgs, device, batch=64):
    """Return {arm: argmax preds} for a shared list of composite images."""
    xs = torch.stack([tf(Image.fromarray(a)) for a in imgs])
    out = {a: [] for a in models}
    for i in range(0, len(xs), batch):
        xb = xs[i:i+batch].to(device)
        for a, m in models.items():
            out[a].append(m(xb).argmax(1).cpu().numpy())
    return {a: np.concatenate(v) for a, v in out.items()}


def spectrum_from_A(A):
    """A: (n_donor, 64) correctness. Return per-cell influence, junta, r*."""
    inf = np.array([np.mean(A != A[:, np.arange(64) ^ (1 << p)]) for p in range(6)])
    g = A.mean(0); coeff = (H64 @ g) / 64.0; power = coeff ** 2
    W = np.array([power[POP == d].sum() for d in range(7)]); Wpos = W[1:].sum() + 1e-12
    flipped_any = (A == 0).any(0)
    r = 7
    for z in range(1, 64):
        if flipped_any[z]:
            r = min(r, int(POP[z]))
    return inf, W[1] / Wpos, (W[1] + W[2]) / Wpos, r


def main(per_class=4, n_cross=4, n_same=2, seed=42):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if device == "cuda" and torch.cuda.mem_get_info()[0] < 2.2e9:
            device = "cpu"
    except Exception:
        pass
    print("device:", device, flush=True)
    rng = random.Random(seed)
    by_c = test_pool()
    models = {a: load(p, device) for a, p in ARMS.items() if Path(p).exists()}
    print("arms:", list(models), flush=True)

    recips = []
    for c, lst in by_c.items():
        lst = list(lst); rng.shuffle(lst); recips += lst[:per_class]
    all_v = [r for lst in by_c.values() for r in lst]

    acc = {a: {"inf": [], "j1": [], "j2": [], "r": [], "clean_ok": 0} for a in models}
    share_recip = {a: {} for a in models}   # ri -> hull_share (paired across arms)
    n_used = 0
    for ri, r in enumerate(recips):
        try:
            y = r["label"]
            recip_rgb = np.asarray(Image.open(r["image_path"]).convert("RGB"))
            rc, vessel = cell_masks(r, region_map=rmap(r))
            bg = _background_fill(recip_rgb, vessel)
            cross_ds = [rng.choice([d for d in all_v if d["label"] != y]) for _ in range(n_cross)]

            # build the model-independent composite cube (cross donors)
            imgs = []
            per_donor_slices = []
            for d in cross_ds:
                drgb_w, daw = warp_donor_cells(d, r)
                comp = assemble_all(recip_rgb, rc, bg, drgb_w, daw)  # 64 imgs, z=0 is index 0
                per_donor_slices.append((len(imgs), len(imgs) + 64))
                imgs.extend(comp)
            P = preds_all(models, imgs, device)

            used = False
            for a in models:
                A = np.zeros((n_cross, 64), np.int8)
                for di, (s, e) in enumerate(per_donor_slices):
                    A[di] = (P[a][s:e] == y).astype(np.int8)
                # condition on this arm getting the clean recipient (z=0) right
                if A[:, 0].mean() < 0.5:      # majority of donors' z=0 (== clean) wrong
                    continue
                acc[a]["clean_ok"] += 1
                inf, j1, j2, rstar = spectrum_from_A(A)
                acc[a]["inf"].append(inf); acc[a]["j1"].append(j1)
                acc[a]["j2"].append(j2); acc[a]["r"].append(rstar)
                share_recip[a][ri] = float(inf[:3].sum() / (inf.sum() + 1e-9))
                used = True
            n_used += int(used)
            if (ri + 1) % 10 == 0:
                print(f"  {ri+1}/{len(recips)}", flush=True)
        except Exception:
            continue

    cell_names = [f"{v}-{l}" for (v, l) in CELLS]
    res = {"cell_order": cell_names, "n_recip_pool": len(recips), "arms": {}}
    print("\n DE-BIASING AUDIT: causal part-influence per arm")
    print(f"{'arm':14s} {'clean_n':>7s} {'hull_share':>10s} {'sup_infl':>8s} {'hull_infl':>9s} "
          f"{'junta<=2':>8s} {'frac_r1':>7s}")
    for a in models:
        inf = np.array(acc[a]["inf"]);
        if len(inf) == 0:
            continue
        m = inf.mean(0)
        hull = m[:3].sum(); sup = m[3:].sum(); share = hull / (hull + sup + 1e-9)
        r = np.array(acc[a]["r"]); fr1 = float((r == 1).mean())
        res["arms"][a] = {
            "clean_n": acc[a]["clean_ok"],
            "influence": m.round(4).tolist(),
            "hull_influence": float(hull), "sup_influence": float(sup),
            "hull_share": float(share),
            "junta_deg2": float(np.mean(acc[a]["j2"])),
            "frac_r1": fr1, "median_r": float(np.median(r)),
        }
        print(f"{a:14s} {acc[a]['clean_ok']:7d} {share:10.3f} {sup:8.3f} {hull:9.3f} "
              f"{np.mean(acc[a]['j2']):8.3f} {fr1:7.3f}")

    # paired-bootstrap CI of (arm - baseline) hull-share over shared recipients
    if "baseline" in models:
        base = share_recip["baseline"]
        rng2 = np.random.RandomState(0)
        res["paired_delta_vs_baseline"] = {}
        print(f"\nbaseline hull-share = {res['arms']['baseline']['hull_share']:.3f}")
        print("paired Δ(hull-share) vs baseline  [95% bootstrap CI], n_paired:")
        for a in models:
            if a == "baseline":
                continue
            shared = sorted(set(base) & set(share_recip[a]))
            if len(shared) < 5:
                continue
            deltas = np.array([share_recip[a][i] - base[i] for i in shared])
            boot = np.array([rng2.choice(deltas, len(deltas), replace=True).mean()
                             for _ in range(5000)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            sig = "" if (lo < 0 < hi) else "  *"
            res["paired_delta_vs_baseline"][a] = {
                "mean": float(deltas.mean()), "lo": float(lo), "hi": float(hi), "n": len(shared)}
            print(f"  {a:14s} {deltas.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]  n={len(shared)}{sig}")

    Path("experiments").mkdir(exist_ok=True)
    Path("experiments/part_lattice_debias_audit.json").write_text(json.dumps(res, indent=1))
    print("wrote experiments/part_lattice_debias_audit.json")


if __name__ == "__main__":
    main()
