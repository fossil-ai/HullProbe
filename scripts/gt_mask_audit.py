"""Camera-ready (reviewer hCSj / AC): does segmentation error propagate into HullProbe?

Re-run the audit on the annotated vessels the SegFormer segmenter never trained on, once with
the segmenter's predicted hull/superstructure maps and once with the human annotations, and
compare per-cell influence and corrected hull-share on identical recipients, donors and
classifier.

Held-out set H: the SegFormer validation split (202 of the 1,014 annotations), listed in
protocol/segformer_split.json (scripts/train_segformer.py draws it with random.seed(42) over the
annotation list; the per-class counts match experiments/segformer_metrics.json).

Arms (donor identities are fixed once and shared by every arm):
  pred : recipient = SegFormer map, donors = SegFormer maps     (the deployed pipeline)
  gt   : recipient = annotation,    donors = annotations         (ground truth end to end)
  gt_r : recipient = annotation,    donors = SegFormer maps      (recipient-side error only)
Donors are drawn from H so that every donor also has an annotation the segmenter never saw:
K_x=4 aspect-compatible cross-class donors (compatible_cross, as in the main audit) and
K_= = 2 same-class controls (random with replacement, as in scripts/part_lattice_spectrum.py).
The vessel silhouette, PCA pose, donor choice and the clean-correct filter do not depend on the
hull/superstructure map, so the arms differ only in where the hull/superstructure boundary lies.

Outputs experiments/gt_mask_audit.json with per-recipient records, per-arm pooled
statistics with recipient-bootstrap CIs, paired bootstrap CIs for gt - pred, and the
segmentation agreement (within-vessel IoU, pixel disagreement) on the same recipients.

Usage (repo root):
  python -m scripts.gt_mask_audit
"""
from __future__ import annotations
import argparse, json, random, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from hullprobe.lattice import cell_masks, CELLS
from hullprobe.lattice_align import precompute_aligned, assemble_all_aligned
from hullprobe.segmenter import load_region_png
from scripts.part_lattice_spectrum import (
    B_MANIFEST, load_model, compatible_cross, classify, H64, POP)

ANNOT = Path("data/annotations")
SPLIT = Path("protocol/segformer_split.json")
PRED = Path("data/predictions")
SEG_METRICS = Path("experiments/segformer_metrics.json")
ARMS = {"pred": ("pred", "pred"), "gt": ("gt", "gt"), "gt_r": ("gt", "pred")}


def stem(rec):
    return Path(rec["image_path"]).stem


def heldout_ids():
    """The SegFormer validation split (never used to train the segmenter)."""
    return json.loads(SPLIT.read_text())["val"]


def load_maps(ids):
    maps = {}
    for s in ids:
        gt_p, pr_p = ANNOT / f"labels/{s}.png", PRED / f"{s}.png"
        assert gt_p.exists() and pr_p.exists(), f"missing map for {s}"
        maps[s] = {"gt": load_region_png(gt_p), "pred": load_region_png(pr_p)}
    return maps


def seg_agreement(gt, pr):
    v = gt > 0
    out = {"disagree_frac": float((gt[v] != pr[v]).mean())}
    for name, k in (("hull", 1), ("sup", 2)):
        a, b = (gt == k) & v, (pr == k) & v
        u = (a | b).sum()
        out[f"{name}_iou"] = float((a & b).sum() / u) if u else float("nan")
    out["sup_frac_gt"] = float((gt[v] == 2).mean())
    out["sup_frac_pred"] = float((pr[v] == 2).mean())
    return out


def influence(A):
    return np.array([np.mean(A != A[:, np.arange(64) ^ (1 << p)]) for p in range(6)])


def walsh_and_radius(A):
    g = A.mean(0); coeff = (H64 @ g) / 64.0; power = coeff ** 2
    W = np.array([power[POP == d].sum() for d in range(7)]); Wpos = W[1:].sum() + 1e-12
    flipped_any = (A == 0).any(0)
    r = 7
    for z in range(1, 64):
        if flipped_any[z]:
            r = min(r, int(POP[z]))
    return (W[1:] / Wpos).tolist(), r


def pooled(C, S):
    corr = C.mean(0) - S.mean(0)
    return corr, corr[:3].sum() / corr.sum()


def arm_stats(rows, arm, idx_sets, n_boot=2000, seed=0):
    """Pooled corrected hull-share, per-cell corrected influence, W1, W<=2, Pr[r=1]."""
    C = np.array([r[arm]["inf_cross"] for r in rows]); S = np.array([r[arm]["inf_same"] for r in rows])
    Wd = np.array([r[arm]["W_by_degree"] for r in rows]); R = np.array([r[arm]["r"] for r in rows])
    def stats(idx):
        corr, hs = pooled(C[idx], S[idx])
        return np.concatenate([[hs, C[idx].mean(0)[:3].sum() / C[idx].mean(0).sum(),
                                Wd[idx, 0].mean(), Wd[idx, :2].sum(1).mean(), (R[idx] == 1).mean()],
                               corr])
    point = stats(np.arange(len(rows)))
    boot = np.array([stats(i) for i in idx_sets])
    names = ["hull_share", "hull_share_raw", "W1", "W_le2", "frac_r1"] + \
            [f"corr_{v}-{l}" for (v, l) in CELLS]
    return {k: {"value": float(point[i]), "ci": [float(np.percentile(boot[:, i], 2.5)),
                                                 float(np.percentile(boot[:, i], 97.5))]}
            for i, k in enumerate(names)}, boot


def summarize(rows, label, n_boot=2000):
    n = len(rows)
    rs = np.random.RandomState(0)
    idx_sets = [rs.randint(0, n, n) for _ in range(n_boot)]   # shared -> paired bootstrap
    out = {"n": n, "arms": {}}
    boots = {}
    for arm in ARMS:
        out["arms"][arm], boots[arm] = arm_stats(rows, arm, idx_sets)
    names = list(out["arms"]["pred"].keys())
    for a, b in (("gt", "pred"), ("gt_r", "pred")):
        d = boots[a] - boots[b]
        pt = np.array([out["arms"][a][k]["value"] - out["arms"][b][k]["value"] for k in names])
        out[f"diff_{a}_minus_{b}"] = {k: {"value": float(pt[i]),
                                          "ci": [float(np.percentile(d[:, i], 2.5)),
                                                 float(np.percentile(d[:, i], 97.5))]}
                                      for i, k in enumerate(names)}
    # per-recipient agreement of the corrected influence profile (gt vs pred)
    cg = np.array([np.array(r["gt"]["inf_cross"]) - np.array(r["gt"]["inf_same"]) for r in rows])
    cp = np.array([np.array(r["pred"]["inf_cross"]) - np.array(r["pred"]["inf_same"]) for r in rows])
    out["per_recipient_corr_profile_pearson"] = float(np.corrcoef(cg.ravel(), cp.ravel())[0, 1])
    seg = np.array([[r["seg"][k] for k in ("disagree_frac", "hull_iou", "sup_iou",
                                             "sup_frac_gt", "sup_frac_pred")] for r in rows])
    out["seg_agreement_mean"] = dict(zip(("disagree_frac", "hull_iou", "sup_iou", "sup_frac_gt",
                                          "sup_frac_pred"), np.nanmean(seg, 0).round(4).tolist()))
    out["seg_disagree_median"] = float(np.median(seg[:, 0]))
    print(f"\n=== {label}: n={n} ===")
    for arm in ARMS:
        a = out["arms"][arm]
        print(f"  {arm:5s} hull-share {a['hull_share']['value']:.3f} "
              f"[{a['hull_share']['ci'][0]:.3f},{a['hull_share']['ci'][1]:.3f}]  "
              f"raw {a['hull_share_raw']['value']:.3f}  W1 {a['W1']['value']:.3f}  "
              f"W<=2 {a['W_le2']['value']:.3f}  Pr[r=1] {a['frac_r1']['value']:.3f}")
    for key in ("diff_gt_minus_pred", "diff_gt_r_minus_pred"):
        d = out[key]["hull_share"]
        print(f"  {key}: {d['value']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]")
    print(f"  seg agreement: {out['seg_agreement_mean']}  median disagree {out['seg_disagree_median']:.3f}")
    print(f"  per-recipient corrected-profile r(gt,pred) = {out['per_recipient_corr_profile_pearson']:.3f}")
    return out


def main(n_cross, n_same, seed, out):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.mem_get_info()[0] < 2.2e9:
        device = "cpu"
    print("device:", device, flush=True)

    # --- held-out annotated set, with classifier labels and split ---
    ids = heldout_ids()
    seg_n = {c: v["n"] for c, v in json.loads(SEG_METRICS.read_text())["per_class_superstructure_iou"].items()}
    pose = {stem(r): r for r in json.loads(Path("data/pose_B.json").read_text())}
    man = {stem(r): r for r in json.loads(Path(B_MANIFEST).read_text())["records"]}
    cls_count = defaultdict(int)
    for s in ids:
        cls_count[pose[s]["class"]] += 1
    mismatch = {c: (cls_count.get(c, 0), n) for c, n in seg_n.items() if cls_count.get(c, 0) != n}
    assert not mismatch, f"held-out split does not match segformer metrics: {mismatch}"
    print(f"held-out split reproduced: {len(ids)} ids, per-class counts match metrics.json", flush=True)

    H = []
    for s in ids:
        r = dict(pose[s]); m = man.get(s)
        if m is None or m.get("label") is None:
            continue
        r["label"] = int(m["label"]); r["source_split"] = m["source_split"]
        H.append(r)
    maps = load_maps([stem(r) for r in H])
    by_c = defaultdict(list)
    for r in H:
        by_c[r["label"]].append(r)
    print(f"H: {len(H)} records ({sum(r['source_split']=='test' for r in H)} in MARVEL test split)",
          flush=True)

    model = load_model(device)
    rng = random.Random(seed)
    rows, skipped = [], defaultdict(int)
    t0 = time.time()
    for ri, r in enumerate(H):
        s, y = stem(r), r["label"]
        try:
            recip_rgb = np.asarray(Image.open(r["image_path"]).convert("RGB"))
            if classify(model, [recip_rgb], device)[0] != y:
                skipped["misclassified"] += 1; continue
            same_pool = [d for d in by_c[y] if stem(d) != s]
            if not same_pool:
                skipped["no_same_class_donor"] += 1; continue
            cross_ds = compatible_cross(r, H, n_cross, rng)
            same_ds = [rng.choice(same_pool) for _ in range(n_same)]
            row = {"recipient": s, "label": y, "split": r["source_split"],
                   "cross_donors": [stem(d) for d in cross_ds],
                   "same_donors": [stem(d) for d in same_ds],
                   "seg": seg_agreement(maps[s]["gt"], maps[s]["pred"])}
            for arm, (rsrc, dsrc) in ARMS.items():
                rm = maps[s][rsrc]
                rc, _ = cell_masks(r, region_map=rm)   # empty cells allowed, as in the main audit

                def A_of(donors):
                    A = np.zeros((len(donors), 64), np.int8)
                    for di, d in enumerate(donors):
                        _, _, drgb, dval = precompute_aligned(d, r, rm, maps[stem(d)][dsrc])
                        imgs = assemble_all_aligned(recip_rgb, rc, drgb, dval)
                        A[di] = (classify(model, imgs, device) == y).astype(np.int8)
                    return A
                Ax, As = A_of(cross_ds), A_of(same_ds)
                Wd, rr = walsh_and_radius(Ax)
                row[arm] = {"inf_cross": influence(Ax).tolist(), "inf_same": influence(As).tolist(),
                            "W_by_degree": Wd, "r": rr,
                            "empty_cells": [f"{v}-{l}" for (v, l) in CELLS if rc[(v, l)].sum() == 0]}
            rows.append(row)
        except Exception as e:
            skipped[f"error:{type(e).__name__}"] += 1
            print(f"  skip {s}: {type(e).__name__}: {e}", flush=True)
            continue
        if (ri + 1) % 10 == 0:
            print(f"  {ri+1}/{len(H)} kept={len(rows)}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"kept {len(rows)}; skipped {dict(skipped)}", flush=True)
    res = {"config": dict(n_cross=n_cross, n_same=n_same, seed=seed, heldout=len(ids),
                          donor_bank="segformer-held-out annotated vessels (H)"),
           "skipped": dict(skipped),
           "all_heldout": summarize(rows, "all held-out (SegFormer val)"),
           "test_split_only": summarize([r for r in rows if r["split"] == "test"],
                                        "held-out AND MARVEL test split"),
           "records": rows}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print(f"wrote {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_cross", type=int, default=4)
    ap.add_argument("--n_same", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="experiments/gt_mask_audit.json")
    a = ap.parse_args()
    main(a.n_cross, a.n_same, a.seed, a.out)
