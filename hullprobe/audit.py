"""Model-agnostic HullProbe audit: composites, the 64-point cube, and its readouts.

The audit needs only a `predict(images) -> labels` function, so any classifier can be scored on the
published recipient/donor lists (protocol/*.json). It reproduces scripts/part_lattice_spectrum.py:

    from hullprobe import audit
    recs = audit.load_records()                          # data/pose_B.json, keyed by vessel id
    pool = audit.load_pool()                             # protocol/attribution_pool_n277.json
    results = [audit.audit_recipient(predict, recs, r) for r in pool["recipients"]]
    print(audit.summarize([x for x in results if x]))   # hull-share, W1, W<=2, Pr[r=1] with CIs

Bit p of a configuration z indexes CELLS[p] = (hull|sup, fwd|mid|aft); z_p = 1 takes cell p from
the donor. A cube A has one row per donor and A[d, z] = 1 iff the composite phi(x, d, z) is still
classified as the recipient's true label.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.linalg import hadamard

from .lattice import CELLS, cell_masks
from .lattice_align import assemble_all_aligned, precompute_aligned
from .segmenter import load_region_png

H64 = hadamard(64).astype(np.float64)          # H[S, z] = (-1)^popcount(S & z)
POP = np.array([bin(i).count("1") for i in range(64)])
HULL = [i for i, (v, _) in enumerate(CELLS) if v == "hull"]


def load_records(pose="data/pose_B.json"):
    """Vessel records (image, silhouette, pose), keyed by vessel id (the image file stem)."""
    return {Path(r["image_path"]).stem: r for r in json.loads(Path(pose).read_text())}


def load_pool(path="protocol/attribution_pool_n277.json"):
    """A published recipient pool: recipients, true labels, and fixed donor identities."""
    return json.loads(Path(path).read_text())


def region_map(rec, region_dir="data/predictions"):
    """Hull/superstructure map of a vessel (0 background, 1 hull, 2 superstructure)."""
    return load_region_png(Path(region_dir) / f"{Path(rec['image_path']).stem}.png")


def composites(recipient, donor, region_dir="data/predictions"):
    """The 64 composites phi(x, d, z) as HxWx3 uint8 arrays, indexed by z."""
    rmap_r, rmap_d = region_map(recipient, region_dir), region_map(donor, region_dir)
    recip_rgb = np.asarray(Image.open(recipient["image_path"]).convert("RGB"))
    cells, _ = cell_masks(recipient, region_map=rmap_r)
    _, _, donor_rgb, donor_valid = precompute_aligned(donor, recipient, rmap_r, rmap_d)
    return assemble_all_aligned(recip_rgb, cells, donor_rgb, donor_valid)


def cube(predict, recipient, label, donors, region_dir="data/predictions"):
    """Correctness cube A (n_donors x 64) for one recipient."""
    A = np.zeros((len(donors), 64), np.int8)
    for i, d in enumerate(donors):
        A[i] = (np.asarray(predict(composites(recipient, d, region_dir))) == label)
    return A


def influence(A):
    """Per-cell flip influence Inf_p = Pr_{d,z}[F(z) != F(z xor e_p)]  (paper Eq. 4)."""
    return np.array([np.mean(A != A[:, np.arange(64) ^ (1 << p)]) for p in range(6)])


def walsh_degrees(A):
    """Fraction of Var_z[g] at interaction degree 1..6 for g = donor-averaged correctness."""
    power = ((H64 @ A.mean(0)) / 64.0) ** 2
    W = np.array([power[POP == k].sum() for k in range(7)])
    return W[1:] / (W[1:].sum() + 1e-12)


def semantic_l0(A):
    """Fewest swapped cells that flip the class with some donor; 7 if none does (paper Eq. 6)."""
    flipped = (A == 0).any(0)
    return min((int(POP[z]) for z in range(1, 64) if flipped[z]), default=7)


def audit_recipient(predict, records, entry, region_dir="data/predictions"):
    """Audit one pool entry; returns None if the classifier misclassifies the clean recipient."""
    r, y = records[entry["recipient"]], int(entry["label"])
    clean = np.asarray(Image.open(r["image_path"]).convert("RGB"))
    if int(np.asarray(predict([clean]))[0]) != y:
        return None
    Ax = cube(predict, r, y, [records[d] for d in entry["cross_donors"]], region_dir)
    As = cube(predict, r, y, [records[d] for d in entry["same_donors"]], region_dir)
    return {"recipient": entry["recipient"], "inf_cross": influence(Ax), "inf_same": influence(As),
            "W_by_degree": walsh_degrees(Ax), "r": semantic_l0(Ax)}


def summarize(results, n_boot=2000, seed=0):
    """Pooled corrected hull-share, W1, W<=2 and Pr[r=1], with 95% recipient-bootstrap CIs."""
    C = np.array([x["inf_cross"] for x in results]); S = np.array([x["inf_same"] for x in results])
    Wd = np.array([x["W_by_degree"] for x in results]); R = np.array([x["r"] for x in results])

    def stats(idx):
        corr = C[idx].mean(0) - S[idx].mean(0)
        return np.array([corr[HULL].sum() / corr.sum(), Wd[idx, 0].mean(),
                         Wd[idx, :2].sum(1).mean(), (R[idx] == 1).mean()])
    n = len(results); rs = np.random.RandomState(seed)
    boot = np.array([stats(rs.randint(0, n, n)) for _ in range(n_boot)])
    point = stats(np.arange(n))
    out = {"n": n}
    for i, k in enumerate(["hull_share", "W1", "W_le2", "frac_r1"]):
        out[k] = {"value": float(point[i]),
                  "ci": [float(np.percentile(boot[:, i], 2.5)), float(np.percentile(boot[:, i], 97.5))]}
    corr = C.mean(0) - S.mean(0)
    out["corrected_influence"] = {f"{v}-{l}": float(c) for (v, l), c in zip(CELLS, corr)}
    return out
