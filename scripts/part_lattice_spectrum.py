"""Part-lattice causal spectrum on the 2x3 anatomical cube.

For each recipient we evaluate the class-evidence function on all 2^6=64 vertices
g(z)=E_donor 1[argmax f(compose_lattice(x,d,z))=y], for cross-class donors (the
intervention) and same-class donors (the artifact control). From g we compute,
exactly:
  - per-cell causal flip-influence Inf_p (cross, and artifact-corrected),
  - Walsh-Hadamard spectrum -> ANOVA variance by interaction degree (junta test),
  - semantic-L0 certificate r(x) = min #cells whose swap flips the class.

Fast: warp each of the 6 donor cells into the recipient frame ONCE per donor,
then assemble 64 composites by numpy mask algebra (matches lattice.compose_lattice).

Writes experiments/part_lattice_spectrum.json.
"""
from __future__ import annotations
import os
import json, random, math
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.linalg import hadamard
import torch, torch.nn as nn
import torchvision.transforms as T
from torchvision.models import convnext_tiny

from hullprobe.lattice import cell_masks, CELLS
from hullprobe.swap import similarity_to_canonical, _affine_coeffs, _background_fill
from hullprobe.segmenter import load_region_png

B_MANIFEST = os.environ.get("MARVEL_B_MANIFEST", "data/marvel_b/manifest_pose_maskpca.json")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
PRED = Path("data/predictions")
CKPT = "experiments/baseline_clf/best.pt"
CLASSES = json.loads(Path("data/classes_B.json").read_text())
tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
H64 = hadamard(64).astype(np.float64)            # H[S,z]=(-1)^popcount(S&z)
POP = np.array([bin(i).count("1") for i in range(64)])

_M = {}
def rmap(rec):
    s = Path(rec["image_path"]).stem
    if s not in _M:
        p = PRED / f"{s}.png"; _M[s] = load_region_png(p) if p.exists() else None
    return _M[s]


def load_model(device):
    m = convnext_tiny(num_classes=26)
    st = torch.load(CKPT, map_location="cpu"); st = st.get("model", st.get("state_dict", st))
    m.load_state_dict({k.replace("module.", ""): v for k, v in st.items()}, strict=False)
    return m.to(device).eval()


def test_pool():
    posB = json.loads(Path("data/pose_B.json").read_text())
    bm = {r["image_path"]: r for r in json.loads(Path(B_MANIFEST).read_text())["records"]}
    by_c = defaultdict(list)
    for r in posB:
        b = bm.get(r["image_path"])
        if (b and b.get("source_split") == "test" and b.get("label") is not None
                and r.get("full_vessel_box_xyxy")
                and (PRED / f"{Path(r['image_path']).stem}.png").exists()):
            r = dict(r); r["label"] = int(b["label"]); by_c[r["label"]].append(r)
    return by_c


from hullprobe.lattice_align import precompute_aligned, assemble_all_aligned

_ASPECT = {}
def _aspect(rec):
    k = rec["image_path"]
    if k not in _ASPECT:
        try:
            _, ves = cell_masks(rec, region_map=rmap(rec))
            ys, xs = np.nonzero(ves)
            _ASPECT[k] = (xs.max() - xs.min() + 1) / max(1, ys.max() - ys.min() + 1)
        except Exception:
            _ASPECT[k] = None
    return _ASPECT[k]


def compatible_cross(r, all_v, k, rng, pool=60):
    """k cross-class donors closest in aspect ratio to r (shape-compatible) -- so the
    per-cell registration is a clean fit, isolating a class change under matched shape."""
    y = r["label"]; ar = _aspect(r)
    cands = [d for d in all_v if d["label"] != y]; rng.shuffle(cands)
    scored = [(abs(_aspect(d) - ar), d) for d in cands[:pool]
              if _aspect(d) is not None and ar is not None]
    scored.sort(key=lambda t: t[0])
    return [d for _, d in scored[:k]] or [rng.choice(cands)]


def warp_donor_cells(donor_rec, recip_rec):
    """Per-cell target-conditioned donor warp (aligned operator): returns
    ([donor_rgb_c HxWx3 | None]*6, [donor_valid_c bool | None]*6)."""
    _rgb, _rc, drgb_c, dval_c = precompute_aligned(
        donor_rec, recip_rec, rmap(recip_rec), rmap(donor_rec))
    return drgb_c, dval_c


def assemble_all(recip_rgb, recip_cells, bg, donor_rgb_w, donor_aw):
    """Return list of 64 composite uint8 images for every vertex z.

    EXACT TARGET-SUPPORT STENCIL: a swapped cell fills the RECIPIENT's own cell
    region with donor texture warped into frame -- so the composite silhouette is
    always the recipient's (no donor overhang, no background holes). This isolates
    the intervention to appearance within a fixed cell support. Residual pixels the
    warped donor does not cover are inpainted from surrounding donor texture.
    """
    # donor_rgb_w, donor_aw are now per-cell lists (aligned operator); delegate.
    return assemble_all_aligned(recip_rgb, recip_cells, donor_rgb_w, donor_aw)


@torch.inference_mode()
def classify(model, imgs, device, batch=64):
    preds = []
    for i in range(0, len(imgs), batch):
        x = torch.stack([tf(Image.fromarray(a)) for a in imgs[i:i+batch]]).to(device)
        preds.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def main(per_class=3, n_cross=4, n_same=2, seed=42, out="experiments/part_lattice_spectrum.json"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        free, _ = torch.cuda.mem_get_info() if device == "cuda" else (0, 0)
        if device == "cuda" and free < 2.2e9:
            device = "cpu"
    except Exception:
        pass
    print("device:", device, flush=True)
    rng = random.Random(seed)
    by_c = test_pool()
    model = load_model(device)

    recips = []
    for c, lst in by_c.items():
        lst = list(lst); rng.shuffle(lst); recips += lst[:per_class]
    all_v = [r for lst in by_c.values() for r in lst]

    per_cell_cross = []; per_cell_same = []
    deg_var = []; junta1 = []; junta2 = []; rvals = []; top_cell = []
    kept = 0
    for ri, r in enumerate(recips):
        try:
            y = r["label"]
            recip_rgb = np.asarray(Image.open(r["image_path"]).convert("RGB"))
            rc, vessel = cell_masks(r, region_map=rmap(r))
            # skip if the clean image is already misclassified (analyze faithful recipients)
            if classify(model, [recip_rgb], device)[0] != y:
                continue
            bg = _background_fill(recip_rgb, vessel)
            cross_ds = compatible_cross(r, all_v, n_cross, rng)
            same_pool = [d for d in by_c[y] if d["image_path"] != r["image_path"]]
            same_ds = [rng.choice(same_pool) for _ in range(n_same)] if same_pool else []

            def donor_correct_matrix(donors):
                A = np.zeros((len(donors), 64), dtype=np.int8)
                for di, d in enumerate(donors):
                    drgb_w, daw = warp_donor_cells(d, r)
                    imgs = assemble_all(recip_rgb, rc, bg, drgb_w, daw)
                    A[di] = (classify(model, imgs, device) == y).astype(np.int8)
                return A
            Ax = donor_correct_matrix(cross_ds)
            As = donor_correct_matrix(same_ds) if same_ds else None

            # per-cell flip influence: Pr_{d,z}[F(z)!=F(z^p)]
            def influence(A):
                inf = np.zeros(6)
                for p in range(6):
                    zp = np.arange(64) ^ (1 << p)
                    inf[p] = np.mean(A != A[:, zp])
                return inf
            infx = influence(Ax); per_cell_cross.append(infx)
            if As is not None: per_cell_same.append(influence(As))

            # g(z)=mean_d correct ; Walsh spectrum -> variance by degree
            g = Ax.mean(0)                                  # in [0,1], len 64
            coeff = (H64 @ g) / 64.0                        # coeff[S]
            power = coeff ** 2
            W = np.array([power[POP == d].sum() for d in range(7)])
            Wpos = W[1:].sum() + 1e-12
            deg_var.append(W[1:] / Wpos)                    # fraction of interaction variance by degree
            junta1.append(W[1] / Wpos); junta2.append((W[1] + W[2]) / Wpos)
            top_cell.append(int(np.argmax(infx)))

            # semantic-L0: min |S| s.t. some cross donor flips composite 1_S
            flipped_any = (Ax == 0).any(0)                  # over donors, per z
            r_star = 7
            for z in range(1, 64):
                if flipped_any[z]:
                    r_star = min(r_star, int(POP[z]))
            rvals.append(r_star)
            kept += 1
            if (ri + 1) % 10 == 0:
                print(f"  {ri+1}/{len(recips)} kept={kept}", flush=True)
        except Exception as e:
            continue

    per_cell_cross = np.array(per_cell_cross)
    per_cell_same = np.array(per_cell_same) if per_cell_same else None
    corrected = per_cell_cross.mean(0) - (per_cell_same.mean(0) if per_cell_same is not None else 0)
    deg_var = np.array(deg_var); rvals = np.array(rvals)

    cell_names = [f"{v}-{l}" for (v, l) in CELLS]
    res = {
        "n_recipients": int(kept), "n_cross": n_cross, "n_same": n_same,
        "cell_order": cell_names,
        "influence_cross_mean": per_cell_cross.mean(0).round(4).tolist(),
        "influence_same_mean": (per_cell_same.mean(0).round(4).tolist() if per_cell_same is not None else None),
        "influence_corrected_mean": np.round(corrected, 4).tolist(),
        "variance_by_degree_mean": deg_var.mean(0).round(4).tolist(),   # deg 1..6
        "junta_deg1_mean": float(np.mean(junta1)), "junta_deg2_mean": float(np.mean(junta2)),
        "semantic_L0_hist": {int(k): int((rvals == k).sum()) for k in sorted(set(rvals.tolist()))},
        "semantic_L0_frac_r1": float((rvals == 1).mean()),
        "semantic_L0_median": float(np.median(rvals)),
        "top_influence_cell_counts": {cell_names[c]: int((np.array(top_cell) == c).sum())
                                      for c in range(6)},
    }
    Path("experiments").mkdir(exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print("\n PART-LATTICE SPECTRUM")
    print("cells:", cell_names)
    print("corrected influence:", res["influence_corrected_mean"])
    print("variance by degree (1..6):", res["variance_by_degree_mean"])
    print(f"junta: deg1={res['junta_deg1_mean']:.3f}  deg<=2={res['junta_deg2_mean']:.3f}")
    print(f"semantic-L0: frac r=1: {res['semantic_L0_frac_r1']:.3f}  median r={res['semantic_L0_median']}  hist={res['semantic_L0_hist']}")
    print("top-influence cell:", res["top_influence_cell_counts"])
    print(f"wrote {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_class", type=int, default=3)
    ap.add_argument("--n_cross", type=int, default=4)
    ap.add_argument("--n_same", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="experiments/part_lattice_spectrum.json")
    a = ap.parse_args()
    main(per_class=a.per_class, n_cross=a.n_cross, n_same=a.n_same, seed=a.seed, out=a.out)
