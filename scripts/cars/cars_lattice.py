"""Cars 2^5 part-lattice: the SAME HullProbe readouts, a car's OWN anatomy.

PARTS = {body, front-face, greenhouse, body-side, wheel}. For a recipient x (class y)
and donor d, compose(x,d,z) replaces the z-selected recipient parts with the donor's,
warped into the recipient's PCA-canonical frame and stenciled onto the recipient's exact
part support (silhouette stays the recipient's). Evaluating the frozen classifier over all
32 patterns gives g(z); we report per-part flip-influence (cross / same-control / corrected),
the front-face share (the car's "superstructure" analogue), the Walsh degree-variance
(near-additivity), and the semantic-L0 radius. Registration is geometric; only the object +
part masks and the classifier are learned/prompted. Mirrors scripts/part_lattice_spectrum.py.
"""
from __future__ import annotations
import sys, json, math, random, argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch, torch.nn as nn
import torchvision.transforms as T
from torchvision import models as tvm
from scipy.linalg import hadamard

PARTS = ["body", "front-face", "greenhouse", "body-side", "wheel"]
N = len(PARTS); V = 1 << N                       # 32 vertices
CANON_W, DECK_ROW = 256.0, 64.0
HV = hadamard(V).astype(np.float64)              # H[S,z] = (-1)^popcount(S&z)
POP = np.array([bin(i).count("1") for i in range(V)])
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])


def load_seg(rec):
    d = np.load(rec["npz"])
    return d["omask"].astype(bool), d["partmap"].astype(np.int64)


def similarity(rec, omask):
    ys, xs = np.where(omask)
    th = math.radians(rec["pca_angle_deg"]); c, s_ = math.cos(th), math.sin(th)
    cx, cy = rec["pca_centroid_x"], rec["pca_centroid_y"]
    lon = (xs - cx) * c + (ys - cy) * s_
    t0, t1 = float(lon.min()), float(lon.max())
    s = CANON_W / max(1e-6, t1 - t0)
    c1 = -s * (c * cx + s_ * cy) - s * t0
    c2 = s * (s_ * cx - c * cy) + DECK_ROW
    return np.array([[s * c, s * s_, c1], [-s * s_, s * c, c2], [0, 0, 1.0]])


def affine_coeffs(M):
    return (M[0, 0], M[0, 1], M[0, 2], M[1, 0], M[1, 1], M[1, 2])


def compose_all(recip_rec, donor_rec):
    """32 composites (uint8 HxW3) for recipient x donor, exact target-support stencil.

    Donor warped into the recipient's canonical frame with an alpha; a swapped part
    fills the recipient's own part support with valid warped-donor texture, keeping the
    recipient silhouette. Where the donor does not cover the support the recipient pixel
    is retained (no black holes)."""
    rrgb = np.asarray(Image.open(recip_rec["image_path"]).convert("RGB"))
    H, W = rrgb.shape[:2]
    romask, rpm = load_seg(recip_rec)
    domask, _ = load_seg(donor_rec)
    drgb = np.asarray(Image.open(donor_rec["image_path"]).convert("RGB"))
    drgba = Image.fromarray(np.dstack([drgb, (domask * 255).astype(np.uint8)]), "RGBA")
    Sr, Sd = similarity(recip_rec, romask), similarity(donor_rec, domask)
    r2d = np.linalg.inv(Sd) @ Sr
    warped = np.asarray(drgba.transform((W, H), Image.AFFINE, affine_coeffs(r2d),
                        resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0)))
    dwarp, dvalid = warped[..., :3], warped[..., 3] > 128
    part_support = [(rpm == p) & dvalid for p in range(N)]   # recip support covered by donor
    imgs = []
    for z in range(V):
        out = rrgb.copy()
        for p in range(N):
            if z & (1 << p):
                out[part_support[p]] = dwarp[part_support[p]]
        imgs.append(out)
    return imgs


@torch.inference_mode()
def classify(model, imgs, device, batch=64):
    preds = []
    for i in range(0, len(imgs), batch):
        x = torch.stack([tf(Image.fromarray(a)) for a in imgs[i:i + batch]]).to(device)
        preds.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def aspect(rec):
    return rec.get("aspect", 2.0)


def compatible_cross(r, pool, k, rng):
    cand = [d for d in pool if d["label"] != r["label"]]
    cand.sort(key=lambda d: abs(aspect(d) - aspect(r)))
    cand = cand[:60]; rng.shuffle(cand)
    return cand[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="experiments/cars_seg.json")
    ap.add_argument("--clf", default="experiments/cars_clf/best.pt")
    ap.add_argument("--n-recip", type=int, default=80)
    ap.add_argument("--n-cross", type=int, default=4)
    ap.add_argument("--n-same", type=int, default=2)
    ap.add_argument("--out", default="experiments/cars_lattice.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg = json.loads(Path(args.manifest).read_text())
    by_c = {}
    for r in seg:
        by_c.setdefault(r["label"], []).append(r)
    ck = torch.load(args.clf, map_location=device)
    n_classes = ck["model"]["classifier.2.weight"].shape[0]
    model = tvm.convnext_tiny(); model.classifier[2] = nn.Linear(model.classifier[2].in_features, n_classes)
    model.load_state_dict(ck["model"]); model.to(device).eval()
    print(f"clf classes={n_classes} test_acc={ck.get('test_acc')}", flush=True)

    rng = random.Random(42)
    # recipients: front-visible, multiple same-class donors available
    cand = [r for r in seg if r.get("has_front") and len(by_c[r["label"]]) >= 2]
    rng.shuffle(cand)
    per_cell_cross, per_cell_same, deg_var, rvals, top_cell = [], [], [], [], []
    kept = 0
    for r in cand:
        if kept >= args.n_recip:
            break
        try:
            y = r["label"]
            rrgb = np.asarray(Image.open(r["image_path"]).convert("RGB"))
            if classify(model, [rrgb], device)[0] != y:      # analyze faithful recipients
                continue
            cross = compatible_cross(r, seg, args.n_cross, rng)
            same_pool = [d for d in by_c[y] if d["image_path"] != r["image_path"]]
            same = [rng.choice(same_pool) for _ in range(args.n_same)] if same_pool else []
            if not cross or not same:
                continue

            def A_of(donors):
                A = np.zeros((len(donors), V), np.int8)
                for di, d in enumerate(donors):
                    A[di] = (classify(model, compose_all(r, d), device) == y).astype(np.int8)
                return A
            Ax = A_of(cross)
            if Ax[:, 0].mean() < 0.5:            # donor at z=0 must mostly keep class
                continue
            As = A_of(same)

            def influence(A):
                inf = np.zeros(N)
                for p in range(N):
                    zp = np.arange(V) ^ (1 << p)
                    inf[p] = np.mean(A != A[:, zp])
                return inf
            ix = influence(Ax); per_cell_cross.append(ix); per_cell_same.append(influence(As))
            g = Ax.mean(0); coeff = (HV @ g) / V; power = coeff ** 2
            W = np.array([power[POP == d].sum() for d in range(N + 1)])
            deg_var.append(W[1:] / (W[1:].sum() + 1e-12))
            top_cell.append(int(np.argmax(ix)))
            flipped_any = (Ax == 0).any(0); rstar = N + 1
            for z in range(1, V):
                if flipped_any[z]:
                    rstar = min(rstar, int(POP[z]))
            rvals.append(rstar); kept += 1
            if kept % 10 == 0:
                print(f"  recip {kept}/{args.n_recip}", flush=True)
        except Exception:
            continue

    pcc = np.array(per_cell_cross); pcs = np.array(per_cell_same)
    corr = pcc.mean(0) - pcs.mean(0)
    deg_var = np.array(deg_var); rvals = np.array(rvals)
    fi = PARTS.index("front-face")
    res = {
        "n_recipients": int(kept), "n_cross": args.n_cross, "n_same": args.n_same,
        "part_order": PARTS,
        "influence_cross_mean": pcc.mean(0).round(4).tolist(),
        "influence_same_mean": pcs.mean(0).round(4).tolist(),
        "influence_corrected_mean": corr.round(4).tolist(),
        "front_face_share": float(corr[fi] / max(1e-9, corr.sum())),
        "front_face_share_raw": float(pcc.mean(0)[fi] / max(1e-9, pcc.mean(0).sum())),
        "variance_by_degree_mean": deg_var.mean(0).round(4).tolist(),   # deg 1..N
        "junta_deg1_mean": float(deg_var.mean(0)[0]),
        "junta_deg_le2_mean": float(deg_var.mean(0)[:2].sum()),
        "semantic_L0_hist": {int(k): int((rvals == k).sum()) for k in sorted(set(rvals.tolist()))},
        "semantic_L0_frac_r1": float((rvals == 1).mean()),
        "semantic_L0_median": float(np.median(rvals)),
        "top_influence_part_counts": {PARTS[c]: int((np.array(top_cell) == c).sum()) for c in range(N)},
    }
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("\n CARS LATTICE")
    print(json.dumps(res, indent=1))
    print("DONE_CARS_LATTICE", flush=True)


if __name__ == "__main__":
    main()
