"""Circularity control for the anatomical (operator) validation (paper App. C).

The operator arm of Sec. 4.2 plants the shortcut with the 6-cell graft (hull <-
donor D0 or D1) and measures reliance with the SAME operator (re-swap D_s -> D_{1-s}).
The concern: the flip rate could read memorization of the two specific
training donors / residual compositing statistics rather than appearance-general
hull reliance.

Resolution: for each de-biaser we ALSO measure the flip under an INDEPENDENT hull
change -- swapping the hull to a THIRD donor D2 whose class was never seen in
training. D2 goes through the identical compositing pipeline (same seam/inpainting),
so any flip it produces is driven by hull APPEARANCE, not the artifact or the two
memorized donors. If the independent-mechanism flip tracks the re-swap flip across
de-biasers (ERM high on both, DFR low on both), the operator measures a
mechanism-independent hull reliance and the plant-and-measure symmetry is not the
whole story.

Writes experiments/phase2_circularity_seed{S}.json.
"""
from __future__ import annotations
import os
import argparse, json, random, copy
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50, ResNet50_Weights

from hullprobe.swap import hull_swap
from hullprobe.segmenter import load_region_png

B_MANIFEST = os.environ.get("MARVEL_B_MANIFEST", "data/marvel_b/manifest_pose_maskpca.json")
POSE_B = "data/pose_B.json"
PRED = Path("data/predictions")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
norm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

_RM = {}
def rmap(path):
    s = Path(path).stem
    if s not in _RM:
        p = PRED / f"{s}.png"
        _RM[s] = load_region_png(p) if p.exists() else None
    return _RM[s]

CACHE = {}
DONORS = {}   # s in {0,1,2} -> donor_rec ; 0/1 seen in training, 2 is unseen

def do_swap(recip_rec, s):
    key = (Path(recip_rec["image_path"]).stem, s)
    if key in CACHE:
        return CACHE[key]
    d = DONORS[s]
    img, _ = hull_swap(recip_rec, d,
                       region_map_recip=rmap(recip_rec["image_path"]),
                       region_map_donor=rmap(d["image_path"]))
    arr = np.asarray(img.convert("RGB").resize((224, 224))).astype(np.uint8)
    CACHE[key] = arr
    return arr


class DS(Dataset):
    def __init__(self, recs, aug=False):
        self.recs, self.aug = recs, aug
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        im = CACHE[(r["stem"], r["s"])]
        if self.aug and random.random() < 0.5:
            im = im[:, ::-1]
        return norm(Image.fromarray(np.ascontiguousarray(im))), r["y"], r["g"], i


def load_pool():
    pose = {r["image_path"]: r for r in json.loads(Path(POSE_B).read_text())}
    recs = json.loads(Path(B_MANIFEST).read_text())["records"]
    by_split_class = {"train": defaultdict(list), "test": defaultdict(list)}
    have = lambda r: (PRED / f"{Path(r['image_path']).stem}.png").exists()
    for r in recs:
        sp = r.get("source_split")
        p = pose.get(r["image_path"])
        if sp in by_split_class and r.get("label") is not None and have(r) and p \
                and p.get("full_vessel_box_xyxy"):
            rr = {**p, "label": int(r["label"]), "source_split": sp,
                  "stem": Path(r["image_path"]).stem}
            by_split_class[sp][int(rr["label"])].append(rr)
    return by_split_class


def make_model():
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m.to(DEVICE)


def train_erm(recs, epochs, weights=None, lr=1e-4, model=None):
    model = model or make_model()
    ds = DS(recs, aug=True)
    if weights is not None:
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
        loader = DataLoader(ds, batch_size=64, sampler=sampler, num_workers=0, drop_last=True)
    else:
        loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        for x, y, g, i in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.autocast("cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    return model


def train_groupdro(recs, epochs, eta=0.01):
    model = make_model()
    loader = DataLoader(DS(recs, aug=True), batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    crit = nn.CrossEntropyLoss(reduction="none")
    q = torch.ones(4, device=DEVICE) / 4
    for ep in range(epochs):
        model.train()
        for x, y, g, i in loader:
            x, y, g = x.to(DEVICE), y.to(DEVICE), g.to(DEVICE)
            opt.zero_grad()
            with torch.autocast("cuda"):
                per = crit(model(x), y)
                gl = torch.zeros(4, device=DEVICE)
                for gi in range(4):
                    m = g == gi
                    if m.any(): gl[gi] = per[m].mean()
                q = q * torch.exp(eta * gl.detach()); q = q / q.sum()
                loss = (q * gl).sum()
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    return model


def train_dfr(base, val_recs):
    model = copy.deepcopy(base)
    for p in model.parameters(): p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2).to(DEVICE)
    byg = defaultdict(list)
    for r in val_recs: byg[r["g"]].append(r)
    n = min(len(v) for v in byg.values())
    bal = [r for g in byg for r in random.sample(byg[g], n)]
    loader = DataLoader(DS(bal, aug=True), batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss()
    model.train()
    for ep in range(20):
        for x, y, g, i in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
    return model


@torch.inference_mode()
def _preds(model, recs):
    model.eval()
    loader = DataLoader(DS(recs), batch_size=128, num_workers=0)
    out = []
    for x, y, g, i in loader:
        out.append(model(x.to(DEVICE)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def evaluate(model, test_recs):
    preds = _preds(model, test_recs)
    ys = np.array([r["y"] for r in test_recs]); gs = np.array([r["g"] for r in test_recs])
    acc = float((preds == ys).mean())
    wga = min(float((preds[gs == gi] == ys[gs == gi]).mean()) for gi in range(4) if (gs == gi).any())
    correct = preds == ys
    # (a) re-swap flip (the operator measure): hull D_s -> D_{1-s}
    reswap = [dict(r, s=1 - r["s"]) for r in test_recs]
    rpred = _preds(model, reswap)
    reswap_flip = float((rpred[correct] != preds[correct]).mean())
    # (b) INDEPENDENT mechanism: hull D_s -> D2 (a third, unseen donor)
    indep = [dict(r, s=2) for r in test_recs]
    ipred = _preds(model, indep)
    indep_flip = float((ipred[correct] != preds[correct]).mean())
    return {"acc": acc, "worst_group_acc": wga,
            "hull_swap_flip": reswap_flip, "indep_donor_flip": indep_flip}


def build(seed, per_class_train, per_class_test, p_align):
    by = load_pool()
    counts = sorted(by["train"], key=lambda c: -min(len(by["train"][c]), len(by["test"].get(c, []))))
    L0, L1 = counts[0], counts[1]
    non_label = [c for c in counts if c not in (L0, L1)]
    donor_classes = [non_label[0], non_label[1], non_label[2]]   # D0, D1 (seen) + D2 (unseen)
    rng = random.Random(seed)
    for s, dc in enumerate(donor_classes):
        pool = [r for r in by["test"][dc]] or [r for r in by["train"][dc]]
        rng.shuffle(pool)
        DONORS[s] = pool[0]

    def make(split, n, align):
        out = []
        for yi, c in enumerate([L0, L1]):
            lst = list(by[split][c]); rng.shuffle(lst); lst = lst[:n]
            for k, r in enumerate(lst):
                s = (yi if rng.random() < p_align else 1 - yi) if align else (k % 2)
                out.append({**r, "y": yi, "s": s, "g": 2 * yi + s})
        rng.shuffle(out); return out

    train = make("train", per_class_train, True)
    test = make("test", per_class_test, False)
    return train, test, (L0, L1), donor_classes


def precompute(recs, need_all):
    kept = []
    for r in recs:
        try:
            do_swap(r, r["s"])
            if need_all:
                do_swap(r, 1 - r["s"])   # for re-swap
                do_swap(r, 2)            # for the independent (unseen donor) mechanism
            kept.append(r)
        except Exception:
            continue
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--per_class_train", type=int, default=250)
    ap.add_argument("--per_class_test", type=int, default=100)
    ap.add_argument("--p_align", type=float, default=0.9)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    train, test, (L0, L1), dcs = build(a.seed, a.per_class_train, a.per_class_test, a.p_align)
    cls = json.loads(Path("data/classes_B.json").read_text())
    print(f"label: {cls[L0]} vs {cls[L1]}  |  donors: {cls[dcs[0]]}(s0) {cls[dcs[1]]}(s1) SEEN, "
          f"{cls[dcs[2]]}(s2) UNSEEN", flush=True)
    train = precompute(train, need_all=False)
    test = precompute(test, need_all=True)
    print(f"kept: train {len(train)} test {len(test)}", flush=True)

    res = {}
    print("== ERM ==", flush=True); erm = train_erm(train, a.epochs); res["ERM"] = evaluate(erm, test); print(res["ERM"], flush=True)
    byg = defaultdict(int)
    for r in train: byg[r["g"]] += 1
    w = torch.tensor([1.0 / byg[r["g"]] for r in train])
    print("== SUBG ==", flush=True); res["SUBG"] = evaluate(train_erm(train, a.epochs, weights=w), test); print(res["SUBG"], flush=True)
    print("== GroupDRO ==", flush=True); res["GroupDRO"] = evaluate(train_groupdro(train, a.epochs), test); print(res["GroupDRO"], flush=True)
    print("== JTT ==", flush=True)
    idm = train_erm(train, max(2, a.epochs // 2)); ip = _preds(idm, train)
    wj = torch.ones(len(train))
    for j, r in enumerate(train):
        if ip[j] != r["y"]: wj[j] = 20.0
    res["JTT"] = evaluate(train_erm(train, a.epochs, weights=wj), test); print(res["JTT"], flush=True)
    print("== DFR ==", flush=True)
    rng = random.Random(a.seed); held = rng.sample(test, len(test) // 2)
    res["DFR"] = evaluate(train_dfr(erm, held), [r for r in test if r not in held]); print(res["DFR"], flush=True)

    print("\n CIRCULARITY CONTROL: re-swap vs independent (unseen-donor) flip")
    print(f"{'method':10s} {'WGA':>6s} {'reswap_flip':>12s} {'indep_flip':>11s}")
    for m, r in res.items():
        print(f"{m:10s} {r['worst_group_acc']:6.3f} {r['hull_swap_flip']:12.3f} {r['indep_donor_flip']:11.3f}")
    a_r = np.array([res[m]["hull_swap_flip"] for m in res])
    a_i = np.array([res[m]["indep_donor_flip"] for m in res])
    if a_r.std() > 0 and a_i.std() > 0:
        print(f"corr(reswap, indep) across methods = {np.corrcoef(a_r, a_i)[0,1]:.3f}")
    out = f"experiments/phase2_circularity_seed{a.seed}.json"
    Path(out).write_text(json.dumps({"labels": [cls[L0], cls[L1]],
                                     "donor_hull_classes": [cls[dcs[0]], cls[dcs[1]], cls[dcs[2]]],
                                     "seed": a.seed, "results": res}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
