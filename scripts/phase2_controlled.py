"""Phase 2: controlled shortcut benchmark + de-biaser audit with the causal probe.

We PLANT a known spurious feature and ask the thesis question directly: when a
de-biaser raises worst-group accuracy, does the model's CAUSAL reliance on the
spurious feature actually drop?

Setup (a clean, artifact-free Waterbirds analogue):
  * Label = the vessel's true class (2 classes; carried by the superstructure).
  * Spurious attribute s in {0,1} = HULL COLOR, planted by tinting the segmenter
    hull region red (s=0) or blue (s=1). Fully controllable, no compositing.
  * Train: s correlates with label (p_align=0.9). Test: balanced (4 equal groups).
  * Worst-group accuracy = the standard proxy. Causal hull-color reliance = P(pred
    flips when we recolor the hull to the other color) over correct test images --
    the direct measurement the field lacks (artifact-free: recolor is exact).

Methods: ERM, SUBG (group balancing), GroupDRO, JTT (group-free), DFR (last-layer).
Writes experiments/phase2_controlled.json.
"""
from __future__ import annotations
import os
import argparse, json, random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50, ResNet50_Weights

from hullprobe.segmenter import load_region_png

B_MANIFEST = os.environ.get("MARVEL_B_MANIFEST", "data/marvel_b/manifest_pose_maskpca.json")
PRED = Path("data/predictions")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RED, BLUE = np.array([205, 45, 45], np.float32), np.array([45, 85, 205], np.float32)
ALPHA = 0.55
norm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

_HM = {}
def hull_mask(path, shape):
    s = Path(path).stem
    if s not in _HM:
        p = PRED / f"{s}.png"
        rm = load_region_png(p) if p.exists() else None
        _HM[s] = rm
    rm = _HM[s]
    if rm is None:
        return None
    import cv2
    rm = cv2.resize(rm.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return rm == 1


SHORTCUT = "hull_color"   # set by main(); one of hull_color | bg_color | watermark


def _region_mask(path, shape):
    """Planted-feature region for the current SHORTCUT (bool HxW)."""
    if SHORTCUT == "watermark":
        m = np.zeros(shape[:2], bool); m[8:38, 8:38] = True; return m   # fixed corner patch
    hm = hull_mask(path, shape)
    if hm is None:
        return None
    if SHORTCUT == "bg_color":     # background = image minus vessel; approx via ~hull-region proxy
        rm = _HM.get(Path(path).stem)
        if rm is None:
            return None
        import cv2
        rm = cv2.resize(rm.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return rm == 0             # non-vessel background
    return hm                      # hull_color


def tinted(rec, s):
    """Load image, tint the shortcut region red (s=0) or blue (s=1); resize 224."""
    im = np.asarray(Image.open(rec["image_path"]).convert("RGB").resize((224, 224))).astype(np.float32)
    m = _region_mask(rec["image_path"], im.shape)
    if m is not None and m.any():
        col = RED if s == 0 else BLUE
        im[m] = (1 - ALPHA) * im[m] + ALPHA * col
    return im.clip(0, 255).astype(np.uint8)


class DS(Dataset):
    def __init__(self, recs, aug=False):
        self.recs = recs
        self.aug = aug
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        im = tinted(r, r["s"])
        if self.aug and random.random() < 0.5:
            im = im[:, ::-1]
        return norm(Image.fromarray(np.ascontiguousarray(im))), r["y"], r["g"], i


def build(seed=42, per_class_train=800, per_class_test=200, p_align=0.9):
    recs = json.loads(Path(B_MANIFEST).read_text())["records"]
    have = lambda r: (PRED / f"{Path(r['image_path']).stem}.png").exists()
    by = {"train": defaultdict(list), "test": defaultdict(list)}
    for r in recs:
        sp = r.get("source_split")
        if sp in by and r.get("label") is not None and have(r):
            by[sp][int(r["label"])].append(r)
    # 2 most populous classes with enough test
    counts = sorted(by["train"], key=lambda c: -min(len(by["train"][c]), len(by["test"].get(c, []))))
    L0, L1 = counts[0], counts[1]
    rng = random.Random(seed)

    def make(split, n, align):
        out = []
        for yi, c in enumerate([L0, L1]):
            lst = list(by[split][c]); rng.shuffle(lst); lst = lst[:n]
            for k, r in enumerate(lst):
                if align:  # train: s correlated with label
                    s = yi if rng.random() < p_align else 1 - yi
                else:      # test: balanced
                    s = k % 2
                out.append({**r, "y": yi, "s": s, "g": 2 * yi + s})
        rng.shuffle(out); return out

    return make("train", per_class_train, True), make("test", per_class_test, False), (L0, L1)


def make_model():
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m.to(DEVICE)


def train_erm(recs, epochs=8, weights=None, model=None, lr=1e-4):
    model = model or make_model()
    ds = DS(recs, aug=True)
    if weights is not None:
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
        loader = DataLoader(ds, batch_size=64, sampler=sampler, num_workers=8, drop_last=True)
    else:
        loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=8, drop_last=True)
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


def train_groupdro(recs, epochs=8, eta=0.01):
    model = make_model()
    loader = DataLoader(DS(recs, aug=True), batch_size=64, shuffle=True, num_workers=8, drop_last=True)
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


def train_dfr(recs, base, val_recs):
    """Freeze base features, retrain fc on a group-balanced set."""
    import copy
    model = copy.deepcopy(base)
    for p in model.parameters(): p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2).to(DEVICE)
    # group-balanced subset
    byg = defaultdict(list)
    for r in val_recs: byg[r["g"]].append(r)
    n = min(len(v) for v in byg.values())
    bal = [r for g in byg for r in random.sample(byg[g], n)]
    loader = DataLoader(DS(bal, aug=True), batch_size=64, shuffle=True, num_workers=6, drop_last=True)
    opt = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss()
    model.train()
    for ep in range(20):
        for x, y, g, i in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
    return model


@torch.inference_mode()
def evaluate(model, test_recs):
    model.eval()
    loader = DataLoader(DS(test_recs), batch_size=128, num_workers=6)
    preds, ys, gs = [], [], []
    for x, y, g, i in loader:
        preds.append(model(x.to(DEVICE)).argmax(1).cpu().numpy()); ys.append(y.numpy()); gs.append(g.numpy())
    preds, ys, gs = np.concatenate(preds), np.concatenate(ys), np.concatenate(gs)
    acc = float((preds == ys).mean())
    wga = min(float((preds[gs == gi] == ys[gs == gi]).mean()) for gi in range(4) if (gs == gi).any())
    # causal hull-color reliance: recolor hull to the OTHER color, measure flip on correct
    correct = preds == ys
    flip_recs = [dict(test_recs[j], s=1 - test_recs[j]["s"]) for j in range(len(test_recs))]
    fl = DataLoader(DS(flip_recs), batch_size=128, num_workers=6)
    fpred = []
    for x, y, g, i in fl:
        fpred.append(model(x.to(DEVICE)).argmax(1).cpu().numpy())
    fpred = np.concatenate(fpred)
    flip = float((fpred[correct] != preds[correct]).mean())   # of correct, fraction flipped by recolor
    return {"acc": acc, "worst_group_acc": wga, "hull_color_flip": flip}


def main():
    global SHORTCUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--shortcut", default="hull_color",
                    choices=["hull_color", "bg_color", "watermark"])
    a = ap.parse_args()
    SHORTCUT = a.shortcut
    print(f"shortcut = {SHORTCUT}", flush=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    train, test, (L0, L1) = build(seed=a.seed)
    cls = json.loads(Path("data/classes_B.json").read_text())
    print(f"labels: {cls[L0]} vs {cls[L1]}; train {len(train)} test {len(test)}", flush=True)
    # group counts
    gc = defaultdict(int)
    for r in train: gc[r["g"]] += 1
    print("train group counts (2y+s):", dict(sorted(gc.items())), flush=True)

    res = {}
    # ERM
    print("== ERM ==", flush=True); erm = train_erm(train, a.epochs); res["ERM"] = evaluate(erm, test)
    print(res["ERM"], flush=True)
    # SUBG: group-balanced sampling weights
    byg = defaultdict(int)
    for r in train: byg[r["g"]] += 1
    w = torch.tensor([1.0 / byg[r["g"]] for r in train])
    print("== SUBG ==", flush=True); res["SUBG"] = evaluate(train_erm(train, a.epochs, weights=w), test)
    print(res["SUBG"], flush=True)
    # GroupDRO
    print("== GroupDRO ==", flush=True); res["GroupDRO"] = evaluate(train_groupdro(train, a.epochs), test)
    print(res["GroupDRO"], flush=True)
    # JTT
    print("== JTT ==", flush=True)
    idm = train_erm(train, max(2, a.epochs // 2))
    idm.eval()
    with torch.inference_mode():
        ld = DataLoader(DS(train), batch_size=128, num_workers=6)
        err = []
        for x, y, g, i in ld:
            pr = idm(x.to(DEVICE)).argmax(1).cpu().numpy()
            for pj, yj, ij in zip(pr, y.numpy(), i.numpy()):
                if pj != yj: err.append(int(ij))
    wj = torch.ones(len(train))
    for ij in err: wj[ij] = 20.0
    res["JTT"] = evaluate(train_erm(train, a.epochs, weights=wj), test)
    print(res["JTT"], flush=True)
    # DFR (reuse ERM base, retrain head on group-balanced test-held split)
    print("== DFR ==", flush=True)
    rng = random.Random(a.seed); heldout = rng.sample(test, len(test) // 2)
    res["DFR"] = evaluate(train_dfr(train, erm, heldout), [r for r in test if r not in heldout])
    print(res["DFR"], flush=True)

    print("\n PHASE 2: CONTROLLED SHORTCUT AUDIT")
    print(f"{'method':10s} {'clean':>7s} {'worst-grp':>10s} {'hull-color-flip(causal)':>24s}")
    for m, r in res.items():
        print(f"{m:10s} {r['acc']:7.3f} {r['worst_group_acc']:10.3f} {r['hull_color_flip']:24.3f}")
    sfx = "" if a.shortcut == "hull_color" else f"_{a.shortcut}"
    out = f"experiments/phase2_controlled{sfx}_seed{a.seed}.json"
    Path(out).write_text(json.dumps({"labels": [cls[L0], cls[L1]], "seed": a.seed,
                                     "shortcut": a.shortcut, "results": res}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
