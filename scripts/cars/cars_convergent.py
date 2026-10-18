"""Convergent validity for the CAR lattice: does the group-label-free reliance measure track
worst-group accuracy, as it does on vessels?

Waterbirds-style controlled shortcut on cars: binary label = one of two car-model groups
(carried by real appearance); spurious attribute s in {0,1} = a colour tint on the lattice parts
{body, body-side}, correlated with the label (p_align=0.9) in train, balanced at test (4 groups).
Worst-group accuracy is the standard proxy; the direct measure is the CAR-lattice recolor-flip
rate (fraction of correct test cars whose prediction flips when those parts are recoloured to the
other tint -- artifact-free, no group labels). Arms: ERM, SUBG, GroupDRO, JTT, DFR x seeds.
Reports corr(WGA, flip). Mirrors scripts/phase2_controlled.py with the maritime hull region
replaced by the car lattice's part support.
"""
from __future__ import annotations
import sys, json, random, argparse, copy
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50, ResNet50_Weights

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RED, BLUE = np.array([205, 45, 45], np.float32), np.array([45, 85, 205], np.float32)
norm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
SEG = {r["image_path"]: r for r in json.load(open("experiments/cars_conv_seg.json"))}
_PM = {}

# Three graded shortcuts (mirrors maritime hull-colour / bg-colour / watermark): different
# regions + strengths so the de-biasers land across the whole worst-group-accuracy range,
# not just two clusters. `painted` is strong (low WGA), `background` medium, `watermark` weak.
SHORTCUTS = {
    "painted":    dict(region="painted",    alpha=0.80, p_align=0.95),
    "background": dict(region="background", alpha=0.75, p_align=0.90),
    "watermark":  dict(region="watermark",  alpha=1.00, p_align=0.85),
}
CURRENT = SHORTCUTS["painted"]


def region_mask(path, shape):
    reg = CURRENT["region"]
    if reg == "watermark":
        m = np.zeros(shape[:2], bool)
        h = int(0.18 * shape[0]); w = int(0.18 * shape[1])
        m[6:6 + h, 6:6 + w] = True                      # fixed top-left patch (small -> weak)
        return m
    if path not in _PM:
        _PM[path] = np.load(SEG[path]["npz"])["partmap"]
    import cv2
    pm = cv2.resize(_PM[path].astype(np.int16), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    if reg == "background":
        return pm == -1                                 # non-car background
    return (pm >= 0) & (pm != 4)                         # painted lattice parts (not wheels)


def tinted(rec, s):
    im = np.asarray(Image.open(rec["image_path"]).convert("RGB").resize((224, 224))).astype(np.float32)
    m = region_mask(rec["image_path"], im.shape)
    if m.any():
        a = CURRENT["alpha"]
        im[m] = (1 - a) * im[m] + a * (RED if s == 0 else BLUE)
    return im.clip(0, 255).astype(np.uint8)


class DS(Dataset):
    def __init__(self, recs, aug=False):
        self.recs = recs; self.aug = aug
    def __len__(self):
        return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]; im = tinted(r, r["s"])
        if self.aug and random.random() < 0.5:
            im = im[:, ::-1]
        return norm(Image.fromarray(np.ascontiguousarray(im))), r["y"], r["g"], i


def build(seed=42, p_align=None):
    p_align = CURRENT["p_align"] if p_align is None else p_align
    recs = json.load(open("experiments/cars_conv_seg.json"))
    by = {"train": defaultdict(list), "test": defaultdict(list)}
    for r in recs:
        by[r["split"]][r["group"]].append(r)
    rng = random.Random(seed)

    def make(split, align):
        out = []
        n = min(len(by[split][0]), len(by[split][1]))
        for yi in [0, 1]:
            lst = list(by[split][yi]); rng.shuffle(lst); lst = lst[:n]
            for k, r in enumerate(lst):
                s = (yi if rng.random() < p_align else 1 - yi) if align else (k % 2)
                out.append({**r, "y": yi, "s": s, "g": 2 * yi + s})
        rng.shuffle(out); return out
    return make("train", True), make("test", False)


def make_model():
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m.to(DEVICE)


def train_erm(recs, epochs=8, weights=None, model=None, lr=1e-4):
    model = model or make_model()
    ds = DS(recs, aug=True)
    if weights is not None:
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
        loader = DataLoader(ds, batch_size=64, sampler=sampler, num_workers=6, drop_last=True)
    else:
        loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=6, drop_last=True)
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
    loader = DataLoader(DS(recs, aug=True), batch_size=64, shuffle=True, num_workers=6, drop_last=True)
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
                    if m.any():
                        gl[gi] = per[m].mean()
                q = q * torch.exp(eta * gl.detach()); q = q / q.sum()
                loss = (q * gl).sum()
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    return model


def train_dfr(base, val_recs, seed):
    model = copy.deepcopy(base)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2).to(DEVICE)
    byg = defaultdict(list)
    for r in val_recs:
        byg[r["g"]].append(r)
    n = min(len(v) for v in byg.values())
    rng = random.Random(seed)
    bal = [r for g in byg for r in rng.sample(byg[g], n)]
    loader = DataLoader(DS(bal, aug=True), batch_size=64, shuffle=True, num_workers=6, drop_last=True)
    opt = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss()
    model.train()
    for ep in range(20):
        for x, y, g, i in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
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
    correct = preds == ys
    flip_recs = [dict(test_recs[j], s=1 - test_recs[j]["s"]) for j in range(len(test_recs))]
    fl = DataLoader(DS(flip_recs), batch_size=128, num_workers=6)
    fpred = np.concatenate([model(x.to(DEVICE)).argmax(1).cpu().numpy() for x, y, g, i in fl])
    flip = float((fpred[correct] != preds[correct]).mean())
    return {"acc": acc, "worst_group_acc": wga, "recolor_flip": flip}


def run_seed(seed, epochs=8):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train, test = build(seed=seed)
    res = {}
    erm = train_erm(train, epochs); res["ERM"] = evaluate(erm, test)
    byg = defaultdict(int)
    for r in train:
        byg[r["g"]] += 1
    w = torch.tensor([1.0 / byg[r["g"]] for r in train])
    res["SUBG"] = evaluate(train_erm(train, epochs, weights=w), test)
    res["GroupDRO"] = evaluate(train_groupdro(train, epochs), test)
    idm = train_erm(train, max(2, epochs // 2)); idm.eval()
    with torch.inference_mode():
        ld = DataLoader(DS(train), batch_size=128, num_workers=6)
        err = []
        for x, y, g, i in ld:
            pr = idm(x.to(DEVICE)).argmax(1).cpu().numpy()
            for pj, yj, ij in zip(pr, y.numpy(), i.numpy()):
                if pj != yj:
                    err.append(int(ij))
    wj = torch.ones(len(train))
    for ij in err:
        wj[ij] = 20.0
    res["JTT"] = evaluate(train_erm(train, epochs, weights=wj), test)
    rng = random.Random(seed); heldout = rng.sample(test, len(test) // 2)
    res["DFR"] = evaluate(train_dfr(erm, heldout, seed), [r for r in test if r not in heldout])
    return res


def main():
    global CURRENT
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--shortcuts", nargs="+", default=["painted", "background", "watermark"])
    ap.add_argument("--out", default="experiments/cars_convergent.json")
    a = ap.parse_args()
    methods = ["ERM", "SUBG", "GroupDRO", "JTT", "DFR"]
    points = []                       # each: {shortcut, method, seed, wga, flip}
    per_shortcut = {}
    for sc in a.shortcuts:
        CURRENT = SHORTCUTS[sc]
        print(f"\n########## SHORTCUT = {sc} ({CURRENT}) ##########", flush=True)
        sc_w, sc_f = [], []
        for sd in a.seeds:
            print(f"== {sc} seed {sd} ==", flush=True)
            r = run_seed(sd, a.epochs)
            for m, v in r.items():
                print(f"  {m:9s} WGA {v['worst_group_acc']:.3f} flip {v['recolor_flip']:.3f}", flush=True)
                points.append({"shortcut": sc, "method": m, "seed": sd,
                               "wga": v["worst_group_acc"], "flip": v["recolor_flip"]})
                sc_w.append(v["worst_group_acc"]); sc_f.append(v["recolor_flip"])
        per_shortcut[sc] = float(np.corrcoef(sc_w, sc_f)[0, 1]) if len(set(sc_w)) > 1 else float("nan")
        print(f"  [{sc}] corr = {per_shortcut[sc]:.3f}", flush=True)
    wga_all = [p["wga"] for p in points]; flip_all = [p["flip"] for p in points]
    agg = {m: {"wga_mean": float(np.mean([p["wga"] for p in points if p["method"] == m])),
               "flip_mean": float(np.mean([p["flip"] for p in points if p["method"] == m]))}
           for m in methods}
    corr = float(np.corrcoef(wga_all, flip_all)[0, 1])
    out = {"n_configs": len(points), "corr_wga_flip": corr, "corr_per_shortcut": per_shortcut,
           "per_method": agg, "points": points}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\n CARS CONVERGENT VALIDITY (pooled {len(points)} configs)")
    print(f"pooled corr(WGA, recolor-flip) = {corr:.3f}; per-shortcut = "
          + ", ".join(f"{k} {v:.3f}" for k, v in per_shortcut.items()))
    xs = np.sort(np.array(wga_all))
    print("WGA spread:", np.round(xs, 2))
    print("DONE_CARS_CONVERGENT", flush=True)


if __name__ == "__main__":
    main()
