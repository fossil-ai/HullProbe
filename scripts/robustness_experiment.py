"""The decisive gate: does registered hull-swap augmentation buy robustness to
hull-shift that the incumbents (CutMix, background-swap) don't?

4 arms, same backbone/schedule/data:
  baseline   : ERM
  cutmix     : VRM incumbent
  bgswap     : spurious-robustness incumbent (keep vessel, randomize background)
  hullswap   : ours (keep superstructure, swap in a registered donor hull; label kept)

Eval: clean top-1 AND hull-shift top-1 (hull blurred / blacked-out at test).
Headline = hullswap vs bgswap on hull-shift. All swaps use the SegFormer masks.
"""

from __future__ import annotations
import os

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from hullprobe.augment import hull_randomize
from hullprobe.segmenter import load_region_png

B_MANIFEST = os.environ.get("MARVEL_B_MANIFEST", "data/marvel_b/manifest_pose_maskpca.json")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
PRED = Path("data/predictions")

_MASK = {}
def mask_of(rec):
    stem = Path(rec["image_path"]).stem
    if stem not in _MASK:
        p = PRED / f"{stem}.png"
        _MASK[stem] = load_region_png(p) if p.exists() else None
    return _MASK[stem]


def bg_swap(rec, donor_rec):
    """Keep the vessel (hull+super), replace the background with the donor's image."""
    region = mask_of(rec)
    img = np.asarray(Image.open(rec["image_path"]).convert("RGB"))
    if region is None or region.shape != img.shape[:2]:
        return Image.fromarray(img)
    vessel = region > 0
    donor = np.asarray(Image.open(donor_rec["image_path"]).convert("RGB").resize(
        (img.shape[1], img.shape[0])))
    return Image.fromarray(np.where(vessel[..., None], img, donor).astype(np.uint8))


def degrade_hull_img(rec, mode, radius=14):
    """Test-time hull-shift: blur or black-out the hull region (SegFormer mask)."""
    region = mask_of(rec)
    img = np.asarray(Image.open(rec["image_path"]).convert("RGB"))
    if region is None or region.shape != img.shape[:2]:
        return Image.fromarray(img)
    hull = region == 1
    out = img.copy()
    if mode == "blur":
        blurred = np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius)))
        out[hull] = blurred[hull]
    elif mode == "blackout":
        out[hull] = 0
    return Image.fromarray(out)


class ArmDataset(Dataset):
    def __init__(self, records, arm, donors, size=224, train=True, p=0.5, seed=0):
        self.records, self.arm, self.donors = records, arm, donors
        self.p, self.rng = p, random.Random(seed)
        if train:
            self.tf = T.Compose([T.Resize((256, 256)), T.RandomCrop(size),
                                 T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
        else:
            self.tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        ya = int(rec["_label"]); yb = ya; lam = 1.0
        img = None
        if self.arm in ("hullswap", "hullswap_soft", "bgswap") and self.rng.random() < self.p:
            donor = self.rng.choice(self.donors)
            try:
                if self.arm in ("hullswap", "hullswap_soft"):
                    img = hull_randomize(rec, donor, region_map_recip=mask_of(rec),
                                         region_map_donor=mask_of(donor))
                    if self.arm == "hullswap_soft":
                        region = mask_of(rec)
                        v = int((region > 0).sum())
                        # soft label by area: donor supplies the hull region
                        lam = float((region == 2).sum() / v) if v else 1.0  # superstructure frac
                        yb = int(donor["_label"])
                else:
                    img = bg_swap(rec, donor)
            except Exception:
                img, yb, lam = None, ya, 1.0
        if img is None:
            img = Image.open(rec["image_path"]).convert("RGB")
        return self.tf(img), ya, yb, float(lam)


def cutmix_batch(x, y, alpha=1.0):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    H, W = x.shape[2:]
    rh, rw = int(H * (1 - lam) ** 0.5), int(W * (1 - lam) ** 0.5)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
    x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, W)
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (H * W))
    return x, y, y[idx], lam


def evaluate(model, records, transform_fn, device, batch=64):
    class E(Dataset):
        def __len__(self): return len(records)
        def __getitem__(self, i):
            return transform_fn(records[i]), int(records[i]["_label"])
    loader = DataLoader(E(), batch_size=batch, num_workers=6, pin_memory=True)
    model.eval(); correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += (pred == y).sum().item(); total += y.numel()
    return correct / max(1, total)


def train_arm(arm, train_recs, donors, device, args):
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, args.n_classes)
    model = model.to(device)
    ds = ArmDataset(train_recs, arm, donors, train=True, p=args.p)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    crit_none = nn.CrossEntropyLoss(label_smoothing=0.05, reduction="none")
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(args.epochs):
        model.train()
        for x, ya, yb, lam in loader:
            x, ya, yb = x.to(device), ya.to(device), yb.to(device)
            lam = lam.to(device).float()
            optim.zero_grad()
            with torch.autocast("cuda"):
                if arm == "cutmix" and np.random.random() < args.p:
                    x, y1, y2, lc = cutmix_batch(x, ya)
                    out = model(x); loss = lc * crit(out, y1) + (1 - lc) * crit(out, y2)
                else:
                    out = model(x)
                    # per-sample soft label (hard arms: lam=1, yb=ya -> plain CE)
                    loss = (lam * crit_none(out, ya) + (1 - lam) * crit_none(out, yb)).mean()
            scaler.scale(loss).backward(); scaler.step(optim); scaler.update()
        sched.step()
        print(f"  [{arm}] epoch {epoch} done", flush=True)
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="experiments/robustness")
    p.add_argument("--n-per-class", type=int, default=250)
    p.add_argument("--test-per-class", type=int, default=60)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--p", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--arms", default="baseline,cutmix,bgswap,hullswap")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    recs = json.loads(Path(B_MANIFEST).read_text())["records"]
    args.n_classes = 1 + max(r["label"] for r in recs)
    have = lambda r: (PRED / f"{Path(r['image_path']).stem}.png").exists()
    by_split = {"train": defaultdict(list), "test": defaultdict(list)}
    for r in recs:
        s = r.get("source_split")
        if s in by_split and r.get("label") is not None and have(r):
            r["_label"] = int(r["label"]); by_split[s][r["label"]].append(r)
    rng = random.Random(args.seed)
    train_recs, test_recs = [], []
    for c, lst in by_split["train"].items():
        rng.shuffle(lst); train_recs += lst[: args.n_per_class]
    for c, lst in by_split["test"].items():
        rng.shuffle(lst); test_recs += lst[: args.test_per_class]
    print(f"train {len(train_recs)} / test {len(test_recs)} / classes {args.n_classes} "
          f"(all with SegFormer masks)")

    clean = lambda r: T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])(
        Image.open(r["image_path"]).convert("RGB"))
    blur = lambda r: T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])(
        degrade_hull_img(r, "blur"))
    black = lambda r: T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])(
        degrade_hull_img(r, "blackout"))

    results = {}
    for arm in args.arms.split(","):
        print(f"=== ARM {arm} ===", flush=True)
        model = train_arm(arm, train_recs, train_recs, device, args)
        res = {"clean": evaluate(model, test_recs, clean, device),
               "hull_blur": evaluate(model, test_recs, blur, device),
               "hull_blackout": evaluate(model, test_recs, black, device)}
        res["blur_retention"] = res["hull_blur"] / res["clean"] if res["clean"] else 0
        res["blackout_retention"] = res["hull_blackout"] / res["clean"] if res["clean"] else 0
        results[arm] = res
        print(f"  {arm}: {json.dumps(res)}", flush=True)
        torch.save({"model": model.state_dict()}, out / f"{arm}.pt")
        (out / "results.json").write_text(json.dumps(results, indent=1))

    print("\n=== RESULTS (top-1) ===")
    print(f"{'arm':10s} {'clean':>7s} {'hull_blur':>10s} {'hull_black':>11s} {'blur_ret':>9s} {'black_ret':>10s}")
    for arm, r in results.items():
        print(f"{arm:10s} {r['clean']:7.3f} {r['hull_blur']:10.3f} {r['hull_blackout']:11.3f} "
              f"{r['blur_retention']:9.3f} {r['blackout_retention']:10.3f}")


if __name__ == "__main__":
    main()
