"""LfF (Learning from Failure; Nam et al. 2020) -- the canonical group-free de-biaser,
added as a 7th arm to the natural-data part-lattice audit (Study 1).

Cited in the paper's related work alongside JTT; this trains it so it appears in the
audit table rather than being cited-but-absent. LfF trains two networks at once:
  - f_B (biased): Generalized Cross-Entropy (GCE, q) -- amplifies easy/spurious cues;
  - f_D (debiased): CE reweighted per-sample by the *relative difficulty*
        W(x) = L_B(x) / (L_B(x) + L_D(x))
    (EMA-smoothed per-sample losses), which upweights the examples f_B fails on.
No group labels are used. The returned/saved model is f_D. Matches the other arms'
recipe exactly (ConvNeXt-Tiny IMAGENET1K init, AdamW lr 2e-4 wd 0.05, cosine, 12 epochs,
class-balanced 250/class) so the arm is comparable. Saves experiments/arms_seed{S}/lff.pt
for part_lattice_debias_multiseed.py to pick up.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from scripts.train_arms_multiseed import build_pools, mkargs, CLEAN
from scripts.robustness_experiment import evaluate, MEAN, STD

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class IdxDS(Dataset):
    """Plain images + sample index (index needed for the per-sample EMA loss buffers).
    Train aug matches ArmDataset (Resize 256 -> RandomCrop 224 -> HFlip)."""
    def __init__(self, recs, size=224):
        self.recs = recs
        self.tf = T.Compose([T.Resize((256, 256)), T.RandomCrop(size),
                             T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        return self.tf(Image.open(r["image_path"]).convert("RGB")), int(r["_label"]), i


def gce(logits, y, q=0.7):
    """Generalized cross-entropy (per-sample): (1 - p_y^q)/q."""
    p = torch.softmax(logits.float(), dim=1)
    py = p[torch.arange(len(y), device=y.device), y].clamp_min(1e-8)
    return (1.0 - py ** q) / q


def build_model(n_classes):
    m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, n_classes)
    return m.to(DEVICE)


def train_lff(train_recs, args, q=0.7, ema=0.7):
    N = len(train_recs)
    fb, fd = build_model(args.n_classes), build_model(args.n_classes)
    loader = DataLoader(IdxDS(train_recs), batch_size=args.batch, shuffle=True,
                        num_workers=3, pin_memory=False, drop_last=True)
    ob = torch.optim.AdamW(fb.parameters(), lr=args.lr, weight_decay=0.05)
    od = torch.optim.AdamW(fd.parameters(), lr=args.lr, weight_decay=0.05)
    sb = torch.optim.lr_scheduler.CosineAnnealingLR(ob, T_max=args.epochs)
    sd = torch.optim.lr_scheduler.CosineAnnealingLR(od, T_max=args.epochs)
    ce_none = nn.CrossEntropyLoss(reduction="none")
    lb = torch.zeros(N, device=DEVICE); ld = torch.zeros(N, device=DEVICE)
    seen = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    for epoch in range(args.epochs):
        fb.train(); fd.train()
        for x, y, idx in loader:
            x, y, idx = x.to(DEVICE), y.to(DEVICE), idx.to(DEVICE)
            zb, zd = fb(x), fd(x)
            ce_b = ce_none(zb.float(), y)     # for the EMA weighting buffer (detached)
            ce_d = ce_none(zd.float(), y)     # f_D's per-sample CE (kept for the loss)
            with torch.no_grad():
                init = ~seen[idx]
                lb[idx] = torch.where(init, ce_b.detach(), ema * lb[idx] + (1 - ema) * ce_b.detach())
                ld[idx] = torch.where(init, ce_d.detach(), ema * ld[idx] + (1 - ema) * ce_d.detach())
                seen[idx] = True
                W = lb[idx] / (lb[idx] + ld[idx] + 1e-8)   # relative difficulty in [0,1]
            loss_b = gce(zb, y, q).mean()      # bias-amplified
            loss_d = (W * ce_d).mean()          # difficulty-reweighted
            ob.zero_grad(); od.zero_grad()
            (loss_b + loss_d).backward()
            ob.step(); od.step()
        sb.step(); sd.step()
        with torch.no_grad():
            frac = W.mean().item() if N else 0.0
        print(f"  [lff] epoch {epoch} done (last-batch mean W={frac:.3f})", flush=True)
    return fd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=250)
    ap.add_argument("--test-per-class", type=int, default=60)
    ap.add_argument("--q", type=float, default=0.7)
    a = ap.parse_args()
    print("device:", DEVICE, flush=True)
    for seed in a.seeds:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        out = Path(f"experiments/arms_seed{seed}"); out.mkdir(parents=True, exist_ok=True)
        if (out / "lff.pt").exists():
            print(f"[seed {seed}] lff.pt exists, skip", flush=True); continue
        train_recs, test_recs, n_classes = build_pools(seed, a.n_per_class, a.test_per_class)
        args = mkargs(n_classes, a.epochs, seed)
        print(f"\n##### SEED {seed}: LfF on {len(train_recs)} train / {len(test_recs)} test #####", flush=True)
        fd = train_lff(train_recs, args, q=a.q)
        acc = evaluate(fd, test_recs, CLEAN, DEVICE)
        print(f"[seed {seed}] lff clean acc {acc:.3f}", flush=True)
        torch.save({"model": fd.state_dict(), "clean": acc}, out / "lff.pt")
    print("\nALL LfF DONE", flush=True)


if __name__ == "__main__":
    main()
