"""AFR (group-free last-layer reweighting) on the NATURAL 26-class setting.

The unifying test: in the controlled study, DFR's last-layer retrain removed the
spurious reliance (head-encoded). DFR needs group labels. AFR (Qiu et al. 2023) is
the group-free version: freeze the ERM features, retrain ONLY the last linear layer
with a weighted loss that upweights the examples the ERM is least confident on (the
shortcut-breaking minority) -- no group labels needed.

If AFR reduces the causal hull-share here -- where augmentation and group-free JTT did
NOT -- then the natural spurious reliance is also head-encoded and removable by
reweighting, unifying both studies. Saves experiments/arms_seed{S}/afr.pt for the
lattice audit to pick up.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import convnext_tiny

from scripts.train_arms_multiseed import build_pools

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_erm(path, n_classes=26):
    m = convnext_tiny(num_classes=n_classes)
    st = torch.load(path, map_location="cpu"); st = st.get("model", st.get("state_dict", st))
    m.load_state_dict({k.replace("module.", ""): v for k, v in st.items()}, strict=False)
    return m.to(DEVICE).eval()


def feats_and_probs(model, recs):
    """Frozen ERM penultimate features (N,768) and softmax probs (N,26)."""
    from scripts.train_arms_multiseed import CLEAN
    from torch.utils.data import Dataset

    class E(Dataset):
        def __len__(self): return len(recs)
        def __getitem__(self, i): return CLEAN(recs[i]), int(recs[i]["_label"])
    loader = DataLoader(E(), batch_size=128, num_workers=8)
    F, P, Y = [], [], []
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(DEVICE)
            z = model.features(x); z = model.avgpool(z)
            z = model.classifier[0](z); z = model.classifier[1](z)   # (N,768)
            logit = model.classifier[2](z)
            F.append(z.cpu()); P.append(torch.softmax(logit, 1).cpu()); Y.append(y)
    return torch.cat(F), torch.cat(P), torch.cat(Y)


def afr_head(F, P, Y, n_classes=26, gamma=2.0, epochs=500, lr=1e-2, wd=1e-3):
    """Retrain a linear head on frozen features with AFR weights."""
    F = F.to(DEVICE); Y = Y.to(DEVICE)
    py = P[torch.arange(len(Y)), Y.cpu()].to(DEVICE)          # ERM confidence in true class
    w = (1.0 - py).clamp(1e-4, 1.0) ** gamma                  # upweight low-confidence-correct
    w = (w / w.mean()).detach()
    head = nn.Linear(F.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss(reduction="none")
    for _ in range(epochs):
        opt.zero_grad()
        loss = (w * crit(head(F), Y)).mean()
        loss.backward(); opt.step()
    return head.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--gamma", type=float, default=2.0)
    a = ap.parse_args()
    for s in a.seeds:
        base = Path(f"experiments/arms_seed{s}/baseline.pt")
        out = Path(f"experiments/arms_seed{s}/afr.pt")
        if not base.exists():
            print(f"skip seed {s}: no baseline"); continue
        train_recs, _test, n_classes = build_pools(s, 250, 60)
        model = load_erm(str(base), n_classes)
        F, P, Y = feats_and_probs(model, train_recs)
        # AFR: reweight the low-confidence tail; report how skewed the weights are
        py = P[torch.arange(len(Y)), Y]
        print(f"[seed {s}] ERM mean conf {py.mean():.3f}; retraining head (gamma={a.gamma})", flush=True)
        new_head = afr_head(F, P, Y, n_classes, gamma=a.gamma)
        model.classifier[2].load_state_dict(new_head.state_dict())
        torch.save({"model": model.state_dict()}, out)
        print(f"[seed {s}] wrote {out}", flush=True)
    print("AFR heads done", flush=True)


if __name__ == "__main__":
    main()
