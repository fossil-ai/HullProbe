"""Multi-seed training of all de-biasing arms + JTT (canonical group-free de-biaser),
for the causal part-lattice audit (Study 1: natural data).

Arms: baseline (ERM), cutmix, bgswap, hullswap (hull-randomization de-biaser), jtt.
Saves experiments/arms_seed{S}/{arm}.pt for S in seeds. Reuses the training machinery
in robustness_experiment.py so the arms match the originals exactly.

JTT (Liu et al. 2021): train an identification ERM model for a few epochs, collect the
training examples it misclassifies, then train a fresh ERM model upweighting that error
set by lambda -- no group labels needed.
"""
from __future__ import annotations
import argparse, json, random, types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from scripts.robustness_experiment import (
    train_arm, evaluate, degrade_hull_img, B_MANIFEST, PRED, MEAN, STD)

_TF = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
def CLEAN(r):  # evaluate() / misclassified() pass a record, not an image
    return _TF(Image.open(r["image_path"]).convert("RGB"))


def build_pools(seed, n_per_class, test_per_class):
    recs = json.loads(Path(B_MANIFEST).read_text())["records"]
    n_classes = 1 + max(r["label"] for r in recs)
    have = lambda r: (PRED / f"{Path(r['image_path']).stem}.png").exists()
    by = {"train": defaultdict(list), "test": defaultdict(list)}
    for r in recs:
        s = r.get("source_split")
        if s in by and r.get("label") is not None and have(r):
            r["_label"] = int(r["label"]); by[s][r["label"]].append(r)
    rng = random.Random(seed)
    train_recs, test_recs = [], []
    for c, lst in by["train"].items():
        rng.shuffle(lst); train_recs += lst[:n_per_class]
    for c, lst in by["test"].items():
        rng.shuffle(lst); test_recs += lst[:test_per_class]
    return train_recs, test_recs, n_classes


def mkargs(n_classes, epochs, seed):
    return types.SimpleNamespace(n_classes=n_classes, epochs=epochs, batch=64,
                                 lr=2e-4, p=0.5, workers=8, seed=seed)


@torch.inference_mode()
def misclassified(model, recs, device, batch=128):
    class E(Dataset):
        def __len__(self): return len(recs)
        def __getitem__(self, i): return CLEAN(recs[i]), i
    loader = DataLoader(E(), batch_size=batch, num_workers=6)
    model.eval(); errs = []
    for x, idx in loader:
        pred = model(x.to(device)).argmax(1).cpu().numpy()
        for p, i in zip(pred, idx.numpy()):
            if p != recs[i]["_label"]:
                errs.append(int(i))
    return errs


def train_jtt(train_recs, device, args, id_epochs=5, lam=20):
    id_args = mkargs(args.n_classes, id_epochs, args.seed)
    print("  [jtt] phase 1: identification ERM", flush=True)
    id_model = train_arm("baseline", train_recs, train_recs, device, id_args)
    errs = misclassified(id_model, train_recs, device)
    print(f"  [jtt] error set: {len(errs)}/{len(train_recs)} -> upweight x{lam}", flush=True)
    del id_model; torch.cuda.empty_cache()
    up = list(train_recs) + [train_recs[i] for i in errs for _ in range(lam - 1)]
    random.Random(args.seed).shuffle(up)
    print(f"  [jtt] phase 2: ERM on upweighted set ({len(up)})", flush=True)
    return train_arm("baseline", up, up, device, args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--arms", default="baseline,cutmix,bgswap,hullswap,jtt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=250)
    ap.add_argument("--test-per-class", type=int, default=60)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    for seed in a.seeds:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        train_recs, test_recs, n_classes = build_pools(seed, a.n_per_class, a.test_per_class)
        args = mkargs(n_classes, a.epochs, seed)
        out = Path(f"experiments/arms_seed{seed}"); out.mkdir(parents=True, exist_ok=True)
        print(f"\n##### SEED {seed}: train {len(train_recs)} / test {len(test_recs)} #####", flush=True)
        results = {}
        for arm in a.arms.split(","):
            if (out / f"{arm}.pt").exists():
                print(f"  [{arm}] exists, skip", flush=True); continue
            print(f"=== seed {seed} ARM {arm} ===", flush=True)
            model = train_jtt(train_recs, device, args) if arm == "jtt" \
                else train_arm(arm, train_recs, train_recs, device, args)
            acc = evaluate(model, test_recs, CLEAN, device)
            results[arm] = {"clean": acc}
            print(f"  [{arm}] clean acc {acc:.3f}", flush=True)
            torch.save({"model": model.state_dict()}, out / f"{arm}.pt")
            (out / "results.json").write_text(json.dumps(results, indent=1))
            del model; torch.cuda.empty_cache()
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
