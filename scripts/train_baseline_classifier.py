"""Train a CLEAN baseline ConvNeXt-Tiny on the 26-class MARVEL-B pool.

The only prior classifier with saved weights was graft-augmented (trained to
ignore composited parts), which biases the causal probe toward null. This trains
a plain ImageNet-pretrained ConvNeXt-Tiny on the standard B train split with no
swap/graft augmentation — a fair judge for the probe.

    python -m scripts.train_baseline_classifier --out experiments/baseline_clf
"""

from __future__ import annotations
import os

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tvm

B_MANIFEST = os.environ.get("MARVEL_B_MANIFEST", "data/marvel_b/manifest_pose_maskpca.json")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def make_model(arch, n_classes):
    if arch == "convnext_tiny":
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n_classes)
    elif arch == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif arch == "vit_b_16":
        m = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads.head = nn.Linear(m.heads.head.in_features, n_classes)
    else:
        raise ValueError(arch)
    return m


class BDataset(Dataset):
    def __init__(self, records, train):
        self.records = records
        if train:
            self.tf = T.Compose([T.Resize((256, 256)), T.RandomCrop(224),
                                 T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
        else:
            self.tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = Image.open(r["image_path"]).convert("RGB")
        return self.tf(img), int(r["label"])


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += (pred == y).sum().item(); total += y.numel()
    return correct / max(1, total)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--manifest", default=B_MANIFEST)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arch", default="convnext_tiny",
                   choices=["convnext_tiny", "resnet50", "vit_b_16"])
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    records = json.loads(Path(args.manifest).read_text())["records"]
    train = [r for r in records if r.get("source_split") == "train" and r.get("label") is not None]
    test = [r for r in records if r.get("source_split") == "test" and r.get("label") is not None]
    n_classes = 1 + max(r["label"] for r in records)
    print(f"train {len(train)} / test {len(test)} / classes {n_classes}")

    model = make_model(args.arch, n_classes).to(device)

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle,
                          num_workers=args.workers, pin_memory=True, drop_last=shuffle)
    train_loader, test_loader = loader(BDataset(train, True), True), loader(BDataset(test, False), False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    best, history = -1.0, []
    for epoch in range(args.epochs):
        model.train(); running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad(); loss = crit(model(x), y); loss.backward(); optim.step()
            running += loss.item()
        sched.step()
        acc = evaluate(model, test_loader, device)
        history.append({"epoch": epoch, "loss": running / len(train_loader), "test_acc": acc})
        print(f"epoch {epoch} loss {running/len(train_loader):.4f} test_acc {acc:.4f}")
        torch.save({"model": model.state_dict(), "epoch": epoch, "test_acc": acc}, out_dir / "last.pt")
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "test_acc": acc}, out_dir / "best.pt")
    (out_dir / "metrics.json").write_text(json.dumps(
        {"best_test_acc": best, "n_classes": n_classes, "history": history}, indent=1))
    print(f"best test acc {best:.4f}")


if __name__ == "__main__":
    main()
