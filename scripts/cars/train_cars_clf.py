"""Train an ImageNet-pretrained ConvNeXt-Tiny on Stanford Cars (196 classes).

Reads data/cars/{train,test}.json ({image_path,label}). Same recipe as the
maritime baseline classifier (AdamW, cosine, label smoothing) so the cars probe
is judged by a fair, un-graft-augmented classifier.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tvm

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def make_model(n):
    m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, n)
    return m


class DS(Dataset):
    def __init__(self, recs, train):
        self.recs = recs
        self.tf = (T.Compose([T.Resize((256, 256)), T.RandomCrop(224), T.RandomHorizontalFlip(),
                              T.ToTensor(), T.Normalize(MEAN, STD)]) if train else
                   T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)]))

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        return self.tf(Image.open(r["image_path"]).convert("RGB")), int(r["label"])


def evaluate(model, loader, device):
    model.eval(); c = t = 0
    with torch.inference_mode():
        for x, y in loader:
            c += (model(x.to(device)).argmax(1).cpu() == y).sum().item(); t += y.numel()
    return c / max(1, t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="experiments/cars_clf")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    torch.manual_seed(42); np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train = json.loads(Path("data/cars/train.json").read_text())
    test = json.loads(Path("data/cars/test.json").read_text())
    n = 1 + max(r["label"] for r in train + test)
    print(f"train {len(train)} / test {len(test)} / classes {n}", flush=True)
    model = make_model(n).to(device)

    def L(ds, sh):
        return DataLoader(ds, batch_size=args.batch, shuffle=sh, num_workers=args.workers,
                          pin_memory=True, drop_last=sh)
    tl, vl = L(DS(train, True), True), L(DS(test, False), False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, hist = -1.0, []
    for e in range(args.epochs):
        model.train(); run = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step(); run += loss.item()
        sch.step(); acc = evaluate(model, vl, device)
        hist.append({"epoch": e, "loss": run / len(tl), "test_acc": acc})
        print(f"epoch {e} loss {run/len(tl):.4f} test_acc {acc:.4f}", flush=True)
        torch.save({"model": model.state_dict(), "epoch": e, "test_acc": acc}, out / "last.pt")
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "epoch": e, "test_acc": acc}, out / "best.pt")
    (out / "metrics.json").write_text(json.dumps({"best_test_acc": best, "n_classes": n, "history": hist}, indent=1))
    print(f"DONE_CLF best_test_acc {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
