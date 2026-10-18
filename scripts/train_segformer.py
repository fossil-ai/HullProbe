"""Fine-tune the ADE20K-pretrained SegFormer for hull/superstructure segmentation.

Relies on the maritime prior (ADE has ship/boat/water/sky), so it fine-tunes on
the accepted gold masks directly — no weak-geometric pretrain needed (that would
just reintroduce the straight-line bias). Class-balanced by vessel class for
appearance coverage; weighted CE so the small superstructure isn't ignored.
Saves best.pt / last.pt / metrics.json.

    python -m scripts.train_segformer --annotations data/annotations/annotations.json \
        --manifest data/pose_B.json --out experiments/segformer
"""

from __future__ import annotations

import argparse
import os
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler

from hullprobe.segformer_segmenter import DEFAULT_CKPT, build_model, logits_at
from hullprobe.segmenter import GoldRegionDataset


def region_iou(pred, target):
    out = {}
    for cls in (1, 2):
        p, t = pred == cls, target == cls
        union = (p | t).sum()
        out[cls] = (p & t).sum() / union if union > 0 else np.nan
    return out


def evaluate(model, loader, device, size):
    model.eval()
    hull, sup = [], []
    with torch.inference_mode():
        for x, y in loader:
            logits = logits_at(model, x.to(device), (size, size))
            pred = logits.argmax(1).cpu().numpy()
            for p, t in zip(pred, y.numpy()):
                iou = region_iou(p, t)
                hull.append(iou[1]); sup.append(iou[2])
    return {"hull_iou": float(np.nanmean(hull)), "superstructure_iou": float(np.nanmean(sup)),
            "mean_iou": float(np.nanmean(hull + sup))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", "--annotations", dest="accepted",
                        default="data/annotations/annotations.json",
                        help="annotation list (repo format: paths relative to --marvel-root and to this file)")
    parser.add_argument("--marvel-root", default=os.environ.get("MARVEL_ROOT", "data/marvel/MARVEL_images1"))
    parser.add_argument("--manifest", required=True, help="for vessel-class balancing")
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--epochs", type=int, default=45)   # paper: 45 epochs, best val mIoU at epoch 33
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--super-weight", type=float, default=2.0,
                        help="CE weight on the superstructure class")
    parser.add_argument("--no-balance", dest="balance", action="store_false", default=True)
    parser.add_argument("--no-hflip", dest="hflip", action="store_false", default=True,
                        help="disable random horizontal flip augmentation")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    entries = json.loads(Path(args.accepted).read_text())
    ann_dir = Path(args.accepted).parent
    for e in entries:  # repo format -> absolute paths (order is kept, so the seeded split is unchanged)
        if "image" in e:
            e["image_path"] = str(Path(args.marvel_root) / e["image"])
            e["label_path"] = str(ann_dir / e["label"])
    if len(entries) < 4:
        raise SystemExit(f"need >=4 accepted labels, got {len(entries)}")
    random.shuffle(entries)
    n_val = max(1, int(args.val_frac * len(entries)))
    val_entries, train_entries = entries[:n_val], entries[n_val:]

    data = json.loads(Path(args.manifest).read_text())
    records = data if isinstance(data, list) else list(data.values())
    stem2class = {Path(r["image_path"]).stem: r.get("class") for r in records}

    model = build_model(args.checkpoint, device=device)

    train_ds = GoldRegionDataset(train_entries, size=args.size, hflip=args.hflip)
    val_ds = GoldRegionDataset(val_entries, size=args.size)
    print(f"hflip augmentation: {args.hflip}")

    sampler = None
    if args.balance:
        classes = [stem2class.get(e["id"]) for e in train_entries]
        counts = Counter(classes)
        weights = [1.0 / np.sqrt(counts[c]) if c else 1.0 for c in classes]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        print(f"class-balanced over {len(counts)} classes (max {max(counts.values())} / min {min(counts.values())})")

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              shuffle=sampler is None, num_workers=args.workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=args.workers, pin_memory=True)

    weight = torch.tensor([0.5, 1.0, args.super_weight], device=device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    print(f"fine-tune SegFormer on {len(train_ds)} gold masks (val {len(val_ds)})")
    best, history = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = logits_at(model, x, (args.size, args.size))
            loss = F.cross_entropy(logits, y, weight=weight)
            optim.zero_grad(); loss.backward(); optim.step()
            running += loss.item()
        sched.step()
        metrics = evaluate(model, val_loader, device, args.size)
        history.append({"epoch": epoch, "loss": running / max(1, len(train_loader)), **metrics})
        print(f"epoch {epoch} loss {running/max(1,len(train_loader)):.4f} "
              f"mIoU {metrics['mean_iou']:.4f} (hull {metrics['hull_iou']:.3f} "
              f"super {metrics['superstructure_iou']:.3f})")
        torch.save({"model": model.state_dict(), "epoch": epoch, "checkpoint": args.checkpoint},
                   out_dir / "last.pt")
        if metrics["mean_iou"] > best:
            best = metrics["mean_iou"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "checkpoint": args.checkpoint},
                       out_dir / "best.pt")

    # ---- Per-class superstructure IoU on val (honest, not the bulk-inflated mean)
    from collections import defaultdict

    from hullprobe.segformer_segmenter import predict_region_map
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device)["model"])
    model.eval()
    stem2rec = {Path(r["image_path"]).stem: r for r in records}
    per = defaultdict(list)
    for e in val_entries:
        rec = stem2rec.get(e["id"])
        if rec is None or not rec.get("full_vessel_box_xyxy"):
            continue
        gold = np.array(Image.open(e["label_path"]))
        pred = predict_region_map(model, rec, device=device, size=args.size)
        if pred.shape != gold.shape:
            continue
        pg, tg = pred == 2, gold == 2
        union = (pg | tg).sum()
        per[stem2class.get(e["id"], "?")].append((pg & tg).sum() / union if union > 0 else np.nan)
    per_class = {c: {"n": len(v), "super_iou": float(np.nanmean(v))} for c, v in sorted(per.items())}
    print("per-class superstructure IoU (val):")
    for c, s in per_class.items():
        flag = "  <-- FAILING" if s["super_iou"] < 0.3 else ""
        print(f"  {c:22s} n={s['n']:2d}  super_iou={s['super_iou']:.3f}{flag}")

    (out_dir / "metrics.json").write_text(json.dumps(
        {"n_accepted": len(entries), "best_mean_iou": best, "checkpoint": args.checkpoint,
         "per_class_superstructure_iou": per_class, "history": history}, indent=1))
    print(json.dumps({"best_mean_iou": best, "n_accepted": len(entries)}, indent=1))


if __name__ == "__main__":
    main()
