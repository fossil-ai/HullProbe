"""Qualitative cars figure: decomposition + cross-class front-face swap + same-class control.

Panels (paper/figures/fig_cars.jpg):
  1. recipient car
  2. its natural 5-part decomposition {body, front-face, greenhouse, body-side, wheel}
  3. cross-class swap: recipient body + DONOR front-face (class changes)
  4. same-class control: identical op with a same-class donor (label preserved, same artifacts)
Reuses the lattice's registration + stencil. Only the object/part masks + classifier are learned.
"""
from __future__ import annotations
import sys, json, random, argparse
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from torchvision import models as tvm

from scripts.cars.cars_lattice import (similarity, affine_coeffs, load_seg, PARTS, N, tf, classify)

COL = [(130, 130, 130), (230, 60, 60), (70, 120, 235), (70, 200, 110), (245, 205, 45)]


def decomp_overlay(rec):
    img = np.asarray(Image.open(rec["image_path"]).convert("RGB"))
    pm = load_seg(rec)[1]
    H, W = pm.shape
    img = np.asarray(Image.fromarray(img).resize((W, H)))
    ov = img.copy()
    for i in range(N):
        m = pm == i
        if m.any():
            ov[m] = (0.5 * np.array(COL[i]) + 0.5 * ov[m]).astype(np.uint8)
    return img, ov


def swap_part(recip_rec, donor_rec, part):
    """Recipient with its `part` replaced by the donor's, stenciled in canonical frame."""
    rrgb = np.asarray(Image.open(recip_rec["image_path"]).convert("RGB"))
    H, W = rrgb.shape[:2]
    romask, rpm = load_seg(recip_rec)
    domask, _ = load_seg(donor_rec)
    drgb = np.asarray(Image.open(donor_rec["image_path"]).convert("RGB"))
    drgba = Image.fromarray(np.dstack([drgb, (domask * 255).astype(np.uint8)]), "RGBA")
    r2d = np.linalg.inv(similarity(donor_rec, domask)) @ similarity(recip_rec, romask)
    warped = np.asarray(drgba.transform((W, H), Image.AFFINE, affine_coeffs(r2d),
                        resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0)))
    sel = (rpm == PARTS.index(part)) & (warped[..., 3] > 128)
    out = rrgb.copy(); out[sel] = warped[..., :3][sel]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="experiments/cars_seg.json")
    ap.add_argument("--clf", default="experiments/cars_clf/best.pt")
    ap.add_argument("--out", default="paper/figures/fig_cars.jpg")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg = json.loads(open(args.manifest).read())
    by_c = {}
    for r in seg:
        by_c.setdefault(r["label"], []).append(r)
    ck = torch.load(args.clf, map_location=device)
    nc = ck["model"]["classifier.2.weight"].shape[0]
    model = tvm.convnext_tiny(); model.classifier[2] = nn.Linear(model.classifier[2].in_features, nc)
    model.load_state_dict(ck["model"]); model.to(device).eval()

    rng = random.Random(args.seed)
    # side-profile recipients (high aspect) give the cleanest roofline swap
    cand = [r for r in seg if r["has_front"] and len(by_c[r["label"]]) >= 2
            and 2.2 <= r.get("aspect", 0) <= 4.0]
    rng.shuffle(cand)
    for r in cand:
        y = r["label"]
        rrgb = np.asarray(Image.open(r["image_path"]).convert("RGB"))
        if classify(model, [rrgb], device)[0] != y:
            continue
        # viewpoint-matched cross donors: aspect within 0.35 of the recipient
        donors = [d for d in seg if d["label"] != y and abs(d.get("aspect", 2) - r.get("aspect", 2)) <= 0.35]
        donors.sort(key=lambda d: abs(d.get("aspect", 2) - r.get("aspect", 2)))
        picked = None
        for d in donors[:40]:
            sw = swap_part(r, d, "greenhouse")
            pred = int(classify(model, [sw], device)[0])
            if pred != y:                      # greenhouse swap changed the class (dominant part)
                picked = (d, sw, pred); break
        if picked is None:
            continue
        d, cross_img, cross_pred = picked
        same_pool = [s for s in by_c[y] if s["image_path"] != r["image_path"]]
        s_don = rng.choice(same_pool)
        same_img = swap_part(r, s_don, "greenhouse")
        orig, ov = decomp_overlay(r)
        # 2x2 grid: [recipient | decomposition] / [cross-class greenhouse swap | same-class control]
        from PIL import ImageDraw, ImageFont
        panels = [orig, ov, cross_img, same_img]
        H, W = orig.shape[:2]
        panels = [np.asarray(Image.fromarray(p).resize((W, H))) for p in panels]
        labels = ["recipient", "5-part decomposition",
                  "cross-class greenhouse swap", "same-class control"]
        lab_h = max(18, H // 12)
        tiled = []
        for p, lab in zip(panels, labels):
            canvas = Image.new("RGB", (W, H + lab_h), (255, 255, 255))
            canvas.paste(Image.fromarray(p), (0, lab_h))
            dr = ImageDraw.Draw(canvas)
            dr.text((4, 2), lab, fill=(0, 0, 0))
            tiled.append(np.asarray(canvas))
        gx = np.full((tiled[0].shape[0], 8, 3), 255, np.uint8)
        gy = np.full((6, 2 * W + 8, 3), 255, np.uint8)
        row1 = np.concatenate([tiled[0], gx, tiled[1]], 1)
        row2 = np.concatenate([tiled[2], gx, tiled[3]], 1)
        grid = np.concatenate([row1, gy, row2], 0)
        Image.fromarray(grid).save(args.out, quality=92)
        print(f"SAVED {args.out} | recip_class={y} cross_donor_class={d['label']} cross_pred={cross_pred} "
              f"same_donor_class={s_don['label']} size={grid.shape[1]}x{grid.shape[0]}")
        return
    print("no flipping example found")


if __name__ == "__main__":
    main()
