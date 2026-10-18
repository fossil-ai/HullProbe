"""Do correlational attribution methods identify/track the planted shortcut the way our
interventional measure does? (Reviewer: compare against Grad-CAM / IG / RISE, quantitatively.)

Controlled hull-color shortcut (Sec. phase2): the hull is tinted red/blue, correlated with
the label; the ground-truth shortcut region is therefore the hull mask. For each de-biaser
(ERM..DFR) we train the phase2 model, then measure, on correctly-classified test images, the
fraction of each attribution method's mass that falls on the hull region (`hull_frac`). A
method that 'sees' the shortcut puts mass there; a method that tracks *reliance* should have
that mass FALL as the model de-biases. We compare each attribution method's
corr(hull_frac, worst-group-acc) across the five de-biasers to the interventional
hull-color-flip, which tracks it at r=-0.996.

Reuses phase2_controlled's dataset + training verbatim. Writes experiments/attribution_compare.json.
"""
from __future__ import annotations
import json, random, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import scripts.phase2_controlled as pc
pc.SHORTCUT = "hull_color"
from scripts.phase2_controlled import (
    build, DS, make_model, train_erm, train_groupdro, train_dfr, evaluate,
    tinted, _region_mask, norm, DEVICE)

torch.backends.cudnn.benchmark = True


# ---------- attribution methods (plain torch, no captum) ----------
def _hull_mask_224(rec):
    m = _region_mask(rec["image_path"], (224, 224, 3))
    return None if m is None else np.asarray(m, bool)


def _input_tensor(rec):
    im = tinted(rec, rec["s"])
    return norm(Image.fromarray(np.ascontiguousarray(im)))


def grad_cam(model, x, target):
    """Grad-CAM on resnet layer4; returns (B,224,224) >=0."""
    x = x.detach().requires_grad_(True)   # force backward to reach layer4 (frozen-backbone DFR)
    acts, grads = {}, {}
    layer = model.layer4
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    model.zero_grad()
    out = model(x)
    sel = out.gather(1, target.view(-1, 1)).sum()
    sel.backward()
    h1.remove(); h2.remove()
    a, g = acts["a"], grads["g"]                       # (B,C,h,w)
    w = g.mean(dim=(2, 3), keepdim=True)               # channel weights
    cam = F.relu((w * a).sum(1))                        # (B,h,w)
    cam = F.interpolate(cam[:, None], size=(224, 224), mode="bilinear", align_corners=False)[:, 0]
    return cam.detach()


def integrated_gradients(model, x, target, steps=32):
    base = torch.zeros_like(x)
    total = torch.zeros_like(x)
    for k in range(1, steps + 1):
        xi = (base + (k / steps) * (x - base)).requires_grad_(True)
        model.zero_grad()
        out = model(xi)
        out.gather(1, target.view(-1, 1)).sum().backward()
        total = total + xi.grad.detach()
    ig = (x - base) * (total / steps)
    return ig.abs().sum(1).detach()                    # (B,224,224)


@torch.inference_mode()
def rise(model, x, target, n_masks=120, grid=7, p=0.5):
    """RISE saliency for a single image x:(1,3,224,224)."""
    B = 40
    sal = torch.zeros(224, 224, device=DEVICE)
    done = 0
    while done < n_masks:
        b = min(B, n_masks - done)
        small = (torch.rand(b, 1, grid, grid, device=DEVICE) < p).float()
        up = F.interpolate(small, size=(224 + grid, 224 + grid), mode="bilinear", align_corners=False)
        sx = torch.randint(0, grid, (1,)).item(); sy = torch.randint(0, grid, (1,)).item()
        masks = up[:, :, sx:sx + 224, sy:sy + 224]     # (b,1,224,224)
        scores = torch.softmax(model(x * masks), 1)[:, target.item()]
        sal += (scores.view(b, 1, 1) * masks[:, 0]).sum(0)
        done += b
    return (sal / n_masks)[None]                        # (1,224,224)


def hull_fraction(amap, hull):
    """Fraction of (non-negative) attribution mass inside the hull region."""
    a = amap.clamp_min(0).cpu().numpy()
    tot = a.sum()
    return float(a[hull].sum() / tot) if tot > 1e-9 else float("nan")


def attribution_scores(model, test_recs, n_img=120, n_rise=70, seed=0):
    """Mean hull-fraction over correctly-classified test images, per attribution method."""
    model.eval()
    rng = random.Random(seed)
    recs = [r for r in test_recs if _hull_mask_224(r) is not None]
    rng.shuffle(recs)
    gc, ig, rs, areas = [], [], [], []
    n_r = 0
    for r in recs:
        if len(gc) >= n_img:
            break
        x = _input_tensor(r)[None].to(DEVICE)
        with torch.inference_mode():
            pred = int(model(x).argmax(1))
        if pred != r["y"]:                              # only correctly-classified
            continue
        hull = _hull_mask_224(r)
        t = torch.tensor([pred], device=DEVICE)
        gc.append(hull_fraction(grad_cam(model, x, t)[0], hull))
        ig.append(hull_fraction(integrated_gradients(model, x, t)[0], hull))
        areas.append(float(hull.mean()))
        if n_r < n_rise:
            rs.append(hull_fraction(rise(model, x, t)[0], hull)); n_r += 1
    f = lambda v: float(np.nanmean(v)) if v else float("nan")
    return {"gradcam": f(gc), "ig": f(ig), "rise": f(rs), "hull_area": f(areas), "n": len(gc)}


def train_method(method, train, test, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if method == "ERM":
        return train_erm(train, 8)
    if method == "SUBG":
        byg = defaultdict(list)
        for j, r in enumerate(train): byg[r["g"]].append(j)
        n = min(len(v) for v in byg.values())
        w = np.zeros(len(train))
        for g in byg:
            for j in random.sample(byg[g], n): w[j] = 1.0
        return train_erm(train, 8, weights=w.tolist())
    if method == "GroupDRO":
        return train_groupdro(train, 8)
    if method == "JTT":
        idm = train_erm(train, 4)
        idm.eval()
        from torch.utils.data import DataLoader
        wr = np.ones(len(train))
        ld = DataLoader(DS(train), batch_size=128, num_workers=6)
        off = 0
        with torch.inference_mode():
            for x, y, g, i in ld:
                pr = idm(x.to(DEVICE)).argmax(1).cpu().numpy()
                for k in range(len(pr)):
                    if pr[k] != int(y[k]): wr[off + k] = 20.0
                off += len(pr)
        del idm; torch.cuda.empty_cache()
        return train_erm(train, 8, weights=wr.tolist())
    if method == "DFR":
        base = train_erm(train, 8)
        return train_dfr(train, base, test)
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-img", type=int, default=120)
    ap.add_argument("--n-rise", type=int, default=70)
    a = ap.parse_args()
    methods = ["ERM", "SUBG", "GroupDRO", "JTT", "DFR"]
    print("device:", DEVICE, "shortcut:", pc.SHORTCUT, flush=True)

    per = {m: {"wga": [], "flip": [], "gradcam": [], "ig": [], "rise": [], "hull_area": []}
           for m in methods}

    def dump():
        agg = {"methods": methods, "per_method": {}}
        for m in methods:
            agg["per_method"][m] = {k: ([float(np.mean(v)), float(np.std(v))] if v else [float("nan"), 0.0])
                                    for k, v in per[m].items()}
        wga = np.array([np.mean(per[m]["wga"]) if per[m]["wga"] else np.nan for m in methods])
        def corr(sig):
            y = np.array([np.mean(per[m][sig]) if per[m][sig] else np.nan for m in methods])
            if np.isnan(wga).any() or np.isnan(y).any() or np.std(y) < 1e-9:
                return float("nan")
            return float(np.corrcoef(wga, y)[0, 1])
        agg["corr_with_wga"] = {s: corr(s) for s in ("flip", "gradcam", "ig", "rise")}
        Path("experiments/attribution_compare.json").write_text(json.dumps(agg, indent=1))
        return agg

    for seed in a.seeds:
        train, test, _ = build(seed=seed)
        for m in methods:
            try:
                model = train_method(m, train, test, seed)
                ev = evaluate(model, test)
                at = attribution_scores(model, test, a.n_img, a.n_rise, seed)
                per[m]["wga"].append(ev["worst_group_acc"]); per[m]["flip"].append(ev["hull_color_flip"])
                for k in ("gradcam", "ig", "rise", "hull_area"):
                    per[m][k].append(at[k])
                print(f"[seed {seed}] {m:9s} wga {ev['worst_group_acc']:.3f} flip {ev['hull_color_flip']:.3f} "
                      f"| hull-frac gradcam {at['gradcam']:.3f} ig {at['ig']:.3f} rise {at['rise']:.3f} "
                      f"(area {at['hull_area']:.3f}, n={at['n']})", flush=True)
                del model; torch.cuda.empty_cache()
                dump()
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[seed {seed}] {m} FAILED: {e}", flush=True)

    agg = dump()
    print("\ncorr(signal, worst-group-acc) across methods:")
    for s in ("flip", "gradcam", "ig", "rise"):
        print(f"  {s:9s} r = {agg['corr_with_wga'][s]:+.3f}")
    print("wrote experiments/attribution_compare.json")


if __name__ == "__main__":
    main()
