"""Run the causal swap probe with a real frozen classifier.

Reads a pose manifest whose records carry a class label (and, ideally, a
`deck_transverse` from build_pose_manifest.py --superstructure-prompt), forms
cross-class recipient/donor pairs plus same-class artifact controls, and asks a
frozen classifier whether swapping the superstructure transfers class identity
more than swapping the hull. Writes the probe summary JSON and a few example
swap visualizations.

Example:
  python -m scripts.causal_swap_probe \
      --manifest data/pose_maskpca.json --checkpoint convnext_tiny.pt \
      --classes classes.json --label-key class --out experiments/probe
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from hullprobe.constants import IMG_SIZE
from hullprobe.probe import run_pair, summarize
from hullprobe.swap import hull_swap, superstructure_swap


def build_classifier(checkpoint, arch, num_classes, device):
    """Return classify(PIL.Image) -> logits (np.ndarray, C) for a frozen model."""
    import torch
    import torchvision.transforms as T
    from torchvision import models

    factory = getattr(models, arch)
    model = factory(num_classes=num_classes)
    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("model", state.get("state_dict", state))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    preprocess = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    @torch.inference_mode()
    def classify(image):
        x = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
        return model(x)[0].float().cpu().numpy()

    return classify


def load_pools(manifest, classes, label_key, min_per_class):
    data = json.loads(Path(manifest).read_text())
    records = data if isinstance(data, list) else list(data.values())
    index = {name: i for i, name in enumerate(classes)}
    pools = defaultdict(list)
    for rec in records:
        label = rec.get(label_key)
        if label in index and "full_vessel_box_xyxy" in rec:
            pools[index[label]].append(rec)
    return {c: recs for c, recs in pools.items() if len(recs) >= min_per_class}


def sample_pairs(pools, n_cross, n_control, seed):
    rng = random.Random(seed)
    classes = sorted(pools)
    cross, control = [], []
    for _ in range(n_cross):
        rc, dc = rng.sample(classes, 2)
        cross.append((rng.choice(pools[rc]), rc, rng.choice(pools[dc]), dc))
    for _ in range(n_control):
        c = rng.choice(classes)
        a, b = rng.sample(pools[c], 2) if len(pools[c]) >= 2 else (pools[c][0], pools[c][0])
        control.append((a, c, b, c))
    return cross, control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--classes", required=True, help="JSON list of class names in index order")
    parser.add_argument("--arch", default="convnext_tiny")
    parser.add_argument("--label-key", default="class")
    parser.add_argument("--predictions-dir", default=None,
                        help="SegFormer hull/superstructure masks (else geometric deck)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-cross", type=int, default=400)
    parser.add_argument("--n-control", type=int, default=200)
    parser.add_argument("--min-per-class", type=int, default=2)
    parser.add_argument("--n-viz", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    classes = json.loads(Path(args.classes).read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pools = load_pools(args.manifest, classes, args.label_key, args.min_per_class)
    if len(pools) < 2:
        raise SystemExit(f"need >=2 populated classes, got {len(pools)}")
    classify = build_classifier(args.checkpoint, args.arch, len(classes), device)
    cross_spec, control_spec = sample_pairs(pools, args.n_cross, args.n_control, args.seed)

    # SegFormer hull/superstructure masks: load per record (cached) and pass to
    # the swap. Falls back to the geometric deck if a mask is missing.
    from hullprobe.segmenter import load_region_png
    pred_dir = Path(args.predictions_dir) if args.predictions_dir else None
    _cache = {}

    def region_map(rec):
        if pred_dir is None:
            return None
        stem = Path(rec["image_path"]).stem
        if stem not in _cache:
            path = pred_dir / f"{stem}.png"
            _cache[stem] = load_region_png(path) if path.exists() else None
        return _cache[stem]

    def evaluate(spec):
        results = []
        for recip, rc, donor, dc in spec:
            try:
                results.append(run_pair(classify, recip, donor, rc, dc,
                                        region_map_recip=region_map(recip),
                                        region_map_donor=region_map(donor)))
            except Exception as exc:
                print(f"skip pair: {exc}")
        return results

    cross = evaluate(cross_spec)
    control = evaluate(control_spec)
    summary = summarize(cross, control, seed=args.seed)

    # A handful of qualitative examples.
    for i, (recip, _rc, donor, _dc) in enumerate(cross_spec[: args.n_viz]):
        try:
            kw = dict(region_map_recip=region_map(recip), region_map_donor=region_map(donor))
            superstructure_swap(recip, donor, **kw)[0].save(out_dir / f"ex{i}_superswap.jpg", quality=90)
            hull_swap(recip, donor, **kw)[0].save(out_dir / f"ex{i}_hullswap.jpg", quality=90)
        except Exception as exc:
            print(f"viz {i} failed: {exc}")

    (out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary.get("verdict", {}), indent=1))
    print(f"summary -> {out_dir/'probe_summary.json'}")


if __name__ == "__main__":
    main()
