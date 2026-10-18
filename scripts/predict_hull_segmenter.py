"""Run the trained hull/superstructure segmenter over a manifest and write a
predicted region map (png, {0=bg,1=hull,2=super}) per record. Feed --out-dir back
to review_server.py --deck-source predictions to review and accept more.

    python -m scripts.predict_hull_segmenter --manifest data/pose_full.json \
        --checkpoint experiments/segmenter/best.pt --out-dir data/predictions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from hullprobe.segmenter import save_region_png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--arch", choices=["segformer", "tinyunet"], default="segformer")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.arch == "segformer":
        from hullprobe.segformer_segmenter import build_model, predict_region_map
        model = build_model(device=device)
        size = args.size or 512
    else:
        from hullprobe.segmenter import TinyUNet, predict_region_map
        model = TinyUNet().to(device)
        size = args.size or 256
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state))
    model.eval()

    data = json.loads(Path(args.manifest).read_text())
    records = [r for r in (data if isinstance(data, list) else data.values())
               if r.get("full_vessel_box_xyxy")]
    if args.limit:
        records = records[: args.limit]

    written = 0
    for rec in tqdm(records, desc="predict"):
        try:
            region = predict_region_map(model, rec, device=device, size=size)
            save_region_png(region, out_dir / f"{Path(rec['image_path']).stem}.png")
            written += 1
        except Exception as exc:
            tqdm.write(f"skip {rec.get('image_path')}: {exc}")
    print(f"wrote {written} region maps to {out_dir}")


if __name__ == "__main__":
    main()
