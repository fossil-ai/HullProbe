"""Audit the hull/superstructure deck split on a sample of vessels.

Produces a visualization gallery (superstructure = red, hull = blue) and a
JSON of area statistics, so we can eyeball whether the GroundedSAM above-deck
prompt + transverse geometry gives a clean, sensible deck line before spending
anything on the swap operator or the causal probe.

Deck line source, in priority order:
  1. --superstructure-prompt : run a live GroundedSAM above-deck pass per image;
  2. record["deck_transverse"] : precomputed by build_pose_manifest.py;
  3. centroid split (T=0) : geometry-only placeholder (clearly wrong, plumbing).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from hullprobe.canonical import deck_split_stats, estimate_deck_geometric, overlay_split


def load_records(manifest):
    data = json.loads(Path(manifest).read_text())
    return data if isinstance(data, list) else list(data.values())


def gallery(images, cols=4, pad=6, bg=(18, 18, 18)):
    if not images:
        return None
    cell_w = max(im.width for im in images)
    cell_h = max(im.height for im in images)
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell_w + (cols + 1) * pad,
                             rows * cell_h + (rows + 1) * pad), bg)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(im, (pad + c * (cell_w + pad), pad + r * (cell_h + pad)))
    return grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-width", type=int, default=360)
    parser.add_argument("--superstructure-prompt", nargs="?",
                        const="superstructure. deckhouse. bridge. funnel. crane. mast. container.",
                        default=None)
    parser.add_argument("--deck-method", choices=["geometric", "manifest", "centroid"],
                        default="geometric",
                        help="geometric: silhouette coverage profile (no models); "
                             "manifest: record['deck_transverse']; centroid: T=0")
    parser.add_argument("--coverage-frac", type=float, default=0.75)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.manifest)
    random.Random(args.seed).shuffle(records)
    records = records[: args.sample]

    deck_seg = None
    if args.superstructure_prompt:
        from hullprobe.segmentation import GroundedSAMVesselSegmenter
        deck_seg = GroundedSAMVesselSegmenter(prompt=args.superstructure_prompt)

    tiles, stats = [], []
    for rec in records:
        try:
            super_mask = None
            if deck_seg is not None:
                image = Image.open(rec["image_path"]).convert("RGB")
                super_mask = deck_seg.predict(image).mask.astype(bool)
                deck = None  # estimated inside from the mask
            elif args.deck_method == "geometric":
                deck = estimate_deck_geometric(rec, coverage_frac=args.coverage_frac)
            elif args.deck_method == "manifest":
                deck = rec.get("deck_transverse")
            else:
                deck = None  # centroid split
            tile = overlay_split(rec, deck=deck, superstructure_mask=super_mask)
            if tile.width > args.max_width:
                scale = args.max_width / tile.width
                tile = tile.resize((args.max_width, int(tile.height * scale)))
            tiles.append(tile)
            s = deck_split_stats(rec, deck=deck, superstructure_mask=super_mask)
            s["image_path"] = rec["image_path"]
            stats.append(s)
        except Exception as exc:
            print(f"skip {rec.get('image_path')}: {exc}")

    grid = gallery(tiles)
    if grid is not None:
        grid.save(out_dir / "deck_split_gallery.jpg", quality=92)
    fracs = [s["superstructure_frac"] for s in stats if "superstructure_frac" in s]
    summary = {
        "n": len(stats),
        "superstructure_frac_mean": float(np.mean(fracs)) if fracs else None,
        "superstructure_frac_median": float(np.median(fracs)) if fracs else None,
        "superstructure_frac_p10_p90": (
            [float(np.percentile(fracs, 10)), float(np.percentile(fracs, 90))]
            if fracs else None
        ),
        "deck_source": ("live_segmenter" if deck_seg else "manifest_or_centroid"),
    }
    (out_dir / "deck_split_stats.json").write_text(
        json.dumps({"summary": summary, "per_image": stats}, indent=1)
    )
    print(json.dumps(summary, indent=1))
    print(f"gallery -> {out_dir/'deck_split_gallery.jpg'}")


if __name__ == "__main__":
    main()
