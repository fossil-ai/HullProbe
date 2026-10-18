"""Segment vessels, fit the shared PCA pose, and (optionally) estimate the deck
line for the hull/superstructure split.

For each image:
  1. GroundedSAM full-vessel mask (prompt "ship. boat. vessel.")  -> identical
     to the reference, so `pca_from_mask` lands in the same canonical frame.
  2. pca_from_mask -> pca_angle_deg / centroid / aspect (reject non-longitudinal
     side views with aspect < 1.40, as the reference does).
  3. Save the full-vessel RGBA crop (so canonical.deck_split_masks can run).
  4. If --superstructure-prompt is given, a second GroundedSAM pass yields the
     above-deck mask; its 95th-percentile transverse coordinate is stored as the
     deck-line estimate.

Records are written as JSON and are directly consumable by hullprobe.canonical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

from hullprobe.canonical import estimate_deck_from_superstructure, estimate_deck_geometric
from hullprobe.pose import pca_from_mask
from hullprobe.segmentation import GroundedSAMVesselSegmenter

SUPERSTRUCTURE_PROMPT = "superstructure. deckhouse. bridge. funnel. crane. mast. container."


def bbox(mask, pad=4):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    height, width = mask.shape
    return [
        int(max(0, xs.min() - pad)), int(max(0, ys.min() - pad)),
        int(min(width, xs.max() + pad + 1)), int(min(height, ys.max() + pad + 1)),
    ]


def save_rgba(image, mask, path):
    box = bbox(mask)
    if box is None:
        return None
    rgba = np.asarray(image.convert("RGBA")).copy()
    rgba[..., 3] = mask.astype(np.uint8) * 255
    crop = Image.fromarray(rgba, "RGBA").crop(tuple(box))
    red, green, blue, alpha = crop.split()
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=0.4))
    Image.merge("RGBA", (red, green, blue, alpha)).save(path)
    return box


def iter_images(images_arg):
    path = Path(images_arg)
    if path.is_dir():
        yield from sorted(
            p.resolve() for p in path.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    else:
        yield from (
            Path(line.strip()).resolve()
            for line in path.read_text().splitlines() if line.strip()
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="image dir or newline-list file")
    parser.add_argument("--out", required=True, help="output manifest JSON")
    parser.add_argument("--region-root", required=True, help="where full-vessel crops are written")
    parser.add_argument("--min-aspect", type=float, default=1.40)
    parser.add_argument("--superstructure-prompt", nargs="?", const=SUPERSTRUCTURE_PROMPT,
                        default=None, help="run a second above-deck pass to estimate the deck line")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=500,
                        help="write the manifest every N processed images")
    parser.add_argument("--label-from-path", action="store_true", default=True,
                        help="store class = parent directory name (MARVEL layout)")
    args = parser.parse_args()

    region_root = Path(args.region_root)
    region_root.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    vessel_seg = GroundedSAMVesselSegmenter()
    deck_seg = (
        GroundedSAMVesselSegmenter(prompt=args.superstructure_prompt)
        if args.superstructure_prompt else None
    )

    records, skipped = [], 0
    images = list(iter_images(args.images))
    if args.limit:
        images = images[: args.limit]
    for image_path in tqdm(images, desc="pose"):
        try:
            image = Image.open(image_path).convert("RGB")
            prediction = vessel_seg.predict(image)
            mask = prediction.mask.astype(bool)
            pca = pca_from_mask(mask)
            if pca is None:
                skipped += 1
                continue
            _axis, centroid, angle, aspect = pca
            if aspect < args.min_aspect:
                skipped += 1
                continue
            ident = Path(image_path).stem
            full_path = region_root / f"{ident}.png"
            full_box = save_rgba(image, mask, full_path)
            record = {
                "image_path": str(image_path),
                "full_vessel_path": str(full_path),
                "full_vessel_box_xyxy": full_box,
                "pca_angle_deg": angle,
                "pca_centroid_x": float(centroid[0]),
                "pca_centroid_y": float(centroid[1]),
                "pca_aspect": aspect,
                "mask_area_frac": float(mask.mean()),
                "detected_label": prediction.label,
            }
            if args.label_from_path:
                record["class"] = Path(image_path).parent.name
            # Default deck line: silhouette-coverage geometry (no extra models,
            # and empirically far more reliable than a SAM part prompt, which
            # returns near-whole-ship masks).
            try:
                record["deck_transverse"] = estimate_deck_geometric(record)
                record["deck_method"] = "geometric"
            except Exception as exc:
                record["deck_transverse_error"] = str(exc)
            if deck_seg is not None:  # optional cross-check
                try:
                    above = deck_seg.predict(image).mask.astype(bool)
                    record["deck_transverse_sam"] = estimate_deck_from_superstructure(record, above)
                except Exception:
                    pass
            records.append(record)
        except Exception as exc:
            skipped += 1
            tqdm.write(f"skip {image_path}: {exc}")
        if args.checkpoint_every and len(records) % args.checkpoint_every == 0 and records:
            Path(args.out).write_text(json.dumps(records, indent=1))

    Path(args.out).write_text(json.dumps(records, indent=1))
    print(f"wrote {len(records)} records to {args.out} ({skipped} skipped)")


if __name__ == "__main__":
    main()
