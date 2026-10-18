"""Vessel silhouettes: the zero-shot ADE20K SegFormer boat/ship mask, one RGBA crop per vessel.

The silhouette fixes each vessel's outline, PCA pose, and the outer boundary of every lattice cell;
the fine-tuned hull/superstructure segmenter only splits it (paper Sec. 3.1, App. K). For each
eligible record of the MARVEL-B pose manifest it runs nvidia/segformer-b2-finetuned-ade-512-512,
sums the probabilities of the boat/ship classes, thresholds at 0.12, keeps the largest connected
component, and saves the vessel as an RGBA crop (alpha = mask, lightly blurred) with its box in
image coordinates. The output manifest is the `--regions` input of import_region_manifests.py.

The released assets already contain these silhouettes (see README); this script regenerates them.

    python -m scripts.build_vessel_silhouettes \
        --manifest data/marvel_b/manifest_pose_maskpca.json \
        --out-dir data/marvel_b/silhouettes --out-manifest data/marvel_b/silhouettes.json
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

SEG_MODEL = "nvidia/segformer-b2-finetuned-ade-512-512"
BOAT_THRESHOLD = 0.12
# eligibility filters applied to the MARVEL-B pose manifest
QUALITY_POOL = {"high", "medium"}
ALLOWED_PCA = {"horizontal_long", "diagonal_long", "vertical_long"}
MIN_ASPECT, MIN_MASK_AREA = 1.4, 0.01


class BoatSegmenter:
    def __init__(self, device):
        self.device = device
        self.processor = SegformerImageProcessor.from_pretrained(SEG_MODEL)
        self.model = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL).to(device).eval()
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.boat_ids = sorted(i for i, n in id2label.items()
                               if any(k in n for k in ("boat", "ship", "vessel")))

    @torch.no_grad()
    def mask(self, image):
        w, h = image.size
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        logits = F.interpolate(self.model(**inputs).logits, size=(h, w), mode="bilinear",
                               align_corners=False)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return (probs[self.boat_ids].sum(0) >= BOAT_THRESHOLD).astype(np.uint8)


def largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    return (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def rgba_crop(image, mask, pad=4):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None, None
    h, w = mask.shape
    box = [int(max(0, xs.min() - pad)), int(max(0, ys.min() - pad)),
           int(min(w, xs.max() + pad + 1)), int(min(h, ys.max() + pad + 1))]
    rgba = np.array(image.convert("RGBA"))
    rgba[:, :, 3] = mask * 255
    r, g, b, a = Image.fromarray(rgba).crop(tuple(box)).split()
    return Image.merge("RGBA", (r, g, b, a.filter(ImageFilter.GaussianBlur(radius=0.4)))), box


def eligible(r):
    return (r.get("hullmix_quality") in QUALITY_POOL and r.get("pca_bin") in ALLOWED_PCA
            and r.get("pca_valid", False) and r.get("pca_angle_deg") is not None
            and r.get("pca_aspect", 0) >= MIN_ASPECT and r.get("mask_area_frac", 0) >= MIN_MASK_AREA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/marvel_b/manifest_pose_maskpca.json")
    ap.add_argument("--out-dir", default="data/marvel_b/silhouettes")
    ap.add_argument("--out-manifest", default="data/marvel_b/silhouettes.json")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    records = [r for r in json.loads(Path(a.manifest).read_text())["records"] if eligible(r)]
    seg = BoatSegmenter(device)
    out, skipped = [], Counter()
    for rec in tqdm(records, desc="silhouettes"):
        image = Image.open(rec["image_path"]).convert("RGB")
        mask = largest_component(seg.mask(image))
        if mask.mean() < MIN_MASK_AREA:
            skipped["small_mask"] += 1
            continue
        crop, box = rgba_crop(image, mask)
        if crop is None:
            skipped["empty"] += 1
            continue
        path = out_dir / f"{Path(rec['image_path']).stem}.png"
        crop.save(path)
        out.append({"file_name": rec["file_name"], "image_path": rec["image_path"],
                    "class_name": rec.get("class_name"), "full_vessel_path": str(path.resolve()),
                    "full_vessel_box_xyxy": box})
    Path(a.out_manifest).write_text(json.dumps({"boat_threshold": BOAT_THRESHOLD,
                                                "skipped": dict(skipped), "records": out}, indent=1))
    print(f"wrote {len(out)} silhouettes to {out_dir}; skipped {dict(skipped)}")


if __name__ == "__main__":
    main()
