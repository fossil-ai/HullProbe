"""Build a hullprobe pose manifest from existing MARVEL-B pose/region artifacts —
no GroundedSAM re-run. Joins the pose (pca_angle/centroid) manifest with the
hull_regions (full-vessel crop + box) manifest on file_name, so the 26-class
benchmark pool that the baseline classifiers were trained on is reused directly.

    python -m scripts.import_region_manifests \
        --pose  data/marvel_b/manifest_pose_maskpca.json \
        --regions data/marvel_b/hull_regions/manifest_hull_regions.json \
        --out data/pose_B.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def records_of(obj):
    if isinstance(obj, dict) and "records" in obj:
        return obj["records"]
    return obj if isinstance(obj, list) else list(obj.values())


def clean_class(name):
    return str(name).strip().strip("'\"") if name is not None else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", required=True)
    parser.add_argument("--regions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--require-valid-pose", action="store_true", default=False,
                        help="keep only records whose pca_valid is truthy")
    args = parser.parse_args()

    pose = {r["file_name"]: r for r in records_of(json.load(open(args.pose)))
            if "file_name" in r}
    regions = records_of(json.load(open(args.regions)))

    out, missing_crop, no_pose = [], 0, 0
    for h in regions:
        key = h.get("file_name")
        p = pose.get(key)
        if p is None:
            no_pose += 1
            continue
        if not h.get("full_vessel_path") or not h.get("full_vessel_box_xyxy"):
            missing_crop += 1
            continue
        if args.require_valid_pose and not p.get("pca_valid", True):
            continue
        if not os.path.exists(h["full_vessel_path"]):
            missing_crop += 1
            continue
        out.append({
            "image_path": p.get("image_path") or h.get("image_path"),
            "full_vessel_path": h["full_vessel_path"],
            "full_vessel_box_xyxy": h["full_vessel_box_xyxy"],
            "pca_angle_deg": p["pca_angle_deg"],
            "pca_centroid_x": p["pca_centroid_x"],
            "pca_centroid_y": p["pca_centroid_y"],
            "pca_aspect": p.get("pca_aspect"),
            "class": clean_class(h.get("class_name") or p.get("class_name")),
            "yaw_deg": p.get("yaw_deg"),
            "file_name": key,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    classes = Counter(r["class"] for r in out)
    print(f"wrote {len(out)} records to {args.out} "
          f"({no_pose} lacked pose, {missing_crop} lacked a crop)")
    print(f"{len(classes)} classes: {dict(sorted(classes.items()))}")


if __name__ == "__main__":
    main()
