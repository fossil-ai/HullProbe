"""Step 1 of the SeaShips second-dataset generality test: turn the VOC detection
dataset into a per-vessel classification set (crop each ship bbox), with the VOC
train/test split. Writes data/seaships/crops/*.jpg, data/seaships/records.json,
data/seaships/classes.json. Later steps: segment the crops, add PCA pose, train, lattice.
"""
from __future__ import annotations
import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

SS = Path(os.environ.get("SEASHIPS_ROOT", "data/seaships_raw"))
OUT = Path("data/seaships"); (OUT / "crops").mkdir(parents=True, exist_ok=True)
MINW, MINH = 180, 70          # skip tiny/truncated boxes that won't segment well
PAD = 0.06                    # small context padding around the bbox


def split_of(stem, sets):
    for name, ids in sets.items():
        if stem in ids:
            return name
    return None


def main():
    sets = {}
    for name in ["train", "test"]:
        f = SS / "ImageSets" / "Main" / f"{name}.txt"
        sets[name] = set(f.read_text().split()) if f.exists() else set()
    class_names, records = [], []
    kept = skipped = 0
    for xml in sorted((SS / "Annotations").glob("*.xml")):
        stem = xml.stem
        sp = split_of(stem, sets)
        if sp is None:
            continue
        root = ET.parse(xml).getroot()
        img_path = SS / "JPEGImages" / f"{stem}.jpg"
        if not img_path.exists():
            continue
        W = int(root.findtext("size/width")); H = int(root.findtext("size/height"))
        objs = root.findall("object")
        for oi, obj in enumerate(objs):
            cname = obj.findtext("name").strip()
            bb = obj.find("bndbox")
            x0, y0 = int(bb.findtext("xmin")), int(bb.findtext("ymin"))
            x1, y1 = int(bb.findtext("xmax")), int(bb.findtext("ymax"))
            w, h = x1 - x0, y1 - y0
            if w < MINW or h < MINH:
                skipped += 1; continue
            px, py = int(w * PAD), int(h * PAD)
            cx0, cy0 = max(0, x0 - px), max(0, y0 - py)
            cx1, cy1 = min(W, x1 + px), min(H, y1 + py)
            if cname not in class_names:
                class_names.append(cname)
            out_name = f"{stem}_{oi}.jpg"
            crop = Image.open(img_path).convert("RGB").crop((cx0, cy0, cx1, cy1))
            crop.save(OUT / "crops" / out_name, quality=94)
            records.append({"image_path": str((OUT / "crops" / out_name).resolve()),
                            "class_name": cname, "source_split": sp})
            kept += 1
    class_names = sorted(class_names)
    cidx = {c: i for i, c in enumerate(class_names)}
    for r in records:
        r["label"] = cidx[r["class_name"]]
    (OUT / "records.json").write_text(json.dumps({"records": records}, indent=1))
    (OUT / "classes.json").write_text(json.dumps(class_names, indent=1))
    from collections import Counter
    print(f"kept {kept} crops, skipped {skipped} tiny; classes {class_names}")
    print("per-split:", Counter(r["source_split"] for r in records))
    print("per-class:", Counter(r["class_name"] for r in records))
    print("wrote data/seaships/{records.json,classes.json,crops/}")


if __name__ == "__main__":
    main()
