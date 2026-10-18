"""Materialize Stanford Cars (Donghyun99 HF mirror) to data/cars/{train,test}/<label>/*.jpg + manifests."""
import json, os
from pathlib import Path
from datasets import load_dataset
ROOT = Path("data/cars")
for split in ["train", "test"]:
    ds = load_dataset("Donghyun99/Stanford-Cars", split=split)
    recs = []
    for i, ex in enumerate(ds):
        img = ex["image"].convert("RGB"); lab = int(ex["label"])
        d = ROOT / split / f"{lab:03d}"; d.mkdir(parents=True, exist_ok=True)
        p = d / f"{i:05d}.jpg"; img.save(p, quality=92)
        recs.append({"image_path": str(p), "label": lab, "class": f"{lab:03d}"})
    (ROOT / f"{split}.json").write_text(json.dumps(recs, indent=0))
    print(f"{split}: {len(recs)} images -> {ROOT/split}")
print("DONE_CARS_DOWNLOAD")
