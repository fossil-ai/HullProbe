"""Segment a 16-model binary-task subset (train+test) for the cars convergent-validity study.
Reuses segment_cars.process to get object + part-maps; the color shortcut is planted on the
lattice parts (body + body-side) at train time."""
import sys, json, os
import numpy as np
from PIL import Image

from scripts.cars.segment_cars import build_models, process

CLASSES = list(range(16))                 # 16 models -> 2 groups of 8
os.makedirs("data/cars/conv_seg", exist_ok=True)
proc, det, sam = build_models()
out = []
for split in ["train", "test"]:
    recs = json.load(open(f"data/cars/{split}.json"))
    sub = [r for r in recs if r["label"] in CLASSES]
    for i, r in enumerate(sub):
        try:
            res = process(proc, det, sam, Image.open(r["image_path"]).convert("RGB"))
        except Exception:
            res = None
        if res is None:
            continue
        npz = f"data/cars/conv_seg/{split}_{os.path.basename(r['image_path'])[:-4]}.npz"
        np.savez_compressed(npz, partmap=res["partmap"])
        out.append(dict(image_path=r["image_path"], model=int(r["label"]),
                        group=(0 if r["label"] < 8 else 1), split=split, npz=npz))
        if (i + 1) % 100 == 0:
            print(f"  {split} {i+1}/{len(sub)} kept={len(out)}", flush=True)
    print(f"{split} done, kept so far {len(out)}", flush=True)
json.dump(out, open("experiments/cars_conv_seg.json", "w"))
ntr = sum(r["split"] == "train" for r in out); nte = sum(r["split"] == "test" for r in out)
print(f"DONE_CONV_SEG total={len(out)} train={ntr} test={nte}", flush=True)
