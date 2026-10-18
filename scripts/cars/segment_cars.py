"""Segment Stanford Cars into their NATURAL part lattice (not the ship grammar).

For each car: GroundedSAM object mask + PCA pose + open-vocab part masks resolved
into a partition over PARTS = {body, front-face, greenhouse, body-side, wheel}.
Saves an .npz {obj_mask, partmap(int8: -1 bg, 0..4 parts)} per image and a manifest
row {image_path, label, pca_angle_deg, pca_centroid_x/y, aspect, npz, parts, coverage}.

Only the object mask + open-vocab parts are learned/prompted; pose + registration are
geometric (reused from hullprobe.pose / swap). Mirrors the maritime pipeline exactly,
with the car's own anatomy instead of hull/superstructure x bow/mid/stern.
"""
from __future__ import annotations
import sys, json, os, random, argparse
import numpy as np
from PIL import Image
import torch

from hullprobe.pose import pca_from_mask
from hullprobe.segmentation import (DEFAULT_GROUNDING_MODEL, DEFAULT_SAM2_CHECKPOINT,
                                   DEFAULT_SAM2_CONFIG)
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

PARTS = ["body", "front-face", "greenhouse", "body-side", "wheel"]   # id 0..4, body=backbone
GROUP = {"wheel": "wheel", "windshield": "greenhouse", "car roof": "greenhouse", "roof": "greenhouse",
         "headlight": "front-face", "grille": "front-face", "bumper": "front-face",
         "car door": "body-side", "door": "body-side"}
OBJ_PROMPT = "car."
PART_PROMPT = "wheel. windshield. headlight. grille. car door. bumper. car roof."
DEV = "cuda"


def build_models():
    proc = AutoProcessor.from_pretrained(str(DEFAULT_GROUNDING_MODEL), local_files_only=True)
    det = AutoModelForZeroShotObjectDetection.from_pretrained(
        str(DEFAULT_GROUNDING_MODEL), local_files_only=True).to(DEV).eval()
    sam = SAM2ImagePredictor(build_sam2(DEFAULT_SAM2_CONFIG, str(DEFAULT_SAM2_CHECKPOINT),
                             device=DEV, apply_postprocessing=True),
                             max_hole_area=64.0, max_sprinkle_area=32.0)
    return proc, det, sam


@torch.inference_mode()
def detect(proc, det, img, prompt, box_t=0.15, text_t=0.12):
    inp = proc(images=img, text=prompt, return_tensors="pt").to(DEV)
    out = det(**inp)
    r = proc.post_process_grounded_object_detection(
        out, inp.input_ids, threshold=box_t, text_threshold=text_t,
        target_sizes=[img.size[::-1]])[0]
    return (r["boxes"].cpu().numpy(), r["scores"].cpu().numpy(),
            [str(l) for l in r.get("text_labels", r["labels"])])


@torch.inference_mode()
def seg_box(sam, box):
    m, _, _ = sam.predict(point_coords=None, point_labels=None, box=box[None, :],
                          multimask_output=False)
    return np.asarray(m[0] if m.ndim == 3 else m, bool)


def canon(label):
    label = label.lower().strip(" .")
    for k, v in GROUP.items():
        if k in label:
            return v
    return None


def process(proc, det, sam, img):
    W, H = img.size
    sam.set_image(np.array(img, copy=True))
    b, s, l = detect(proc, det, img, OBJ_PROMPT)
    if len(b) == 0:
        return None
    areas = np.array([(x2 - x0) * (y2 - y0) for x0, y0, x2, y2 in b])
    oi = int(np.argmax(np.array(s) * areas ** 0.3))
    omask = seg_box(sam, b[oi])
    afrac = float(omask.mean())
    if not (0.04 <= afrac <= 0.93):
        return None
    pca = pca_from_mask(omask)
    if pca is None:
        return None
    _axis, centroid, angle, aspect = pca
    if not (1.35 <= aspect <= 6.0):          # reject head-on/rear (round) and slivers
        return None
    pb, ps, pl = detect(proc, det, img, PART_PROMPT)
    partmap = np.zeros((H, W), np.int8)       # 0 = body backbone
    present = {}
    for j in np.argsort(ps):                  # ascending score: higher paints last (wins)
        cp = canon(pl[j])
        if cp is None or cp == "body":
            continue
        m = seg_box(sam, pb[j]) & omask
        if m.sum() < 0.003 * omask.sum():
            continue
        partmap[m] = PARTS.index(cp)
        present[cp] = present.get(cp, 0) + int(m.sum())
    partmap[~omask] = -1
    named = int((partmap > 0).sum())
    coverage = named / max(1, int(omask.sum()))
    n_named = len(present)
    if n_named < 3 or coverage < 0.45:
        return None
    return dict(omask=omask, partmap=partmap, angle=float(angle),
                cx=float(centroid[0]), cy=float(centroid[1]), aspect=float(aspect),
                parts=sorted(present.keys()), coverage=round(coverage, 3),
                has_front=("front-face" in present))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--out", default="data/cars/seg")
    ap.add_argument("--manifest", default="experiments/cars_seg.json")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    recs = json.loads(open(f"data/cars/{args.split}.json").read())
    by_c = {}
    for r in recs:
        by_c.setdefault(r["label"], []).append(r)
    rng = random.Random(42)
    pool = []
    for c, lst in by_c.items():
        lst = list(lst); rng.shuffle(lst); pool += lst[:args.per_class]
    print(f"segmenting {len(pool)} images ({args.per_class}/class)", flush=True)
    proc, det, sam = build_models()
    out = []
    done = {r["image_path"] for r in (json.loads(open(args.manifest).read())
            if os.path.exists(args.manifest) else [])}
    for i, r in enumerate(pool):
        if r["image_path"] in done:
            continue
        try:
            img = Image.open(r["image_path"]).convert("RGB")
            res = process(proc, det, sam, img)
            if res is None:
                continue
            npz = os.path.join(args.out, os.path.basename(r["image_path"]).replace(".jpg", ".npz"))
            np.savez_compressed(npz, omask=res["omask"], partmap=res["partmap"])
            out.append(dict(image_path=r["image_path"], label=r["label"],
                            pca_angle_deg=res["angle"], pca_centroid_x=res["cx"],
                            pca_centroid_y=res["cy"], aspect=res["aspect"], npz=npz,
                            parts=res["parts"], coverage=res["coverage"], has_front=res["has_front"]))
        except Exception as e:
            continue
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pool)} kept={len(out)}", flush=True)
            open(args.manifest, "w").write(json.dumps(out, indent=0))
    open(args.manifest, "w").write(json.dumps(out, indent=0))
    fronts = sum(r["has_front"] for r in out)
    print(f"DONE_SEG kept={len(out)} with_front={fronts} -> {args.manifest}", flush=True)


if __name__ == "__main__":
    main()
