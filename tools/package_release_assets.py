"""Build the GitHub Release assets from a working HullProbe installation (maintainers only).

Writes to --out:
  hullprobe_manifests.tar.gz     pose_B.json and manifest_pose_maskpca.json with machine paths
                                 replaced by ${MARVEL_ROOT} / ${HULLPROBE_ASSETS} placeholders
  hullprobe_silhouettes.tar.gz   one RGBA PNG per vessel; the alpha channel is the zero-shot vessel
                                 mask (RGB is zeroed, which the loader ignores, to keep files small)
  hullprobe_region_maps.tar.gz   fine-tuned SegFormer hull/superstructure maps (0 bg, 1 hull, 2 sup)
  segformer_b2_hullsup.pt        hull/superstructure segmenter (best validation checkpoint)
  convnext_tiny_marvel.pt        attribution classifier (77.6%); optionally resnet50 / vit_b_16
  SHA256SUMS

Users install them with scripts/setup_assets.py.

    python tools/package_release_assets.py --pose data/pose_B.json \
        --manifest /path/to/manifest_pose_maskpca.json --region-maps data/predictions \
        --marvel-root /path/to/MARVEL_images1 --segformer experiments/segformer/best.pt \
        --classifier experiments/baseline_clf/best.pt --out release_assets
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def stem(p):
    return Path(p).stem


def placeholder(path, marvel_root):
    root = str(Path(marvel_root).resolve()).rstrip("/") + "/"
    if not path.startswith(root):
        raise ValueError(f"{path} is not under MARVEL root {root}")
    return "${MARVEL_ROOT}/" + path[len(root):]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tar_dir(src, dst):
    with tarfile.open(dst, "w:gz") as t:
        t.add(src, arcname=src.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose", default="data/pose_B.json")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--region-maps", default="data/predictions")
    ap.add_argument("--marvel-root", required=True)
    ap.add_argument("--segformer", default="experiments/segformer/best.pt")
    ap.add_argument("--classifier", default="experiments/baseline_clf/best.pt")
    ap.add_argument("--extra-classifier", nargs=2, action="append", default=[],
                    metavar=("NAME", "CKPT"), help="e.g. resnet50_marvel experiments/baseline_clf_resnet50/best.pt")
    ap.add_argument("--out", default="release_assets")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stage = out / "_stage"; shutil.rmtree(stage, ignore_errors=True)

    pose = json.loads(Path(a.pose).read_text())
    stems = [stem(r["image_path"]) for r in pose]
    assert len(set(stems)) == len(stems), "vessel ids (image stems) must be unique"

    # 1. silhouettes (alpha only) and region maps, keyed by vessel id
    sil = stage / "silhouettes"; rmaps = stage / "region_maps"
    sil.mkdir(parents=True); rmaps.mkdir(parents=True)
    for r in tqdm(pose, desc="silhouettes + region maps"):
        s = stem(r["image_path"])
        alpha = np.asarray(Image.open(r["full_vessel_path"]).convert("RGBA"))[:, :, 3]
        rgba = np.zeros(alpha.shape + (4,), np.uint8); rgba[:, :, 3] = alpha
        Image.fromarray(rgba, "RGBA").save(sil / f"{s}.png", optimize=True)
        shutil.copy2(Path(a.region_maps) / f"{s}.png", rmaps / f"{s}.png")

    # 2. manifests with placeholders
    man = stage / "manifests"; man.mkdir(parents=True)
    pose_out = []
    for r in pose:
        r = dict(r)
        r["image_path"] = placeholder(r["image_path"], a.marvel_root)
        r["full_vessel_path"] = "${HULLPROBE_ASSETS}/silhouettes/" + stem(r["image_path"]) + ".png"
        pose_out.append(r)
    (man / "pose_B.json").write_text(json.dumps(pose_out))
    m = json.loads(Path(a.manifest).read_text())
    m["dataset_root"] = "${MARVEL_ROOT}"
    for r in m["records"]:
        r["image_path"] = placeholder(r["image_path"], a.marvel_root)
    (man / "manifest_pose_maskpca.json").write_text(json.dumps(m))
    for f in man.iterdir():
        assert "/home/" not in f.read_text(), f"machine path left in {f}"

    # 3. archives, checkpoints, checksums
    for d, name in ((man, "hullprobe_manifests"), (sil, "hullprobe_silhouettes"),
                    (rmaps, "hullprobe_region_maps")):
        tar_dir(d, out / f"{name}.tar.gz")
    shutil.copy2(a.segformer, out / "segformer_b2_hullsup.pt")
    shutil.copy2(a.classifier, out / "convnext_tiny_marvel.pt")
    for name, ckpt in a.extra_classifier:
        shutil.copy2(ckpt, out / f"{name}.pt")
    shutil.rmtree(stage)
    files = sorted(p for p in out.iterdir() if p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files))
    for p in files:
        print(f"{p.stat().st_size / 2**20:8.1f} MB  {p.name}")


if __name__ == "__main__":
    main()
