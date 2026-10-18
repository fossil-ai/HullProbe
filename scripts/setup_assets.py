"""Install the released HullProbe assets into this checkout.

Given the files downloaded from the GitHub Release (in --assets) and your local copy of the MARVEL
images (--marvel-root, the folder containing train/ and test/), this
  - extracts the manifest, silhouette and region-map archives into --assets,
  - writes data/pose_B.json and data/marvel_b/manifest_pose_maskpca.json with the ${MARVEL_ROOT} and
    ${HULLPROBE_ASSETS} placeholders replaced by absolute local paths,
  - links region maps to data/predictions/ and checkpoints to the paths the scripts read
    (experiments/segformer/best.pt, experiments/baseline_clf/best.pt, ...).
Every script then runs with its defaults.

    python -m scripts.setup_assets --assets ~/Downloads/hullprobe_assets \
        --marvel-root /data/MARVEL/MARVEL_images1
"""
from __future__ import annotations
import argparse
import json
import os
import tarfile
from pathlib import Path

CHECKPOINTS = {  # released file -> path the scripts load
    "segformer_b2_hullsup.pt": "experiments/segformer/best.pt",
    "convnext_tiny_marvel.pt": "experiments/baseline_clf/best.pt",
    "resnet50_marvel.pt": "experiments/baseline_clf_resnet50/best.pt",
    "vit_b16_marvel.pt": "experiments/baseline_clf_vit_b_16/best.pt",
}


def link(src, dst):
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    dst.symlink_to(Path(src).resolve())


def fill(path, marvel_root, assets):
    return path.replace("${MARVEL_ROOT}", marvel_root).replace("${HULLPROBE_ASSETS}", assets)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", required=True, help="folder with the downloaded release files")
    ap.add_argument("--marvel-root", default=os.environ.get("MARVEL_ROOT"),
                    help="MARVEL image root containing train/ and test/ (or set MARVEL_ROOT)")
    a = ap.parse_args()
    if not a.marvel_root:
        raise SystemExit("pass --marvel-root or set MARVEL_ROOT")
    assets = Path(a.assets).expanduser().resolve()
    marvel = str(Path(a.marvel_root).expanduser().resolve())

    for name in ("manifests", "silhouettes", "region_maps"):
        tgz = assets / f"hullprobe_{name}.tar.gz"
        if not (assets / name).is_dir():
            if not tgz.exists():
                raise SystemExit(f"missing {tgz}")
            print(f"extracting {tgz.name} ...")
            with tarfile.open(tgz) as t:
                try:
                    t.extractall(assets, filter="data")
                except TypeError:  # Python without tarfile extraction filters
                    t.extractall(assets)

    pose = json.loads((assets / "manifests/pose_B.json").read_text())
    for r in pose:
        r["image_path"] = fill(r["image_path"], marvel, str(assets))
        r["full_vessel_path"] = fill(r["full_vessel_path"], marvel, str(assets))
    man = json.loads((assets / "manifests/manifest_pose_maskpca.json").read_text())
    man["dataset_root"] = marvel
    for r in man["records"]:
        r["image_path"] = fill(r["image_path"], marvel, str(assets))
    Path("data/marvel_b").mkdir(parents=True, exist_ok=True)
    Path("data/pose_B.json").write_text(json.dumps(pose))
    Path("data/marvel_b/manifest_pose_maskpca.json").write_text(json.dumps(man))

    if Path("data/predictions").is_symlink():
        Path("data/predictions").unlink()
    if not Path("data/predictions").exists():
        Path("data/predictions").symlink_to(assets / "region_maps")
    for f, dst in CHECKPOINTS.items():
        if (assets / f).exists():
            link(assets / f, dst)

    missing = [r["image_path"] for r in pose[:200] if not Path(r["image_path"]).exists()]
    print(f"wrote data/pose_B.json ({len(pose)} vessels) and data/marvel_b/manifest_pose_maskpca.json")
    print("checkpoints:", [d for f, d in CHECKPOINTS.items() if (assets / f).exists()])
    if missing:
        print(f"WARNING: {len(missing)}/200 sampled images not found under {marvel}, e.g. {missing[0]}")
    else:
        print("MARVEL images found. Default paths are ready; e.g. python -m scripts.headline_spectrum")


if __name__ == "__main__":
    main()
