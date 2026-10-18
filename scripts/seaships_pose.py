"""Step 3: compute the PCA longitudinal axis + centroid per SeaShips crop from its
segmenter vessel mask, and keep only crops whose hull/superstructure decomposition is
non-degenerate (all six lattice cells non-empty). Writes data/seaships/pose.json (records
with pca_angle_deg, pca_centroid_x/y) -- the manifest the lattice consumes."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

from PIL import Image
from hullprobe.segmenter import load_region_png
from hullprobe.lattice import cell_masks, CELLS

PRED = Path("data/seaships/predictions")
VES = Path("data/seaships/vessel"); VES.mkdir(parents=True, exist_ok=True)
MIN_CELL = 150   # px per lattice cell to be usable


def main():
    recs = json.loads(Path("data/seaships/records.json").read_text())["records"]
    out = []
    degen = nomask = 0
    for r in recs:
        stem = Path(r["image_path"]).stem
        p = PRED / f"{stem}.png"
        if not p.exists():
            nomask += 1; continue
        rm = load_region_png(p)
        vessel = rm > 0
        ys, xs = np.nonzero(vessel)
        if xs.size < 500:
            degen += 1; continue
        pts = np.stack([xs, ys], 1).astype(np.float64)
        c = pts.mean(0)
        cov = np.cov((pts - c).T)
        w, v = np.linalg.eigh(cov)
        axis = v[:, -1]                                   # principal (longitudinal) axis
        angle = math.degrees(math.atan2(axis[1], axis[0]))
        # write an RGBA vessel crop (alpha = segmenter foreground) for full_vessel_alpha
        rgb = np.asarray(Image.open(r["image_path"]).convert("RGB"))
        if rm.shape != rgb.shape[:2]:
            rm = np.asarray(Image.fromarray(rm).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST))
            vessel = rm > 0
        rgba = np.dstack([rgb, (vessel * 255).astype(np.uint8)])
        vpath = VES / f"{stem}.png"
        Image.fromarray(rgba, "RGBA").save(vpath)
        rec = dict(r, pca_angle_deg=float(angle), pca_centroid_x=float(c[0]),
                   pca_centroid_y=float(c[1]), full_vessel_path=str(vpath.resolve()),
                   full_vessel_box_xyxy=[0, 0, int(rm.shape[1]), int(rm.shape[0])])
        try:                                              # require all 6 cells present
            cells, _ = cell_masks(rec, region_map=rm)
            if min(int(cells[cc].sum()) for cc in CELLS) < MIN_CELL:
                degen += 1; continue
        except Exception:
            degen += 1; continue
        out.append(rec)
    from collections import Counter
    (Path("data/seaships/pose.json")).write_text(json.dumps({"records": out}, indent=1))
    print(f"usable {len(out)} / {len(recs)} (skipped {degen} degenerate, {nomask} no-mask)")
    print("per-split:", Counter(r["source_split"] for r in out))
    print("per-class:", Counter(r["class_name"] for r in out))
    print("wrote data/seaships/pose.json")


if __name__ == "__main__":
    main()
