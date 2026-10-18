"""fig_decomp.png -- Fig. 1: the two decompositions and their 2x3 product.

2x2 panel: original | hull/superstructure  //  fwd/mid/aft thirds | six cells.
Uses the audit's own region map (data/predictions/<id>.png) and lattice.cell_masks, so the
cells drawn are exactly the cells the audit swaps. Every panel has its own legend, and the
six-cell panel uses colors that appear in neither of the two factor panels, so the cells are not
read as segmentation classes. Longitudinal thirds are labeled
fwd/mid/aft, not bow/stern: they are canonical-frame thirds, not a detected bow.

Usage (repo root):  python paper/make_fig_decomp.py
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from hullprobe.lattice import cell_masks, CELLS
from hullprobe.segmenter import load_region_png

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "fig_decomp.png"
VID = "581061"

# factor panels keep the colors used across the paper (hull red, superstructure blue)
C_HULL, C_SUP = "#C0392B", "#2E6FBF"
C_FWD, C_MID, C_AFT = "#2CA02C", "#FF8C00", "#9467BD"
# product panel: six colors disjoint from the five above
C_CELL = {("hull", "fwd"): "#F2C80F", ("hull", "mid"): "#8C564B", ("hull", "aft"): "#E377C2",
          ("sup", "fwd"): "#17BECF", ("sup", "mid"): "#BCBD22", ("sup", "aft"): "#8A8A8A"}


def rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32)


def overlay(img, masks_colors, alpha=0.55):
    out = img.astype(np.float32).copy()
    for m, c in masks_colors:
        out[m] = (1 - alpha) * img[m] + alpha * rgb(c)
    return out.clip(0, 255).astype(np.uint8)


def main():
    rec = next(r for r in json.loads((ROOT / "data/pose_B.json").read_text())
               if Path(r["image_path"]).stem == VID)
    img = np.asarray(Image.open(rec["image_path"]).convert("RGB"))
    cells, vessel = cell_masks(rec, region_map=load_region_png(ROOT / f"data/predictions/{VID}.png"))
    hull = cells[("hull", "fwd")] | cells[("hull", "mid")] | cells[("hull", "aft")]
    sup = vessel & ~hull
    thirds = [cells[("hull", l)] | cells[("sup", l)] for l in ("fwd", "mid", "aft")]

    panels = [
        (img, None),
        (overlay(img, [(hull, C_HULL), (sup, C_SUP)]),
         [(C_HULL, "hull"), (C_SUP, "superstructure")]),
        (overlay(img, [(thirds[0], C_FWD), (thirds[1], C_MID), (thirds[2], C_AFT)]),
         [(C_FWD, "fwd"), (C_MID, "mid"), (C_AFT, "aft")]),
        (overlay(img, [(cells[c], C_CELL[c]) for c in CELLS], alpha=0.6),
         # legend fills column-major: order so rows read hull / sup and columns fwd/mid/aft
         [(C_CELL[(v, l)], f"{v}-{l}") for l in ("fwd", "mid", "aft") for v in ("hull", "sup")]),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(4.0, 4.02), gridspec_kw=dict(wspace=0.04, hspace=0.05))
    for ax, (im, leg) in zip(axs.ravel(), panels):
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4); s.set_color("#888888")
        if leg:
            ncol = 3 if len(leg) > 2 else 2
            ax.legend(handles=[Patch(color=c, label=l) for c, l in leg], loc="lower center",
                      ncol=ncol, fontsize=5.2, framealpha=0.88, handlelength=1.0,
                      handletextpad=0.35, columnspacing=0.6, borderpad=0.3, labelspacing=0.25)
    fig.savefig(OUT, dpi=400, bbox_inches="tight", pad_inches=0.01)
    print("wrote", OUT, "| cells px:", {f"{c[0]}-{c[1]}": int(cells[c].sum()) for c in CELLS})


if __name__ == "__main__":
    main()
