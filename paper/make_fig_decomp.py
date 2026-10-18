"""fig_decomp.png -- Fig. 1: the two decompositions and their 2x3 product.

2x2 panel: original | hull/superstructure  //  fwd/mid/aft thirds | six cells, in the paper's
palette. Masks come from the audit's own region map and lattice.cell_masks, so the cells drawn are
exactly the cells the audit swaps; they are upsampled with per-label smoothing (no staircased
edges), and the six cells are separated by thin white borders. Longitudinal thirds are labeled
fwd/mid/aft: they are canonical-frame thirds, not a detected bow and stern.

Usage (repo root):  python paper/make_fig_decomp.py   (needs the release assets; see README)
"""
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from hullprobe.lattice import CELLS, cell_masks
from hullprobe.segmenter import load_region_png

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "fig_decomp.png"
VID = "581061"
UP = 4                                   # render at 4x the 256 px source
LEG = dict(fontsize=4.6, handlelength=0.9, handleheight=0.7, handletextpad=0.3, columnspacing=0.7,
           borderpad=0.3, labelspacing=0.2, framealpha=0.85, edgecolor="none")

# original palette (scripts/decompose_image.py, submission Fig. 1)
C_HULL, C_SUP = "#C82828", "#286EC8"
C_THIRD = {1: "#2CA02C", 2: "#FF8C00", 3: "#945ABE"}                  # fwd, mid, aft
C_CELL = {("hull", "fwd"): "#D62728", ("hull", "mid"): "#FF7F0E", ("hull", "aft"): "#9467BD",
          ("sup", "fwd"): "#1F77B4", ("sup", "mid"): "#2CA02C", ("sup", "aft"): "#17BECF"}


def rgb(h):
    return np.array(matplotlib.colors.to_rgb(h)) * 255


def smooth_labels(label, n):
    """Upsample an integer label map by UP with per-label blurring, then argmax (a clean partition)."""
    h, w = label.shape
    prob = []
    for k in range(n):
        m = cv2.resize((label == k).astype(np.float32), (w * UP, h * UP), interpolation=cv2.INTER_LINEAR)
        prob.append(cv2.GaussianBlur(m, (0, 0), UP * 0.6))
    return np.argmax(np.stack(prob), 0)


def overlay(img, label, colors, alpha):
    out = img.astype(np.float32).copy()
    for k, c in colors.items():
        m = label == k
        out[m] = (1 - alpha) * img[m] + alpha * rgb(c)
    return out.clip(0, 255).astype(np.uint8)


def borders(label, keep):
    """Pixels where the label changes between two labels in `keep` (internal cell borders)."""
    b = np.zeros(label.shape, bool)
    for dy, dx in ((0, 1), (1, 0)):
        a, c = label[: label.shape[0] - dy, : label.shape[1] - dx], label[dy:, dx:]
        e = (a != c) & np.isin(a, keep) & np.isin(c, keep)
        b[: label.shape[0] - dy, : label.shape[1] - dx] |= e
    return cv2.dilate(b.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def main():
    rec = next(r for r in json.loads((ROOT / "data/pose_B.json").read_text())
               if Path(r["image_path"]).stem == VID)
    src = Image.open(rec["image_path"]).convert("RGB")
    img = np.asarray(src.resize((src.width * UP, src.height * UP), Image.LANCZOS))
    cells, vessel = cell_masks(rec, region_map=load_region_png(ROOT / f"data/predictions/{VID}.png"))

    cell_lab = np.zeros(vessel.shape, np.int32)               # 0 background, 1..6 cells
    for i, c in enumerate(CELLS):
        cell_lab[cells[c]] = i + 1
    L = smooth_labels(cell_lab, 7)
    hull = np.isin(L, [1, 2, 3]); sup = np.isin(L, [4, 5, 6])
    third = np.where(L == 0, 0, (L - 1) % 3 + 1)              # 1 fwd, 2 mid, 3 aft

    fig, axs = plt.subplots(2, 2, figsize=(4.0, 4.0), gridspec_kw=dict(wspace=0.03, hspace=0.03))
    axs = axs.ravel()

    # (a) original
    axs[0].imshow(img)

    # (b) hull / superstructure
    hs = np.where(hull, 1, np.where(sup, 2, 0))
    axs[1].imshow(overlay(img, hs, {1: C_HULL, 2: C_SUP}, 0.55))
    axs[1].legend(handles=[Patch(color=C_HULL, label="hull"), Patch(color=C_SUP, label="superstructure")],
                  loc="lower center", ncol=2, **LEG)

    # (c) fwd / mid / aft thirds
    axs[2].imshow(overlay(img, third, C_THIRD, 0.55))
    axs[2].legend(handles=[Patch(color=C_THIRD[k], label=n) for k, n in ((1, "fwd"), (2, "mid"), (3, "aft"))],
                  loc="lower center", ncol=3, **LEG)

    # (d) six cells
    d_img = overlay(img, L, {i + 1: C_CELL[c] for i, c in enumerate(CELLS)}, 0.55)
    d_img[borders(L, list(range(1, 7)))] = (255, 255, 255)
    axs[3].imshow(d_img)
    axs[3].legend(handles=[Patch(color=C_CELL[(v, l)], label=f"{v}-{l}")
                           for l in ("fwd", "mid", "aft") for v in ("hull", "sup")],
                  loc="lower center", ncol=3, **LEG)

    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4); s.set_color("#999999")
    fig.savefig(OUT, dpi=350, bbox_inches="tight", pad_inches=0.01)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
