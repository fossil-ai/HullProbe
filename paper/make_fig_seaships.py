"""fig_seaships.jpg -- qualitative transfer to SeaShips (second dataset).

Shows that the MARVEL-trained SegFormer segmenter and the swap operator transfer ZERO-SHOT
to SeaShips: 4 vessels (distinct types), each as Original / 6-cell decomposition /
cross-class hull swap. Supports the numerical second-dataset replication (Sec. 5.3).
Run from repo root: PYTHONPATH=. <venv-py> paper/make_fig_seaships.py
"""
import json, random
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scripts.part_lattice_spectrum as pls
pls.PRED = Path("data/seaships/predictions")
from hullprobe.lattice import cell_masks, CELLS
from hullprobe.swap import _background_fill

CLASSES = json.loads(Path("data/seaships/classes.json").read_text())
recs = json.loads(Path("data/seaships/pose.json").read_text())["records"]
by_stem = {Path(r["image_path"]).stem: r for r in recs}
test = [r for r in recs if r.get("source_split") == "test"]
# six-cell palette shared with Fig. 1 (make_fig_decomp.py), in CELLS order
COLORS = np.array([[0xF2,0xC8,0x0F],[0x8C,0x56,0x4B],[0xE3,0x77,0xC2],
                   [0x17,0xBE,0xCF],[0xBC,0xBD,0x22],[0x8A,0x8A,0x8A]], np.float32)
OUT = Path("paper/figures/fig_seaships.jpg")
SEL = ["000647_0", "006861_0", "000305_0", "004513_0"]   # bulk, fishing, ore, passenger
DONOR_RANK = {"000305_0": 1, "004513_0": 0}   # which aspect-compatible donor (0=closest)

def resize(a, W=210, H=140):
    return np.asarray(Image.fromarray(np.asarray(a).astype(np.uint8)).resize((W, H)))

def panels(stem):
    r = by_stem[stem]
    rc, ves = cell_masks(r, region_map=pls.rmap(r))
    img = np.asarray(Image.open(r["image_path"]).convert("RGB"))
    over = img.astype(np.float32).copy()
    for i, c in enumerate(CELLS):
        over[rc[c]] = 0.45 * img[rc[c]] + 0.55 * COLORS[i]
    bg = _background_fill(img, ves)
    rng = random.Random(1234)
    donors = pls.compatible_cross(r, test, 4, rng)
    donor = donors[min(DONOR_RANK.get(stem, 0), len(donors) - 1)]
    drgb, dval = pls.warp_donor_cells(donor, r)
    swap = pls.assemble_all(img, rc, bg, drgb, dval)[7]     # z=0b000111 full hull swap
    return (resize(img), resize(over), resize(swap),
            CLASSES[r["label"]], CLASSES[donor["label"]])

rows = ["Original", "Anatomy\n(zero-shot)", "Hull swap"]
fig, axes = plt.subplots(3, 4, figsize=(7.0, 3.5))
for j, stem in enumerate(SEL):
    orig, over, swap, cls, dcls = panels(stem)
    for i, im in enumerate([orig, over, swap]):
        ax = axes[i, j]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#cccccc")
    axes[0, j].set_title(cls, fontsize=8)
    axes[2, j].set_xlabel(f"hull $\\leftarrow$ {dcls}", fontsize=6.8)
for i, name in enumerate(rows):
    axes[i, 0].set_ylabel(name, fontsize=8)
fig.subplots_adjust(left=0.075, right=0.995, top=0.92, bottom=0.09, wspace=0.05, hspace=0.12)
fig.savefig(OUT, dpi=220)
plt.close(fig)
print("wrote", OUT, "classes:", [panels(s)[3] for s in SEL])
