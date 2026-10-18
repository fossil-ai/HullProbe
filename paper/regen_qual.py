"""Regenerate fig_qualitative with the FIXED exact-stencil operator (full hull swap).
Silhouette is the recipient's by construction -> report IoU as a check."""
import json, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from hullprobe.lattice import cell_masks, CELLS
from hullprobe.swap import _background_fill
from hullprobe.segmenter import load_region_png
from scripts.part_lattice_spectrum import rmap, warp_donor_cells, assemble_all, test_pool

HULL_Z = 0b000111  # bits 0,1,2 = hull-fwd/mid/aft  (full hull swap)
by_c = test_pool()
all_v = [r for lst in by_c.values() for r in lst]
rng = random.Random(5)
CLASSES = json.loads(Path("data/classes_B.json").read_text())
rows, ious = [], []
for c in [CLASSES.index(n) for n in ["Tankers", "Bulk Carrier", "Container Ship"] if n in CLASSES]:
    for r in rng.sample(by_c[c], min(8, len(by_c[c]))):
        try:
            recip_rgb = np.asarray(Image.open(r["image_path"]).convert("RGB"))
            rc, vessel = cell_masks(r, region_map=rmap(r))
            bg = _background_fill(recip_rgb, vessel)
            dcross = rng.choice([d for d in all_v if d["label"] != c])
            dsame = rng.choice([d for d in by_c[c] if d["image_path"] != r["image_path"]])
            comps = {}
            for tag, d in [("cross", dcross), ("same", dsame)]:
                drgb_w, daw = warp_donor_cells(d, r)
                comps[tag] = assemble_all(recip_rgb, rc, bg, drgb_w, daw)[HULL_Z]
            clean = recip_rgb
            rows.append((Image.fromarray(clean), Image.fromarray(comps["cross"]),
                         Image.fromarray(comps["same"]), CLASSES[c], CLASSES[dcross["label"]]))
            break
        except Exception:
            continue
# montage
TS = 200
pad = 6
n = len(rows)
W = 3 * TS + 4 * pad
H = n * (TS + 22) + pad
canvas = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(canvas)
for i, (a, b, cc, ry, dy) in enumerate(rows):
    y = pad + i * (TS + 22)
    for j, im in enumerate([a, b, cc]):
        canvas.paste(im.resize((TS, TS)), (pad + j * (TS + pad), y + 16))
    d.text((pad, y + 2), f"recipient: {ry}", fill="black")
    d.text((pad + TS + pad, y + 2), f"hull <- {dy} (cross)", fill="#a00000")
    d.text((pad + 2 * (TS + pad), y + 2), "hull <- same class (control)", fill="#004000")
canvas.save("paper/figures/fig_qualitative.jpg", quality=92)
print(f"wrote fig_qualitative.jpg ({n} rows), exact-stencil -> silhouette = recipient by construction")
