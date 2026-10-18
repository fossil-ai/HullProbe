"""Class-conditional hull-share (reviewer point 5): a global 39% may conceal classes for
which hull evidence is legitimate (identity-bearing hull) vs. suspicious. We compute the
artifact-corrected hull-share separately per recipient class."""
from __future__ import annotations
import json, random
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

from hullprobe.lattice import cell_masks
from hullprobe.swap import _background_fill
from scripts.part_lattice_spectrum import (
    test_pool, rmap, warp_donor_cells, assemble_all, classify, load_model, compatible_cross)
from scripts.part_lattice_debias_audit import spectrum_from_A


def main(per_class=6, K=4):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if device == "cuda" and torch.cuda.mem_get_info()[0] < 2.5e9:
            device = "cpu"
    except Exception:
        pass
    model = load_model(device)
    by_c = test_pool(); all_v = [r for lst in by_c.values() for r in lst]
    CLS = json.loads(Path("data/classes_B.json").read_text())
    rng = random.Random(42)
    cross_by, same_by = defaultdict(list), defaultdict(list)   # class -> list of 6-vec influence
    for c, lst in by_c.items():
        lst = list(lst); rng.shuffle(lst)
        for r in lst[:per_class]:
            try:
                y = r["label"]
                recip = np.asarray(Image.open(r["image_path"]).convert("RGB"))
                rc, vessel = cell_masks(r, region_map=rmap(r))
                bg = _background_fill(recip, vessel)
                cross = compatible_cross(r, all_v, K, rng)
                sp = sorted([d for d in by_c[y] if d["image_path"] != r["image_path"]],
                            key=lambda d: 0)[:K] or None
                same = rng.sample([d for d in by_c[y] if d["image_path"] != r["image_path"]],
                                  min(K, len(by_c[y]) - 1)) if len(by_c[y]) > 1 else None
                if not same:
                    continue

                def A_of(dz):
                    A = np.zeros((len(dz), 64), np.int8)
                    for di, d in enumerate(dz):
                        drgb, dval = warp_donor_cells(d, r)
                        A[di] = (classify(model, assemble_all(recip, rc, bg, drgb, dval), device) == y)
                    return A
                Ax = A_of(cross)
                if Ax[:, 0].mean() < 0.5:
                    continue
                cross_by[y].append(spectrum_from_A(Ax)[0])
            except Exception:
                continue
    rows = []
    for c in sorted(cross_by):
        if len(cross_by[c]) < 3:
            continue
        cr = np.array(cross_by[c]).mean(0)               # pooled raw cross influence (stable)
        hs = float(cr[:3].sum() / cr.sum()) if cr.sum() > 1e-6 else float("nan")
        rows.append((CLS[c], hs, len(cross_by[c])))
    rows.sort(key=lambda t: -t[1])
    print("\n CLASS-CONDITIONAL HULL-SHARE")
    for name, hs, n in rows:
        print(f"  {name:28s} {hs:5.2f}  (n={n})")
    hsv = np.array([r[1] for r in rows])
    print(f"\nrange {np.nanmin(hsv):.2f}-{np.nanmax(hsv):.2f}, mean {np.nanmean(hsv):.2f}, "
          f"std {np.nanstd(hsv):.2f} over {len(rows)} classes")
    Path("experiments/perclass_hullshare.json").write_text(json.dumps(
        {"rows": [{"class": n, "hull_share": hs, "n": k} for n, hs, k in rows]}, indent=1))
    print("wrote experiments/perclass_hullshare.json")


if __name__ == "__main__":
    main()
