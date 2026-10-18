"""Hull/superstructure CONFLICT experiment (paper Sec. 5.3: is hull sensitivity also hull
sufficiency?). We create a conflict by swapping in a cross-class donor hull: the hull now
says the donor class, while the superstructure (true identity) says the recipient class.
We ask which the classifier follows.

  super-follow  : pred == recipient class  (correct; ignores the misleading hull)
  hull-follow   : pred == donor class       (shortcut-driven ERROR caused by the hull)
  other         : pred == some third class

If the hull-share measure predicts the per-class hull-follow error rate, then it identifies
exactly when hull reliance is harmful -- turning "reliance" into "predicted error under
component conflict." Writes experiments/conflict.json.
"""
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

HULL_Z = 0b000111   # swap all three hull cells -> donor hull, recipient superstructure


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
    per_class_cnt = defaultdict(lambda: [0, 0, 0, 0])   # class -> [super, hull, other, total]
    tot = [0, 0, 0, 0]
    for c, lst in by_c.items():
        lst = list(lst); rng.shuffle(lst)
        for r in lst[:per_class]:
            try:
                y = r["label"]
                recip = np.asarray(Image.open(r["image_path"]).convert("RGB"))
                rc, vessel = cell_masks(r, region_map=rmap(r))
                # require the model to get the clean recipient right
                if classify(model, [recip], device)[0] != y:
                    continue
                bg = _background_fill(recip, vessel)
                for d in compatible_cross(r, all_v, K, rng):
                    dy = d["label"]
                    if dy == y:
                        continue
                    drgb, dval = warp_donor_cells(d, r)
                    pred = int(classify(model, [assemble_all(recip, rc, bg, drgb, dval)[HULL_Z]], device)[0])
                    k = 0 if pred == y else (1 if pred == dy else 2)
                    per_class_cnt[y][k] += 1; per_class_cnt[y][3] += 1
                    tot[k] += 1; tot[3] += 1
            except Exception:
                continue

    def rates(v):
        t = max(1, v[3]); return v[0] / t, v[1] / t, v[2] / t, v[3]
    gs, gh, go, gn = rates(tot)
    print("\n HULL/SUPERSTRUCTURE CONFLICT (cross-class hull swap)")
    print(f"GLOBAL (n={gn}):  super-follow {gs:.3f}  hull-follow(ERR) {gh:.3f}  other {go:.3f}")

    # per-class hull-follow error rate, correlate with per-class hull-share
    hs = {row["class"]: row["hull_share"]
          for row in json.loads(Path("experiments/perclass_hullshare.json").read_text())["rows"]}
    rows = []
    for c, v in per_class_cnt.items():
        if v[3] >= 6:
            s, h, o, n = rates(v); rows.append((CLS[c], h, s, hs.get(CLS[c]), n))
    rows.sort(key=lambda t: -t[1])
    print("\nclass                         hull-follow  super-follow  hull-share  n")
    for name, h, s, sh, n in rows:
        shs = f"{sh:.2f}" if sh is not None and sh == sh else "  --"
        print(f"  {name:26s}   {h:6.2f}      {s:6.2f}      {shs}     {n}")
    valid = [(h, sh) for _, h, _, sh, _ in rows if sh is not None and sh == sh]
    if len(valid) > 3:
        H = np.array([x[0] for x in valid]); SH = np.array([x[1] for x in valid])
        rr = float(np.corrcoef(H, SH)[0, 1])
        print(f"\ncorr(per-class hull-follow error, per-class hull-share) = {rr:+.3f}")
    Path("experiments/conflict.json").write_text(json.dumps({
        "global": {"super_follow": gs, "hull_follow_err": gh, "other": go, "n": gn},
        "per_class": [{"class": nm, "hull_follow": h, "super_follow": s, "hull_share": sh, "n": n}
                      for nm, h, s, sh, n in rows]}, indent=1))
    print("wrote experiments/conflict.json")


if __name__ == "__main__":
    main()
