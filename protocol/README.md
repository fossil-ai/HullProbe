# Protocol files

A HullProbe score depends on the intervention distribution, so these files fix every choice that
affects the number. With them, the swap cubes are regenerated exactly from the public images.

| File | What it fixes |
|---|---|
| `attribution_pool_n277.json` | The headline pool (paper Secs. 5.1&ndash;5.2): 277 MARVEL-B test recipients with their true labels, 4 cross-class donors (nearest in aspect ratio among 60 seeded candidates) and 2 same-class controls each. Used by `scripts/audit_classifier.py` and `hullprobe.audit`. |
| `gt_mask_heldout_n194.json` | The segmentation-error audit (Sec. 4.3, App. A): the segmenter's held-out annotated vessels that enter the audit, with donors drawn from the same held-out set. |
| `segformer_split.json` | The hull/superstructure segmenter's 812/202 train/validation split of `data/annotations/`. |

Vessel ids are MARVEL image ids (file stems); labels index `data/classes_B.json`.
