"""Validate the causal probe end-to-end with synthetic classifiers whose class
signal lives, by construction, in a *known* component. The probe must report
'superstructure defines class' when the classifier keys on the superstructure,
and must NOT report it when the classifier keys on the hull.
"""

import numpy as np

from hullprobe.probe import run_pair, summarize
from tests.conftest import make_vessel

BG = np.array([30, 90, 140], dtype=np.float32)

CLASSES = [
    {"hull": (180, 120, 90), "super": (240, 240, 240)},
    {"hull": (90, 150, 90), "super": (250, 210, 120)},
    {"hull": (140, 90, 160), "super": (120, 200, 240)},
]
SUPER_SIG = np.array([c["super"] for c in CLASSES], dtype=np.float32)
HULL_SIG = np.array([c["hull"] for c in CLASSES], dtype=np.float32)


def _region_color(image, y_slice):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)[y_slice]
    fg = np.linalg.norm(arr - BG, axis=-1) > 30.0
    if not fg.any():
        return BG.copy()
    return arr[fg].mean(axis=0)


def _classifier(signatures, y_slice):
    def classify(image):
        color = _region_color(image, y_slice)
        return -np.linalg.norm(signatures - color, axis=1)  # logits
    return classify


def _build_vessels(tmp_path):
    vessels = []
    for i, c in enumerate(CLASSES):
        vessels.append(make_vessel(tmp_path, f"c{i}", hue=c["hull"], super_hue=c["super"]))
    return vessels


def _cross_pairs(classify, vessels):
    pairs = []
    for r in range(len(vessels)):
        for d in range(len(vessels)):
            if r == d:
                continue
            pairs.append(run_pair(classify, vessels[r], vessels[d], r, d))
    return pairs


def test_probe_detects_superstructure_classifier(tmp_path):
    vessels = _build_vessels(tmp_path)
    classify = _classifier(SUPER_SIG, slice(0, 60))  # keys on superstructure band
    pairs = _cross_pairs(classify, vessels)
    result = summarize(pairs, n_boot=2000)
    assert result["verdict"]["superstructure_defines_class"] is True
    assert result["cross_class"]["delta_donor"]["lo"] > 0
    assert result["cross_class"]["super_follows_donor_rate"] > 0.8
    assert result["cross_class"]["hull_keeps_recip_rate"] > 0.8


def test_probe_rejects_hull_classifier(tmp_path):
    vessels = _build_vessels(tmp_path)
    classify = _classifier(HULL_SIG, slice(60, 130))  # keys on hull band
    pairs = _cross_pairs(classify, vessels)
    result = summarize(pairs, n_boot=2000)
    # A hull-driven classifier must NOT look like 'superstructure defines class'.
    assert result["verdict"]["superstructure_defines_class"] is False
    assert result["cross_class"]["delta_donor"]["mean"] < 0
