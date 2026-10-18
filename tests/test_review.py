"""Review GUI backend: accept/reject/skip/back and resumable persistence."""

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

# Load review_server as a module (it lives under scripts/, not the package).
_spec = importlib.util.spec_from_file_location(
    "review_server",
    Path(__file__).resolve().parents[1] / "scripts" / "review_server.py",
)
review_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(review_server)
Reviewer = review_server.Reviewer


def test_accept_saves_label_and_records_decision(tmp_path, vessel_a, vessel_b):
    records = [vessel_a, vessel_b]
    acc = tmp_path / "accepted"
    r = Reviewer(records, str(acc), "geometric", None)

    r.decide(0, "accept")
    assert len(r.accepted) == 1
    label_path = Path(r.accepted[0]["label_path"])
    assert label_path.exists()
    region = np.asarray(Image.open(label_path))
    assert set(np.unique(region)).issubset({0, 1, 2})

    r.decide(1, "reject")
    assert r.state()["rejected"] == 1
    assert r.state()["accepted"] == 1


def test_decisions_persist_and_resume(tmp_path, vessel_a, vessel_b):
    records = [vessel_a, vessel_b]
    acc = tmp_path / "accepted"
    r1 = Reviewer(records, str(acc), "geometric", None)
    r1.decide(0, "accept")

    # A fresh Reviewer on the same dir resumes at the first undecided image.
    r2 = Reviewer(records, str(acc), "geometric", None)
    assert r2.i == 1
    assert len(r2.accepted) == 1


def test_deck_nudge_changes_superstructure_area(tmp_path, vessel_a, vessel_b):
    r = Reviewer([vessel_a, vessel_b], str(tmp_path / "acc"), "geometric", None)
    base = int((np.asarray(r.region_map(0)) == 2).sum())
    r.decide(0, "down")  # lower the deck -> more superstructure
    lower = int((np.asarray(r.region_map(0)) == 2).sum())
    r.decide(0, "up"); r.decide(0, "up")  # net one up from base -> less
    higher = int((np.asarray(r.region_map(0)) == 2).sum())
    assert lower > base > higher
    # nudging does not advance the image
    assert r.decide(0, "down") == 0


def test_deck_rotation_changes_split_and_stays(tmp_path, vessel_a, vessel_b):
    r = Reviewer([vessel_a, vessel_b], str(tmp_path / "acc"), "geometric", None)
    base = np.asarray(r.region_map(0)).copy()
    assert r.decide(0, "rotate_cw") == 0        # rotating does not advance
    rotated = np.asarray(r.region_map(0))
    assert not np.array_equal(base, rotated)     # the split actually tilts
    # rotating back to zero restores the original split
    r.decide(0, "rotate_ccw")
    assert np.array_equal(base, np.asarray(r.region_map(0)))


class _FakeSAM:
    """Stub SAM: returns the box interior, or the top third for a point."""
    def set_image(self, pil, key=None):
        self.shape = (pil.size[1], pil.size[0])
    def predict(self, points=None, labels=None, box=None):
        m = np.zeros(self.shape, dtype=bool)
        if box is not None:
            x0, y0, x1, y1 = [int(v) for v in box]
            m[y0:y1, x0:x1] = True
        else:
            m[: self.shape[0] // 3, :] = True
        return m


def test_sam_mode_labels_from_clicks(tmp_path, vessel_a):
    r = Reviewer([vessel_a], str(tmp_path / "acc"), "geometric", None,
                 label_mode="sam", sam=_FakeSAM())
    # no clicks yet -> falls back to a geometric hint (still a valid map)
    assert set(np.unique(np.asarray(r.region_map(0)))).issubset({0, 1, 2})
    r.add_point(0, 0.5, 0.1, 1)               # click near the top
    rm = np.asarray(r.region_map(0))
    assert (rm == 2).sum() > 0                  # superstructure comes from SAM
    r.decide(0, "accept")
    assert len(r.accepted) == 1
    r.decide(0, "clearpoints")
    assert r.points[r.rec_id(0)] == []


def test_advance_skips_already_decided(tmp_path):
    from tests.conftest import make_vessel
    vessels = [make_vessel(tmp_path, f"v{i}") for i in range(3)]
    r = Reviewer(vessels, str(tmp_path / "acc"), "geometric", None)
    r.decisions[r.rec_id(1)] = "reject"     # middle already decided
    assert r.decide(0, "accept") == 2        # skips the decided middle
    # skip also lands on an undecided image, never a decided one
    r2 = Reviewer(vessels, str(tmp_path / "acc2"), "geometric", None)
    r2.decisions[r2.rec_id(1)] = "accept"
    assert r2.decide(0, "skip") == 2


def test_sam_box_adds_then_subtracts(tmp_path, vessel_a):
    r = Reviewer([vessel_a], str(tmp_path / "acc"), "geometric", None,
                 label_mode="sam", sam=_FakeSAM())
    r.add_box(0, 0.4, 0.0, 0.65, 0.55, 1)          # box -> superstructure
    added = int((np.asarray(r.region_map(0)) == 2).sum())
    assert added > 0
    r.add_box(0, 0.4, 0.0, 0.65, 0.55, 0)          # same box -> hull (subtract)
    assert int((np.asarray(r.region_map(0)) == 2).sum()) < added


def test_back_and_skip(tmp_path, vessel_a, vessel_b):
    r = Reviewer([vessel_a, vessel_b], str(tmp_path / "acc"), "geometric", None)
    assert r.decide(0, "skip") == 1        # advance, no decision saved
    assert r.decide(1, "back") == 0        # go back
    assert "accept" not in r.decisions.values()
