"""Shared constants for hullprobe.

Canonical-frame sizes are identical to the reference implementation so a vessel registers into the
same frame consistently. Data paths point, by default, at the shared MARVEL
data used by the reference implementation; override HULLPROBE_DATA_ROOT to relocate.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Default to the shared MARVEL data root used by the reference implementation; override with env var.
DATA_ROOT = Path(
    os.environ.get("HULLPROBE_DATA_ROOT", REPO_ROOT / "data")
)

POSE_MANIFEST = DATA_ROOT / "pose_maskpca.json"
REGION_ROOT = DATA_ROOT / "hull_superstructure"

EXPERIMENTS_DIR = REPO_ROOT / "experiments"
VIZ_DIR = REPO_ROOT / "viz"
CACHE_DIR = REPO_ROOT / ".cache"

# Classifier input frame (unchanged from the reference implementation).
IMG_SIZE = 224

# Canonical composite frame the aligner operates in (W x H) — identical to
# the reference implementation ALIGN_IMG_W / ALIGN_IMG_H. Do not change: aligner parity depends on it.
ALIGN_IMG_W, ALIGN_IMG_H = 256, 128

# The two transverse components the hull/superstructure swap operates on.
COMPONENT_NAMES = ("hull", "superstructure")

SEED = 42
