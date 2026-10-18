"""Canonical hull frame: the hull <-> superstructure (below/above-deck) split.

The vessel is registered with the *same* PCA pose as the reference implementation (see `pose.py`).
The longitudinal axis gives bow/center/stern; its perpendicular — the transverse
axis — gives the deck-line split this repo is built around:

    T(x, y) = -(x - cx) sin(theta) + (y - cy) cos(theta)     (pose.transverse_coord)

    above-deck (superstructure):  T <= deck
    below-deck (hull):            T >  deck

The deck threshold is the one quantity a full-vessel mask cannot give on its own
— it needs the above-deck silhouette. Two ways to obtain it here:

  * `deck=None`  -> centroid split (T=0). A geometry-only placeholder; wrong for
    real vessels (superstructure is a minority of the height), useful only for
    plumbing/tests.
  * `estimate_deck_from_superstructure(...)` -> a data-driven threshold from a
    superstructure-prompted SAM mask (the real path; see build_pose_manifest.py).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .pose import longitudinal_coord, transverse_coord


def full_vessel_alpha(rec):
    """Reconstruct the saved full-vessel alpha in original image coordinates.

    Mirrors the reference regions.complete_region_masks' alpha reconstruction so the
    coordinate frame matches exactly.
    """
    with Image.open(rec["image_path"]) as image:
        width, height = image.size
    full_box = [int(value) for value in rec["full_vessel_box_xyxy"]]
    full_crop = Image.open(rec["full_vessel_path"]).convert("RGBA").split()[3]
    canvas = Image.new("L", (width, height), 0)
    canvas.paste(full_crop, (full_box[0], full_box[1]))
    return np.asarray(canvas, dtype=np.uint8), width, height


def _grids(width, height):
    ys, xs = np.indices((height, width), dtype=np.float32)
    return xs, ys


def estimate_deck_from_superstructure(rec, superstructure_mask):
    """Transverse threshold separating above-deck from hull.

    `superstructure_mask` is a boolean array (H, W) of above-deck pixels, e.g.
    from a GroundedSAM pass prompted for deckhouse/bridge/crane/mast/containers.
    The deck line is placed at the lowest (largest-T) superstructure pixel: the
    point where the above-deck structure meets the main deck.
    """
    ys, xs = np.where(superstructure_mask)
    if len(xs) == 0:
        raise ValueError("empty superstructure mask; cannot estimate deck line")
    t = transverse_coord(
        xs.astype(np.float32), ys.astype(np.float32),
        rec["pca_angle_deg"], rec["pca_centroid_x"], rec["pca_centroid_y"],
    )
    return float(np.percentile(t, 95.0))


def estimate_deck_geometric(rec, coverage_frac=0.90, bins=48, smooth=3):
    """Deck line from silhouette geometry alone — no part segmentation.

    The hull is the full-length band near the waterline; the superstructure is
    the narrower structure above it. Scanning the transverse coverage profile
    from the top (sky) downward, the deck line is the first transverse level
    where longitudinal coverage reaches `coverage_frac` of its maximum — i.e.
    where the silhouette becomes "full length" and the hull begins.

    Robust for vessels whose above-deck structure is longitudinally localized
    (tanker, bulk, cargo, tug). Vessels whose superstructure spans the whole
    length (cruise/passenger, fully-stacked container) will place the deck high;
    that is a known limit of a silhouette-only estimate.
    """
    full_alpha, width, height = full_vessel_alpha(rec)
    ys, xs = np.where(full_alpha > 12)
    if len(xs) < 60:
        raise ValueError("vessel alpha too small for deck estimation")
    t = transverse_coord(
        xs.astype(np.float64), ys.astype(np.float64),
        rec["pca_angle_deg"], rec["pca_centroid_x"], rec["pca_centroid_y"],
    )
    edges = np.linspace(t.min(), t.max(), bins + 1)
    counts, _ = np.histogram(t, bins=edges)  # ~longitudinal coverage per T level
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        counts = np.convolve(counts, kernel, mode="same")
    threshold = coverage_frac * counts.max()
    # Scan top (low T = above deck) -> bottom; first "full" level is the deck.
    hit = np.argmax(counts >= threshold)
    return float(edges[hit])


def deck_split_masks(rec, deck=None):
    """Return {"hull": L-image, "superstructure": L-image} partitioning the full
    vessel alpha at transverse threshold `deck` (T=0 / centroid if None).

    hull | superstructure exactly covers the vessel alpha, by construction.
    """
    full_alpha, width, height = full_vessel_alpha(rec)
    xs, ys = _grids(width, height)
    t = transverse_coord(
        xs, ys, rec["pca_angle_deg"], rec["pca_centroid_x"], rec["pca_centroid_y"]
    )
    threshold = 0.0 if deck is None else float(deck)
    above = t <= threshold
    masks = {
        "superstructure": np.where(above, full_alpha, 0).astype(np.uint8),
        "hull": np.where(~above, full_alpha, 0).astype(np.uint8),
    }
    return {name: Image.fromarray(mask, "L") for name, mask in masks.items()}


def longitudinal_projection_image(rec):
    """Per-pixel longitudinal coordinate over the image, for diagnostics/overlays.

    Identical formula to the reference regions.complete_region_masks; provided so the
    shared frame can be visualized alongside the transverse deck split.
    """
    _full_alpha, width, height = full_vessel_alpha(rec)
    xs, ys = _grids(width, height)
    return longitudinal_coord(
        xs, ys, rec["pca_angle_deg"], rec["pca_centroid_x"], rec["pca_centroid_y"]
    )


# Region-map class ids (background / hull / superstructure) for the segmenter.
BACKGROUND, HULL, SUPERSTRUCTURE = 0, 1, 2


def hull_superstructure_masks(rec, deck=None, superstructure_mask=None, region_map=None,
                              angle=None):
    """Boolean (hull, superstructure) masks that exactly partition the vessel.

    Priority of the deck source:
      1. `region_map` — a trained segmenter's per-pixel {bg, hull, super} map
         (curved, appearance-aware). Used directly: super = vessel & (map==SUPER).
      2. `superstructure_mask` — a GroundedSAM above-deck mask; used only to
         *locate* a straight deck line via `estimate_deck_from_superstructure`.
      3. `deck` — an explicit straight transverse threshold (e.g. from
         `estimate_deck_geometric`).
      4. none — centroid split (T=0), plumbing only.

    In all cases hull | superstructure == the vessel alpha, disjoint.
    """
    full_alpha, width, height = full_vessel_alpha(rec)
    vessel = full_alpha > 12

    if region_map is not None:
        rm = np.asarray(region_map)
        if rm.shape != vessel.shape:
            rm = np.asarray(
                Image.fromarray(rm.astype(np.uint8)).resize(
                    (width, height), Image.Resampling.NEAREST
                )
            )
        above = rm == SUPERSTRUCTURE
    else:
        if superstructure_mask is not None and deck is None:
            deck = estimate_deck_from_superstructure(rec, superstructure_mask)
        ang = rec["pca_angle_deg"] if angle is None else float(angle)
        xs, ys = _grids(width, height)
        t = transverse_coord(
            xs, ys, ang, rec["pca_centroid_x"], rec["pca_centroid_y"]
        )
        threshold = 0.0 if deck is None else float(deck)
        above = t <= threshold

    superstructure = vessel & above
    hull = vessel & ~above
    # Exact-partition invariant: hull | superstructure == vessel, disjoint.
    assert not np.any(hull & superstructure)
    assert np.array_equal(hull | superstructure, vessel)
    return hull, superstructure


def region_map_geometric(rec, coverage_frac=0.75):
    """HxW uint8 map {0=bg, 1=hull, 2=superstructure} from the geometric deck —
    the weak label the segmenter bootstraps from and the GUI reviews."""
    deck = estimate_deck_geometric(rec, coverage_frac=coverage_frac)
    hull, superstructure = hull_superstructure_masks(rec, deck=deck)
    region = np.zeros(hull.shape, dtype=np.uint8)
    region[hull] = HULL
    region[superstructure] = SUPERSTRUCTURE
    return region


def region_map_from_masks(hull, superstructure):
    region = np.zeros(hull.shape, dtype=np.uint8)
    region[hull] = HULL
    region[superstructure] = SUPERSTRUCTURE
    return region


def deck_split_stats(rec, deck=None, superstructure_mask=None, region_map=None):
    """Area diagnostics for a hull/superstructure split (for the audit)."""
    hull, superstructure = hull_superstructure_masks(
        rec, deck=deck, superstructure_mask=superstructure_mask, region_map=region_map
    )
    vessel_area = int(hull.sum() + superstructure.sum())
    if vessel_area == 0:
        return {"vessel_area": 0, "superstructure_frac": 0.0, "hull_frac": 0.0}
    return {
        "vessel_area": vessel_area,
        "superstructure_area": int(superstructure.sum()),
        "hull_area": int(hull.sum()),
        "superstructure_frac": float(superstructure.sum() / vessel_area),
        "hull_frac": float(hull.sum() / vessel_area),
    }


def overlay_split(rec, deck=None, superstructure_mask=None, region_map=None, alpha=0.45,
                  tint_hull=True, tint_super=True):
    """RGB visualization: superstructure tinted red, hull tinted blue, over the
    original image. Set tint_hull=False (SAM mode) to show only the painted
    superstructure over a clean image — a truly blank reset state."""
    hull, superstructure = hull_superstructure_masks(
        rec, deck=deck, superstructure_mask=superstructure_mask, region_map=region_map
    )
    base = np.asarray(Image.open(rec["image_path"]).convert("RGB"), dtype=np.float32)
    tint = np.zeros_like(base)
    paint = np.zeros(hull.shape, dtype=bool)
    if tint_super:
        tint[superstructure] = np.array([255.0, 40.0, 40.0])
        paint |= superstructure
    if tint_hull:
        tint[hull] = np.array([40.0, 120.0, 255.0])
        paint |= hull
    painted = np.where(paint[..., None], (1.0 - alpha) * base + alpha * tint, base)
    return Image.fromarray(np.clip(painted, 0, 255).astype(np.uint8), "RGB")
