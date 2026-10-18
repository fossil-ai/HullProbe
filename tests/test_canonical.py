"""Hull/superstructure decomposition invariants."""

import numpy as np

from hullprobe.canonical import (
    deck_split_stats,
    estimate_deck_geometric,
    hull_superstructure_masks,
    overlay_split,
)


def test_hull_superstructure_exactly_partition_vessel(vessel_a):
    hull, superstructure = hull_superstructure_masks(vessel_a)
    assert not np.any(hull & superstructure)
    # Their union is the vessel alpha (the function asserts this internally too).
    assert hull.sum() > 0 and superstructure.sum() > 0


def test_deck_line_puts_superstructure_above_hull(vessel_a):
    """With the fixture's deck line, the deckhouse is the superstructure and the
    big rectangle is the hull (hull has the larger area)."""
    hull, superstructure = hull_superstructure_masks(vessel_a)
    assert hull.sum() > superstructure.sum()


def test_superstructure_mask_estimates_a_deck_line(vessel_a):
    """Passing an above-deck mask routes through estimate_deck_from_superstructure."""
    _hull, superstructure = hull_superstructure_masks(vessel_a)
    hull2, super2 = hull_superstructure_masks(
        vessel_a, superstructure_mask=superstructure
    )
    # Re-deriving the deck from the true superstructure recovers a similar split.
    assert super2.sum() > 0 and hull2.sum() > 0


def test_geometric_deck_recovers_hull_superstructure_boundary(vessel_a):
    """The silhouette-coverage deck line separates the full-length hull from the
    localized deckhouse: hull ends up larger than superstructure."""
    deck = estimate_deck_geometric(vessel_a)
    hull, superstructure = hull_superstructure_masks(vessel_a, deck=deck)
    assert hull.sum() > superstructure.sum()
    # And the split is non-degenerate (both components present).
    assert superstructure.sum() > 0


def test_stats_and_overlay_run(vessel_a):
    stats = deck_split_stats(vessel_a)
    assert 0.0 < stats["superstructure_frac"] < 1.0
    image = overlay_split(vessel_a)
    assert image.size[0] > 0
