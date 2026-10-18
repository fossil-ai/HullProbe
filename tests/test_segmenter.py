"""Segmenter model, datasets, prediction, and region-map integration."""

import numpy as np
import torch

from hullprobe.canonical import HULL, SUPERSTRUCTURE, hull_superstructure_masks, region_map_geometric
from hullprobe.segmenter import (
    TinyUNet,
    WeakRegionDataset,
    predict_region_map,
)
from hullprobe.swap import superstructure_swap


def test_tinyunet_forward_shape():
    model = TinyUNet()
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 3, 64, 64)


def test_weak_dataset_yields_valid_labels(vessel_a):
    ds = WeakRegionDataset([vessel_a], size=128)
    image, label = ds[0]
    assert image.shape == (3, 128, 128)
    assert label.shape == (128, 128)
    assert set(np.unique(label.numpy())).issubset({0, 1, 2})


def test_predict_region_map_partitions_vessel(vessel_a):
    model = TinyUNet()
    region = predict_region_map(model, vessel_a, device="cpu", size=128)
    from hullprobe.canonical import full_vessel_alpha

    full_alpha, _w, _h = full_vessel_alpha(vessel_a)
    vessel = full_alpha > 12
    # Every vessel pixel is hull or superstructure; no vessel pixel is background.
    assert set(np.unique(region[vessel])).issubset({HULL, SUPERSTRUCTURE})
    assert (region[~vessel] == 0).all()


def test_region_map_threads_into_masks_and_swap(vessel_a, vessel_b):
    rm_a = region_map_geometric(vessel_a)
    rm_b = region_map_geometric(vessel_b)
    hull, sup = hull_superstructure_masks(vessel_a, region_map=rm_a)
    assert not np.any(hull & sup)
    assert hull.sum() > 0 and sup.sum() > 0
    # Swap consumes region maps directly.
    out, info = superstructure_swap(
        vessel_a, vessel_b, region_map_recip=rm_a, region_map_donor=rm_b
    )
    assert out.size[0] > 0 and info["donor_paint_area"] >= 0
