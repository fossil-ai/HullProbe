"""Hull/superstructure semantic segmenter (3 classes: bg / hull / superstructure).

Bootstraps from the weak geometric deck labels, then fine-tunes on the
human-accepted set from the review GUI (active learning). Being appearance-based
and trained over the whole dataset, it learns a *curved, per-column* deck that
the straight-line geometric heuristic cannot express.

The predicted region map plugs straight into `canonical.hull_superstructure_masks`
and `swap.compose_swap` via their `region_map=` argument.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .canonical import HULL, SUPERSTRUCTURE, full_vessel_alpha, region_map_geometric

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 3


def _norm(image_rgb, size):
    im = image_rgb.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr).permute(2, 0, 1).float()


# --------------------------------------------------------------------------- #
# Model: a small UNet.
# --------------------------------------------------------------------------- #
class _Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, width=32):
        super().__init__()
        w = width
        self.d1 = _Block(3, w)
        self.d2 = _Block(w, w * 2)
        self.d3 = _Block(w * 2, w * 4)
        self.d4 = _Block(w * 4, w * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(w * 8, w * 4, 2, stride=2)
        self.u3 = _Block(w * 8, w * 4)
        self.up2 = nn.ConvTranspose2d(w * 4, w * 2, 2, stride=2)
        self.u2 = _Block(w * 4, w * 2)
        self.up1 = nn.ConvTranspose2d(w * 2, w, 2, stride=2)
        self.u1 = _Block(w * 2, w)
        self.head = nn.Conv2d(w, num_classes, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        x = self.u3(torch.cat([self.up3(c4), c3], 1))
        x = self.u2(torch.cat([self.up2(x), c2], 1))
        x = self.u1(torch.cat([self.up1(x), c1], 1))
        return self.head(x)


# --------------------------------------------------------------------------- #
# Datasets.
# --------------------------------------------------------------------------- #
class WeakRegionDataset(torch.utils.data.Dataset):
    """(image, geometric region map) — the bootstrap training signal."""

    def __init__(self, records, size=256):
        self.records = records
        self.size = size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        image = Image.open(rec["image_path"])
        label = region_map_geometric(rec)
        label = np.array(
            Image.fromarray(label).resize((self.size, self.size), Image.NEAREST)
        )
        return _norm(image, self.size), torch.from_numpy(label).long()


class GoldRegionDataset(torch.utils.data.Dataset):
    """(image, accepted region-map png) — the human-verified training signal.
    With hflip=True, randomly mirrors image+label (a mirrored ship is still a
    valid ship), which effectively doubles the data."""

    def __init__(self, entries, size=256, hflip=False):
        self.entries = entries  # [{"image_path":..., "label_path":...}]
        self.size = size
        self.hflip = hflip

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        import random

        entry = self.entries[i]
        image = Image.open(entry["image_path"]).convert("RGB")
        label_img = Image.open(entry["label_path"]).resize((self.size, self.size), Image.NEAREST)
        if self.hflip and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label_img = label_img.transpose(Image.FLIP_LEFT_RIGHT)
        label = np.array(label_img)
        return _norm(image, self.size), torch.from_numpy(label).long()


# --------------------------------------------------------------------------- #
# Prediction.
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def predict_region_map(model, rec, device="cpu", size=256):
    """Predict a full-resolution {bg, hull, superstructure} map for a record,
    restricted to the vessel alpha (partition preserved: inside the vessel every
    pixel is hull or superstructure, never background)."""
    model.eval()
    image = Image.open(rec["image_path"]).convert("RGB")
    width, height = image.size
    x = _norm(image, size).unsqueeze(0).to(device)
    logits = model(x)[0]  # (3, size, size)
    up = F.interpolate(logits.unsqueeze(0), size=(height, width),
                       mode="bilinear", align_corners=False)[0]
    full_alpha, _w, _h = full_vessel_alpha(rec)
    vessel = full_alpha > 12
    # Inside the vessel, decide hull vs superstructure only (ignore bg channel).
    hull_vs_super = torch.stack([up[HULL], up[SUPERSTRUCTURE]], 0).argmax(0).cpu().numpy()
    region = np.zeros((height, width), dtype=np.uint8)
    region[vessel] = np.where(hull_vs_super[vessel] == 1, SUPERSTRUCTURE, HULL)
    return region


def save_region_png(region, path):
    Image.fromarray(region.astype(np.uint8), "L").save(path)


def load_region_png(path):
    return np.asarray(Image.open(path))
