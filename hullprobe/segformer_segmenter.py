"""Hull/superstructure segmenter built on SegFormer pretrained on ADE20K.

ADE20K already contains ship / boat / water / sea / sky, so the encoder parses
maritime scenes out of the box — we only fine-tune it to split the ship into
hull vs. superstructure (3 classes: bg / hull / superstructure). That maritime
prior is why this generalizes across cruise tiers, masts and container stacks
from a few hundred masks, where a from-scratch UNet overfit to bulk carriers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .canonical import HULL, SUPERSTRUCTURE, full_vessel_alpha
from .segmenter import _norm  # ImageNet normalization (SegFormer uses the same)

DEFAULT_CKPT = "nvidia/segformer-b2-finetuned-ade-512-512"
NUM_CLASSES = 3


def build_model(checkpoint=DEFAULT_CKPT, num_labels=NUM_CLASSES, device="cpu"):
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(
        checkpoint, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    return model.to(device)


def logits_at(model, pixel_values, size):
    """SegFormer logits (B,C,H/4,W/4) upsampled to `size` (H,W)."""
    logits = model(pixel_values=pixel_values).logits
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


@torch.inference_mode()
def predict_region_map(model, rec, device="cpu", size=512):
    """Full-resolution {bg, hull, superstructure} map for a record, restricted to
    the vessel alpha (inside the vessel every pixel is hull or superstructure)."""
    model.eval()
    image = Image.open(rec["image_path"]).convert("RGB")
    width, height = image.size
    x = _norm(image, size).unsqueeze(0).to(device)
    logits = logits_at(model, x, (height, width))[0]  # (3, H, W)
    full_alpha, _w, _h = full_vessel_alpha(rec)
    vessel = full_alpha > 12
    hull_vs_super = torch.stack([logits[HULL], logits[SUPERSTRUCTURE]], 0).argmax(0).cpu().numpy()
    region = np.zeros((height, width), dtype=np.uint8)
    region[vessel] = np.where(hull_vs_super[vessel] == 1, SUPERSTRUCTURE, HULL)
    return region
