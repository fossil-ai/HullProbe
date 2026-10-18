"""Interactive SAM2 point-prompt segmentation for the review GUI.

Unlike the GroundingDINO text prompt (which returned near-whole-ship masks), a
few point clicks on the superstructure give a pixel-accurate, freeform mask —
masts, tiers, deckhouses and all. That's the unbiased label the segmenter needs.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .segmentation import DEFAULT_SAM2_CHECKPOINT, DEFAULT_SAM2_CONFIG


class InteractiveSAM:
    def __init__(self, checkpoint=DEFAULT_SAM2_CHECKPOINT, config=DEFAULT_SAM2_CONFIG,
                 device=None):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        want = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.device = torch.device(want)
            model = build_sam2(config, str(checkpoint), device=self.device, apply_postprocessing=True)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and want != "cpu":
                print(f"SAM2: CUDA OOM (GPU busy?), falling back to CPU — {exc}")
                torch.cuda.empty_cache()
                self.device = torch.device("cpu")
                model = build_sam2(config, str(checkpoint), device=self.device, apply_postprocessing=True)
            else:
                raise
        self.predictor = SAM2ImagePredictor(model, max_hole_area=64.0, max_sprinkle_area=32.0)
        self._key = None

    def set_image(self, pil_image, key=None):
        """Embed an image once; skip if the same key is already embedded."""
        if key is not None and key == self._key:
            return
        self.predictor.set_image(np.array(pil_image.convert("RGB"), copy=True))
        self._key = key

    @torch.inference_mode()
    def predict(self, points=None, labels=None, box=None):
        """Segment from a point prompt and/or a box prompt (image pixels).
        box = [x0, y0, x1, y1]. Returns a boolean mask (H, W) or None."""
        kwargs = {}
        if points:
            kwargs["point_coords"] = np.asarray(points, dtype=np.float32)
            kwargs["point_labels"] = np.asarray(labels, dtype=np.int32)
        if box is not None:
            kwargs["box"] = np.asarray(box, dtype=np.float32)
        if not kwargs:
            return None
        # A box picks one object; points alone use multimask + best score.
        masks, scores, _ = self.predictor.predict(
            multimask_output=(box is None), **kwargs
        )
        if masks.ndim == 4:
            masks = masks[:, 0]
        best = int(np.argmax(scores)) if masks.shape[0] > 1 else 0
        return np.asarray(masks[best], dtype=bool)
