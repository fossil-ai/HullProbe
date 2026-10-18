"""Complete vessel-instance segmentation using local GroundingDINO + SAM2.

Ported verbatim from a reference vessel-segmentation implementation. The only
intended extension for hullprobe is prompting for above-deck parts (e.g.
``prompt="superstructure. deckhouse. bridge. funnel. crane. mast. container."``)
to obtain a hull vs. superstructure split; the vessel prompt default is
unchanged so full-vessel pose estimation stays identical to the reference.
"""

from __future__ import annotations
import os

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


DEFAULT_GROUNDING_MODEL = Path(
    os.environ.get("GROUNDING_DINO_DIR", "checkpoints/grounding-dino-tiny")
)
DEFAULT_SAM2_CHECKPOINT = Path(
    os.environ.get("SAM2_CHECKPOINT", "checkpoints/sam2.1_hiera_large.pt")
)
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


@dataclass
class VesselMaskPrediction:
    mask: np.ndarray
    box_xyxy: list[float]
    detection_score: float
    sam_score: float
    label: str


class GroundedSAMVesselSegmenter:
    """Open-vocabulary detector followed by vessel-instance segmentation."""

    def __init__(
        self,
        grounding_model=DEFAULT_GROUNDING_MODEL,
        sam2_checkpoint=DEFAULT_SAM2_CHECKPOINT,
        sam2_config=DEFAULT_SAM2_CONFIG,
        device=None,
        box_threshold=0.18,
        text_threshold=0.20,
        prompt="ship. boat. vessel.",
    ):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.prompt = prompt
        self.processor = AutoProcessor.from_pretrained(
            str(grounding_model), local_files_only=True
        )
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(grounding_model), local_files_only=True
        ).to(self.device).eval()
        sam_model = build_sam2(
            sam2_config, str(sam2_checkpoint), device=self.device, apply_postprocessing=True
        )
        self.sam = SAM2ImagePredictor(
            sam_model, max_hole_area=64.0, max_sprinkle_area=32.0
        )

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> VesselMaskPrediction:
        image = image.convert("RGB")
        inputs = self.processor(
            images=image, text=self.prompt, return_tensors="pt"
        ).to(self.device)
        outputs = self.detector(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        if len(result["boxes"]) == 0:
            raise RuntimeError("GroundingDINO found no vessel box")

        boxes = result["boxes"].detach().cpu().numpy()
        detection_scores = result["scores"].detach().cpu().numpy()
        labels = [str(label) for label in result["labels"]]
        image_area = float(image.width * image.height)

        # Prefer a confident, substantial, image-centered vessel detection. This
        # rejects tiny secondary boats while retaining the dominant MARVEL vessel.
        ranking = []
        for index, (box, score) in enumerate(zip(boxes, detection_scores)):
            x0, y0, x1, y1 = box
            area_fraction = max(0.0, (x1 - x0) * (y1 - y0) / image_area)
            center_x = 0.5 * (x0 + x1) / max(1.0, image.width)
            center_penalty = abs(center_x - 0.5)
            ranking.append((float(score) * (area_fraction ** 0.35) - 0.05 * center_penalty, index))
        ranking.sort(reverse=True)

        # SAM2's normalization kernel may write into its NumPy input.  PIL can
        # expose a read-only view, so provide an owned array explicitly.
        self.sam.set_image(np.array(image, copy=True))
        candidates = []
        for _rank, index in ranking[: min(3, len(ranking))]:
            masks, sam_scores, _logits = self.sam.predict(
                point_coords=None,
                point_labels=None,
                box=boxes[index][None, :],
                multimask_output=True,
            )
            if masks.ndim == 4:
                masks = masks[:, 0]
            for mask, sam_score in zip(masks, sam_scores):
                mask = np.asarray(mask, dtype=bool)
                area_fraction = float(mask.mean())
                # SAM IoU score plus weak priors: the vessel should be neither a
                # tiny fragment nor most of the image, and should fill its box.
                box = boxes[index]
                box_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
                fill = float(mask.sum() / box_area)
                plausible = 0.01 <= area_fraction <= 0.65
                quality = float(sam_score) + 0.15 * min(fill, 1.0)
                if not plausible:
                    quality -= 1.0
                candidates.append((quality, index, mask, float(sam_score)))
        if not candidates:
            raise RuntimeError("SAM2 produced no vessel masks")
        _quality, index, mask, sam_score = max(candidates, key=lambda item: item[0])
        return VesselMaskPrediction(
            mask=mask,
            box_xyxy=[float(value) for value in boxes[index]],
            detection_score=float(detection_scores[index]),
            sam_score=sam_score,
            label=labels[index],
        )
