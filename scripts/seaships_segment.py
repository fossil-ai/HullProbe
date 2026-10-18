"""Step 2: segment SeaShips crops with the MARVEL-trained SegFormer, using its OWN
3-way argmax (bg/hull/superstructure) -- no external vessel alpha needed (SeaShips has
none). This also tests whether the hull/superstructure decomposition TRANSFERS to a new
dataset. Writes data/seaships/predictions/{stem}.png ({0=bg,1=hull,2=sup})."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from hullprobe.segformer_segmenter import build_model, logits_at
from hullprobe.segmenter import _norm, save_region_png

OUT = Path("data/seaships/predictions"); OUT.mkdir(parents=True, exist_ok=True)
CKPT = "experiments/segformer/best.pt"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if device == "cuda" and torch.cuda.mem_get_info()[0] < 2.0e9:
            device = "cpu"
    except Exception:
        pass
    model = build_model(device=device)
    st = torch.load(CKPT, map_location=device)
    model.load_state_dict(st.get("model", st)); model.eval()
    recs = json.loads(Path("data/seaships/records.json").read_text())["records"]
    print(f"segmenting {len(recs)} crops on {device}", flush=True)
    written = 0
    with torch.inference_mode():
        for i, r in enumerate(recs):
            try:
                img = Image.open(r["image_path"]).convert("RGB"); W, H = img.size
                x = _norm(img, 512).unsqueeze(0).to(device)
                logits = logits_at(model, x, (H, W))[0]          # (3,H,W): bg/hull/sup
                region = logits.argmax(0).cpu().numpy().astype(np.uint8)
                save_region_png(region, OUT / f"{Path(r['image_path']).stem}.png")
                written += 1
            except Exception as e:
                print("skip", r["image_path"], e)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(recs)}", flush=True)
    print(f"wrote {written} region maps to {OUT}", flush=True)


if __name__ == "__main__":
    main()
