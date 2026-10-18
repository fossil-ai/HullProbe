"""Score any MARVEL-B classifier with HullProbe on a published recipient pool.

Loads a torchvision classifier (26 MARVEL-B classes, 224x224 ImageNet normalization), audits it on
the fixed recipients and donors of protocol/attribution_pool_n277.json, and reports the corrected
hull-share, interaction spectrum, and single-swap fragility with 95% bootstrap CIs. Recipients the
classifier misclassifies are skipped, as in the paper (each backbone on its own correct recipients).

    python -m scripts.audit_classifier --arch convnext_tiny --checkpoint experiments/baseline_clf/best.pt

For other architectures or preprocessing, write a `predict(list_of_HxWx3_uint8) -> labels` function
and call hullprobe.audit directly (see its docstring).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image

from hullprobe import audit

TF = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_classifier(arch, checkpoint, num_classes, device):
    model = getattr(tvm, arch)(num_classes=num_classes)
    st = torch.load(checkpoint, map_location="cpu")
    st = st.get("model", st.get("state_dict", st))
    model.load_state_dict({k.replace("module.", ""): v for k, v in st.items()}, strict=False)
    return model.to(device).eval()


def make_predict(model, device, batch=64):
    @torch.inference_mode()
    def predict(images):
        out = []
        for i in range(0, len(images), batch):
            x = torch.stack([TF(Image.fromarray(a)) for a in images[i:i + batch]]).to(device)
            out.append(model(x).argmax(1).cpu().numpy())
        return np.concatenate(out)
    return predict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="convnext_tiny", help="torchvision model builder name")
    ap.add_argument("--checkpoint", default="experiments/baseline_clf/best.pt")
    ap.add_argument("--pool", default="protocol/attribution_pool_n277.json")
    ap.add_argument("--limit", type=int, default=None, help="audit only the first N recipients")
    ap.add_argument("--out", default=None, help="optional JSON output")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predict = make_predict(load_classifier(a.arch, a.checkpoint, 26, device), device)
    records, pool = audit.load_records(), audit.load_pool(a.pool)
    entries = pool["recipients"][:a.limit]
    results = []
    for i, e in enumerate(entries, 1):
        res = audit.audit_recipient(predict, records, e)
        if res is not None:
            results.append(res)
        if i % 25 == 0:
            print(f"  {i}/{len(entries)}  kept {len(results)}", flush=True)
    summary = audit.summarize(results)
    for k in ("hull_share", "W1", "W_le2", "frac_r1"):
        v = summary[k]
        print(f"{k:10s} {v['value']:.3f}  [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]")
    print(f"n = {summary['n']} of {len(entries)} recipients (rest misclassified by this model)")
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
