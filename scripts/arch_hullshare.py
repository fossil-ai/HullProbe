"""Architecture-generality of the causal hull-share (paper Sec. 5.5).
Computes the artifact-corrected hull-share for ConvNeXt-Tiny, ResNet-50, and ViT-B/16
baselines (each trained on the same MARVEL-B split), reusing the validated pooled
estimator from sensitivity.py. Writes experiments/arch_hullshare.json."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torchvision import models as tvm

from scripts.sensitivity import hull_share
from scripts.part_lattice_spectrum import test_pool

ARCHS = [
    ("ConvNeXt-T", "convnext_tiny", "experiments/baseline_clf/best.pt"),
    ("ResNet-50",  "resnet50",      "experiments/baseline_clf_resnet50/best.pt"),
    ("ViT-B/16",   "vit_b_16",      "experiments/baseline_clf_vit_b_16/best.pt"),
]


def build(arch, ckpt, device):
    m = {"convnext_tiny": tvm.convnext_tiny, "resnet50": tvm.resnet50,
         "vit_b_16": tvm.vit_b_16}[arch](num_classes=26)
    st = torch.load(ckpt, map_location="cpu"); st = st.get("model", st.get("state_dict", st))
    m.load_state_dict({k.replace("module.", ""): v for k, v in st.items()}, strict=False)
    return m.to(device).eval()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    by_c = test_pool(); all_v = [r for lst in by_c.values() for r in lst]
    res = {}
    print(f"{'arch':12s} {'clean-acc':>9s} {'hull-share':>11s} {'CI':>16s}")
    for name, arch, ckpt in ARCHS:
        if not Path(ckpt).exists():
            print(f"{name:12s}  missing {ckpt}"); continue
        acc = None
        mj = Path(ckpt).parent / "metrics.json"
        if mj.exists():
            acc = json.loads(mj.read_text()).get("best_test_acc")
        m = build(arch, ckpt, device)
        r = hull_share(m, by_c, all_v, device, mode="compatible", K=4, per_class=4)
        res[name] = {"arch": arch, "acc": acc, **r}
        ci = f"[{r['ci'][0]:.2f},{r['ci'][1]:.2f}]"
        print(f"{name:12s} {acc if acc else 0:9.3f} {r['hull_share']:11.3f} {ci:>16s}")
        del m; torch.cuda.empty_cache()
    Path("experiments/arch_hullshare.json").write_text(json.dumps(res, indent=1))
    print("wrote experiments/arch_hullshare.json")


if __name__ == "__main__":
    main()
