"""HullProbe: measuring interventional part reliance in maritime vessel classifiers.

    pose            PCA pose and the longitudinal fwd/mid/aft thirds
    canonical       hull/superstructure masks inside the vessel silhouette
    lattice         the 2x3 part lattice {hull, sup} x {fwd, mid, aft}
    lattice_align   per-cell registration and the 64 composites phi(x, d, z)
    audit           model-agnostic cube, influence, Walsh spectrum, semantic-L0, hull-share
    probe           component-level swap-probe statistics
    segmenter       region-map I/O; segformer_segmenter: the hull/superstructure SegFormer
"""

from . import canonical, constants, pose

__version__ = "1.0.0"
__all__ = ["pose", "canonical", "constants"]
