"""Causal swap probe: does the superstructure define the vessel class?

For a recipient r (class y_r) and a cross-class donor d (class y_d != y_r):

    super_img = recip hull + donor superstructure     (superstructure_swap)
    hull_img  = recip superstructure + donor hull      (hull_swap)

If the superstructure carries class identity and the hull does not, then
swapping the superstructure should pull the classifier toward the donor's class
far more than swapping the hull does:

    delta_donor = P(y_d | super_img) - P(y_d | hull_img)      > 0
    delta_recip = P(y_r | hull_img)  - P(y_r | super_img)      > 0

Same-class swaps (y_d == y_r) form the artifact floor: any prediction change
there is due to seams/fill, not content, and is reported alongside so the
cross-class effect is read against it.

`classify` is injected — any callable PIL.Image -> logits (np.ndarray, C) — so
this module is fully testable without a real network.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .swap import hull_swap, superstructure_swap


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


@dataclass
class PairResult:
    recip_class: int
    donor_class: int
    p_super: np.ndarray  # softmax over classes for the superstructure-swap image
    p_hull: np.ndarray   # softmax over classes for the hull-swap image

    @property
    def delta_donor(self):
        return float(self.p_super[self.donor_class] - self.p_hull[self.donor_class])

    @property
    def delta_recip(self):
        return float(self.p_hull[self.recip_class] - self.p_super[self.recip_class])

    @property
    def super_follows_donor(self):
        return int(int(np.argmax(self.p_super)) == self.donor_class)

    @property
    def hull_follows_donor(self):
        return int(int(np.argmax(self.p_hull)) == self.donor_class)

    @property
    def hull_keeps_recip(self):
        return int(int(np.argmax(self.p_hull)) == self.recip_class)

    @property
    def super_keeps_recip(self):
        return int(int(np.argmax(self.p_super)) == self.recip_class)


def run_pair(classify, recip_rec, donor_rec, recip_class, donor_class, **swap_kwargs):
    """Build both swaps for one (recip, donor) pair and score them."""
    super_img, _ = superstructure_swap(recip_rec, donor_rec, **swap_kwargs)
    hull_img, _ = hull_swap(recip_rec, donor_rec, **swap_kwargs)
    p_super = softmax(classify(super_img))
    p_hull = softmax(classify(hull_img))
    return PairResult(recip_class, donor_class, p_super, p_hull)


def paired_bootstrap_ci(values, n_boot=5000, seed=42, alpha=0.05):
    """Percentile bootstrap CI for the mean of paired per-sample values."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lo": float(np.percentile(boot_means, 100 * alpha / 2)),
        "hi": float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
        "n": int(len(values)),
    }


def summarize(cross_pairs, control_pairs=None, n_boot=5000, seed=42):
    """Aggregate cross-class pairs (and optional same-class controls) into the
    probe's headline statistics with bootstrap CIs."""
    def block(pairs):
        if not pairs:
            return None
        return {
            "delta_donor": paired_bootstrap_ci([p.delta_donor for p in pairs], n_boot, seed),
            "delta_recip": paired_bootstrap_ci([p.delta_recip for p in pairs], n_boot, seed),
            "super_follows_donor_rate": float(np.mean([p.super_follows_donor for p in pairs])),
            "hull_follows_donor_rate": float(np.mean([p.hull_follows_donor for p in pairs])),
            "hull_keeps_recip_rate": float(np.mean([p.hull_keeps_recip for p in pairs])),
            "super_keeps_recip_rate": float(np.mean([p.super_keeps_recip for p in pairs])),
            "n": len(pairs),
        }

    out = {"cross_class": block(cross_pairs)}
    if control_pairs:
        # For same-class controls, "donor class" == recip class; the informative
        # quantity is how often each swap flips *away* from the true class
        # (artifact-induced instability).
        out["same_class_control"] = {
            "super_flip_away_rate": float(np.mean([1 - p.super_keeps_recip for p in control_pairs])),
            "hull_flip_away_rate": float(np.mean([1 - p.hull_keeps_recip for p in control_pairs])),
            "n": len(control_pairs),
        }
    # Headline: superstructure carries donor identity more than hull does, and
    # the CI on that gap excludes zero.
    cc = out["cross_class"]
    if cc is not None:
        out["verdict"] = {
            "superstructure_defines_class": bool(
                cc["delta_donor"]["lo"] > 0 and cc["delta_recip"]["lo"] > 0
            ),
            "donor_transfer_gap": cc["delta_donor"]["mean"],
        }
    return out
