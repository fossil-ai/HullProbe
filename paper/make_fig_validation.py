"""fig_validation.pdf: convergent validity at two levels, as two parallel
multi-shortcut scatters.

Left  (a) appearance shortcuts  -- hull-colour, background, watermark recolours,
          the measurement *principle*.  35 configs (3 shortcuts x seeds x 5 de-biasers).
Right (b) anatomical shortcuts  -- three donor-hull grafts, the swap *operator* itself.
          45 configs (3 shortcuts x 3 seeds x 5 de-biasers).
Each panel: individual (WGA, reliance) points coloured by shortcut, a dashed
least-squares fit, and the pooled Pearson r.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
import figstyle as fs

fs.apply()
FIG = Path(__file__).resolve().parent / "figures"
EXP = Path(__file__).resolve().parents[1] / "experiments"


def load(name):
    return json.loads((EXP / name).read_text())


def flipval(v):
    return v[[k for k in v if "flip" in k][0]]


APP = [("hull color", [42, 43, 44], "phase2_controlled_seed"),
       ("background",  [42, 43],     "phase2_controlled_bg_color_seed"),
       ("watermark",   [42, 43],     "phase2_controlled_watermark_seed")]
appearance = []
for lbl, seeds, pre in APP:
    pts = [(v["worst_group_acc"], flipval(v)) for s in seeds
           for v in load(f"{pre}{s}.json")["results"].values()]
    appearance.append((lbl, pts))

ANAT = [("Veh-Carrier / Ro-ro", "scA"), ("Container / Wood-Chips", "scB"),
        ("Fishing / Yacht", "scC")]
anatomical = []
for lbl, tag in ANAT:
    pts = [(v["worst_group_acc"], v["hull_swap_flip"]) for s in (42, 43, 44)
           for v in load(f"phase2_anatomical_{tag}_seed{s}.json")["results"].values()]
    anatomical.append((lbl, pts))


def draw(ax, groups, title, ylabel):
    ax_x, ax_y = [], []
    for (lbl, pts), col in zip(groups, fs.SHORTCUT):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.scatter(xs, ys, s=22, color=col, edgecolors="white", linewidths=0.4, zorder=3)
        ax_x += xs; ax_y += ys
    ax_x = np.array(ax_x); ax_y = np.array(ax_y)
    b, a = np.polyfit(ax_x, ax_y, 1)
    xf = np.array([ax_x.min(), ax_x.max()])
    ax.plot(xf, b * xf + a, "--", color="#8a8a8a", lw=1.0, zorder=2)
    r, _ = pearsonr(ax_x, ax_y)
    ax.text(0.95, 0.95, f"$r={r:.2f}$", transform=ax.transAxes, ha="right",
            va="top", fontsize=10)
    ax.text(0.95, 0.865, f"$n={len(ax_x)}$", transform=ax.transAxes, ha="right",
            va="top", fontsize=7.5, color="#777777")
    ax.set_xlabel("worst-group accuracy (proxy)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8.5)
    ax.set_xlim(ax_x.min() - 0.04, ax_x.max() + 0.04); ax.set_ylim(-0.02, 0.72)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                      markersize=5, label=l) for (l, _), c in zip(groups, fs.SHORTCUT)]
    ax.legend(handles=handles, fontsize=6.4, loc="lower left", labelspacing=0.3)
    fs.despine(ax)
    return r, len(ax_x)


fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 2.9))
rA, nA = draw(axA, appearance, "(a) principle: appearance recolor",
              "interventional reliance (recolor flip)")
rB, nB = draw(axB, anatomical, "(b) operator: anatomical part-graft",
              "interventional reliance (hull-swap flip)")
fig.tight_layout(w_pad=1.6)
fig.savefig(FIG / "fig_validation.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"(a) appearance r={rA:.3f} n={nA}   (b) anatomical r={rB:.3f} n={nB}")
