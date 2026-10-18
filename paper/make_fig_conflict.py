"""fig_conflict.pdf -- hull/superstructure conflict (Sec. 5.4).

Top: with a cross-class donor hull grafted so the hull "says" the donor class while the
superstructure still says the true class, which class does the classifier follow?
Bottom: per class, hull SENSITIVITY (share) vs the rate at which the hull is actually
SUFFICIENT to set the prediction; every class sits far below the y=x line.
Data: experiments/conflict.json.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import figstyle as fs

fs.apply()
FIG = Path(__file__).resolve().parent / "figures"
d = json.loads((Path(__file__).resolve().parents[1] / "experiments/conflict.json").read_text())
g = d["global"]
pc = [c for c in d["per_class"] if c.get("hull_share") is not None]

fig = plt.figure(figsize=(3.35, 3.4))
gs = fig.add_gridspec(2, 1, height_ratios=[0.42, 2.5], hspace=0.32)

# ---- top: stacked outcome bar --------------------------------------------
axT = fig.add_subplot(gs[0])
segs = [("follows superstructure", g["super_follow"], fs.BLUE),
        ("other class",            g["other"],        fs.LGRAY),
        ("follows donor hull",     g["hull_follow_err"], fs.RED)]
left = 0.0
for lab, val, col in segs:
    axT.barh(0, val, left=left, color=col, height=0.42,
             edgecolor="white", linewidth=1.0)
    if val > 0.10:
        axT.text(left + val / 2, 0, f"{val*100:.0f}%", ha="center", va="center",
                 color=("white" if col != fs.LGRAY else "#3a3a3a"), fontsize=9)
    left += val
# clean label for the thin donor-hull segment: centred just above it
axT.text(1 - g["hull_follow_err"] / 2, 0.30, f"{g['hull_follow_err']*100:.1f}%",
         ha="center", va="bottom", color=fs.RED, fontsize=8)
axT.set_xlim(0, 1); axT.set_ylim(-0.32, 0.55); axT.axis("off")
axT.set_title("Under conflict, which class does the classifier follow?  ($n=476$)",
              fontsize=7.6, pad=3)
handles = [Patch(facecolor=c, label=l) for l, _, c in segs]
axT.legend(handles=handles, fontsize=6.4, loc="upper center",
           bbox_to_anchor=(0.5, -0.18), ncol=3, handlelength=1.0,
           columnspacing=1.1)

# ---- bottom: per-class sensitivity vs sufficiency rate -------------------
axB = fig.add_subplot(gs[1])
xs = [c["hull_share"] for c in pc]; ys = [c["hull_follow"] for c in pc]
axB.plot([0, 0.7], [0, 0.7], ls="--", color="#b3b3b3", lw=1.0, zorder=1)
axB.text(0.355, 0.40, "sensitivity = sufficiency", fontsize=6.6, color="#9a9a9a",
         rotation=45, rotation_mode="anchor", ha="left", va="bottom")
axB.scatter(xs, ys, s=30, color=fs.BLUE, zorder=3, edgecolor="white", lw=0.5,
            label=f"vessel class ($n={len(pc)}$)")
axB.set_xlabel("hull sensitivity (share)")
axB.set_ylabel("hull sufficiency (rate)")
axB.set_xlim(0, 0.7); axB.set_ylim(0, 0.7); axB.set_aspect("equal")
axB.legend(loc="upper left")
fs.despine(axB)

fig.savefig(FIG / "fig_conflict.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("wrote fig_conflict.pdf; n_class =", len(pc))
