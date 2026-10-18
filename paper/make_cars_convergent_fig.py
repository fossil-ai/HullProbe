"""fig_cars_convergent.pdf: convergent validity for cars.

Scatter of the group-label-free car part-reliance (recolor-flip on the {body,body-side}
parts) vs worst-group accuracy, one point per de-biaser x seed, coloured by de-biaser, with the
pooled Pearson r and a dashed least-squares fit. The car analogue of Fig.~5.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import figstyle as fs

fs.apply()
FIG = Path(__file__).resolve().parent / "figures"
EXP = Path(__file__).resolve().parents[1] / "experiments"
d = json.loads((EXP / "cars_convergent.json").read_text())

METHODS = ["ERM", "SUBG", "JTT", "GroupDRO", "DFR"]
COL = {"ERM": "#1a1a1a", "SUBG": "#4C72B0", "JTT": "#DD8452",
       "GroupDRO": "#55A868", "DFR": "#C44E52"}

pts = d["points"]                     # list of {shortcut, method, seed, wga, flip}
fig, ax = plt.subplots(figsize=(3.3, 2.7))
xs, ys = [], []
for m in METHODS:
    mp = [p for p in pts if p["method"] == m]
    mx = [p["wga"] for p in mp]; my = [p["flip"] for p in mp]
    xs += mx; ys += my
    ax.scatter(mx, my, s=15, color=COL[m], alpha=0.55, edgecolors="none", zorder=3)
    ax.scatter([np.mean(mx)], [np.mean(my)], s=68, color=COL[m], edgecolors="white",
               linewidths=1.0, zorder=4)

xs, ys = np.array(xs), np.array(ys)
b, a = np.polyfit(xs, ys, 1)
xr = np.linspace(xs.min() - 0.02, xs.max() + 0.02, 50)
ax.plot(xr, a + b * xr, "--", color="#8C8C8C", lw=1.0, zorder=2)
r = d["corr_wga_flip"]; n = d["n_configs"]
ax.text(0.96, 0.94, f"$r = {r:.2f}$\n$n = {n}$", transform=ax.transAxes, ha="right", va="top",
        fontsize=8.5, color="#1a1a1a")
ax.set_xlabel("worst-group accuracy (proxy, needs group labels)")
ax.set_ylabel("car part-reliance  (recolor flip)")
handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COL[m], markersize=6, label=m)
           for m in METHODS]
ax.legend(handles=handles, loc="lower left", fontsize=6.5, ncol=1)
fig.tight_layout()
fig.savefig(FIG / "fig_cars_convergent.pdf", bbox_inches="tight")
print("SAVED fig_cars_convergent.pdf | r =", round(r, 3), "n =", n)
