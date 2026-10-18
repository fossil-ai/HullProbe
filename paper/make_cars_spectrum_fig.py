"""fig_cars_spectrum.pdf: near-additivity + single-swap fragility for the CAR lattice.

(a) Walsh variance by interaction degree over the 2^5 cube (main effects dominate).
(b) Per-part corrected influence (greenhouse dominates; wheels/body ~0 under the control).
The car analogue of the maritime Fig.~4, computed from experiments/cars_lattice.json.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

fs.apply()
FIG = Path(__file__).resolve().parent / "figures"
EXP = Path(__file__).resolve().parents[1] / "experiments"
d = json.loads((EXP / "cars_lattice.json").read_text())

parts = d["part_order"]
deg = d["variance_by_degree_mean"]                     # len 5 (degrees 1..5)
corr = d["influence_corrected_mean"]
cross = d["influence_cross_mean"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.5))

# (a) degree variance
degs = list(range(1, len(deg) + 1))
ax1.bar(degs, deg, color="#4C72B0", width=0.6)
ax1.set_xlabel("interaction degree $k$")
ax1.set_ylabel("fraction of variance $W_k$")
ax1.set_xticks(degs)
ax1.text(0.95, 0.9, f"$W_1{{=}}{deg[0]:.2f}$\n$W_{{\\leq 2}}{{=}}{deg[0]+deg[1]:.2f}$",
         transform=ax1.transAxes, ha="right", va="top", fontsize=8.5)
ax1.set_title("(a) near-additive over anatomy", fontsize=9)

# (b) per-part corrected influence
order = np.argsort(corr)[::-1]
names = [parts[i] for i in order]
cvals = [corr[i] for i in order]
xvals = [cross[i] for i in order]
y = np.arange(len(names))[::-1]
ax2.barh(y, xvals, color="#C7C7C7", height=0.6, label="raw (cross-class)")
ax2.barh(y, cvals, color="#55A868", height=0.6, label="artifact-corrected")
ax2.set_yticks(y); ax2.set_yticklabels(names, fontsize=8)
ax2.set_xlabel("part flip-influence")
ax2.legend(fontsize=6.5, loc="lower right")
ax2.set_title("(b) greenhouse carries the class", fontsize=9)

fig.tight_layout()
fig.savefig(FIG / "fig_cars_spectrum.pdf", bbox_inches="tight")
print("SAVED fig_cars_spectrum.pdf | W1=%.2f W<=2=%.2f | greenhouse corr=%.2f"
      % (deg[0], deg[0] + deg[1], corr[parts.index("greenhouse")]))
