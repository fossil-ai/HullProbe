"""fig_semL0.pdf -- semantic-L0 radius distribution.
Data: experiments/part_lattice_spectrum.json (semantic_L0_hist).
r=1..6 realistic part swaps to flip; the 'never' bin is r>6 (no flip within six cells).
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import figstyle as fs

fs.apply()
FIG = Path(__file__).resolve().parent / "figures"
S = json.loads((Path(__file__).resolve().parents[1] /
                "experiments/part_lattice_spectrum.json").read_text())
hist = {int(k): v for k, v in S["semantic_L0_hist"].items()}
tot = sum(hist.values())
xs = [1, 2, 3, 4, 5, 6, 7]                 # 7 = never flips within the six cells
ys = [hist.get(k, 0) / tot for k in xs]

fig, ax = plt.subplots(figsize=(3.1, 2.3))
ax.bar(range(7), ys, width=0.7, color=fs.BLUE, edgecolor="white", linewidth=0.6,
       zorder=3)
for i, y in enumerate(ys):
    if y:
        ax.text(i, y + 0.012, f"{y:.0%}", ha="center", va="bottom", fontsize=6.8,
                color="#333333")
ax.set_xlabel("semantic-$L_0$ radius $r(x)$")
ax.set_ylabel("fraction of vessels")
ax.set_xticks(range(7))
ax.set_xticklabels(["1", "2", "3", "4", "5", "6", "never"])
ax.set_ylim(0, max(ys) * 1.20)
ax.grid(axis="x", visible=False)
ax.text(0.96, 0.92, f"{ys[0]:.0%} at $r=1$", transform=ax.transAxes,
        ha="right", va="top", fontsize=7.5)
fs.despine(ax)
fig.tight_layout()
fig.savefig(FIG / "fig_semL0.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"wrote fig_semL0.pdf  ({ys[0]:.0%} at r=1)")
