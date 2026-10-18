"""fig_degree.pdf -- Fourier-Walsh variance by interaction degree (near-additivity).
Data: experiments/part_lattice_spectrum.json (variance_by_degree_mean, degrees 1..6).
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import figstyle as fs

fs.apply()
FIG = Path(__file__).resolve().parent / "figures"
S = json.loads((Path(__file__).resolve().parents[1] /
                "experiments/part_lattice_spectrum.json").read_text())
deg = S["variance_by_degree_mean"]           # degrees 1..6
d12 = deg[0] + deg[1]

fig, ax = plt.subplots(figsize=(3.1, 2.3))
bars = ax.bar(range(1, 7), deg, width=0.68, color=fs.BLUE,
              edgecolor="white", linewidth=0.6, zorder=3)
for x, v in zip(range(1, 7), deg):
    ax.text(x, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=6.8,
            color="#333333")
ax.set_xlabel("interaction degree $|S|$")
ax.set_ylabel("fraction of variance")
ax.set_xticks(range(1, 7))
ax.set_ylim(0, deg[0] * 1.18)
ax.grid(axis="x", visible=False)
ax.text(0.96, 0.92, f"degree $\\leq 2$: {d12:.2f}", transform=ax.transAxes,
        ha="right", va="top", fontsize=7.5)
fs.despine(ax)
fig.tight_layout()
fig.savefig(FIG / "fig_degree.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"wrote fig_degree.pdf  (deg1={deg[0]:.2f}, deg<=2={d12:.2f})")
