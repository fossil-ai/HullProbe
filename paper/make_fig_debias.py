"""fig_debias.pdf -- de-biasing audit, two independent axes (Sec. 5.7 / App.).
Left:  hull-share (balance axis, which parts).  Right: Pr[r=1] (concentration axis).
Data: experiments/part_lattice_debias_multiseed.json.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import figstyle as fs

fs.apply()
FIG = Path(__file__).resolve().parent / "figures"
D = json.loads((Path(__file__).resolve().parents[1] /
                "experiments/part_lattice_debias_multiseed.json").read_text())["arms"]
arms = ["baseline", "cutmix", "hullswap", "jtt", "lff", "afr", "bgswap"]
lbl = ["ERM", "CutMix", "hull-rand", "JTT", "LfF", "AFR", "bg-swap"]
share = [D[a]["hull_share_mean"] for a in arms]
sstd = [D[a]["hull_share_std"] for a in arms]
dstd = [D[a]["delta_vs_base_std"] for a in arms]
delta = [D[a]["delta_vs_base_mean"] for a in arms]
frac = [D[a]["frac_r1_mean"] for a in arms]
sig = [i > 0 and (abs(delta[i]) - 2 * dstd[i]) > 0 for i in range(len(arms))]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.4))

# left: balance axis (hull-share)
cols = [fs.GRAY] + [fs.RED if sig[i] else fs.MUTED for i in range(1, len(arms))]
a1.bar(range(len(arms)), share, yerr=sstd, capsize=2.5, color=cols,
       edgecolor="white", linewidth=0.6, error_kw=dict(lw=0.8, ecolor="#555"), zorder=3)
a1.axhline(share[0], color="#555", ls="--", lw=0.7, zorder=2)
for i in range(len(arms)):
    if sig[i]:
        a1.text(i, share[i] + sstd[i] + 0.012, "*", ha="center", fontsize=11)
a1.set_xticks(range(len(arms)))
a1.set_xticklabels(lbl, rotation=25, ha="right")
a1.set_ylabel("hull-share (balance axis)")
a1.set_ylim(0, 0.56)
a1.set_title("which parts: spurious balance", fontsize=8.5)
a1.grid(axis="x", visible=False)
fs.despine(a1)

# right: concentration axis (Pr[r=1])
a2.bar(range(len(arms)), frac, width=0.72, color=fs.GREEN,
       edgecolor="white", linewidth=0.6, zorder=3)
a2.axhline(frac[0], color="#555", ls="--", lw=0.7, zorder=2)
a2.set_xticks(range(len(arms)))
a2.set_xticklabels(lbl, rotation=25, ha="right")
a2.set_ylabel(r"$\Pr[r=1]$ (concentration axis)")
a2.set_ylim(0, 1)
a2.set_title("how many parts: fragility", fontsize=8.5)
a2.grid(axis="x", visible=False)
fs.despine(a2)

fig.tight_layout(w_pad=1.6)
fig.savefig(FIG / "fig_debias.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("wrote fig_debias.pdf")
