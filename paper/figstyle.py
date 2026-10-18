"""Shared sleek figure style for the HullProbe paper plots (seaborn-based).

Every plot generator does:
    import figstyle as fs
    fs.apply()
so all figures share one clean, consistent look.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# coordinated muted palette
BLUE, GREEN, RED = "#4C72B0", "#55A868", "#C44E52"
GRAY, LGRAY, MUTED = "#8C8C8C", "#CBCBCB", "#9DB0C4"
SHORTCUT = [BLUE, GREEN, RED]
INK = "#1a1a1a"


def apply():
    sns.set_theme(context="paper", style="whitegrid", font="DejaVu Sans")
    plt.rcParams.update({
        "figure.dpi": 220, "savefig.dpi": 220,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.linewidth": 0.8, "axes.edgecolor": "#666666",
        "axes.labelcolor": INK, "axes.labelsize": 8.5, "axes.labelpad": 3,
        "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlepad": 6,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#E9E9E9", "grid.linewidth": 0.6,
        "xtick.color": "#666666", "ytick.color": "#666666",
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "font.size": 8, "text.color": INK,
        "legend.frameon": False, "legend.fontsize": 7,
        "legend.handletextpad": 0.4, "legend.borderpad": 0.2,
    })


def despine(ax):
    sns.despine(ax=ax)
