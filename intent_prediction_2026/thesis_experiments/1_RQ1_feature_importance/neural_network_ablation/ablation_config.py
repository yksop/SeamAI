"""
ablation_config.py — shared paths and plot style for the ablation analysis.

Used by select_gru_finalists.py and plot_ablation_figures.py (stages 2-3).
All generated output goes under results/.
"""

from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
RESULTS_CSV = RESULTS_DIR / "results_all.csv"   # written by train_ablation.py
TABLE_DIR = RESULTS_DIR                          # finalist CSVs land here
FIG_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------
TTE_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5]

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
        "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
    })
