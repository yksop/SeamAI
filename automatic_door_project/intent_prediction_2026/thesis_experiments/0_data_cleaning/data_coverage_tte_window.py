#!/usr/bin/env python3
"""
data_coverage_tte_window.py

Compute and visualise the number of usable samples from MasterData for
every combination of observation window and time-to-event (TTE) horizon.

This answers the question: "How much data do we lose as we increase the
prediction horizon or the observation window?"

A sample is usable for a given (TTE, window) pair when the trajectory is
long enough that T_predict = T_event - TTE >= window_frames - 1, i.e. the
full observation window fits before the prediction point.

Outputs (saved to 0_data_cleaning/results/):
    - data_coverage_matrix.csv        — pivot table: rows = TTE, cols = window
    - data_coverage_detailed.csv      — long format with enter/pass breakdown
    - data_coverage_heatmap.png       — heatmap visualisation
    - data_coverage_lines.png         — line plot (one curve per window)

Run:
    python 0_data_cleaning/data_coverage_tte_window.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                              # pipeline root
DATA_ROOT = ROOT / "MasterData"
OUT_DIR = HERE / "results"
OUT_DIR.mkdir(exist_ok=True)

# Make utils/ importable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.dataset_utils import load_all_samples, is_valid_sample
from utils.feature_extractor import is_sample_usable

# ---------------------------------------------------------------------------
# SETTINGS — edit these to change the run
# ---------------------------------------------------------------------------
TTE_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
WINDOW_VALUES = [0.5, 1.0, 1.5]


def main():
    # ------------------------------------------------------------------
    # Load all valid samples
    # ------------------------------------------------------------------
    print(f"Loading samples from {DATA_ROOT} ...")
    all_samples = load_all_samples(str(DATA_ROOT))
    samples = [s for s in all_samples if is_valid_sample(s)]

    n_enter = sum(1 for s in samples if s["label"] == "enter")
    n_pass = sum(1 for s in samples if s["label"] == "pass")
    n_total = len(samples)
    print(f"  Valid samples: {n_total}  (enter={n_enter}, pass={n_pass})")

    # ------------------------------------------------------------------
    # Compute usability for every (TTE, window) pair
    # ------------------------------------------------------------------
    rows = []
    for win in WINDOW_VALUES:
        for tte in TTE_VALUES:
            usable = [s for s in samples if is_sample_usable(s, tte, win)]
            n_usable = len(usable)
            n_enter_usable = sum(1 for s in usable if s["label"] == "enter")
            n_pass_usable = sum(1 for s in usable if s["label"] == "pass")
            rows.append({
                "window_s": win,
                "tte_s": tte,
                "n_total": n_total,
                "n_usable": n_usable,
                "n_enter": n_enter_usable,
                "n_pass": n_pass_usable,
                "coverage_pct": 100.0 * n_usable / n_total,
                "enter_pct": 100.0 * n_enter_usable / n_enter,
                "pass_pct": 100.0 * n_pass_usable / n_pass,
                "class_ratio_enter": n_enter_usable / max(n_usable, 1),
            })

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  DATA COVERAGE: usable samples per (TTE, window) combination")
    print("=" * 72)

    pivot = df.pivot(index="tte_s", columns="window_s", values="n_usable")
    pivot_pct = df.pivot(index="tte_s", columns="window_s", values="coverage_pct")

    print("\nAbsolute counts:")
    print(pivot.to_string())
    print(f"\n(Total valid samples: {n_total})")

    print("\nCoverage (%):")
    print(pivot_pct.round(1).to_string())

    # ------------------------------------------------------------------
    # Save CSVs
    # ------------------------------------------------------------------
    pivot.to_csv(OUT_DIR / "data_coverage_matrix.csv")
    df.to_csv(OUT_DIR / "data_coverage_detailed.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'data_coverage_matrix.csv'}")
    print(f"Saved: {OUT_DIR / 'data_coverage_detailed.csv'}")

    # ------------------------------------------------------------------
    # Plot 1: Heatmap
    # ------------------------------------------------------------------
    _apply_style()

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot_pct.values, aspect="auto", cmap="YlOrRd_r",
                   vmin=0, vmax=100)

    ax.set_xticks(range(len(WINDOW_VALUES)))
    ax.set_xticklabels([f"{w:.2g}" for w in WINDOW_VALUES])
    ax.set_yticks(range(len(TTE_VALUES)))
    ax.set_yticklabels([f"{t:.1f}" for t in TTE_VALUES])
    ax.set_xlabel("Observation window (s)")
    ax.set_ylabel("Time-to-event (s)")
    ax.set_title("Data coverage (%)")

    # Annotate cells with count and percentage
    for i, tte in enumerate(TTE_VALUES):
        for j, win in enumerate(WINDOW_VALUES):
            n = int(pivot.iloc[i, j])
            pct = pivot_pct.iloc[i, j]
            colour = "white" if pct < 50 else "black"
            ax.text(j, i, f"{n}\n({pct:.0f}%)", ha="center", va="center",
                    fontsize=8, color=colour, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Coverage (%)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "data_coverage_heatmap.png", dpi=300)
    plt.close(fig)
    print(f"Saved: data_coverage_heatmap.png")

    # ------------------------------------------------------------------
    # Plot 2: Line plot (one curve per window)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(WINDOW_VALUES)))
    for idx, win in enumerate(WINDOW_VALUES):
        subset = df[df["window_s"] == win]
        ax.plot(subset["tte_s"], subset["coverage_pct"],
                "o-", color=colours[idx], label=f"w = {win:.2g} s",
                markersize=6, linewidth=1.8)

    ax.set_xlabel("Time-to-event (s)")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Dataset coverage vs. TTE for different observation windows")
    ax.legend(title="Obs. window", loc="lower left")
    ax.set_ylim(25, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "data_coverage_lines.png", dpi=300)
    plt.close(fig)
    print(f"Saved: data_coverage_lines.png")

    # ------------------------------------------------------------------
    # Plot 3: Class balance across TTE (for w = 0.5 s)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    w_ref = 0.5
    subset = df[df["window_s"] == w_ref].copy()

    bar_width = 0.35
    x_pos = np.arange(len(subset))
    ax.bar(x_pos - bar_width / 2, subset["n_enter"], bar_width,
           label="enter", color="#4878CF")
    ax.bar(x_pos + bar_width / 2, subset["n_pass"], bar_width,
           label="pass", color="#E07B54")

    for i, row in enumerate(subset.itertuples()):
        ax.text(i, max(row.n_enter, row.n_pass) + 8,
                f"{row.n_usable}", ha="center", fontsize=8, color="grey")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{t:.1f}" for t in subset["tte_s"]])
    ax.set_xlabel("Time-to-event (s)")
    ax.set_ylabel("Number of samples")
    ax.set_title(f"Class balance across TTE (window = {w_ref} s)")
    ax.legend()
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "data_coverage_class_balance.png", dpi=300)
    plt.close(fig)
    print(f"Saved: data_coverage_class_balance.png")

    print("\nDone.")


def _apply_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
        "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
    })


if __name__ == "__main__":
    main()
