#!/usr/bin/env python3
"""
Generate publication-quality figures for RQ2 edge deployment results.

Produces six PDF/PNG figures:
  1. Perception frontend comparison (grouped bar: PyTorch vs TensorRT, no MediaPipe)
  2. Perception latency percentiles (mean, p50, p95, p99 for TRT frontends)
  3. Full pipeline FPS vs balanced accuracy (scatter + Pareto frontier)
  4. Latency breakdown per pipeline configuration (stacked horizontal bars)
  5. Pipeline latency percentiles (p50, p95, p99 for all 35 combos)
  6. GRU overhead by input dimension (showing ~1ms regardless of dim)

Usage:
    python plot_rq2_results.py
"""

import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "outputs" / "results"
FIGURES = HERE / "outputs" / "figures"
FIGURES.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

# Colour palette
C_PYTORCH   = "#4878CF"
C_TRT       = "#D65F5F"
C_PARETO    = "#2CA02C"
C_TARGET    = "#D65F5F"
C_MEAN      = "#4878CF"
C_P50       = "#6ACC65"
C_P95       = "#D65F5F"
C_P99       = "#B47CC7"

TIER_COLOURS = {
    "Core-4":     "#4878CF",
    "Traj-6":     "#6ACC65",
    "Head angle": "#D65F5F",
    "Torso+head": "#B47CC7",
    "Full body":  "#C4AD66",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
perc_df  = pd.read_csv(RESULTS / "perception_benchmark.csv")
pipe_df  = pd.read_csv(RESULTS / "pipeline_benchmark.csv")
pareto_df = pd.read_csv(RESULTS / "pipeline_pareto.csv")

# ---------------------------------------------------------------------------
# Helper: GRU short name
# ---------------------------------------------------------------------------
def gru_short(name):
    if "core4" in name.lower() or "Core-4" in name:
        return "Core-4"
    elif "traj6" in name.lower() or "Trajectory" in name:
        return "Traj-6"
    elif "head_angle" in name.lower() or "head (angle)" in name.lower():
        return "Head angle"
    elif "torso_head" in name.lower() or "torso_head" in name:
        return "Torso+head"
    elif "full_body" in name.lower() or "full body" in name.lower():
        return "Full body"
    return name

pipe_df["gru_short"] = pipe_df["gru_model"].apply(gru_short)
pipe_df["is_trt"]    = pipe_df["perception"].str.contains("TRT")
pipe_df["is_mp"]     = pipe_df["perception"].str.contains("MediaPipe")

# Merge balanced accuracy from pareto file
ACC_KNOWN = {
    "A4_core4": 0.874,
    "A2_traj6": 0.8754,
    "B1_head_angle": 0.877,
    "B4_torso_head_raw": 0.877,
    "B6_full_body_raw": 0.877,
}
if "mean_bal_acc" not in pipe_df.columns:
    acc_map = pareto_df.set_index("experiment_key")["mean_bal_acc"].to_dict()
    pipe_df["mean_bal_acc"] = pipe_df["experiment_key"].map(acc_map)
for ek, acc in ACC_KNOWN.items():
    mask = (pipe_df["experiment_key"] == ek) & pipe_df["mean_bal_acc"].isna()
    pipe_df.loc[mask, "mean_bal_acc"] = acc

# Known PyTorch perception FPS from step5 run
PT_PERC = {
    "YOLOv8n":      {"fps": 46.5, "mean_ms": 21.5, "p50_ms": 21.4, "p95_ms": 22.3, "p99_ms": 24.2, "std_ms": 0.6},
    "YOLOv11n":     {"fps": 37.2, "mean_ms": 26.9, "p50_ms": 26.8, "p95_ms": 27.5, "p99_ms": 29.5, "std_ms": 0.5},
    "YOLOv8n-pose": {"fps": 40.1, "mean_ms": 24.9, "p50_ms": 24.8, "p95_ms": 25.5, "p99_ms": 27.8, "std_ms": 0.5},
    "YOLOv11n-pose":{"fps": 33.1, "mean_ms": 30.2, "p50_ms": 30.1, "p95_ms": 30.9, "p99_ms": 33.0, "std_ms": 0.5},
}

base_models = ["YOLOv8n", "YOLOv11n", "YOLOv8n-pose", "YOLOv11n-pose"]
trt_names   = [f"{m} [TRT]" for m in base_models]


# ===================================================================
# FIGURE 1 — Perception frontend: PyTorch vs TensorRT FPS comparison
# ===================================================================
fig1, ax1 = plt.subplots(figsize=(6.5, 3.5))

x = np.arange(len(base_models))
width = 0.35

pt_fps  = [PT_PERC[m]["fps"] for m in base_models]
trt_fps = []
for m in trt_names:
    row = perc_df[perc_df["name"] == m]
    trt_fps.append(row["fps"].values[0] if len(row) > 0 else 0)

bars_pt  = ax1.bar(x - width/2, pt_fps, width, label="PyTorch FP32",
                   color=C_PYTORCH, edgecolor="white", linewidth=0.5)
bars_trt = ax1.bar(x + width/2, trt_fps, width, label="TensorRT FP16",
                   color=C_TRT, edgecolor="white", linewidth=0.5)

ax1.axhline(y=30, color=C_TARGET, linestyle="--", linewidth=1.0,
            alpha=0.6, label="30 FPS target", zorder=1)

# Value labels
for bars in [bars_pt, bars_trt]:
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 1.2,
                 f"{h:.0f}", ha="center", va="bottom", fontsize=8)

# Speedup annotations
for i in range(len(base_models)):
    speedup = trt_fps[i] / pt_fps[i]
    mid_y = max(pt_fps[i], trt_fps[i]) + 8
    ax1.annotate(f"{speedup:.1f}x", xy=(x[i], mid_y), ha="center",
                 fontsize=8, color="#555555", style="italic")

# MediaPipe footnote instead of bar
ax1.annotate("YOLOv8n + MediaPipe (two-stage): 14.9 FPS",
             xy=(0.98, 0.04), xycoords="axes fraction", ha="right",
             fontsize=7.5, color="#666666", style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                       edgecolor="#cccccc", alpha=0.8))

xtick_labels = ["YOLOv8n\n(det)", "YOLOv11n\n(det)",
                "YOLOv8n-pose\n(det+pose)", "YOLOv11n-pose\n(det+pose)"]
ax1.set_xticks(x)
ax1.set_xticklabels(xtick_labels)
ax1.set_ylabel("Throughput (FPS)")
ax1.set_title("Perception Frontend: PyTorch vs TensorRT FP16 on Jetson Orin Nano")
ax1.legend(loc="upper right", framealpha=0.9)
ax1.set_ylim(0, max(trt_fps) * 1.25)

fig1.tight_layout()
fig1.savefig(FIGURES / "rq2_perception_comparison.png")
print(f"[1/6] Saved: rq2_perception_comparison")
plt.close(fig1)


# ===================================================================
# FIGURE 2 — Perception latency percentiles (TRT frontends)
# ===================================================================
fig2, ax2 = plt.subplots(figsize=(7, 3.8))

labels_perc = ["YOLOv8n\n[TRT]", "YOLOv11n\n[TRT]",
               "YOLOv8n-pose\n[TRT]", "YOLOv11n-pose\n[TRT]"]
x2 = np.arange(len(trt_names))
w = 0.18

means = perc_df.set_index("name").loc[trt_names, "mean_ms"].values
p50s  = perc_df.set_index("name").loc[trt_names, "p50_ms"].values
p95s  = perc_df.set_index("name").loc[trt_names, "p95_ms"].values
p99s  = perc_df.set_index("name").loc[trt_names, "p99_ms"].values
stds  = perc_df.set_index("name").loc[trt_names, "std_ms"].values

ax2.bar(x2 - 1.5*w, means, w, label="Mean", color=C_MEAN,
        edgecolor="white", linewidth=0.5)
ax2.bar(x2 - 0.5*w, p50s, w, label="p50 (median)", color=C_P50,
        edgecolor="white", linewidth=0.5)
ax2.bar(x2 + 0.5*w, p95s, w, label="p95", color=C_P95,
        edgecolor="white", linewidth=0.5)
ax2.bar(x2 + 1.5*w, p99s, w, label="p99", color=C_P99,
        edgecolor="white", linewidth=0.5)

# Error bars on mean (1 std)
ax2.errorbar(x2 - 1.5*w, means, yerr=stds, fmt="none", ecolor="black",
             capsize=3, capthick=0.8, linewidth=0.8, zorder=5)

# Value labels on all bars
for offset, vals in [(-1.5*w, means), (-0.5*w, p50s),
                     (0.5*w, p95s), (1.5*w, p99s)]:
    for i, v in enumerate(vals):
        ax2.text(x2[i] + offset, v + 0.15, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=7)

ax2.axhline(y=33.33, color=C_TARGET, linestyle="--", linewidth=1.0,
            alpha=0.6, label="33.3 ms (30 FPS)")

ax2.set_xticks(x2)
ax2.set_xticklabels(labels_perc)
ax2.set_ylabel("Latency (ms)")
ax2.set_title("TensorRT FP16 Perception Latency Distribution")
ax2.legend(loc="upper left", framealpha=0.9, ncol=3)
ax2.set_ylim(0, max(p99s) * 1.25)

fig2.tight_layout()
fig2.savefig(FIGURES / "rq2_perception_percentiles.png")
print(f"[2/6] Saved: rq2_perception_percentiles")
plt.close(fig2)


# ===================================================================
# FIGURE 3 — Pipeline FPS vs Balanced Accuracy (scatter + Pareto)
# ===================================================================
fig3, ax3 = plt.subplots(figsize=(7, 4.5))

for gru_name, colour in TIER_COLOURS.items():
    subset = pipe_df[pipe_df["gru_short"] == gru_name]
    for _, row in subset.iterrows():
        if row["is_mp"]:
            marker = "^"
        elif row["is_trt"]:
            marker = "s"
        else:
            marker = "o"
        ax3.scatter(row["total_fps"], row["mean_bal_acc"],
                    c=colour, marker=marker, s=55, edgecolors="white",
                    linewidth=0.4, zorder=3)

pareto_sorted = pareto_df.sort_values("total_fps")
ax3.plot(pareto_sorted["total_fps"], pareto_sorted["mean_bal_acc"],
         color=C_PARETO, linewidth=1.5, linestyle="-", alpha=0.7,
         zorder=2, label="Pareto frontier")

ax3.axvline(x=30, color=C_TARGET, linestyle="--", linewidth=1.0,
            alpha=0.8, label="30 FPS target")

legend_elements = []
for gru_name, colour in TIER_COLOURS.items():
    legend_elements.append(Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=colour, markersize=7,
                                  label=gru_name))
legend_elements.append(Line2D([0], [0], color="w", label=""))
legend_elements.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="#888", markersize=6,
                              label="PyTorch FP32"))
legend_elements.append(Line2D([0], [0], marker="s", color="w",
                              markerfacecolor="#888", markersize=6,
                              label="TensorRT FP16"))
legend_elements.append(Line2D([0], [0], marker="^", color="w",
                              markerfacecolor="#888", markersize=6,
                              label="Two-stage (MP)"))
legend_elements.append(Line2D([0], [0], color=C_PARETO, linewidth=1.5,
                              label="Pareto frontier"))
legend_elements.append(Line2D([0], [0], color=C_TARGET, linestyle="--",
                              linewidth=1.0, label="30 FPS target"))

ax3.legend(handles=legend_elements, loc="center left",
           bbox_to_anchor=(1.02, 0.5), framealpha=0.9)
ax3.set_xlabel("End-to-end throughput (FPS)")
ax3.set_ylabel("Mean balanced accuracy")
ax3.set_title("Pipeline Throughput vs. Prediction Accuracy on Jetson Orin Nano")
ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

fig3.tight_layout()
fig3.savefig(FIGURES / "rq2_pipeline_fps_vs_accuracy.png")
print(f"[3/6] Saved: rq2_pipeline_fps_vs_accuracy")
plt.close(fig3)


# ===================================================================
# FIGURE 4 — Latency breakdown (stacked horizontal bar)
# ===================================================================
pipe_sorted = pipe_df.sort_values("total_fps", ascending=True).copy()

def make_label(row):
    return f"{row['perception']}  +  {row['gru_short']} (d={row['input_dim']})"

pipe_sorted["label"] = pipe_sorted.apply(make_label, axis=1)

fig4, ax4 = plt.subplots(figsize=(8, 10))
y_pos = np.arange(len(pipe_sorted))
bar_h = 0.7

perc_ms = pipe_sorted["perception_mean_ms"].values
feat_ms = pipe_sorted["features_mean_ms"].values
gru_ms  = pipe_sorted["gru_mean_ms"].values

ax4.barh(y_pos, perc_ms, bar_h, label="Perception",
         color="#4878CF", edgecolor="white", linewidth=0.3)
ax4.barh(y_pos, feat_ms, bar_h, left=perc_ms, label="Feature extraction",
         color="#6ACC65", edgecolor="white", linewidth=0.3)
ax4.barh(y_pos, gru_ms, bar_h, left=perc_ms + feat_ms, label="GRU inference",
         color="#D65F5F", edgecolor="white", linewidth=0.3)

ax4.axvline(x=33.33, color=C_TARGET, linestyle="--", linewidth=1.0,
            alpha=0.8, label="33.3 ms (30 FPS)")

for i, (_, row) in enumerate(pipe_sorted.iterrows()):
    fps = row["total_fps"]
    colour = "#2CA02C" if fps >= 30 else "#D65F5F"
    ax4.text(row["total_mean_ms"] + 0.8, i, f"{fps:.0f} FPS",
             va="center", ha="left", fontsize=7, color=colour, weight="bold")

ax4.set_yticks(y_pos)
ax4.set_yticklabels(pipe_sorted["label"].values, fontsize=7.5)
ax4.set_xlabel("Latency per frame (ms)")
ax4.set_title("End-to-End Latency Breakdown by Pipeline Configuration")
ax4.legend(loc="lower right", framealpha=0.9)
ax4.set_xlim(0, max(pipe_sorted["total_mean_ms"]) * 1.15)

fig4.tight_layout()
fig4.savefig(FIGURES / "rq2_latency_breakdown.png")
print(f"[4/6] Saved: rq2_latency_breakdown")
plt.close(fig4)


# ===================================================================
# FIGURE 5 — Pipeline latency percentiles (p50, p95, p99)
#   Grouped bar chart for all configs, excluding MediaPipe to fit
# ===================================================================
pipe_no_mp = pipe_df[~pipe_df["is_mp"]].sort_values("total_fps", ascending=True).copy()
pipe_no_mp["label"] = pipe_no_mp.apply(make_label, axis=1)

fig5, ax5 = plt.subplots(figsize=(8, 9))

y5 = np.arange(len(pipe_no_mp))
bh = 0.25

p50_vals  = pipe_no_mp["total_p50_ms"].values
p95_vals  = pipe_no_mp["total_p95_ms"].values
p99_vals  = pipe_no_mp["total_p99_ms"].values
mean_vals = pipe_no_mp["total_mean_ms"].values
std_vals  = pipe_no_mp["total_std_ms"].values

ax5.barh(y5 + bh, p99_vals, bh, label="p99", color=C_P99,
         edgecolor="white", linewidth=0.3)
ax5.barh(y5, p95_vals, bh, label="p95", color=C_P95,
         edgecolor="white", linewidth=0.3)
ax5.barh(y5 - bh, p50_vals, bh, label="p50 (median)", color=C_P50,
         edgecolor="white", linewidth=0.3)

# Mean as a marker with std error bar
ax5.errorbar(mean_vals, y5, xerr=std_vals, fmt="d", color="black",
             markersize=3.5, capsize=2, capthick=0.6, linewidth=0.6,
             label="Mean ± std", zorder=5)

ax5.axvline(x=33.33, color=C_TARGET, linestyle="--", linewidth=1.0,
            alpha=0.6, label="33.3 ms (30 FPS)")

ax5.set_yticks(y5)
ax5.set_yticklabels(pipe_no_mp["label"].values, fontsize=7)
ax5.set_xlabel("Latency per frame (ms)")
ax5.set_title("Pipeline Latency Percentiles (Excluding MediaPipe Two-Stage)")
ax5.legend(loc="lower right", framealpha=0.9)
ax5.set_xlim(0, max(p99_vals) * 1.12)

fig5.tight_layout()
fig5.savefig(FIGURES / "rq2_pipeline_percentiles.png")
print(f"[5/6] Saved: rq2_pipeline_percentiles")
plt.close(fig5)


# ===================================================================
# FIGURE 6 — GRU inference overhead by input dimension
#   Shows that GRU latency is ~1 ms regardless of input dim (4 to 34)
# ===================================================================
fig6, ax6 = plt.subplots(figsize=(5.5, 3.5))

# One data point per pipeline config
for gru_name, colour in TIER_COLOURS.items():
    subset = pipe_df[pipe_df["gru_short"] == gru_name]
    dims   = subset["input_dim"].values
    gru_t  = subset["gru_mean_ms"].values
    # Jitter x slightly to avoid overlap
    jitter = np.random.default_rng(42).uniform(-0.3, 0.3, len(dims))
    ax6.scatter(dims + jitter, gru_t, c=colour, s=40, alpha=0.7,
                edgecolors="white", linewidth=0.3, label=gru_name, zorder=3)

# Horizontal reference lines
overall_mean = pipe_df["gru_mean_ms"].mean()
overall_max  = pipe_df["gru_mean_ms"].max()
overall_min  = pipe_df["gru_mean_ms"].min()
ax6.axhline(y=overall_mean, color="black", linestyle="-", linewidth=0.8,
            alpha=0.5, zorder=2)
ax6.annotate(f"Overall mean: {overall_mean:.2f} ms",
             xy=(35, overall_mean), ha="right", va="bottom",
             fontsize=8, color="#333333")

# Also show what fraction of total pipeline the GRU is
avg_frac = (pipe_df["gru_mean_ms"] / pipe_df["total_mean_ms"]).mean() * 100
ax6.annotate(f"GRU = {avg_frac:.1f}% of total pipeline latency (avg)",
             xy=(0.5, 0.03), xycoords="axes fraction", ha="center",
             fontsize=8, color="#666666", style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                       edgecolor="#cccccc", alpha=0.8))

ax6.set_xlabel("GRU input dimension (features per frame)")
ax6.set_ylabel("GRU inference time (ms)")
ax6.set_title("GRU Inference Overhead by Feature Complexity")
ax6.set_xticks([4, 6, 7, 18, 34])
ax6.set_xticklabels(["4\n(Core-4)", "6\n(Traj-6)", "7\n(Head angle)",
                      "18\n(Torso+head)", "34\n(Full body)"])

# De-duplicate legend entries
handles, labels = ax6.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax6.legend(by_label.values(), by_label.keys(), loc="upper left",
           framealpha=0.9)

ax6.set_ylim(0, max(pipe_df["gru_mean_ms"]) * 1.6)

fig6.tight_layout()
fig6.savefig(FIGURES / "rq2_gru_overhead.png")
print(f"[6/6] Saved: rq2_gru_overhead")
plt.close(fig6)

print("\nAll 6 figures generated successfully.")
