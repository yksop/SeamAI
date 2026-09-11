#!/usr/bin/env python3
"""
plot_ablation_figures.py — Stage 3: generate the RQ1 ablation figures.

Reads the structured ablation results (results/results_all.csv) and writes the
comparison figures to results/figures/.

Figures produced:
  1. Group A: Trajectory representation comparison (HC-30, Traj-6, Core-3, Core-4)
  2. Group B: Effect of adding pose to Traj-6 (angle vs raw, 3 body regions)
  3. Group C: Core features + pose combinations
  4. Model comparison (MLP vs CNN vs GRU) across all experiments
  5. Angle vs raw comparison (Group B pairs)
  6. TTE sensitivity: accuracy over prediction horizons
  7. Final ranking: all 14 experiments × 3 models, sorted
  8. Heatmap: experiment × TTE balanced accuracy (GRU only)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "results_all.csv"
FIG_DIR = HERE / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

# Colours
MODEL_COLORS = {"mlp": "#E07B54", "gru": "#4878CF", "cnn": "#6ACC65"}
MODEL_LABELS = {"mlp": "MLP", "gru": "GRU", "cnn": "CNN"}

GROUP_COLORS = {
    "A": "#4878CF",
    "B": "#6ACC65",
    "C": "#D65F5F",
}

# Short labels for readability
SHORT_LABELS = {
    "HC-30 (LogReg features)": "HC-30",
    "Trajectory base (6 per-frame)": "Traj-6",
    "Core-3 (RF survival: dist, closure, rel_angle)": "Core-3",
    "Core-4 (RF survival: dist, closure, vy, rel_angle)": "Core-4",
    "Traj-6 + head (angle)": "Head (angle)",
    "Traj-6 + head (raw)": "Head (raw)",
    "Traj-6 + torso_head (angle)": "Torso+head (angle)",
    "Traj-6 + torso_head (raw)": "Torso+head (raw)",
    "Traj-6 + full_body (angle)": "Full body (angle)",
    "Traj-6 + full_body (raw)": "Full body (raw)",
    "Core-3 + head (raw)": "C3 + head (raw)",
    "Core-3 + torso_head (raw)": "C3 + torso+head (raw)",
    "Core-4 + head (raw)": "C4 + head (raw)",
    "Core-4 + torso_head (raw)": "C4 + torso+head (raw)",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(RESULTS)
df["short_label"] = df["label"].map(SHORT_LABELS).fillna(df["label"])

TTE_VALUES = sorted(df["tte_s"].unique())
REFERENCE_TTE = 1.5

print(f"Loaded {len(df)} rows: {df['experiment_key'].nunique()} experiments, "
      f"{df['model'].nunique()} models, {len(TTE_VALUES)} TTE values\n")


# ---------------------------------------------------------------------------
# Helper: compute mean balanced accuracy across TTE for a subset
# ---------------------------------------------------------------------------
def mean_acc_across_tte(sub):
    """Mean bal_acc across TTE values for groupby results."""
    return sub.groupby("tte_s")["bal_acc"].mean().mean()


# ===================================================================
# FIGURE 1 — Group A: Trajectory representations
# ===================================================================
group_a = df[df["group"] == "A"]
exps_a = ["A2_traj6", "A4_core4", "A3_core3", "A1_hc30"]
labels_a = ["Traj-6\n(sequential)", "Core-4\n(sequential)", "Core-3\n(sequential)", "HC-30\n(aggregated)"]

fig1, ax1 = plt.subplots(figsize=(8, 4))
x = np.arange(len(exps_a))
w = 0.25

for i, model in enumerate(["gru", "cnn", "mlp"]):
    means = []
    stds = []
    for ek in exps_a:
        sub = group_a[(group_a["experiment_key"] == ek) & (group_a["model"] == model)]
        # Mean across all TTE values
        means.append(sub["bal_acc"].mean())
        stds.append(sub["bal_acc_std"].mean())
    ax1.bar(x + (i - 1) * w, means, w, label=MODEL_LABELS[model],
            color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5)
    ax1.errorbar(x + (i - 1) * w, means, yerr=stds, fmt="none",
                 ecolor="black", capsize=3, capthick=0.6, linewidth=0.6)

ax1.set_xticks(x)
ax1.set_xticklabels(labels_a)
ax1.set_ylabel("Mean balanced accuracy (across TTE)")
ax1.set_title("Group A: Trajectory Representation Comparison")
ax1.legend()
ax1.set_ylim(0.77, 0.94)

fig1.tight_layout()
fig1.savefig(FIG_DIR / "01_group_a_trajectory.png")
print("[1/8] Group A trajectory comparison")
plt.close(fig1)


# ===================================================================
# FIGURE 2 — Group B: Adding pose to Traj-6
# ===================================================================
group_b = df[df["group"] == "B"]
# Include Traj-6 baseline for comparison
traj6 = df[df["experiment_key"] == "A2_traj6"]
group_b_with_base = pd.concat([traj6, group_b])

exps_b = ["A2_traj6", "B1_head_angle", "B2_head_raw",
          "B3_torso_head_angle", "B4_torso_head_raw",
          "B5_full_body_angle", "B6_full_body_raw"]
labels_b = ["Traj-6\n(baseline)", "Head\n(angle)", "Head\n(raw)",
            "Torso+head\n(angle)", "Torso+head\n(raw)",
            "Full body\n(angle)", "Full body\n(raw)"]

fig2, ax2 = plt.subplots(figsize=(9, 4))
x2 = np.arange(len(exps_b))

for i, model in enumerate(["gru", "cnn", "mlp"]):
    means = []
    stds = []
    for ek in exps_b:
        sub = group_b_with_base[(group_b_with_base["experiment_key"] == ek) &
                                (group_b_with_base["model"] == model)]
        means.append(sub["bal_acc"].mean())
        stds.append(sub["bal_acc_std"].mean())
    ax2.bar(x2 + (i - 1) * w, means, w, label=MODEL_LABELS[model],
            color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5)
    ax2.errorbar(x2 + (i - 1) * w, means, yerr=stds, fmt="none",
                 ecolor="black", capsize=3, capthick=0.6, linewidth=0.6)

# Separate baseline visually
ax2.axvline(x=0.55, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
ax2.text(0.55, ax2.get_ylim()[1], "  + pose features →", fontsize=8,
         color="grey", va="top")

ax2.set_xticks(x2)
ax2.set_xticklabels(labels_b, fontsize=9)
ax2.set_ylabel("Mean balanced accuracy (across TTE)")
ax2.set_title("Group B: Effect of Adding Pose Features to Traj-6 Baseline")
ax2.legend()
ax2.set_ylim(0.77, 0.94)

fig2.tight_layout()
fig2.savefig(FIG_DIR / "02_group_b_pose_effect.png")
print("[2/8] Group B pose effect")
plt.close(fig2)


# ===================================================================
# FIGURE 3 — Group C: Core features + pose
# ===================================================================
group_c = df[df["group"] == "C"]
core_baselines = df[df["experiment_key"].isin(["A3_core3", "A4_core4"])]
group_c_with_base = pd.concat([core_baselines, group_c])

exps_c = ["A3_core3", "C1_core3_head_raw", "C2_core3_torso_head_raw",
          "A4_core4", "C3_core4_head_raw", "C4_core4_torso_head_raw"]
labels_c = ["Core-3", "C3+head\n(raw)", "C3+torso\n+head (raw)",
            "Core-4", "C4+head\n(raw)", "C4+torso\n+head (raw)"]

fig3, ax3 = plt.subplots(figsize=(8, 4))
x3 = np.arange(len(exps_c))

for i, model in enumerate(["gru", "cnn", "mlp"]):
    means = []
    stds = []
    for ek in exps_c:
        sub = group_c_with_base[(group_c_with_base["experiment_key"] == ek) &
                                (group_c_with_base["model"] == model)]
        means.append(sub["bal_acc"].mean())
        stds.append(sub["bal_acc_std"].mean())
    ax3.bar(x3 + (i - 1) * w, means, w, label=MODEL_LABELS[model],
            color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5)
    ax3.errorbar(x3 + (i - 1) * w, means, yerr=stds, fmt="none",
                 ecolor="black", capsize=3, capthick=0.6, linewidth=0.6)

# Separate Core-3 from Core-4 block
ax3.axvline(x=2.55, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)

ax3.set_xticks(x3)
ax3.set_xticklabels(labels_c, fontsize=9)
ax3.set_ylabel("Mean balanced accuracy (across TTE)")
ax3.set_title("Group C: Core Feature Sets + Pose Augmentation")
ax3.legend()
ax3.set_ylim(0.82, 0.94)

fig3.tight_layout()
fig3.savefig(FIG_DIR / "03_group_c_core_pose.png")
print("[3/8] Group C core + pose")
plt.close(fig3)


# ===================================================================
# FIGURE 4 — Model comparison across all experiments
# ===================================================================
# For each model, average bal_acc across TTE per experiment
model_summary = (df.groupby(["experiment_key", "model", "short_label", "input_dim"])
                   ["bal_acc"].mean().reset_index())

fig4, ax4 = plt.subplots(figsize=(9, 5))

# Sort experiments by GRU performance
gru_order = (model_summary[model_summary["model"] == "gru"]
             .sort_values("bal_acc", ascending=True)["experiment_key"].values)

y4 = np.arange(len(gru_order))
bh = 0.25

for i, model in enumerate(["gru", "cnn", "mlp"]):
    vals = []
    for ek in gru_order:
        sub = model_summary[(model_summary["experiment_key"] == ek) &
                            (model_summary["model"] == model)]
        vals.append(sub["bal_acc"].values[0] if len(sub) > 0 else 0)
    ax4.barh(y4 + (i - 1) * bh, vals, bh, label=MODEL_LABELS[model],
             color=MODEL_COLORS[model], edgecolor="white", linewidth=0.3)

# Labels
ylabels = []
for ek in gru_order:
    row = model_summary[model_summary["experiment_key"] == ek].iloc[0]
    ylabels.append(f"{row['short_label']} (d={row['input_dim']})")

ax4.set_yticks(y4)
ax4.set_yticklabels(ylabels, fontsize=9)
ax4.set_xlabel("Mean balanced accuracy (across TTE)")
ax4.set_title("Model Comparison Across All Feature Configurations")
ax4.legend(loc="lower right")

fig4.tight_layout()
fig4.savefig(FIG_DIR / "04_model_comparison.png")
print("[4/8] Model comparison")
plt.close(fig4)


# ===================================================================
# FIGURE 5 — Angle vs Raw (Group B pairs)
# ===================================================================
pairs = [
    ("B1_head_angle", "B2_head_raw", "Head"),
    ("B3_torso_head_angle", "B4_torso_head_raw", "Torso+head"),
    ("B5_full_body_angle", "B6_full_body_raw", "Full body"),
]

fig5, axes5 = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)

for ax, (ek_angle, ek_raw, region_name) in zip(axes5, pairs):
    for model in ["gru", "cnn", "mlp"]:
        angle_data = df[(df["experiment_key"] == ek_angle) & (df["model"] == model)]
        raw_data = df[(df["experiment_key"] == ek_raw) & (df["model"] == model)]

        ax.plot(angle_data["tte_s"], angle_data["bal_acc"], "o-",
                color=MODEL_COLORS[model], label=f"{MODEL_LABELS[model]} (angle)",
                markersize=5, linewidth=1.5)
        ax.plot(raw_data["tte_s"], raw_data["bal_acc"], "s--",
                color=MODEL_COLORS[model], label=f"{MODEL_LABELS[model]} (raw)",
                markersize=5, linewidth=1.5, alpha=0.7)

    ax.set_xlabel("TTE (s)")
    ax.set_title(region_name)
    #ax.invert_xaxis()

axes5[0].set_ylabel("Balanced accuracy")
axes5[0].legend(fontsize=7, loc="lower left")
fig5.suptitle("Angle vs Raw Pose Representation by Body Region", fontsize=13, y=1.02)

fig5.tight_layout()
fig5.savefig(FIG_DIR / "05_angle_vs_raw.png")
print("[5/8] Angle vs raw")
plt.close(fig5)


# ===================================================================
# FIGURE 6 — TTE sensitivity: accuracy over prediction horizons
# ===================================================================
# Show top experiments (GRU only) across TTE values
gru_df = df[df["model"] == "gru"]

# Pick representative experiments from each group
tte_exps = ["A2_traj6", "A4_core4", "B1_head_angle",
            "B4_torso_head_raw", "B6_full_body_raw"]
tte_colors = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66"]

fig6, ax6 = plt.subplots(figsize=(7, 4))

for ek, color in zip(tte_exps, tte_colors):
    sub = gru_df[gru_df["experiment_key"] == ek].sort_values("tte_s")
    if sub.empty:
        continue   # this experiment isn't in the current run (e.g. a partial ablation)
    short = SHORT_LABELS.get(sub["label"].iloc[0], ek)
    ax6.plot(sub["tte_s"], sub["bal_acc"], "o-", color=color,
             label=f"{short} (d={sub['input_dim'].iloc[0]})",
             markersize=6, linewidth=1.5)
    ax6.fill_between(sub["tte_s"],
                     sub["bal_acc"] - sub["bal_acc_std"],
                     sub["bal_acc"] + sub["bal_acc_std"],
                     alpha=0.15, color=color)

ax6.set_xlabel("Time to event (s)")
ax6.set_ylabel("Balanced accuracy")
ax6.set_title("GRU Performance Across Prediction Horizons")
ax6.legend(loc="lower right", fontsize=8)

fig6.tight_layout()
fig6.savefig(FIG_DIR / "06_tte_sensitivity.png")
print("[6/8] TTE sensitivity")
plt.close(fig6)


# ===================================================================
# FIGURE 7 — Final ranking: all experiments at reference TTE
# ===================================================================
ref = df[df["tte_s"] == REFERENCE_TTE].copy()

fig7, ax7 = plt.subplots(figsize=(9, 6))

# Rank by GRU bal_acc at reference TTE
gru_ref = ref[ref["model"] == "gru"].sort_values("bal_acc", ascending=True)
exp_order = gru_ref["experiment_key"].values

y7 = np.arange(len(exp_order))
bh = 0.25

for i, model in enumerate(["gru", "cnn", "mlp"]):
    vals = []
    errs = []
    for ek in exp_order:
        sub = ref[(ref["experiment_key"] == ek) & (ref["model"] == model)]
        vals.append(sub["bal_acc"].values[0] if len(sub) > 0 else 0)
        errs.append(sub["bal_acc_std"].values[0] if len(sub) > 0 else 0)
    bars = ax7.barh(y7 + (i - 1) * bh, vals, bh, label=MODEL_LABELS[model],
                    color=MODEL_COLORS[model], edgecolor="white", linewidth=0.3,
                    xerr=errs, capsize=2, error_kw={"linewidth": 0.6})

ylabels7 = []
for ek in exp_order:
    row = ref[ref["experiment_key"] == ek].iloc[0]
    ylabels7.append(f"{SHORT_LABELS.get(row['label'], row['label'])} (d={row['input_dim']})")

ax7.set_yticks(y7)
ax7.set_yticklabels(ylabels7, fontsize=9)
ax7.set_xlabel("Balanced accuracy")
ax7.set_title(f"Ranking at TTE = {REFERENCE_TTE}s (with CV std)")
ax7.legend(loc="lower right")

fig7.tight_layout()
fig7.savefig(FIG_DIR / "07_ranking_tte15.png")
print("[7/8] Final ranking")
plt.close(fig7)


# ===================================================================
# FIGURE 8 — Heatmap: GRU bal_acc by experiment × TTE
# ===================================================================
gru_pivot = gru_df.pivot_table(index="experiment_key", columns="tte_s",
                                values="bal_acc", aggfunc="first")

# Sort by mean accuracy
gru_pivot["mean"] = gru_pivot.mean(axis=1)
gru_pivot = gru_pivot.sort_values("mean", ascending=True)
gru_pivot = gru_pivot.drop(columns="mean")

# Map to short labels
idx_labels = []
for ek in gru_pivot.index:
    row = gru_df[gru_df["experiment_key"] == ek].iloc[0]
    idx_labels.append(f"{SHORT_LABELS.get(row['label'], row['label'])} (d={row['input_dim']})")

fig8, ax8 = plt.subplots(figsize=(7, 5))
im = ax8.imshow(gru_pivot.values, aspect="auto", cmap="RdYlGn",
                vmin=gru_pivot.values.min() - 0.01,
                vmax=gru_pivot.values.max() + 0.01)

# Labels
ax8.set_xticks(range(len(gru_pivot.columns)))
ax8.set_xticklabels([f"{t:.1f}s" for t in gru_pivot.columns])
ax8.set_yticks(range(len(idx_labels)))
ax8.set_yticklabels(idx_labels, fontsize=9)
ax8.set_xlabel("Time to event (s)")
ax8.set_title("GRU Balanced Accuracy: Experiment × TTE")

# Annotate cells
for i in range(len(gru_pivot)):
    for j in range(len(gru_pivot.columns)):
        val = gru_pivot.values[i, j]
        ax8.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8,
                 color="black" if 0.78 < val < 0.95 else "white")

plt.colorbar(im, ax=ax8, shrink=0.8, label="Balanced accuracy")

fig8.tight_layout()
fig8.savefig(FIG_DIR / "08_heatmap_gru.png")
print("[8/8] GRU heatmap")
plt.close(fig8)


# ===================================================================
# Summary table
# ===================================================================
print("\n" + "=" * 70)
print("SUMMARY: GRU mean balanced accuracy across TTE (sorted)")
print("=" * 70)
summary = (gru_df.groupby(["experiment_key", "short_label", "input_dim"])
                 .agg(mean_acc=("bal_acc", "mean"),
                      mean_std=("bal_acc_std", "mean"))
                 .reset_index()
                 .sort_values("mean_acc", ascending=False))
for _, r in summary.iterrows():
    print(f"  {r['short_label']:25s}  dim={r['input_dim']:2.0f}  "
          f"acc={r['mean_acc']:.4f} ± {r['mean_std']:.4f}")

print(f"\nAll 8 figures saved to: {FIG_DIR}/")
