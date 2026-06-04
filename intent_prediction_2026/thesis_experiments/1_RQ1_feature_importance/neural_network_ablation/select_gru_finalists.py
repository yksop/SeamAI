#!/usr/bin/env python3
"""
Stage 2 — Select GRU finalists for RQ2 edge deployment.

The handoff from RQ1 (feature importance) to RQ2 (edge efficiency): it reads the
structured ablation results from train_ablation.py and selects one GRU model per
complexity tier.

Why GRU?
    GRU and CNN tied for highest mean balanced accuracy (0.874) across all
    experiments, but GRU is the natural choice for sequential trajectory
    data and is the standard architecture in pedestrian prediction literature.
    Fixing the architecture isolates the effect of *feature complexity* on
    edge inference latency, which is the core question of RQ2.

Why tiers?
    RQ2 asks: how much visual information can we extract and still run at
    30 FPS on edge hardware?  To answer this, we need models spanning the
    full input-dimension range.  Complexity tiers ensure one finalist per
    level, from minimal (4 features) to maximum (34 features).

Selection criterion:
    robust_score = mean_bal_acc - 0.5 * mean_cv_std - 0.5 * std_across_tte

    This balances raw accuracy with cross-validation stability and temporal
    consistency across TTE horizons.

Outputs (all under results/):
    results/gru_finalists.csv           — the 5 finalists (consumed by RQ2)
    results/gru_all_scored.csv          — all GRU configs ranked
    results/gru_finalists_per_tte.csv   — per-TTE breakdown
    results/figures/gru_finalists.png    — visual summary
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from ablation_config import FIG_DIR, TABLE_DIR, TTE_VALUES, style

# ---------------------------------------------------------------------------
# Paths — reads the ablation results written by train_ablation.py
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
RESULTS_CSV = HERE / "results" / "results_all.csv"

# ---------------------------------------------------------------------------
# SETTINGS — selection weights (edit these)
# ---------------------------------------------------------------------------
ALPHA = 0.5          # penalty for within-TTE CV instability
BETA = 0.5           # penalty for across-TTE sensitivity
ARCHITECTURE = "gru"

# Tier 5 exclusion: HC-30 uses single-frame aggregated features (seq_len=1),
# which is a fundamentally different inference paradigm from the real-time
# per-frame pipeline that RQ2 benchmarks.
TIER5_EXCLUDED = {"A1_hc30"}

COMPLEXITY_TIERS = [
    ("Tier 1: RF-minimal", 1, 4,
     "RF backward-elimination survivors (Core-3, Core-4). "
     "Tests whether the minimal feature set suffices for edge deployment."),
    ("Tier 2: Trajectory baseline", 5, 6,
     "Full 6-feature per-frame trajectory baseline. "
     "Reference: no pose, no RF reduction."),
    ("Tier 3: Compact + pose", 7, 10,
     "Trajectory + targeted pose (e.g., head angle or head raw). "
     "Tests whether a single pose signal justifies the added edge cost."),
    ("Tier 4: Medium pose", 11, 20,
     "Extended pose: torso + head landmarks or full-body angles. "
     "Intermediate complexity level."),
    ("Tier 5: Full body (sequential only)", 21, 100,
     "Maximum-complexity sequential model (e.g., full-body raw, 34D). "
     "Stress test for edge hardware. Excludes HC-30 (single-frame)."),
]

TIER_COLORS = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd", "#d62728"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    style()

    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} not found.")
        print("Run train_ablation.py first to generate structured ablation results.")
        return

    df = pd.read_csv(RESULTS_CSV)
    gru_df = df[df["model"] == ARCHITECTURE].copy()
    print(f"Loaded {len(gru_df)} GRU rows "
          f"({gru_df['experiment_key'].nunique()} experiments)")

    # --- Score each GRU configuration ---
    scored_rows = []
    for exp_key, grp in gru_df.groupby("experiment_key"):
        grp = grp.sort_values("tte_s")
        input_dim = int(grp["input_dim"].iloc[0])
        label = grp["label"].iloc[0] if "label" in grp.columns else exp_key
        feat_list = grp["feature_list"].iloc[0] if "feature_list" in grp.columns else ""

        mean_acc = grp["bal_acc"].mean()
        std_tte = grp["bal_acc"].std()
        mean_cv_std = grp["bal_acc_std"].mean()
        max_cv_std = grp["bal_acc_std"].max()
        min_acc = grp["bal_acc"].min()
        max_acc = grp["bal_acc"].max()
        robust = mean_acc - ALPHA * mean_cv_std - BETA * std_tte

        # Assign tier
        tier_label, tier_idx = "Unassigned", -1
        for i, (tlabel, lo, hi, _) in enumerate(COMPLEXITY_TIERS):
            if lo <= input_dim <= hi:
                tier_label, tier_idx = tlabel, i
                break

        scored_rows.append({
            "experiment_key": exp_key,
            "label": label,
            "input_dim": input_dim,
            "feature_list": feat_list,
            "tier": tier_label,
            "tier_idx": tier_idx,
            "mean_bal_acc": round(mean_acc, 4),
            "std_across_tte": round(std_tte, 4),
            "mean_cv_std": round(mean_cv_std, 4),
            "max_cv_std": round(max_cv_std, 4),
            "min_bal_acc": round(min_acc, 4),
            "max_bal_acc": round(max_acc, 4),
            "robust_score": round(robust, 4),
        })

    scored = pd.DataFrame(scored_rows).sort_values(
        "robust_score", ascending=False
    ).reset_index(drop=True)

    scored.to_csv(TABLE_DIR / "gru_all_scored.csv", index=False)

    # --- Select best per complexity tier ---
    finalists = []
    print(f"\n{'=' * 72}")
    print(f"  GRU FINALIST SELECTION FOR RQ2 EDGE DEPLOYMENT")
    print(f"  Score = mean_bal_acc - {ALPHA} * mean_cv_std - {BETA} * std_across_tte")
    print(f"{'=' * 72}")

    for i, (tier_label, lo, hi, rationale) in enumerate(COMPLEXITY_TIERS):
        tier_df = scored[scored["tier_idx"] == i].copy()

        # Tier 5: exclude single-frame aggregated inputs
        if TIER5_EXCLUDED and "sequential only" in tier_label.lower():
            tier_df = tier_df[~tier_df["experiment_key"].isin(TIER5_EXCLUDED)]

        print(f"\n  {tier_label}  (dim {lo}-{hi})")
        print(f"  Rationale: {rationale}")

        if tier_df.empty:
            print(f"    No GRU configurations in this tier.")
            continue

        for _, row in tier_df.iterrows():
            print(f"    {row['experiment_key']:30s}  dim={row['input_dim']:2d}  "
                  f"acc={row['mean_bal_acc']:.4f}  "
                  f"robust={row['robust_score']:.4f}")

        winner = tier_df.iloc[0]
        winner_dict = winner.to_dict()
        winner_dict["tier_rationale"] = rationale
        winner_dict["selection_reason"] = (
            f"Best robust score ({winner['robust_score']:.4f}) "
            f"among {len(tier_df)} GRU configs in {tier_label}"
        )
        finalists.append(winner_dict)
        print(f"  -> SELECTED: {winner['experiment_key']}")

    out = pd.DataFrame(finalists)
    out.to_csv(TABLE_DIR / "gru_finalists.csv", index=False)

    # --- Per-TTE table ---
    tte_rows = []
    for _, fin in out.iterrows():
        sub = gru_df[
            gru_df["experiment_key"] == fin["experiment_key"]
        ].sort_values("tte_s")
        for _, r in sub.iterrows():
            tte_rows.append({
                "experiment_key": fin["experiment_key"],
                "label": fin["label"],
                "input_dim": int(fin["input_dim"]),
                "tier": fin["tier"],
                "tte_s": r["tte_s"],
                "bal_acc": round(r["bal_acc"], 4),
                "bal_acc_std": round(r["bal_acc_std"], 4),
                "roc_auc": round(r.get("roc_auc", 0), 4),
                "f1": round(r.get("f1", 0), 4),
            })

    tte_df = pd.DataFrame(tte_rows)
    tte_df.to_csv(TABLE_DIR / "gru_finalists_per_tte.csv", index=False)

    # --- Figure: two-panel audit ---
    fig, (ax_bar, ax_tte) = plt.subplots(1, 2, figsize=(15, 5))

    # Left: bar chart of finalists
    bar_labels = [
        f"{r['label']}  (dim={int(r['input_dim'])})"
        for _, r in out.iterrows()
    ]
    colors = [TIER_COLORS[int(r["tier_idx"])] for _, r in out.iterrows()]
    y_pos = range(len(out))

    ax_bar.barh(
        y_pos, out["mean_bal_acc"], xerr=out["mean_cv_std"],
        color=colors, capsize=4, edgecolor="white", linewidth=0.6,
    )
    for yp, (_, r) in zip(y_pos, out.iterrows()):
        ax_bar.text(
            r["mean_bal_acc"] + r["mean_cv_std"] + 0.003, yp,
            f"robust={r['robust_score']:.3f}",
            va="center", fontsize=8, color="#555",
        )
    ax_bar.set_yticks(list(y_pos))
    ax_bar.set_yticklabels(bar_labels, fontsize=10)
    ax_bar.invert_yaxis()
    lo = max(0, (out["mean_bal_acc"] - out["mean_cv_std"]).min() - 0.04)
    hi = min(1, (out["mean_bal_acc"] + out["mean_cv_std"]).max() + 0.08)
    ax_bar.set_xlim(lo, hi)
    ax_bar.set_xlabel("Mean Balanced Accuracy")
    ax_bar.set_title("RQ2 GRU Finalists by Complexity Tier")

    # Right: accuracy over TTE
    for _, fin_row in out.iterrows():
        sub = tte_df[
            tte_df["experiment_key"] == fin_row["experiment_key"]
        ].sort_values("tte_s")
        if sub.empty:
            continue
        tidx = int(fin_row["tier_idx"])
        ax_tte.errorbar(
            sub["tte_s"], sub["bal_acc"], yerr=sub["bal_acc_std"],
            marker="o", lw=2, capsize=4, color=TIER_COLORS[tidx],
            label=f"{fin_row['label']} (dim={int(fin_row['input_dim'])})",
        )

    ax_tte.set_xlabel("Time to Event (s)")
    ax_tte.set_ylabel("Balanced Accuracy")
    ax_tte.set_xticks(TTE_VALUES)
    ax_tte.set_title("Finalist Performance over TTE")
    ax_tte.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "gru_finalists.png")
    plt.close()

    # --- Console summary ---
    print(f"\n{'=' * 72}")
    print(f"  SELECTED {len(finalists)} GRU FINALISTS FOR RQ2")
    print(f"{'=' * 72}")
    for i, (_, r) in enumerate(out.iterrows(), 1):
        print(f"\n  #{i}  {r['label']}  [{r['experiment_key']}]")
        print(f"       dim={int(r['input_dim'])}  tier={r['tier']}")
        print(f"       mean_acc={r['mean_bal_acc']:.4f}  "
              f"robust={r['robust_score']:.4f}")
        print(f"       features: {r['feature_list'][:70]}...")

    print(f"\n  Outputs:")
    print(f"    {TABLE_DIR / 'gru_finalists.csv'}")
    print(f"    {TABLE_DIR / 'gru_finalists_per_tte.csv'}")
    print(f"    {TABLE_DIR / 'gru_all_scored.csv'}")
    print(f"    {FIG_DIR / 'gru_finalists.png'}")
    print()
    print("  These finalists feed into 2_RQ2_RQ3_edge_deployment/")
    print("  Run step1_select_finalists.py there to continue.")


if __name__ == "__main__":
    main()
