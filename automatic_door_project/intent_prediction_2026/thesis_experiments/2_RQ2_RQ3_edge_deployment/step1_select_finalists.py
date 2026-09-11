#!/usr/bin/env python3
"""
Step 1 — Confirm and load RQ2 GRU finalists.

Reads the finalist selection from the RQ1 analysis pipeline
(select_gru_finalists.py) and produces a local copy
with any RQ2-specific annotations (tier rationale, perception requirements).

If select_gru_finalists has not been run yet, this script falls back to reading
results/results_all.csv and re-running the selection logic locally.

Output:
    outputs/results/finalists.csv
    outputs/results/finalists_per_tte.csv
    outputs/figures/finalists_overview.png
    outputs/figures/finalists_over_tte.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    RQ1_FINALISTS_CSV, RQ1_RESULTS_CSV,
    RESULTS_DIR, FIG_DIR,
    TTE_VALUES, COMPLEXITY_TIERS, TIER_COLORS, TIER_NEEDS_POSE,
    style,
)

# Selection parameters (same as select_gru_finalists)
ALPHA = 0.5
BETA = 0.5
TIER5_EXCLUDED = {"A1_hc30"}


def load_finalists_from_rq1():
    """Load pre-computed finalists from select_gru_finalists."""
    if RQ1_FINALISTS_CSV.exists():
        df = pd.read_csv(RQ1_FINALISTS_CSV)
        print(f"  Loaded {len(df)} finalists from RQ1 (select_gru_finalists)")
        return df
    return None


def select_finalists_locally():
    """Re-run selection logic if select_gru_finalists output is not available."""
    print(f"  WARNING: {RQ1_FINALISTS_CSV} not found.")
    print(f"  Re-running selection from {RQ1_RESULTS_CSV}")

    if not RQ1_RESULTS_CSV.exists():
        print(f"  ERROR: {RQ1_RESULTS_CSV} also not found!")
        print(f"  Run train_ablation.py and select_gru_finalists.py first.")
        return None

    df = pd.read_csv(RQ1_RESULTS_CSV)
    gru = df[df["model"] == "gru"].copy()

    scored = []
    for key, g in gru.groupby("experiment_key"):
        dim = int(g["input_dim"].iloc[0])
        label = g["label"].iloc[0] if "label" in g.columns else key
        feat_list = g["feature_list"].iloc[0] if "feature_list" in g.columns else ""

        mean_acc = g["bal_acc"].mean()
        std_tte = g["bal_acc"].std()
        mean_cv = g["bal_acc_std"].mean()
        robust = mean_acc - ALPHA * mean_cv - BETA * std_tte

        tier_label, tier_idx = "Unassigned", -1
        for i, (tl, lo, hi) in enumerate(COMPLEXITY_TIERS):
            if lo <= dim <= hi:
                tier_label, tier_idx = tl, i
                break

        scored.append({
            "experiment_key": key, "label": label, "input_dim": dim,
            "feature_list": feat_list, "tier": tier_label, "tier_idx": tier_idx,
            "mean_bal_acc": round(mean_acc, 4),
            "std_across_tte": round(std_tte, 4),
            "mean_cv_std": round(mean_cv, 4),
            "robust_score": round(robust, 4),
        })

    scored_df = pd.DataFrame(scored).sort_values("robust_score", ascending=False)

    finalists = []
    for i, (tier_label, lo, hi) in enumerate(COMPLEXITY_TIERS):
        tier = scored_df[scored_df["tier_idx"] == i]
        if "sequential only" in tier_label.lower():
            tier = tier[~tier["experiment_key"].isin(TIER5_EXCLUDED)]
        if tier.empty:
            continue
        w = tier.iloc[0].to_dict()
        w["selection_reason"] = f"Best robust score in {tier_label}"
        finalists.append(w)

    return pd.DataFrame(finalists)


def main():
    style()

    print("=" * 72)
    print("  STEP 1: Load / confirm RQ2 GRU finalists")
    print("=" * 72)

    finalists = load_finalists_from_rq1()
    if finalists is None:
        finalists = select_finalists_locally()
    if finalists is None or finalists.empty:
        print("  No finalists available. Aborting.")
        return

    # Annotate with perception requirements
    finalists["needs_pose"] = finalists["tier_idx"].map(
        lambda x: TIER_NEEDS_POSE.get(int(x), True)
    )

    # Save local copy
    finalists.to_csv(RESULTS_DIR / "finalists.csv", index=False)

    # Per-TTE table
    if RQ1_RESULTS_CSV.exists():
        df = pd.read_csv(RQ1_RESULTS_CSV)
        gru = df[df["model"] == "gru"]
        tte_rows = []
        for _, fin in finalists.iterrows():
            sub = gru[gru["experiment_key"] == fin["experiment_key"]].sort_values("tte_s")
            for _, r in sub.iterrows():
                tte_rows.append({
                    "experiment_key": fin["experiment_key"],
                    "label": fin["label"],
                    "input_dim": int(fin["input_dim"]),
                    "tier": fin["tier"],
                    "tte_s": r["tte_s"],
                    "bal_acc": round(r["bal_acc"], 4),
                    "bal_acc_std": round(r["bal_acc_std"], 4),
                })
        tte_df = pd.DataFrame(tte_rows)
        tte_df.to_csv(RESULTS_DIR / "finalists_per_tte.csv", index=False)

    # --- Figure 1: bar overview ---
    fig, ax = plt.subplots(figsize=(12, 4.5))
    bar_labels = [
        f"{r['label']}  (dim={int(r['input_dim'])})"
        for _, r in finalists.iterrows()
    ]
    colors = [
        TIER_COLORS[int(r["tier_idx"])] if int(r["tier_idx"]) >= 0 else "#999"
        for _, r in finalists.iterrows()
    ]
    y_pos = range(len(finalists))
    ax.barh(
        y_pos, finalists["mean_bal_acc"], xerr=finalists["mean_cv_std"],
        color=colors, capsize=4, edgecolor="white", linewidth=0.6,
    )
    for yp, (_, r) in zip(y_pos, finalists.iterrows()):
        ax.text(
            r["mean_bal_acc"] + r["mean_cv_std"] + 0.003, yp,
            f"robust={r['robust_score']:.3f}",
            va="center", fontsize=8, color="#555",
        )
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(bar_labels, fontsize=10)
    ax.invert_yaxis()
    lo = max(0, (finalists["mean_bal_acc"] - finalists["mean_cv_std"]).min() - 0.04)
    hi = min(1, (finalists["mean_bal_acc"] + finalists["mean_cv_std"]).max() + 0.08)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Mean Balanced Accuracy (error = mean CV std)")
    ax.set_title("RQ2 Finalists: Best GRU per Complexity Tier")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "finalists_overview.png")
    plt.close()

    # --- Figure 2: accuracy over TTE ---
    if (RESULTS_DIR / "finalists_per_tte.csv").exists():
        tte_df = pd.read_csv(RESULTS_DIR / "finalists_per_tte.csv")
        fig, ax = plt.subplots(figsize=(10, 6))
        for _, fin in finalists.iterrows():
            sub = tte_df[
                tte_df["experiment_key"] == fin["experiment_key"]
            ].sort_values("tte_s")
            if sub.empty:
                continue
            tidx = int(fin["tier_idx"])
            ax.errorbar(
                sub["tte_s"], sub["bal_acc"], yerr=sub["bal_acc_std"],
                marker="o", lw=2, capsize=4, color=TIER_COLORS[tidx],
                label=f"{fin['label']} (dim={int(fin['input_dim'])})",
            )
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Time to Event (s)")
        ax.set_ylabel("Balanced Accuracy")
        ax.set_xticks(TTE_VALUES)
        ax.set_title("RQ2 Finalists: Performance over TTE")
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "finalists_over_tte.png")
        plt.close()

    # Console summary
    print(f"\n  {len(finalists)} finalists confirmed:")
    for i, (_, r) in enumerate(finalists.iterrows(), 1):
        pose_str = "needs pose" if r["needs_pose"] else "bbox only"
        print(f"    #{i}  {r['label']:45s}  dim={int(r['input_dim']):2d}  "
              f"robust={r['robust_score']:.4f}  [{pose_str}]")

    print(f"\n  Outputs: {RESULTS_DIR / 'finalists.csv'}")
    print(f"           {FIG_DIR / 'finalists_overview.png'}")


if __name__ == "__main__":
    main()
