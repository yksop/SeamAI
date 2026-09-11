#!/usr/bin/env python3
"""
validate_hyperparams.py  —  Hyperparameter sensitivity check

Validates that the fixed hyperparameters used in the RQ1 ablation study
(hidden_size=16, dropout=0.1) were a reasonable choice by running a small
grid search on the five RQ2 finalist models.

Motivation
----------
The RQ1 ablation fixed hyperparameters across all 14 experiments to ensure
fair comparison — observed differences reflect feature selection, not tuning.
This script verifies post-hoc that the chosen values are near-optimal and
that conclusions are robust within a reasonable hyperparameter range.

Grid
----
  hidden_size : [8, 16, 32]
  dropout     : [0.0, 0.1, 0.2]

  = 9 combinations × 5 finalists × 5 TTE = 225 cells
  Each cell: 5-fold × 3 repeats = 15 evaluations (same CV as RQ1)

The script reuses the exact same training loop, data builders, and CV
protocol as train_ablation.py to ensure comparability.

Output
------
  outputs/results/hp_grid_results.csv       — full results (one row per cell)
  outputs/results/hp_sensitivity.csv        — summary: best HP per finalist
  outputs/results/hp_sensitivity.png        — heatmap of hidden × dropout per model

Run
---
    cd 2_RQ2_RQ3_edge_deployment
    python step3_validate_hyperparams.py

    # To run a quick test (1 TTE, fewer CV repeats):
    python step3_validate_hyperparams.py --quick
"""

import argparse
import copy
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Paths (resolved via config.py)
# ---------------------------------------------------------------------------
from config import (
    DATA_ROOT, RESULTS_DIR, FIG_DIR,
    TTE_VALUES,
    EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, RANDOM_STATE,
)

from utils.dataset_utils import load_all_samples
from utils.models import IntentGRU
from utils.sequence_dataset import SequenceDataset, normalise

from train_ablation import (
    build_experiment_data,
    EXPERIMENTS,
    WINDOW_SECONDS,
)

CV_FOLDS = 5
CV_REPEATS = 3

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------------------------
HIDDEN_SIZES = [8, 16, 32]
DROPOUTS = [0.0, 0.1, 0.2]

# Finalist experiment keys (loaded from CSV or defaults)
FINALISTS_CSV = RESULTS_DIR / "finalists.csv"


def load_finalist_keys():
    import pandas as pd
    if FINALISTS_CSV.exists():
        return pd.read_csv(FINALISTS_CSV)["experiment_key"].tolist()
    # Fallback: the 5 GRU finalists from RQ1 (select_gru_finalists)
    return [
        "A4_core4", "A2_traj6", "B1_head_angle",
        "B4_torso_head_raw", "B6_full_body_raw",
    ]


# ---------------------------------------------------------------------------
# Training (identical to train_ablation.train_one_fold but parameterised)
# ---------------------------------------------------------------------------
def class_weight(y_train):
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)


def train_one_fold(model, X_fold_train, y_fold_train, X_fold_eval, y_fold_eval,
                   fold_seed):
    """Train one fold — mirrors train_ablation.train_one_fold exactly."""
    X_sub_train, X_sub_val, y_sub_train, y_sub_val = train_test_split(
        X_fold_train, y_fold_train,
        test_size=0.15,
        stratify=y_fold_train,
        random_state=fold_seed,
    )

    X_sub_train_n, X_sub_val_n, mean, std = normalise(X_sub_train, X_sub_val)

    X_fold_eval_n = (X_fold_eval - mean) / std
    np.nan_to_num(X_fold_eval_n, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    shuffle_gen = torch.Generator().manual_seed(fold_seed)
    train_loader = DataLoader(
        SequenceDataset(X_sub_train_n, y_sub_train),
        batch_size=BATCH_SIZE, shuffle=True, generator=shuffle_gen,
    )
    sub_val_loader = DataLoader(
        SequenceDataset(X_sub_val_n, y_sub_val),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    eval_loader = DataLoader(
        SequenceDataset(X_fold_eval_n, y_fold_eval),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight(y_sub_train))

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 1
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in sub_val_loader:
                val_loss += criterion(
                    model(X_batch.to(DEVICE)), y_batch.to(DEVICE)
                ).item()
        val_loss /= max(len(sub_val_loader), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            break

    model.load_state_dict(best_weights)

    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in eval_loader:
            all_logits.append(model(X_batch.to(DEVICE)).cpu())
            all_labels.append(y_batch)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    return {
        "bal_acc": balanced_accuracy_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs),
        "best_epoch": best_epoch,
    }


def run_cv(X, y, input_dim, hidden_size, dropout, label):
    """Repeated stratified k-fold CV for one HP combination."""
    bal_accs, roc_aucs = [], []

    for repeat in range(CV_REPEATS):
        repeat_seed = RANDOM_STATE + repeat * 100
        cv = StratifiedKFold(
            n_splits=CV_FOLDS, shuffle=True, random_state=repeat_seed
        )

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            fold_seed = repeat_seed + fold
            torch.manual_seed(fold_seed)

            model = IntentGRU(input_dim, hidden_size=hidden_size, dropout=dropout)
            metrics = train_one_fold(model, X_train, y_train, X_val, y_val,
                                     fold_seed)

            bal_accs.append(metrics["bal_acc"])
            roc_aucs.append(metrics["roc_auc"])

    return {
        "bal_acc_mean": float(np.mean(bal_accs)),
        "bal_acc_std": float(np.std(bal_accs)),
        "roc_auc_mean": float(np.mean(roc_aucs)),
        "roc_auc_std": float(np.std(roc_aucs)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter sensitivity validation for RQ2 finalists"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: only TTE=1.0, CV_REPEATS=1 (for testing)",
    )
    args = parser.parse_args()

    global CV_REPEATS

    tte_values = TTE_VALUES
    if args.quick:
        tte_values = [1.0]
        CV_REPEATS = 1

    finalist_keys = load_finalist_keys()
    n_hp = len(HIDDEN_SIZES) * len(DROPOUTS)
    n_total = len(finalist_keys) * len(tte_values) * n_hp

    print("=" * 72)
    print("  HYPERPARAMETER SENSITIVITY VALIDATION")
    print(f"  Device        : {DEVICE}")
    print(f"  Finalists     : {len(finalist_keys)}")
    print(f"  TTE values    : {tte_values}")
    print(f"  Hidden sizes  : {HIDDEN_SIZES}")
    print(f"  Dropouts      : {DROPOUTS}")
    print(f"  Grid size     : {n_hp} combinations per model per TTE")
    print(f"  Total cells   : {n_total}")
    print(f"  CV            : {CV_REPEATS}×{CV_FOLDS}-fold = "
          f"{CV_REPEATS * CV_FOLDS} evals per cell")
    if args.quick:
        print(f"  ** QUICK MODE — reduced for testing **")
    print("=" * 72)

    # Validate keys
    for key in finalist_keys:
        if key not in EXPERIMENTS:
            print(f"ERROR: '{key}' not in EXPERIMENTS")
            return

    # Load data
    samples = load_all_samples(DATA_ROOT)
    print(f"\nLoaded {len(samples)} samples")

    if not samples:
        print("No samples found — check DATA_ROOT.")
        return

    # CSV output
    csv_path = RESULTS_DIR / "hp_grid_results.csv"
    fieldnames = [
        "experiment_key", "label", "input_dim", "tte_s",
        "hidden_size", "dropout",
        "bal_acc_mean", "bal_acc_std", "roc_auc_mean", "roc_auc_std",
        "is_default",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        cell_num = 0

        for exp_key in finalist_keys:
            exp = EXPERIMENTS[exp_key]
            label = exp["label"]

            print(f"\n{'=' * 72}")
            print(f"  {label}  [{exp_key}]")
            print(f"{'=' * 72}")

            for tte in tte_values:
                X, y, input_dim, feat_names = build_experiment_data(
                    exp, samples, tte
                )

                if len(y) == 0 or len(np.unique(y)) < 2:
                    print(f"  TTE={tte:.1f}s — skipped (insufficient data)")
                    continue

                print(f"\n  TTE = {tte:.1f}s  (n={len(y)}, dim={input_dim})")

                for hidden in HIDDEN_SIZES:
                    for dropout in DROPOUTS:
                        cell_num += 1
                        is_default = (hidden == 16 and dropout == 0.1)
                        tag = " ← default" if is_default else ""

                        t0 = time.time()
                        res = run_cv(X, y, input_dim, hidden, dropout,
                                     f"{exp_key} h={hidden} d={dropout}")
                        elapsed = time.time() - t0

                        print(f"    [{cell_num:3d}/{n_total}]  "
                              f"h={hidden:2d}  d={dropout:.1f}  "
                              f"bal_acc={res['bal_acc_mean']:.4f} "
                              f"±{res['bal_acc_std']:.4f}  "
                              f"({elapsed:.0f}s){tag}")

                        row = {
                            "experiment_key": exp_key,
                            "label": label,
                            "input_dim": input_dim,
                            "tte_s": tte,
                            "hidden_size": hidden,
                            "dropout": dropout,
                            "bal_acc_mean": round(res["bal_acc_mean"], 4),
                            "bal_acc_std": round(res["bal_acc_std"], 4),
                            "roc_auc_mean": round(res["roc_auc_mean"], 4),
                            "roc_auc_std": round(res["roc_auc_std"], 4),
                            "is_default": is_default,
                        }
                        writer.writerow(row)
                        f.flush()

    print(f"\n  Full results → {csv_path}")

    # --- Summary: compare default vs best per finalist --------------------
    import pandas as pd

    df = pd.read_csv(csv_path)

    summary_rows = []
    for exp_key in finalist_keys:
        sub = df[df["experiment_key"] == exp_key]
        if sub.empty:
            continue

        # Aggregate across TTE for each HP combo
        hp_agg = sub.groupby(["hidden_size", "dropout"]).agg(
            mean_acc=("bal_acc_mean", "mean"),
            mean_std=("bal_acc_std", "mean"),
        ).reset_index()

        # Default row
        default = hp_agg[
            (hp_agg["hidden_size"] == 16) & (hp_agg["dropout"] == 0.1)
        ]
        default_acc = default["mean_acc"].values[0] if not default.empty else 0

        # Best row
        best_idx = hp_agg["mean_acc"].idxmax()
        best = hp_agg.loc[best_idx]

        # Worst row
        worst_idx = hp_agg["mean_acc"].idxmin()
        worst = hp_agg.loc[worst_idx]

        summary_rows.append({
            "experiment_key": exp_key,
            "label": sub["label"].iloc[0],
            "input_dim": int(sub["input_dim"].iloc[0]),
            "default_acc": round(default_acc, 4),
            "best_hidden": int(best["hidden_size"]),
            "best_dropout": best["dropout"],
            "best_acc": round(best["mean_acc"], 4),
            "delta_vs_default": round(best["mean_acc"] - default_acc, 4),
            "worst_acc": round(worst["mean_acc"], 4),
            "range": round(best["mean_acc"] - worst["mean_acc"], 4),
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "hp_sensitivity.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Summary → {summary_path}")

    # --- Console summary ---------------------------------------------------
    print(f"\n{'=' * 72}")
    print("  HYPERPARAMETER SENSITIVITY SUMMARY")
    print(f"  Grid: hidden ∈ {HIDDEN_SIZES}, dropout ∈ {DROPOUTS}")
    print(f"  Default: hidden=16, dropout=0.1")
    print(f"{'=' * 72}")

    for _, r in summary.iterrows():
        print(f"\n  {r['label']}  (dim={r['input_dim']})")
        print(f"    Default (h=16, d=0.1):  {r['default_acc']:.4f}")
        print(f"    Best   (h={r['best_hidden']}, d={r['best_dropout']}):  "
              f"{r['best_acc']:.4f}  (Δ = {r['delta_vs_default']:+.4f})")
        print(f"    Worst:                  {r['worst_acc']:.4f}")
        print(f"    Range:                  {r['range']:.4f}")

    max_delta = summary["delta_vs_default"].abs().max()
    print(f"\n  Maximum |Δ| vs default: {max_delta:.4f}")

    if max_delta < 0.02:
        print("  → Results are ROBUST: default hyperparameters are within "
              "2 pp of the best combination for all finalists.")
    elif max_delta < 0.05:
        print("  → Results are MODERATELY sensitive: some finalists could "
              "benefit from tuning, but differences are small.")
    else:
        print("  → WARNING: Results are SENSITIVE to hyperparameters. "
              "Consider re-running the ablation with tuned values.")

    # --- Heatmap plot ------------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
            "font.size": 11,
        })

        n_models = len(finalist_keys)
        fig, axes = plt.subplots(
            1, n_models, figsize=(4 * n_models, 3.5), sharey=True
        )
        if n_models == 1:
            axes = [axes]

        for ax, exp_key in zip(axes, finalist_keys):
            sub = df[df["experiment_key"] == exp_key]
            if sub.empty:
                continue

            # Pivot: mean across TTE
            hp_agg = sub.groupby(["hidden_size", "dropout"])[
                "bal_acc_mean"
            ].mean().reset_index()

            pivot = hp_agg.pivot(
                index="hidden_size", columns="dropout", values="bal_acc_mean"
            )

            im = ax.imshow(
                pivot.values, cmap="YlGn", aspect="auto",
                vmin=pivot.values.min() - 0.005,
                vmax=pivot.values.max() + 0.005,
            )

            # Annotate cells
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    is_def = (
                        pivot.index[i] == 16
                        and pivot.columns[j] == 0.1
                    )
                    weight = "bold" if is_def else "normal"
                    ax.text(
                        j, i, f"{val:.3f}",
                        ha="center", va="center", fontsize=9,
                        fontweight=weight,
                    )

            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{d:.1f}" for d in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel("Dropout")

            label = sub["label"].iloc[0]
            short = label[:25] + "…" if len(label) > 25 else label
            ax.set_title(f"{short}\n(dim={int(sub['input_dim'].iloc[0])})",
                         fontsize=10)

        axes[0].set_ylabel("Hidden size")
        fig.suptitle(
            "Hyperparameter Sensitivity — Mean Balanced Accuracy\n"
            "(bold = default h=16, d=0.1)",
            fontsize=12, y=1.05,
        )
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "hp_sensitivity.png")
        plt.close()
        print(f"\n  Heatmap → {RESULTS_DIR / 'hp_sensitivity.png'}")

    except ImportError:
        print("  matplotlib not available — skipping heatmap")

    print()


if __name__ == "__main__":
    main()
