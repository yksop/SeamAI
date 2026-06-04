#!/usr/bin/env python3
"""
rf_feature_importance.py
========================

Feature importance analysis using Random Forest with SHAP (TreeSHAP).

Pipeline
--------
1.  Train a Random Forest with 5-fold stratified cross-validation.
    SHAP values are computed on each fold's *validation* set, so every
    sample is explained exactly once using a model that never saw it
    during training.
2.  Aggregate SHAP importance across all folds.  Additionally compute
    per-fold importance vectors to assess ranking stability (Spearman).
3.  Iterative backward elimination (Díaz-Uriarte & Alvarez de Andrés,
    2006):  at each step the 20 % least important features are removed,
    SHAP is recomputed on the reduced set, and the RF is re-evaluated.
    The optimal feature set is the one with highest balanced accuracy.
4.  Per-TTE plots: individual feature ranking, base-signal aggregation,
    family aggregation, SHAP beeswarm, rank stability, correlation
    matrix, elimination curve, confusion matrix.
5.  Cross-TTE analysis:
        – Consensus ranking heatmap (mean SHAP across TTE values).
        – Optimal feature survival (which features survive backward
          elimination across all TTE horizons).
        – Family importance trends across TTE.

Usage
-----
    python rf_feature_importance.py

Outputs are saved to OUTPUT_DIR (default: "results/").
"""

import os
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy.stats import spearmanr, rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score,  # ADDED
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # pipeline root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset_utils import build_feature_matrix, load_all_samples
from utils.feature_extractor import (
    get_feature_families,
    get_feature_names,
)

# ──────────────────────────────────────────────────────────────────────────────
# SETTINGS — edit these to change the run
# ──────────────────────────────────────────────────────────────────────────────

DATA_ROOT = str(PROJECT_ROOT / "MasterData")

TTE_SECONDS    = [0.5, 1.0, 1.5, 2.0, 2.5]
WINDOW_SECONDS = [0.5]

N_TREES        = 300
CV_FOLDS       = 5
RANDOM_STATE   = 42

FRACTION_DROPPED = 0.2     # Díaz-Uriarte: drop 20 % least important per step
MIN_FEATURES     = 2       # Stop when fewer than this many features remain

OUTPUT_DIR     = str(SCRIPT_DIR / "results")

BASE_SIGNAL_NAMES = [
    "dist_to_door",
    "closure_rate",
    "vx",
    "vy",
    "abs_speed",
    "step_displacement",
    "heading_angle",
    "rel_angle_to_door",
    "aspect_ratio",
    "scale_change_rate",
]


# ──────────────────────────────────────────────────────────────────────────────
# SHAP helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_shap_matrix(shap_values):
    """
    Normalise SHAP output to shape (n_samples, n_features) for class 1
    ("enter").  Handles the different formats that shap.TreeExplainer can
    return depending on library version.
    """
    if isinstance(shap_values, list):
        arr = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
        return arr

    arr = np.asarray(shap_values)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[2] > 1:
            return arr[:, :, 1]
        return arr[:, :, 0]

    raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 – Cross-validation with integrated SHAP
# ──────────────────────────────────────────────────────────────────────────────

def run_cv_with_shap(X, y):
    """
    5-fold stratified CV that also computes SHAP on each fold's validation
    set.  This avoids explaining training-set predictions (which may
    reflect overfitting) and provides K independent importance estimates
    for stability analysis.

    Returns
    -------
    cv_results : dict
        Mean and std of accuracy, balanced accuracy, ROC-AUC.
    all_shap_values : ndarray, shape (n_samples, n_features)
        Per-sample SHAP values (each sample explained once, when held out).
    per_fold_importance : list of K ndarrays, each shape (n_features,)
        mean(|SHAP|) computed within each fold's validation set.
    all_y_pred : ndarray, shape (n_samples,)
        Out-of-fold predictions (for aggregated confusion matrix).
    all_y_proba : ndarray, shape (n_samples,)
        Out-of-fold predicted probabilities for class "enter".
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    all_shap_values = np.zeros_like(X, dtype=float)
    all_y_pred      = np.zeros(len(y), dtype=int)
    all_y_proba     = np.zeros(len(y), dtype=float)

    per_fold_importance = []
    accuracies          = []
    balanced_accuracies = []
    roc_aucs            = []
    f1_scores           = []  # ADDED

    print(f"Running {CV_FOLDS}-fold CV with SHAP ...")

    for fold_nr, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        rf = RandomForestClassifier(
            n_estimators  = N_TREES,
            max_features  = "sqrt",
            class_weight  = "balanced",
            random_state  = RANDOM_STATE,
            n_jobs        = -1,
        )
        rf.fit(X_train, y_train)

        y_pred  = rf.predict(X_val)
        y_proba = rf.predict_proba(X_val)[:, 1]

        accuracies.append(accuracy_score(y_val, y_pred))
        balanced_accuracies.append(balanced_accuracy_score(y_val, y_pred))
        roc_aucs.append(roc_auc_score(y_val, y_proba))
        f1_scores.append(f1_score(y_val, y_pred))  # ADDED

        all_y_pred[val_idx]  = y_pred
        all_y_proba[val_idx] = y_proba

        # SHAP on validation set only
        explainer   = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_val, check_additivity=False)
        shap_matrix = _safe_shap_matrix(shap_values)

        all_shap_values[val_idx] = shap_matrix
        per_fold_importance.append(np.mean(np.abs(shap_matrix), axis=0))

        print(f"  Fold {fold_nr:2d}: "
              f"acc={accuracies[-1]:.4f}  "
              f"bal_acc={balanced_accuracies[-1]:.4f}  "
              f"auc={roc_aucs[-1]:.4f}")

    cv_results = {
        "acc_mean"     : np.mean(accuracies),
        "acc_std"      : np.std(accuracies),
        "bal_acc_mean" : np.mean(balanced_accuracies),
        "bal_acc_std"  : np.std(balanced_accuracies),
        "auc_mean"     : np.mean(roc_aucs),
        "auc_std"      : np.std(roc_aucs),
        "f1_mean"      : np.mean(f1_scores),   # ADDED
        "f1_std"       : np.std(f1_scores),     # ADDED
    }

    print(f"\nCV results (mean +/- std across {CV_FOLDS} folds):")
    print(f"  Accuracy          : {cv_results['acc_mean']:.4f} +/- {cv_results['acc_std']:.4f}")
    print(f"  Balanced accuracy : {cv_results['bal_acc_mean']:.4f} +/- {cv_results['bal_acc_std']:.4f}")
    print(f"  ROC-AUC           : {cv_results['auc_mean']:.4f} +/- {cv_results['auc_std']:.4f}")
    print(f"  F1 score          : {cv_results['f1_mean']:.4f} +/- {cv_results['f1_std']:.4f}")  # ADDED

    return cv_results, all_shap_values, per_fold_importance, all_y_pred, all_y_proba


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 – Rank stability
# ──────────────────────────────────────────────────────────────────────────────

def compute_rank_stability(per_fold_importance):
    """
    Assess how stable the SHAP ranking is across CV folds.

    Returns
    -------
    rank_std        : ndarray, shape (n_features,)
        Standard deviation of each feature's rank across folds.
    importance_std  : ndarray, shape (n_features,)
        Standard deviation of raw importance scores across folds.
    mean_spearman   : float
        Mean pairwise Spearman correlation between fold rankings.
    """
    fold_ranks = np.array([rankdata(-imp) for imp in per_fold_importance])
    rank_std   = np.std(fold_ranks, axis=0)

    imp_array     = np.array(per_fold_importance)
    importance_std = np.std(imp_array, axis=0)

    # Pairwise Spearman between folds
    n_folds = len(per_fold_importance)
    rhos = []
    for i, j in combinations(range(n_folds), 2):
        rho, _ = spearmanr(per_fold_importance[i], per_fold_importance[j])
        rhos.append(rho)
    mean_spearman = float(np.mean(rhos))

    return rank_std, importance_std, mean_spearman


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2b – Base-signal and family aggregation
# ──────────────────────────────────────────────────────────────────────────────

def compute_base_signal_importance(importances):
    """
    Aggregate importance by base signal (sum over mean, var, latest).

    Each of the 10 base signals has three derived features.  Summing
    them shows the total credit attributed to each underlying quantity,
    avoiding the dilution caused by SHAP splitting credit among
    correlated variants.
    """
    result = {}
    for i, signal in enumerate(BASE_SIGNAL_NAMES):
        idx = [i * 3, i * 3 + 1, i * 3 + 2]
        result[signal] = float(np.sum(importances[idx]))
    return result


def compute_family_scores(importances):
    """Mean feature importance per family (normalised by family size)."""
    families = get_feature_families()
    return {
        name: float(np.mean([importances[i] for i in indices]))
        for name, indices in families.items()
    }


def compute_family_scores_std(shap_matrix):
    """
    Std of per-sample mean-family importance.

    For each sample compute mean(|SHAP|) over the family's features,
    then return the std of that series across samples.
    """
    families = get_feature_families()
    return {
        name: float(np.std(np.mean(np.abs(shap_matrix[:, indices]), axis=1)))
        for name, indices in families.items()
    }


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2c – Correlation diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def analyse_correlations(X, feature_names, threshold=0.8):
    """
    Compute feature correlation matrix and flag highly correlated pairs.

    Returns
    -------
    corr_matrix : ndarray (n_features, n_features)
    high_pairs  : list of (feat_i, feat_j, r)
    """
    corr_matrix = np.corrcoef(X.T)
    n = len(feature_names)
    high_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            r = corr_matrix[i, j]
            if abs(r) >= threshold:
                high_pairs.append((feature_names[i], feature_names[j], r))

    if high_pairs:
        print(f"\n  Feature pairs with |r| >= {threshold}:")
        for fi, fj, r in sorted(high_pairs, key=lambda t: -abs(t[2])):
            print(f"    {fi:<30s}  <->  {fj:<30s}  r = {r:+.3f}")
    else:
        print(f"\n  No feature pairs with |r| >= {threshold}.")

    return corr_matrix, high_pairs


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 – Iterative backward elimination (Díaz-Uriarte & Alvarez de Andrés)
# ──────────────────────────────────────────────────────────────────────────────

def _cv_balanced_accuracy(X, y, feature_idx):
    """
    Evaluate RF on a given subset of features using stratified k-fold CV.

    Returns mean and std of balanced accuracy across folds.
    """
    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        rf = RandomForestClassifier(
            n_estimators=N_TREES, max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        rf.fit(X[train_idx][:, feature_idx], y[train_idx])
        y_pred = rf.predict(X[val_idx][:, feature_idx])
        scores.append(balanced_accuracy_score(y[val_idx], y_pred))
    return float(np.mean(scores)), float(np.std(scores))


def _cv_shap_importance(X, y, feature_idx):
    """
    Compute SHAP importance for a given subset of features using
    stratified k-fold CV (SHAP on validation set only).

    Returns importance vector of length len(feature_idx).
    """
    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    n_sub = len(feature_idx)
    shap_accum = np.zeros(n_sub, dtype=float)
    sample_count = 0

    for train_idx, val_idx in cv.split(X, y):
        rf = RandomForestClassifier(
            n_estimators=N_TREES, max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        rf.fit(X[train_idx][:, feature_idx], y[train_idx])
        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(X[val_idx][:, feature_idx],
                                   check_additivity=False)
        sv_matrix = _safe_shap_matrix(sv)
        shap_accum += np.sum(np.abs(sv_matrix), axis=0)
        sample_count += len(val_idx)

    return shap_accum / sample_count


def run_diaz_uriarte(X, y, initial_importances, feature_names):
    """
    Iterative backward elimination following Díaz-Uriarte & Alvarez de
    Andrés (2006).

    At each iteration the least important fraction (FRACTION_DROPPED) of
    the remaining features is removed.  SHAP importances are recomputed
    on the reduced feature set so that the ranking adapts as features
    disappear.  The procedure terminates when the number of remaining
    features falls below MIN_FEATURES.

    The optimal feature set is the one that achieved the highest balanced
    accuracy during the elimination.  If several sets tie, the smallest
    is preferred (parsimony).

    Returns
    -------
    elimination_history : list of dicts
        One entry per iteration with keys:
            n_features, feature_idx, feature_names, bal_acc_mean,
            bal_acc_std, dropped_names
    best_step : int
        Index into elimination_history for the optimal feature set.
    """
    n_total = X.shape[1]
    current_idx = np.arange(n_total)
    current_imp = initial_importances.copy()

    history = []

    # ── Step 0: evaluate full feature set ────────────────────────────────
    acc_mean, acc_std = _cv_balanced_accuracy(X, y, current_idx)
    history.append({
        "n_features"    : len(current_idx),
        "feature_idx"   : current_idx.copy(),
        "feature_names" : [feature_names[i] for i in current_idx],
        "bal_acc_mean"  : acc_mean,
        "bal_acc_std"   : acc_std,
        "dropped_names" : [],
    })
    print(f"  Step 0 : {len(current_idx):3d} features  "
          f"bal_acc = {acc_mean:.4f} +/- {acc_std:.4f}")

    step = 1
    while len(current_idx) > MIN_FEATURES:
        # Number of features to drop this round
        n_drop = max(1, int(np.floor(len(current_idx) * FRACTION_DROPPED)))
        if len(current_idx) - n_drop < MIN_FEATURES:
            n_drop = len(current_idx) - MIN_FEATURES
        if n_drop <= 0:
            break

        # Rank by current importance (ascending) and drop the worst
        order = np.argsort(current_imp)          # least important first
        drop_local = order[:n_drop]
        keep_local = order[n_drop:]

        dropped_names = [feature_names[current_idx[i]] for i in drop_local]

        # Update bookkeeping
        current_idx = current_idx[keep_local]
        current_imp = current_imp[keep_local]

        # Evaluate reduced set
        acc_mean, acc_std = _cv_balanced_accuracy(X, y, current_idx)

        history.append({
            "n_features"    : len(current_idx),
            "feature_idx"   : current_idx.copy(),
            "feature_names" : [feature_names[i] for i in current_idx],
            "bal_acc_mean"  : acc_mean,
            "bal_acc_std"   : acc_std,
            "dropped_names" : dropped_names,
        })
        print(f"  Step {step:<2d}: {len(current_idx):3d} features  "
              f"bal_acc = {acc_mean:.4f} +/- {acc_std:.4f}  "
              f"dropped: {dropped_names}")

        # Recompute SHAP on reduced set for next iteration
        if len(current_idx) > MIN_FEATURES:
            current_imp = _cv_shap_importance(X, y, current_idx)

        step += 1

    # ── Select optimal set (highest accuracy; ties → fewest features) ────
    best_step = 0
    best_acc  = history[0]["bal_acc_mean"]
    for i, h in enumerate(history):
        if (h["bal_acc_mean"] > best_acc + 1e-6) or \
           (abs(h["bal_acc_mean"] - best_acc) < 1e-6
            and h["n_features"] < history[best_step]["n_features"]):
            best_acc  = h["bal_acc_mean"]
            best_step = i

    print(f"\n  Optimal set at step {best_step}: "
          f"{history[best_step]['n_features']} features, "
          f"bal_acc = {history[best_step]['bal_acc_mean']:.4f}")
    print(f"  Features: {history[best_step]['feature_names']}")

    return history, best_step


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3b – Top-vs-bottom feature validation (AlSagri & Ykhlef, 2020)
# ──────────────────────────────────────────────────────────────────────────────

def run_top_bottom_validation(X, y, global_importance, feature_names,
                              k_values=None):
    """
    Validate the SHAP ranking by comparing RF performance when using
    only the *k* most important vs. the *k* least important features,
    against the baseline with all features.

    A meaningful ranking should show that top-*k* features retain most
    of the baseline accuracy, while bottom-*k* features perform
    substantially worse.

    Following AlSagri & Ykhlef (2020), who removed the 5 most/least
    important features and compared against the full-feature baseline.

    Parameters
    ----------
    k_values : list of int or None
        Feature-set sizes to test.  Defaults to [5, 7, 10].

    Returns
    -------
    validation_results : list of dicts
        One entry per *k* with baseline, top-k, and bottom-k accuracy.
    """
    n_features = X.shape[1]

    if k_values is None:
        k_values = [k for k in [5, 7, 10] if k < n_features]

    # Rank features by SHAP importance (descending)
    ranked_idx = np.argsort(global_importance)[::-1]

    # Baseline: all features
    baseline_mean, baseline_std = _cv_balanced_accuracy(
        X, y, np.arange(n_features)
    )

    print(f"\n  Top-vs-Bottom Validation (AlSagri & Ykhlef, 2020)")
    print(f"  Baseline (all {n_features} features): "
          f"bal_acc = {baseline_mean:.4f} +/- {baseline_std:.4f}")
    print(f"  {'k':>4}  {'Top-k acc':>10}  {'Bottom-k acc':>12}  "
          f"{'Delta top':>10}  {'Delta bottom':>12}")
    print("  " + "-" * 55)

    results = []
    for k in k_values:
        top_idx    = ranked_idx[:k]
        bottom_idx = ranked_idx[-k:]

        top_mean, top_std       = _cv_balanced_accuracy(X, y, top_idx)
        bottom_mean, bottom_std = _cv_balanced_accuracy(X, y, bottom_idx)

        delta_top    = top_mean - baseline_mean
        delta_bottom = bottom_mean - baseline_mean

        print(f"  {k:>4}  {top_mean:>10.4f}  {bottom_mean:>12.4f}  "
              f"{delta_top:>+10.4f}  {delta_bottom:>+12.4f}")

        results.append({
            "k"                   : k,
            "baseline_bal_acc"    : baseline_mean,
            "baseline_std"        : baseline_std,
            "top_k_bal_acc"       : top_mean,
            "top_k_std"           : top_std,
            "bottom_k_bal_acc"    : bottom_mean,
            "bottom_k_std"        : bottom_std,
            "delta_top"           : delta_top,
            "delta_bottom"        : delta_bottom,
            "top_features"        : [feature_names[i] for i in top_idx],
            "bottom_features"     : [feature_names[i] for i in bottom_idx],
        })

    return results


def plot_top_bottom_validation(validation_results, tte, window, output_dir):
    """
    Grouped bar chart comparing top-k vs. bottom-k accuracy against
    the all-features baseline for each tested k.
    """
    if not validation_results:
        return

    k_values       = [r["k"] for r in validation_results]
    baseline       = validation_results[0]["baseline_bal_acc"]
    baseline_std   = validation_results[0]["baseline_std"]
    top_accs       = [r["top_k_bal_acc"] for r in validation_results]
    top_stds       = [r["top_k_std"] for r in validation_results]
    bottom_accs    = [r["bottom_k_bal_acc"] for r in validation_results]
    bottom_stds    = [r["bottom_k_std"] for r in validation_results]

    x = np.arange(len(k_values))
    bar_width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))

    # Baseline reference line
    ax.axhline(baseline, color="#555555", linestyle="--", linewidth=1.2,
               label=f"Baseline (all features): {baseline:.4f}", zorder=1)
    ax.axhspan(baseline - baseline_std, baseline + baseline_std,
               alpha=0.1, color="#555555", zorder=0)

    # Top-k bars
    ax.bar(x - bar_width / 2, top_accs, bar_width, yerr=top_stds,
           color="#55A868", edgecolor="white", capsize=4,
           label="Top-k features", zorder=2)

    # Bottom-k bars
    ax.bar(x + bar_width / 2, bottom_accs, bar_width, yerr=bottom_stds,
           color="#C44E52", edgecolor="white", capsize=4,
           label="Bottom-k features", zorder=2)

    # Value annotations
    for i, (tv, bv) in enumerate(zip(top_accs, bottom_accs)):
        ax.text(i - bar_width / 2, tv + 0.005, f"{tv:.3f}",
                ha="center", va="bottom", fontsize=8)
        ax.text(i + bar_width / 2, bv + 0.005, f"{bv:.3f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"k = {k}" for k in k_values], fontsize=10)
    ax.set_ylabel(f"{CV_FOLDS}-fold CV Balanced Accuracy")
    ax.set_title(
        f"Top-k vs. Bottom-k Feature Validation\n"
        f"TTE = {tte}s  |  window = {window}s"
    )
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(axis="y", alpha=0.3)

    # Set y-axis to give some room
    all_vals = top_accs + bottom_accs + [baseline]
    y_min = max(0.4, min(all_vals) - 0.05)
    ax.set_ylim(y_min, min(1.0, max(all_vals) + 0.04))

    plt.tight_layout()
    path = os.path.join(output_dir,
                        f"top_bottom_validation_{_slug(tte, window)}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def save_top_bottom_csv(validation_results, tte, window, output_dir):
    """Save top-vs-bottom validation results to CSV."""
    rows = []
    for r in validation_results:
        rows.append({
            "k"                : r["k"],
            "baseline_bal_acc" : r["baseline_bal_acc"],
            "baseline_std"     : r["baseline_std"],
            "top_k_bal_acc"    : r["top_k_bal_acc"],
            "top_k_std"        : r["top_k_std"],
            "bottom_k_bal_acc" : r["bottom_k_bal_acc"],
            "bottom_k_std"     : r["bottom_k_std"],
            "delta_top"        : r["delta_top"],
            "delta_bottom"     : r["delta_bottom"],
            "top_features"     : "; ".join(r["top_features"]),
            "bottom_features"  : "; ".join(r["bottom_features"]),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir,
                        f"top_bottom_validation_{_slug(tte, window)}.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 – Plots
# ──────────────────────────────────────────────────────────────────────────────

def _slug(tte, window):
    """Short string for filenames, e.g. 'tte1_0s_win0_5s'."""
    return f"tte{tte:.1f}s_win{window:.1f}s".replace(".", "_")


FAMILY_COLORS = {
    "Goal-Oriented"        : "#C44E52",
    "Kinematic"            : "#55A868",
    "Trajectory & Orient." : "#DD8452",
    "Bounding Box"         : "#4C72B0",
}


def plot_individual_features(importances, feature_names, importance_std,
                             tte, window, output_dir):
    """Horizontal bar chart of all 30 features with error bars."""
    sorted_idx   = np.argsort(importances)
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_vals  = importances[sorted_idx]
    sorted_errs  = importance_std[sorted_idx] if importance_std is not None else None

    fig, ax = plt.subplots(figsize=(8, 10))
    bars = ax.barh(sorted_names, sorted_vals, xerr=sorted_errs,
                   color="#8172B2", edgecolor="white", capsize=2)

    ax.set_xlabel("SHAP importance (mean |SHAP value|)")
    ax.set_title(
        f"Feature Importance – SHAP\n"
        f"TTE = {tte}s  |  window = {window}s"
    )

    for bar, val in zip(bars, sorted_vals):
        ax.text(val + 0.0003, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir,
                        f"importance_individual_shap_{_slug(tte, window)}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_base_signal_importance(base_scores, tte, window, output_dir):
    """Bar chart of the 10 base signals (sum of mean + var + latest)."""
    sorted_items = sorted(base_scores.items(), key=lambda x: x[1])
    names  = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, scores, color="#55A868", edgecolor="white")

    ax.set_xlabel("Summed SHAP importance (mean + var + latest)")
    ax.set_title(
        f"Base Signal Importance – SHAP\n"
        f"TTE = {tte}s  |  window = {window}s"
    )

    for bar, val in zip(bars, scores):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir,
                        f"importance_base_signal_{_slug(tte, window)}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_family_aggregation(importances, per_fold_importance, tte, window,
                            output_dir):
    """Family-level bar chart with error bars from per-fold variation."""
    families = get_feature_families()
    family_names = list(families.keys())

    global_family = {
        name: float(np.mean([importances[i] for i in indices]))
        for name, indices in families.items()
    }

    fold_family_scores = []
    for fold_imp in per_fold_importance:
        fold_family_scores.append([
            float(np.mean([fold_imp[i] for i in families[name]]))
            for name in family_names
        ])
    fold_family_scores = np.array(fold_family_scores)
    family_std = np.std(fold_family_scores, axis=0)

    sorted_pairs = sorted(zip(family_names, range(len(family_names))),
                          key=lambda t: global_family[t[0]])
    s_names = [t[0] for t in sorted_pairs]
    s_idx   = [t[1] for t in sorted_pairs]
    s_vals  = [global_family[n] for n in s_names]
    s_errs  = family_std[s_idx]
    s_colors = [FAMILY_COLORS.get(n, "#999999") for n in s_names]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(s_names, s_vals, xerr=s_errs, color=s_colors,
            edgecolor="white", capsize=3)

    ax.set_xlabel("Mean SHAP importance per feature")
    ax.set_title(
        f"Feature Importance by Family – SHAP\n"
        f"TTE = {tte}s  |  window = {window}s"
    )

    for i, val in enumerate(s_vals):
        ax.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir,
                        f"importance_by_family_shap_{_slug(tte, window)}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_shap_beeswarm(shap_values, X, feature_names, tte, window,
                       output_dir):
    """SHAP beeswarm / summary plot showing direction-of-effect."""
    fig, ax = plt.subplots(figsize=(9, 10))
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      show=False, max_display=30)
    plt.title(f"SHAP Summary – TTE = {tte}s  |  window = {window}s",
              fontsize=11, pad=12)
    plt.tight_layout()
    path = os.path.join(output_dir,
                        f"shap_beeswarm_{_slug(tte, window)}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_rank_stability(rank_std, feature_names, tte, window, output_dir):
    """Bar chart of rank standard deviation across folds."""
    sorted_idx = np.argsort(rank_std)[::-1]
    names = [feature_names[i] for i in sorted_idx]
    vals  = rank_std[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 10))
    colors = ["#C44E52" if v > 5 else "#DD8452" if v > 3 else "#55A868"
              for v in vals]
    ax.barh(names, vals, color=colors, edgecolor="white")

    ax.set_xlabel("Rank std across CV folds (lower = more stable)")
    ax.set_title(
        f"Feature Rank Stability – SHAP\n"
        f"TTE = {tte}s  |  window = {window}s"
    )
    ax.axvline(3, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(output_dir,
                        f"rank_stability_{_slug(tte, window)}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_correlation_matrix(corr_matrix, feature_names, tte, window,
                            output_dir):
    """Feature correlation heatmap."""
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        xticklabels=feature_names,
        yticklabels=feature_names,
        cmap="RdBu_r",
        center=0, vmin=-1, vmax=1,
        linewidths=0.3,
        annot=False,
        ax=ax,
    )
    ax.set_title(
        f"Feature Correlation Matrix\n"
        f"TTE = {tte}s  |  window = {window}s",
        fontsize=11,
    )
    ax.tick_params(axis="both", labelsize=7)
    plt.tight_layout()
    path = os.path.join(output_dir,
                        f"correlation_matrix_{_slug(tte, window)}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_elimination_curve(history, best_step, tte, window, output_dir):
    """
    Line plot of balanced accuracy vs. number of remaining features,
    showing the full iterative backward elimination trajectory.
    The optimal feature set is highlighted.
    """
    n_feats = [h["n_features"] for h in history]
    accs    = [h["bal_acc_mean"] for h in history]
    stds    = [h["bal_acc_std"] for h in history]

    accs_arr = np.array(accs)
    stds_arr = np.array(stds)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(n_feats, accs_arr - stds_arr, accs_arr + stds_arr,
                    alpha=0.2, color="#4C72B0")
    ax.plot(n_feats, accs, "o-", color="#4C72B0", linewidth=2, markersize=6,
            label="Balanced accuracy")

    # Mark optimal
    ax.plot(n_feats[best_step], accs[best_step], "D", color="#C44E52",
            markersize=10, zorder=5,
            label=f"Optimal: {n_feats[best_step]} features "
                  f"({accs[best_step]:.4f})")

    ax.set_xlabel("Number of features remaining")
    ax.set_ylabel(f"{CV_FOLDS}-fold CV Balanced Accuracy")
    ax.set_title(
        f"Backward Elimination (Díaz-Uriarte)\n"
        f"TTE = {tte}s  |  window = {window}s"
    )
    ax.legend(fontsize=9)
    ax.invert_xaxis()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir,
                        f"elimination_curve_{_slug(tte, window)}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix_cv(y_true, y_pred, tte, window, bal_acc,
                             output_dir):
    """
    Confusion matrix aggregated from out-of-fold predictions.

    Every sample is predicted exactly once (when held out), so this CM
    reflects generalisation performance across the full dataset.
    """
    # labels=[0, 1] keeps the matrix 2x2 even if a class is absent (tiny data),
    # and the guarded division shows 0% instead of NaN for an empty row.
    cm = sk_confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm.astype(float), row_sums,
                       out=np.zeros((2, 2), dtype=float),
                       where=row_sums != 0) * 100.0

    annot = np.empty_like(cm, dtype=object)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f} %)"

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_pct, annot=annot, fmt="", cmap="Blues",
        xticklabels=["pass", "enter"],
        yticklabels=["pass", "enter"],
        vmin=0, vmax=100,
        linewidths=0.5, linecolor="white", ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(
        f"Confusion Matrix (out-of-fold) – all 30 features\n"
        f"TTE = {tte}s  |  bal_acc = {bal_acc:.4f}"
    )
    plt.tight_layout()
    path = os.path.join(output_dir,
                        f"confusion_matrix_{_slug(tte, window)}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Cross-TTE family plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_cross_tte_family_importance(summary_rows, window, output_dir):
    """Grouped bar chart: family importance across TTE values."""
    rows = sorted(
        [r for r in summary_rows if r["window"] == window],
        key=lambda r: r["tte"],
    )
    if len(rows) < 2:
        return

    family_names = list(get_feature_families().keys())
    tte_labels   = [f"TTE {r['tte']:.1f}s" for r in rows]
    n_ttes       = len(rows)
    n_families   = len(family_names)

    scores = np.array([
        [row["family_scores_shap"][f] for f in family_names]
        for row in rows
    ])

    has_std = all("family_scores_shap_std" in r for r in rows)
    stds = None
    if has_std:
        stds = np.array([
            [row["family_scores_shap_std"][f] for f in family_names]
            for row in rows
        ])

    bar_width = 0.18
    x = np.arange(n_ttes)

    fig, ax = plt.subplots(figsize=(max(8, n_ttes * 1.8), 5))

    for fi, family in enumerate(family_names):
        offset = (fi - n_families / 2.0 + 0.5) * bar_width
        ax.bar(x + offset, scores[:, fi], width=bar_width, label=family,
               color=FAMILY_COLORS.get(family, f"C{fi}"), edgecolor="white")
        if stds is not None:
            ax.errorbar(x + offset, scores[:, fi], yerr=stds[:, fi],
                        fmt="none", ecolor="#555555", elinewidth=1.2,
                        capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(tte_labels, fontsize=10)
    ax.set_ylabel("Mean SHAP importance per feature", fontsize=10)
    ax.set_title(
        f"Feature Family Importance Across TTE – SHAP\n(window = {window}s)",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fname = (f"family_importance_cross_tte_shap_win{window:.1f}s"
             .replace(".", "_") + ".png")
    path = os.path.join(output_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Cross-TTE consensus ranking & heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_cross_tte_heatmap(summary_rows, window, output_dir):
    """
    Heatmap of per-feature SHAP importance across all TTE values.

    Also computes and prints a consensus ranking (mean importance across
    TTE values) and saves a CSV with per-feature mean rank.
    """
    rows = sorted(
        [r for r in summary_rows if r["window"] == window],
        key=lambda r: r["tte"],
    )
    if len(rows) < 2:
        return

    feature_names = rows[0]["feature_names"]
    n_features    = len(feature_names)
    tte_labels    = [f"{r['tte']:.1f}s" for r in rows]

    # Build importance matrix: (n_tte, n_features)
    imp_matrix = np.array([r["shap_importances"] for r in rows])

    # Consensus ranking: mean importance across TTE values
    mean_importance = np.mean(imp_matrix, axis=0)
    consensus_order = np.argsort(mean_importance)[::-1]

    # Reorder features by consensus rank for the heatmap
    ordered_names = [feature_names[i] for i in consensus_order]
    ordered_matrix = imp_matrix[:, consensus_order].T  # (n_features, n_tte)

    # ── Heatmap ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 1.5), 10))
    im = ax.imshow(ordered_matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(tte_labels)))
    ax.set_xticklabels(tte_labels, fontsize=10)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(ordered_names, fontsize=8)
    ax.set_xlabel("Time-to-Event (TTE)")
    ax.set_title(
        f"Feature Importance Across TTE – SHAP\n"
        f"(ordered by consensus rank, window = {window}s)",
        fontsize=11,
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Mean |SHAP value|", fontsize=9)

    plt.tight_layout()
    fname = f"cross_tte_heatmap_win{window:.1f}s".replace(".", "_") + ".png"
    path = os.path.join(output_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ── Consensus ranking CSV ────────────────────────────────────────────────
    rank_matrix = np.array([
        rankdata(-imp_matrix[t], method="min")
        for t in range(len(rows))
    ])  # (n_tte, n_features)
    mean_rank = np.mean(rank_matrix, axis=0)

    consensus_rows = []
    for i in consensus_order:
        consensus_rows.append({
            "consensus_rank" : int(rankdata(-mean_importance, method="min")[i]),
            "feature"        : feature_names[i],
            "mean_importance" : float(mean_importance[i]),
            "mean_rank"       : float(mean_rank[i]),
            **{f"importance_tte{rows[t]['tte']:.1f}s": float(imp_matrix[t, i])
               for t in range(len(rows))},
            **{f"rank_tte{rows[t]['tte']:.1f}s": int(rank_matrix[t, i])
               for t in range(len(rows))},
        })

    df = pd.DataFrame(consensus_rows)
    csv_path = os.path.join(
        output_dir,
        f"consensus_ranking_win{window:.1f}s".replace(".", "_") + ".csv"
    )
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Print top 10
    print(f"\n  Consensus ranking (top 10, window = {window}s):")
    print(f"  {'Rank':<6} {'Feature':<35} {'Mean Imp':>10} {'Mean Rank':>10}")
    print("  " + "-" * 63)
    for row in consensus_rows[:10]:
        print(f"  {row['consensus_rank']:<6} {row['feature']:<35} "
              f"{row['mean_importance']:>10.5f} {row['mean_rank']:>10.1f}")


# ──────────────────────────────────────────────────────────────────────────────
# Cross-TTE optimal feature intersection
# ──────────────────────────────────────────────────────────────────────────────

def analyse_optimal_feature_intersection(summary_rows, window, output_dir):
    """
    Identify features that appear in the optimal set across multiple TTE
    values.  The 'universal core' is the intersection of all optimal sets.
    A frequency table shows how many TTE values each feature survives.
    """
    rows = sorted(
        [r for r in summary_rows if r["window"] == window],
        key=lambda r: r["tte"],
    )
    if len(rows) < 2:
        return

    feature_names = rows[0]["feature_names"]
    tte_labels    = [f"{r['tte']:.1f}s" for r in rows]
    n_ttes        = len(rows)

    # Count how often each feature appears in an optimal set
    freq = {f: 0 for f in feature_names}
    for r in rows:
        for f in r["optimal_features"]:
            freq[f] += 1

    # Universal core: present in ALL optimal sets
    core = sorted([f for f, c in freq.items() if c == n_ttes])

    # Frequency-ordered list
    freq_sorted = sorted(freq.items(), key=lambda x: (-x[1], x[0]))

    print(f"\n  Universal core features (in optimal set for all "
          f"{n_ttes} TTE values):")
    if core:
        for f in core:
            print(f"    - {f}")
    else:
        print("    (none – no feature survives all TTE eliminations)")

    # ── Survival heatmap ─────────────────────────────────────────────────────
    # Binary matrix: 1 if feature is in optimal set for that TTE
    survival = np.zeros((len(feature_names), n_ttes), dtype=int)
    for t, r in enumerate(rows):
        opt_set = set(r["optimal_features"])
        for i, f in enumerate(feature_names):
            if f in opt_set:
                survival[i, t] = 1

    # Order by frequency (most universal first)
    freq_order = [feature_names.index(f) for f, _ in freq_sorted]
    ordered_names = [feature_names[i] for i in freq_order]
    ordered_survival = survival[freq_order]

    fig, ax = plt.subplots(figsize=(max(5, n_ttes * 1.2), 10))
    ax.imshow(ordered_survival, aspect="auto", cmap="Greens", vmin=0, vmax=1)

    for i in range(len(ordered_names)):
        for j in range(n_ttes):
            if ordered_survival[i, j]:
                ax.text(j, i, "\u2713", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")

    ax.set_xticks(range(n_ttes))
    ax.set_xticklabels(tte_labels, fontsize=10)
    ax.set_yticks(range(len(ordered_names)))
    ax.set_yticklabels(ordered_names, fontsize=8)
    ax.set_xlabel("Time-to-Event (TTE)")
    ax.set_title(
        f"Feature Survival in Optimal Sets Across TTE\n"
        f"(window = {window}s)",
        fontsize=11,
    )

    # Add frequency annotation on the right
    for i, (f, _) in enumerate(freq_sorted):
        ax.text(n_ttes + 0.3, i, f"{freq[f]}/{n_ttes}",
                ha="left", va="center", fontsize=8)

    plt.tight_layout()
    fname = (f"optimal_feature_survival_win{window:.1f}s"
             .replace(".", "_") + ".png")
    path = os.path.join(output_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    surv_rows = []
    for f, count in freq_sorted:
        row = {
            "feature"          : f,
            "n_tte_optimal"    : count,
            "fraction_optimal" : count / n_ttes,
        }
        for t, r in enumerate(rows):
            row[f"in_optimal_tte{r['tte']:.1f}s"] = (
                f in set(r["optimal_features"])
            )
        surv_rows.append(row)

    df = pd.DataFrame(surv_rows)
    csv_path = os.path.join(
        output_dir,
        f"optimal_feature_survival_win{window:.1f}s".replace(".", "_") + ".csv"
    )
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    return core, freq


# ──────────────────────────────────────────────────────────────────────────────
# CSV output
# ──────────────────────────────────────────────────────────────────────────────

def save_results_csv(shap_importances, importance_std, rank_std,
                     feature_names, elimination_history, best_step,
                     base_scores, tte, window, output_dir):
    """
    Save three CSV files:
      1. Per-feature importance with stability metrics.
      2. Per base-signal aggregated importance.
      3. Ablation results.
    """
    families = get_feature_families()
    family_lookup = {}
    for name, indices in families.items():
        for idx in indices:
            family_lookup[idx] = name

    # -- Feature importance CSV ------------------------------------------------
    rows = []
    for i, fname in enumerate(feature_names):
        rows.append({
            "feature"          : fname,
            "family"           : family_lookup.get(i, "-"),
            "shap_importance"  : float(shap_importances[i]),
            "importance_std"   : float(importance_std[i]),
            "rank_std"         : float(rank_std[i]),
        })

    df_imp = pd.DataFrame(rows)
    df_imp["shap_rank"] = (df_imp["shap_importance"]
                           .rank(method="min", ascending=False).astype(int))
    df_imp = df_imp.sort_values("shap_rank").reset_index(drop=True)

    imp_path = os.path.join(output_dir,
                            f"feature_importance_{_slug(tte, window)}.csv")
    df_imp.to_csv(imp_path, index=False)
    print(f"  Saved: {imp_path}")

    # -- Base signal CSV -------------------------------------------------------
    bs_rows = [{"signal": sig, "summed_shap": val}
               for sig, val in base_scores.items()]
    df_bs = pd.DataFrame(bs_rows)
    df_bs["signal_rank"] = (df_bs["summed_shap"]
                            .rank(method="min", ascending=False).astype(int))
    df_bs = df_bs.sort_values("signal_rank").reset_index(drop=True)

    bs_path = os.path.join(output_dir,
                           f"base_signal_importance_{_slug(tte, window)}.csv")
    df_bs.to_csv(bs_path, index=False)
    print(f"  Saved: {bs_path}")

    # -- Elimination history CSV -----------------------------------------------
    elim_rows = []
    for step_i, h in enumerate(elimination_history):
        elim_rows.append({
            "step"         : step_i,
            "n_features"   : h["n_features"],
            "bal_acc_mean" : h["bal_acc_mean"],
            "bal_acc_std"  : h["bal_acc_std"],
            "dropped"      : ", ".join(h["dropped_names"]),
            "remaining"    : ", ".join(h["feature_names"]),
            "is_optimal"   : (step_i == best_step),
        })
    df_elim = pd.DataFrame(elim_rows)
    elim_path = os.path.join(output_dir,
                             f"elimination_history_{_slug(tte, window)}.csv")
    df_elim.to_csv(elim_path, index=False)
    print(f"  Saved: {elim_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Single-run orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run_one(samples, feature_names, tte, window):
    """
    Full pipeline for one (TTE, window) combination.

    Returns a summary dict for the cross-TTE table, or None if too few
    samples are available.
    """
    print("\n" + "=" * 60)
    print(f"  TTE = {tte}s  |  window = {window}s")
    print("=" * 60)

    X, y = build_feature_matrix(samples, tte_seconds=tte,
                                window_seconds=window)
    print(f"  Usable samples : {X.shape[0]}  "
          f"(pass={int((y == 0).sum())}, enter={int((y == 1).sum())})")

    if X.shape[0] < 20:
        print("  Too few samples - skipping.")
        return None

    run_dir = os.path.join(OUTPUT_DIR, _slug(tte, window))
    os.makedirs(run_dir, exist_ok=True)

    # ── Step 1: CV + SHAP ────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("STEP 1 – Cross-validation with integrated SHAP")
    print("-" * 60)
    cv_results, all_shap, per_fold_imp, y_pred, y_proba = \
        run_cv_with_shap(X, y)

    global_importance = np.mean(np.abs(all_shap), axis=0)

    # ── Step 2: Stability + aggregation ──────────────────────────────────────
    print("\n" + "-" * 60)
    print("STEP 2 – Ranking stability & aggregation")
    print("-" * 60)
    rank_std, importance_std, mean_spearman = \
        compute_rank_stability(per_fold_imp)
    print(f"  Mean pairwise Spearman rho across folds: {mean_spearman:.4f}")

    ranked = np.argsort(global_importance)[::-1]
    print(f"\n  Top 10 features (SHAP, aggregated across folds):")
    print(f"  {'Rank':<6} {'Feature':<35} {'Importance':>12} {'Rank Std':>10}")
    print("  " + "-" * 65)
    for rank, idx in enumerate(ranked[:10], start=1):
        print(f"  {rank:<6} {feature_names[idx]:<35} "
              f"{global_importance[idx]:>12.5f} {rank_std[idx]:>10.2f}")

    base_scores = compute_base_signal_importance(global_importance)

    print(f"\n  Base signal importance (sum of mean + var + latest):")
    for sig, val in sorted(base_scores.items(), key=lambda x: -x[1]):
        print(f"    {sig:<25s}  {val:.4f}")

    # ── Step 2b: Correlation diagnostics ─────────────────────────────────────
    corr_matrix, high_pairs = analyse_correlations(X, feature_names)

    # ── Step 3: Iterative backward elimination (Díaz-Uriarte) ──────────────
    print("\n" + "-" * 60)
    print("STEP 3 – Iterative backward elimination (Díaz-Uriarte)")
    print("-" * 60)
    elim_history, best_step = run_diaz_uriarte(
        X, y, global_importance, feature_names
    )

    # ── Step 3b: Top-vs-bottom validation (AlSagri & Ykhlef) ────────────────
    print("\n" + "-" * 60)
    print("STEP 3b – Top-vs-bottom feature validation")
    print("-" * 60)
    top_bottom_results = run_top_bottom_validation(
        X, y, global_importance, feature_names
    )
    plot_top_bottom_validation(top_bottom_results, tte, window, run_dir)
    save_top_bottom_csv(top_bottom_results, tte, window, run_dir)

    # ── Step 4: Plots ────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("STEP 4 – Generating plots")
    print("-" * 60)

    plot_individual_features(global_importance, feature_names,
                             importance_std, tte, window, run_dir)
    plot_base_signal_importance(base_scores, tte, window, run_dir)
    plot_family_aggregation(global_importance, per_fold_imp, tte, window,
                            run_dir)
    plot_shap_beeswarm(all_shap, X, feature_names, tte, window, run_dir)
    plot_rank_stability(rank_std, feature_names, tte, window, run_dir)
    plot_correlation_matrix(corr_matrix, feature_names, tte, window, run_dir)
    plot_elimination_curve(elim_history, best_step, tte, window, run_dir)

    bal_acc_oof = balanced_accuracy_score(y, y_pred)
    plot_confusion_matrix_cv(y, y_pred, tte, window, bal_acc_oof, run_dir)

    # ── Save CSVs ────────────────────────────────────────────────────────────
    save_results_csv(global_importance, importance_std, rank_std,
                     feature_names, elim_history, best_step,
                     base_scores, tte, window, run_dir)

    family_scores_shap     = compute_family_scores(global_importance)
    family_scores_shap_std = compute_family_scores_std(all_shap)

    optimal = elim_history[best_step]

    return {
        "tte"                    : tte,
        "window"                 : window,
        "n_samples"              : X.shape[0],
        "bal_acc_mean"           : cv_results["bal_acc_mean"],
        "bal_acc_std"            : cv_results["bal_acc_std"],
        "f1_mean"                : cv_results["f1_mean"],    # ADDED
        "f1_std"                 : cv_results["f1_std"],     # ADDED
        "top_feature_shap"       : feature_names[ranked[0]],
        "mean_spearman_rho"      : mean_spearman,
        "optimal_n_features"     : optimal["n_features"],
        "optimal_bal_acc"        : optimal["bal_acc_mean"],
        "optimal_features"       : optimal["feature_names"],
        "family_scores_shap"     : family_scores_shap,
        "family_scores_shap_std" : family_scores_shap_std,
        # Per-feature data for cross-TTE analysis
        "shap_importances"       : global_importance,
        "feature_names"          : feature_names,
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    tte_list    = TTE_SECONDS    if isinstance(TTE_SECONDS,    list) else [TTE_SECONDS]
    window_list = WINDOW_SECONDS if isinstance(WINDOW_SECONDS, list) else [WINDOW_SECONDS]

    print("=" * 60)
    print("  RF Feature Importance – SHAP (CV-integrated)")
    print(f"  TTE values : {tte_list}")
    print(f"  Windows    : {window_list}")
    print("=" * 60)

    print("\nLoading samples ...")
    samples = load_all_samples(DATA_ROOT)
    print(f"  Total samples loaded: {len(samples)}")

    feature_names = get_feature_names()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_rows = []
    for window in window_list:
        for tte in tte_list:
            result = run_one(samples, feature_names, tte, window)
            if result is not None:
                summary_rows.append(result)

    if not summary_rows:
        print("\nNo experiments completed.")
        return

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY ACROSS ALL TTE / WINDOW COMBINATIONS")
    print("=" * 70)
    print(f"  {'TTE':>5}  {'Win':>5}  {'N':>5}  {'Bal Acc':>8}  "
          f"{'Spearman':>9}  {'Opt #feat':>10}  {'Opt Acc':>9}  Top(SHAP)")
    print("  " + "-" * 75)
    for r in summary_rows:
        print(
            f"  {r['tte']:>5.1f}  {r['window']:>5.1f}  {r['n_samples']:>5d}  "
            f"{r['bal_acc_mean']:>8.4f}  "
            f"{r['mean_spearman_rho']:>9.4f}  "
            f"{r['optimal_n_features']:>10d}  "
            f"{r['optimal_bal_acc']:>9.4f}  "
            f"{r['top_feature_shap']}"
        )

    # Save summary CSV
    df = pd.DataFrame([
        {k: v for k, v in r.items()
         if k not in {"family_scores_shap", "family_scores_shap_std",
                      "optimal_features", "shap_importances",
                      "feature_names"}}
        for r in summary_rows
    ])
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\n  Summary CSV : {summary_path}")

    # ── Cross-TTE analyses ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CROSS-TTE ANALYSIS")
    print("=" * 70)

    for window in window_list:
        print(f"\n--- Window = {window}s ---")

        # Family importance across TTE
        print("\n  Generating cross-TTE family importance plot ...")
        plot_cross_tte_family_importance(summary_rows, window, OUTPUT_DIR)

        # Consensus ranking heatmap
        print("\n  Generating cross-TTE consensus heatmap ...")
        plot_cross_tte_heatmap(summary_rows, window, OUTPUT_DIR)

        # Optimal feature intersection / survival
        print("\n  Analysing optimal feature intersection ...")
        analyse_optimal_feature_intersection(summary_rows, window, OUTPUT_DIR)

    print(f"\n  All outputs : {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()