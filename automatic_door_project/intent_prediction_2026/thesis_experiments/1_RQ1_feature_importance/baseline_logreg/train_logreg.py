#!/usr/bin/env python3
"""
train_logreg.py

Training and evaluation script for the baseline Logistic Regression model.

This script loads pedestrian trajectory samples from MasterData, extracts
handcrafted features, trains a classifier, and reports evaluation metrics.

--- What this script does, step by step ---

1. Load all valid JSON samples from MasterData/enter and MasterData/pass.
   Folders named "exit", "removed", or "Corrupt" are skipped because those
   contain samples that were rejected during the data cleaning step.

2. For each sample, call extract_features() from trajectory_feature_extractor.py
   to get a 30-dimensional feature vector.

3. Stack all feature vectors into a matrix X and a label vector y.

4. Train a Logistic Regression classifier.
   - A StandardScaler is applied first so that all features have
     similar numeric scale (Z-score normalisation).
   - class_weight="balanced" compensates for class imbalance.

5. Evaluate the model in two ways:
   - 5-fold cross-validation: gives a stable estimate of average performance.
   - Holdout split (80/20): gives concrete predictions for reporting metrics.

6. Repeat for each observation window in WINDOW_SECONDS and each Time-To-Event
   (TTE) horizon listed in TTE_VALUES.

--- How to run ---

    python train_logreg.py

Adjust the CONFIG section below to change TTE values, window length, etc.
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # pipeline root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset_utils import (
    build_feature_matrix,
    load_all_samples,
    print_sample_counts_per_tte,
)
from utils.feature_extractor import (
    get_feature_families,
    get_feature_names,
)


# ──────────────────────────────────────────────────────────────────────────────
# SETTINGS — edit these to change the run
# ──────────────────────────────────────────────────────────────────────────────

DATA_ROOT      = str(PROJECT_ROOT / "MasterData")
OUTPUT_DIR     = str(SCRIPT_DIR / "results")
# One value or a list, e.g. 0.5 or [0.5, 1.0, 1.5]
WINDOW_SECONDS = [0.5]         # observation window(s) in seconds
TTE_VALUES     = [0.5, 1.0, 1.5, 2.0, 2.5] # Time-To-Event horizons to test
TEST_SIZE      = 0.2                       # fraction of data used as test set
CV_FOLDS       = 5                         # number of cross-validation folds
RANDOM_STATE   = 42                        # makes results reproducible


# Model

def make_model():
    """
    Create the classifier as a scikit-learn Pipeline with two steps:

    Step 1 – StandardScaler:
        Standardises each feature to zero mean and unit variance (Z-score).
        This is necessary because Logistic Regression is sensitive to the
        scale of features. Without this, a feature measured in pixels
        (e.g. distance = 300) would dominate one measured in radians (e.g. 0.5).

    Step 2 – LogisticRegression:
        A linear binary classifier (pass vs enter).
        class_weight="balanced" makes the model penalise errors on the
        minority class more heavily, which helps when the classes are
        not equally represented in the data.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

# Step 5 – Evaluate the model

def run_cross_validation(X, y):
    """
    Evaluate the model with Stratified K-Fold cross-validation.

    In K-fold CV the data is split into K equal parts (folds). The model
    is trained on K-1 folds and evaluated on the remaining fold. This is
    repeated K times so every sample is used for evaluation exactly once.

    "Stratified" means each fold preserves the original class ratio, which
    is important when the dataset is imbalanced.

    Reporting mean ± std across folds is more reliable than a single
    train/test split because it is less sensitive to which samples ended
    up in the test set by chance.
    """
    print(f"\n  --- {CV_FOLDS}-fold Cross-Validation ---")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    accuracies        = []
    balanced_accs     = []
    roc_aucs          = []
    f1_scores         = []

    for fold_num, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = make_model()
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]  # probability of "enter"

        acc     = accuracy_score(y_val, y_pred)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        auc     = roc_auc_score(y_val, y_proba)
        f1      = f1_score(y_val, y_pred, zero_division=0)

        accuracies.append(acc)
        balanced_accs.append(bal_acc)
        roc_aucs.append(auc)
        f1_scores.append(f1)

        print(f"    Fold {fold_num}: accuracy={acc:.4f}  bal_acc={bal_acc:.4f}  F1={f1:.4f}  ROC-AUC={auc:.4f}")

    results = {
        "acc_mean": float(np.mean(accuracies)),
        "acc_std": float(np.std(accuracies)),
        "bal_acc_mean": float(np.mean(balanced_accs)),
        "bal_acc_std": float(np.std(balanced_accs)),
        "f1_mean": float(np.mean(f1_scores)),
        "f1_std": float(np.std(f1_scores)),
        "auc_mean": float(np.mean(roc_aucs)),
        "auc_std": float(np.std(roc_aucs)),
    }

    print(f"\n  Mean accuracy          : {results['acc_mean']:.4f} ± {results['acc_std']:.4f}")
    print(f"  Mean balanced accuracy : {results['bal_acc_mean']:.4f} ± {results['bal_acc_std']:.4f}")
    print(f"  Mean F1 score          : {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
    print(f"  Mean ROC-AUC           : {results['auc_mean']:.4f} ± {results['auc_std']:.4f}")
    return results


def run_holdout_evaluation(X, y):
    """
    Train on 80% of the data and evaluate on the remaining 20%.

    This gives a concrete set of predictions to report alongside the
    cross-validation numbers. The holdout split uses stratification so
    both splits have the same class ratio.

    Reported metrics:
      - Accuracy          : fraction of all predictions that are correct
      - Balanced Accuracy : average recall across classes (fairer when imbalanced)
      - ROC-AUC           : area under the ROC curve (threshold-independent)
      - Precision         : of all predicted "enter", how many were truly "enter"?
      - Recall            : of all true "enter", how many did we catch?
      - F1 score          : harmonic mean of precision and recall
      - Confusion matrix  : breaks down correct and incorrect predictions by class
    """
    print("\n  --- Holdout Evaluation (80% train / 20% test) ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = make_model()
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n  Train size : {len(y_train)} samples")
    print(f"  Test size  : {len(y_test)} samples")

    print(f"\n  Accuracy          : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Balanced Accuracy : {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC           : {roc_auc_score(y_test, y_proba):.4f}")

    # --- Confusion matrix ---
    # Rows = true label, columns = predicted label
    # [0,0] = correct "pass",  [1,1] = correct "enter"
    # [0,1] = false "enter",   [1,0] = missed "enter"
    # labels=[0, 1] forces a 2x2 matrix even if the (small) test split happens
    # to contain only one class — otherwise confusion_matrix returns 1x1.
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("\n  Confusion matrix (rows=true, cols=predicted):")
    print(f"                    Pred pass    Pred enter")
    print(f"  True pass   :       {cm[0][0]:4d}          {cm[0][1]:4d}")
    print(f"  True enter  :       {cm[1][0]:4d}          {cm[1][1]:4d}")

    # --- Per-class precision, recall, F1 ---
    print("\n  Per-class report:")
    report = classification_report(y_test, y_pred, target_names=["pass", "enter"], digits=4)
    print(report)

    return {
        "acc": float(accuracy_score(y_test, y_pred)),
        "bal_acc": float(balanced_accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, y_proba)),
        "cm_tn": int(cm[0][0]),
        "cm_fp": int(cm[0][1]),
        "cm_fn": int(cm[1][0]),
        "cm_tp": int(cm[1][1]),
        "report": report,
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def _slug(tte, window):
    return f"tte{tte:.1f}s_win{window:.1f}s".replace(".", "_")


def plot_coefficient_importance(model, feature_names, tte, window, output_dir):
    """
    Horizontal bar chart of the standardised logistic regression coefficients.

    Because the features are Z-scored before fitting, the absolute coefficient
    value is directly comparable across features and reflects how much each
    feature contributes to the log-odds of "enter".
    """
    coefs = model.named_steps["logreg"].coef_[0]

    families = get_feature_families()
    family_lookup = {}
    for family_name, indices in families.items():
        for idx in indices:
            family_lookup[idx] = family_name

    family_colors = {
        "Goal-Oriented":        "#C44E52",
        "Kinematic":            "#55A868",
        "Trajectory & Orient.": "#DD8452",
        "Bounding Box":         "#4C72B0",
    }

    sorted_idx = np.argsort(np.abs(coefs))
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_vals = coefs[sorted_idx]
    bar_colors = [family_colors.get(family_lookup.get(i, ""), "#888888")
                  for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(9, 10))
    bars = ax.barh(sorted_names, sorted_vals, color=bar_colors, edgecolor="white")

    for bar, val in zip(bars, sorted_vals):
        offset = 0.01 * np.sign(val) if val != 0 else 0.01
        ax.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", fontsize=7,
        )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardised coefficient (positive → enter)")
    ax.set_title(
        f"Logistic Regression Coefficients\n"
        f"TTE = {tte}s  |  window = {window}s"
    )

    handles = [
        plt.Line2D([0], [0], color=c, lw=8, label=name)
        for name, c in family_colors.items()
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(output_dir, f"coef_importance_{_slug(tte, window)}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_family_coefficient_importance(model, tte, window, output_dir):
    """
    Bar chart of mean |coefficient| aggregated per feature family.
    """
    coefs = np.abs(model.named_steps["logreg"].coef_[0])
    families = get_feature_families()

    family_colors = {
        "Goal-Oriented":        "#C44E52",
        "Kinematic":            "#55A868",
        "Trajectory & Orient.": "#DD8452",
        "Bounding Box":         "#4C72B0",
    }

    family_scores = {
        name: float(np.mean(coefs[indices]))
        for name, indices in families.items()
    }

    sorted_items = sorted(family_scores.items(), key=lambda x: x[1])
    names = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]
    colors = [family_colors.get(n, "#888888") for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names, scores, color=colors, edgecolor="white")

    for i, val in enumerate(scores):
        ax.text(val + 0.002, i, f"{val:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Mean |coefficient| per feature")
    ax.set_title(
        f"Feature Family Importance – Logistic Regression\n"
        f"TTE = {tte}s  |  window = {window}s"
    )
    plt.tight_layout()
    path = os.path.join(output_dir, f"family_importance_{_slug(tte, window)}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix_heatmap(y_test, y_pred, tte, window, output_dir):
    """
    Seaborn heatmap showing counts and row-normalised percentages.
    """
    # labels=[0, 1] forces a 2x2 matrix even if the (small) test split happens
    # to contain only one class — otherwise confusion_matrix returns 1x1.
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    # Row-normalise to percentages; guard against empty rows (a class with no
    # true samples in a tiny test split) so we show 0% instead of NaN.
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm.astype(float), row_sums,
                       out=np.zeros((2, 2), dtype=float),
                       where=row_sums != 0) * 100.0

    annot = np.empty_like(cm, dtype=object)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

    bal_acc = balanced_accuracy_score(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_pct, annot=annot, fmt="",
        cmap="Blues",
        xticklabels=["pass", "enter"],
        yticklabels=["pass", "enter"],
        vmin=0, vmax=100,
        linewidths=0.5, linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(
        f"Confusion Matrix – Logistic Regression\n"
        f"TTE = {tte}s  |  bal_acc = {bal_acc:.4f}"
    )
    plt.tight_layout()
    path = os.path.join(output_dir, f"confusion_matrix_{_slug(tte, window)}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curve(model, X_test, y_test, tte, window, output_dir):
    """
    ROC curve with AUC value displayed.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name="LogReg")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
    ax.set_title(
        f"ROC Curve – Logistic Regression\n"
        f"TTE = {tte}s  |  window = {window}s"
    )
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(output_dir, f"roc_curve_{_slug(tte, window)}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_performance_vs_tte(summary_rows, output_dir):
    """
    Line plot showing how balanced accuracy and ROC-AUC degrade as TTE increases.

    Groups by observation window so each window gets its own set of lines.
    """
    df = pd.DataFrame(summary_rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for window, group in df.groupby("window"):
        group = group.sort_values("tte")
        ttes = group["tte"].values

        ax1.errorbar(
            ttes, group["cv_bal_acc_mean"].values,
            yerr=group["cv_bal_acc_std"].values,
            marker="o", capsize=4, label=f"window = {window}s",
        )
        ax2.errorbar(
            ttes, group["cv_auc_mean"].values,
            yerr=group["cv_auc_std"].values,
            marker="s", capsize=4, label=f"window = {window}s",
        )

    for ax, ylabel, title in [
        (ax1, "Balanced Accuracy (5-fold CV)", "Balanced Accuracy vs TTE"),
        (ax2, "ROC-AUC (5-fold CV)", "ROC-AUC vs TTE"),
    ]:
        ax.set_xlabel("Time-To-Event (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.set_ylim(0.5, 1.02)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Logistic Regression Baseline – Performance vs Prediction Horizon", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, "performance_vs_tte.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_cross_tte_family_importance(summary_rows, window, feature_names, output_dir):
    """
    Grouped bar chart: TTE on x-axis, one coloured bar per feature family.

    Shows at a glance how the relative importance of each feature family
    shifts as the prediction horizon grows.
    """
    rows = sorted(
        [r for r in summary_rows if r["window"] == window],
        key=lambda r: r["tte"],
    )
    if len(rows) < 2:
        return

    families = get_feature_families()
    family_names = list(families.keys())
    family_colors = {
        "Goal-Oriented":        "#C44E52",
        "Kinematic":            "#55A868",
        "Trajectory & Orient.": "#DD8452",
        "Bounding Box":         "#4C72B0",
    }

    tte_labels = [f"{r['tte']:.1f}s" for r in rows]
    n_ttes = len(rows)
    n_families = len(family_names)

    scores = np.zeros((n_ttes, n_families))
    for i, row in enumerate(rows):
        abs_coefs = np.abs(row["coefs"])
        for j, family in enumerate(family_names):
            scores[i, j] = float(np.mean(abs_coefs[families[family]]))

    bar_width = 0.18
    x = np.arange(n_ttes)

    fig, ax = plt.subplots(figsize=(max(8, n_ttes * 1.8), 5))

    for fi, family in enumerate(family_names):
        offset = (fi - n_families / 2.0 + 0.5) * bar_width
        ax.bar(
            x + offset, scores[:, fi], width=bar_width,
            label=family,
            color=family_colors.get(family, f"C{fi}"),
            edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tte_labels, fontsize=10)
    ax.set_xlabel("Time-To-Event (s)")
    ax.set_ylabel("Mean |coefficient| per feature")
    ax.set_title(
        f"Feature Family Importance Across TTE – Logistic Regression\n"
        f"(window = {window}s)"
    )
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    fname = f"family_importance_cross_tte_win{window:.1f}s".replace(".", "_") + ".png"
    path = os.path.join(output_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_cross_tte_family_heatmap(summary_rows, window, output_dir):
    """
    Heatmap version of family importance across TTE.

    Rows = feature families, columns = TTE, values = mean |coefficient|
    within each family.
    """
    rows = sorted(
        [r for r in summary_rows if r["window"] == window],
        key=lambda r: r["tte"],
    )
    if len(rows) < 2:
        return

    families = get_feature_families()
    family_names = list(families.keys())
    tte_labels = [f"TTE {r['tte']:.1f}s" for r in rows]

    scores = np.zeros((len(family_names), len(rows)))
    for j, row in enumerate(rows):
        abs_coefs = np.abs(row["coefs"])
        for i, family in enumerate(family_names):
            scores[i, j] = float(np.mean(abs_coefs[families[family]]))

    annot = np.vectorize(lambda x: f"{x:.2f}")(scores)

    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 1.4), max(3.2, len(family_names) * 0.8)))
    sns.heatmap(
        scores,
        annot=annot, fmt="",
        cmap="YlOrRd",
        xticklabels=tte_labels,
        yticklabels=family_names,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Mean |coefficient| per feature"},
        ax=ax,
    )
    ax.set_xlabel("Time-To-Event (s)")
    ax.set_ylabel("Feature Family")
    ax.set_title(
        f"Feature Family Importance Heatmap Across TTE – Logistic Regression\n"
        f"(window = {window}s)"
    )
    plt.tight_layout()

    fname = f"family_heatmap_cross_tte_win{window:.1f}s".replace(".", "_") + ".png"
    path = os.path.join(output_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_cross_tte_feature_heatmap(summary_rows, window, feature_names, output_dir,
                                   top_n=15):
    """
    Heatmap with the top-N features (by max |coefficient| across any TTE) on
    the y-axis and TTE values on the x-axis.

    Cell colour = signed coefficient (diverging around 0), with the signed
    value annotated inside each cell so direction is immediately visible.
    """
    rows = sorted(
        [r for r in summary_rows if r["window"] == window],
        key=lambda r: r["tte"],
    )
    if len(rows) < 2:
        return

    families = get_feature_families()
    family_lookup = {}
    for family_name, indices in families.items():
        for idx in indices:
            family_lookup[idx] = family_name

    family_colors = {
        "Goal-Oriented":        "#C44E52",
        "Kinematic":            "#55A868",
        "Trajectory & Orient.": "#DD8452",
        "Bounding Box":         "#4C72B0",
    }

    tte_labels = [f"TTE {r['tte']:.1f}s" for r in rows]
    coef_matrix = np.array([r["coefs"] for r in rows])  # shape: (n_ttes, 30)

    max_abs = np.max(np.abs(coef_matrix), axis=0)
    top_indices = np.argsort(max_abs)[::-1][:top_n]

    subset = coef_matrix[:, top_indices].T  # shape: (top_n, n_ttes)
    subset_names = [feature_names[i] for i in top_indices]

    annot = np.empty_like(subset, dtype=object)
    for i in range(subset.shape[0]):
        for j in range(subset.shape[1]):
            annot[i, j] = f"{subset[i, j]:+.2f}"

    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 1.5), top_n * 0.45 + 2))
    vmax = np.max(np.abs(subset))
    sns.heatmap(
        subset,
        annot=annot, fmt="",
        cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        xticklabels=tte_labels,
        yticklabels=subset_names,
        linewidths=0.5, linecolor="white",
        ax=ax,
        cbar_kws={"label": "coefficient (signed)"},
    )

    # Colour y-tick labels by family
    for tick_label, feat_idx in zip(ax.get_yticklabels(), top_indices):
        family = family_lookup.get(feat_idx, "")
        tick_label.set_color(family_colors.get(family, "black"))
        tick_label.set_fontweight("bold")

    ax.set_title(
        f"Top-{top_n} Feature Coefficients Across TTE – Logistic Regression\n"
        f"(window = {window}s, diverging colour scale centered at 0)"
    )
    plt.tight_layout()

    fname = f"feature_heatmap_cross_tte_win{window:.1f}s".replace(".", "_") + ".png"
    path = os.path.join(output_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def save_coefficient_csv(model, feature_names, tte, window, output_dir):
    """
    Save a CSV with the standardised coefficients and their family membership.
    """
    coefs = model.named_steps["logreg"].coef_[0]
    families = get_feature_families()
    family_lookup = {}
    for family_name, indices in families.items():
        for idx in indices:
            family_lookup[idx] = family_name

    rows = []
    for i, name in enumerate(feature_names):
        rows.append({
            "feature": name,
            "family": family_lookup.get(i, "—"),
            "coefficient": float(coefs[i]),
            "abs_coefficient": float(abs(coefs[i])),
        })

    df = pd.DataFrame(rows).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    path = os.path.join(output_dir, f"coefficients_{_slug(tte, window)}.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main – run all experiments
# ──────────────────────────────────────────────────────────────────────────────

def main():
    window_list = (
        list(WINDOW_SECONDS)
        if isinstance(WINDOW_SECONDS, (list, tuple))
        else [WINDOW_SECONDS]
    )

    print("=" * 60)
    print(f"  Loading samples from: {DATA_ROOT}")
    print(f"  Observation window(s): {window_list}s")
    print(f"  TTE values          : {TTE_VALUES}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load all valid samples once (the same samples are reused for every TTE/window)
    samples = load_all_samples(DATA_ROOT)

    n_enter = sum(1 for s in samples if s.get("label") == "enter")
    n_pass  = sum(1 for s in samples if s.get("label") == "pass")
    print(f"\nLoaded {len(samples)} samples total  (enter={n_enter}, pass={n_pass})")

    if len(samples) == 0:
        print("No samples found. Please check the DATA_ROOT path.")
        return

    feature_names = get_feature_names()
    summary_rows = []

    for window in window_list:
        # Show how many samples survive each TTE threshold for this window.
        # Samples whose trajectory is too short are excluded from that TTE's
        # dataset. This is important to report: a large drop at high TTE
        # means results are based on fewer, potentially easier samples.
        print_sample_counts_per_tte(samples, TTE_VALUES, window)

        for tte in TTE_VALUES:
            print("\n" + "=" * 60)
            print(f"  TTE = {tte:.1f}s  |  window = {window}s")
            print("=" * 60)

            X, y = build_feature_matrix(
                samples, tte_seconds=tte, window_seconds=window
            )
            print(f"\n  Feature matrix shape: {X.shape}  (samples × features)")

            unique, counts = np.unique(y, return_counts=True)
            class_counts = dict(zip(unique.tolist(), counts.tolist()))
            print(f"  Class distribution  : pass={class_counts.get(0, 0)}, enter={class_counts.get(1, 0)}")

            cv_res = run_cross_validation(X, y)
            hold_res = run_holdout_evaluation(X, y)

            coefs = hold_res["model"].named_steps["logreg"].coef_[0].copy()

            summary_rows.append({
                "tte": float(tte),
                "window": float(window),
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "cv_acc_mean": cv_res["acc_mean"],
                "cv_acc_std": cv_res["acc_std"],
                "cv_bal_acc_mean": cv_res["bal_acc_mean"],
                "cv_bal_acc_std": cv_res["bal_acc_std"],
                "cv_f1_mean": cv_res["f1_mean"],
                "cv_f1_std": cv_res["f1_std"],
                "cv_auc_mean": cv_res["auc_mean"],
                "cv_auc_std": cv_res["auc_std"],
                "hold_acc": hold_res["acc"],
                "hold_bal_acc": hold_res["bal_acc"],
                "hold_f1": hold_res["f1"],
                "hold_auc": hold_res["auc"],
                "cm_tn": hold_res["cm_tn"],
                "cm_fp": hold_res["cm_fp"],
                "cm_fn": hold_res["cm_fn"],
                "cm_tp": hold_res["cm_tp"],
                "coefs": coefs,
            })

            # Per-(TTE, window) plots
            run_dir = os.path.join(OUTPUT_DIR, _slug(tte, window))
            os.makedirs(run_dir, exist_ok=True)

            model = hold_res["model"]
            plot_coefficient_importance(model, feature_names, tte, window, run_dir)
            plot_family_coefficient_importance(model, tte, window, run_dir)
            plot_confusion_matrix_heatmap(
                hold_res["y_test"], hold_res["y_pred"], tte, window, run_dir,
            )
            plot_roc_curve(model, hold_res["X_test"], hold_res["y_test"],
                           tte, window, run_dir)
            save_coefficient_csv(model, feature_names, tte, window, run_dir)

    # Save summary CSV (exclude numpy arrays that don't serialise well)
    csv_rows = [{k: v for k, v in r.items() if k != "coefs"} for r in summary_rows]
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    pd.DataFrame(csv_rows).to_csv(summary_path, index=False)

    # Cross-TTE plots (only useful if multiple TTE values ran)
    if len(summary_rows) > 1:
        plot_performance_vs_tte(summary_rows, OUTPUT_DIR)

        for window in window_list:
            plot_cross_tte_family_importance(
                summary_rows, window, feature_names, OUTPUT_DIR,
            )
            plot_cross_tte_family_heatmap(
                summary_rows, window, OUTPUT_DIR,
            )
            plot_cross_tte_feature_heatmap(
                summary_rows, window, feature_names, OUTPUT_DIR,
            )

    print("\n" + "=" * 60)
    print("  All experiments done.")
    print(f"  Summary saved to: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()