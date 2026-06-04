#!/usr/bin/env python3
"""
train_ablation.py

Structured ablation study for pedestrian door-entry intention prediction
using neural networks (MLP, GRU, CNN).

This script builds on insights from the logistic regression and random forest
baselines.  Experiments are organised into four groups that each answer a
specific research question:

  GROUP A — Cross-model anchoring
      Do neural networks improve over the logistic-regression baseline when
      given the exact same features?  Does temporal modelling (GRU/CNN) add
      value beyond single-frame classification (MLP)?

      A1  HC-30           : The full 30-dim aggregated feature vector used by
                            logistic regression, presented as a single-frame
                            input.  Meaningful mainly for MLP (direct comparison
                            to LogReg); GRU/CNN see only 1 timestep.
      A2  traj-6          : The 6-feature per-frame engineered trajectory
                            baseline (dist_to_door, closure_rate, vx, vy,
                            rel_angle_to_door, bbox_h).
      A3  core-3          : The 3 per-frame signals whose aggregated statistics
                            survived ALL 5 TTE windows in random-forest
                            iterative SHAP analysis: dist_to_door, closure_rate,
                            rel_angle_to_door.
      A4  core-4          : Core-3 plus vy (present in 4/5 TTE windows).

  GROUP B — Progressive pose ablation  (on the 6-feature trajectory base)
      Does body-pose information improve prediction?  Which body regions
      matter, and does raw landmark representation outperform hand-crafted
      angles?

      Tested progressively:
        trajectory only → + head → + torso_head → + full_body
      In both angle and raw-landmark representations.

  GROUP C — Insight-driven combinations
      Can a small, interpretable feature set (core-3 or core-4 from RF)
      combined with targeted pose features match the full trajectory base?

      C1  core-3 + head (raw)
      C2  core-3 + torso_head (raw)
      C3  core-4 + head (raw)
      C4  core-4 + torso_head (raw)

Fixed parameters
----------------
  Observation window : 0.5 s   (all experiments)
  TTE values         : 0.5, 1.0, 1.5, 2.0, 2.5 s
  CV                 : 5-fold × 3 repeats = 15 evaluations per cell
  Hyperparameters    : hidden=16, dropout=0.1, lr=1e-3, patience=15

Output
------
  results/results_all.csv — one row per (experiment, TTE, model).
"""

import copy
import csv
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

# ──────────────────────────────────────────────────────────────────────────────
# Resolve repo root and make utils/ importable
# ──────────────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]  # pipeline root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.dataset_utils import load_all_samples, print_sample_counts_per_tte
from utils.models import IntentCNN, IntentGRU, IntentMLP
from utils.sequence_dataset import (
    ALL_FEATURE_NAMES,
    SequenceDataset,
    TRAJ_FEATURE_NAMES,
    build_sequences,
    compute_all_pose_features,
    get_input_dim,
    get_seq_len,
    load_pose_sidecar,
    normalise,
    smooth_pose_features,
)
from utils.feature_extractor import (
    DEFAULT_FPS,
    extract_features,
    find_event_frame,
    get_coordinates,
    get_feature_names,
    is_sample_usable,
    compute_distance_to_door,
    compute_closure_rate,
    compute_velocity,
    compute_heading_and_relative_angle,
)


# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────

DATA_ROOT = str(_ROOT / "MasterData")
_RESULTS  = _HERE / "results"


# ──────────────────────────────────────────────────────────────────────────────
# SETTINGS — edit these to change the ablation  (window held fixed at 0.5 s)
# ──────────────────────────────────────────────────────────────────────────────

WINDOW_SECONDS = 0.5                         # fixed observation window
TTE_VALUES     = [0.5, 1.0, 1.5, 2.0, 2.5]  # prediction horizons

# Hyperparameters (same as first ablation study for reproducibility)
CV_FOLDS                = 5
CV_REPEATS              = 3
RANDOM_STATE            = 42
EARLY_STOPPING_PATIENCE = 15
HIDDEN_SIZE  = 16
DROPOUT      = 0.1
EPOCHS       = 150
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-4

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENT DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────
#
# Each experiment is a dict with:
#   key          : unique identifier (used in results CSV)
#   group        : A / B / C
#   label        : human-readable name
#   builder      : "hc30" | "seq" | "seq_subset"
#   pose_mode    : (for builder="seq")  key into ALL_FEATURE_NAMES
#   traj_cols    : (for builder="seq_subset") which trajectory columns to keep
#   pose_features: (for builder="seq_subset") additional pose feature names
#   models       : list of model keys to evaluate ("mlp", "gru", "cnn")
# ──────────────────────────────────────────────────────────────────────────────

# Full trajectory feature names (per-frame, as in sequence_dataset.py)
_FULL_TRAJ_NAMES = list(TRAJ_FEATURE_NAMES)  # 6 features

# Core-3: the 3 per-frame signals whose aggregated statistics survive ALL 5
# TTE windows in the random-forest iterative SHAP analysis.
#   closure_rate_latest  → per-frame: closure_rate
#   dist_to_door_latest  → per-frame: dist_to_door
#   rel_angle_to_door_latest + rel_angle_to_door_mean → per-frame: rel_angle_to_door
_CORE3_NAMES = ["dist_to_door", "closure_rate", "rel_angle_to_door"]

# Core-4: Core-3 plus vy  (vy_latest survived 4/5 TTEs, closure_rate_mean
# is already captured by the core-3 temporal signal).
_CORE4_NAMES = ["dist_to_door", "closure_rate", "vy", "rel_angle_to_door"]


def _raw_xy(*landmark_names):
    """Expand landmark names into [name_x, name_y, ...] feature list."""
    return [f"{n}_{c}" for n in landmark_names for c in ("x", "y")]


# Pose feature lists (for subset builder)
_HEAD_RAW   = _raw_xy("left_ear", "right_ear")                        # 4
_TORSO_HEAD_RAW = _raw_xy("left_hip", "right_hip",
                          "left_shoulder", "right_shoulder",
                          "left_ear", "right_ear")                     # 12
_HEAD_ANGLE     = ["head_angle"]                                       # 1
_TORSO_HEAD_ANGLE = ["hip_angle", "shoulder_angle", "head_angle"]      # 3
_FULL_BODY_ANGLE  = ["hip_angle", "shoulder_angle", "head_angle",
                     "elbow_angle", "wrist_angle",
                     "knee_angle", "ankle_angle"]                      # 7
_FULL_BODY_RAW    = _raw_xy("left_hip", "right_hip",
                            "left_shoulder", "right_shoulder",
                            "left_ear", "right_ear",
                            "left_elbow", "right_elbow",
                            "left_wrist", "right_wrist",
                            "left_knee", "right_knee",
                            "left_ankle", "right_ankle")               # 28


EXPERIMENTS = OrderedDict()

# ── GROUP A: Cross-model anchoring ──────────────────────────────────────────

EXPERIMENTS["A1_hc30"] = {
    "group": "A", "label": "HC-30 (LogReg features)",
    "builder": "hc30",
    # CNN excluded: hc30 produces a single-frame input (seq_len=1), making
    # temporal convolution meaningless.  It also triggers an MPS backend
    # assertion on Apple Silicon (Conv1d on length-1 tensors).
    "models": ["mlp", "gru"],
}

EXPERIMENTS["A2_traj6"] = {
    "group": "A", "label": "Trajectory base (6 per-frame)",
    "builder": "seq", "pose_mode": "trajectory-only",
    "models": ["mlp", "gru", "cnn"],
}

EXPERIMENTS["A3_core3"] = {
    "group": "A", "label": "Core-3 (RF survival: dist, closure, rel_angle)",
    "builder": "seq_subset",
    "traj_cols": _CORE3_NAMES, "pose_features": [],
    "models": ["mlp", "gru", "cnn"],
}

EXPERIMENTS["A4_core4"] = {
    "group": "A", "label": "Core-4 (RF survival: dist, closure, vy, rel_angle)",
    "builder": "seq_subset",
    "traj_cols": _CORE4_NAMES, "pose_features": [],
    "models": ["mlp", "gru", "cnn"],
}

# ── GROUP B: Progressive pose ablation (on 6-feature trajectory base) ──────

EXPERIMENTS["B1_head_angle"] = {
    "group": "B", "label": "Traj-6 + head (angle)",
    "builder": "seq", "pose_mode": "head",
    "models": ["mlp", "gru", "cnn"],
}

EXPERIMENTS["B2_head_raw"] = {
    "group": "B", "label": "Traj-6 + head (raw)",
    "builder": "seq", "pose_mode": "head_raw",
    "models": ["mlp", "gru", "cnn"],
}

EXPERIMENTS["B3_torso_head_angle"] = {
    "group": "B", "label": "Traj-6 + torso_head (angle)",
    "builder": "seq", "pose_mode": "torso_head",
    "models": ["mlp", "gru", "cnn"],
}

EXPERIMENTS["B4_torso_head_raw"] = {
    "group": "B", "label": "Traj-6 + torso_head (raw)",
    "builder": "seq", "pose_mode": "torso_head_raw",
    "models": ["mlp", "gru", "cnn"],
}

EXPERIMENTS["B5_full_body_angle"] = {
    "group": "B", "label": "Traj-6 + full_body (angle)",
    "builder": "seq", "pose_mode": "full_body",
    "models": ["mlp", "gru", "cnn"],
}

EXPERIMENTS["B6_full_body_raw"] = {
    "group": "B", "label": "Traj-6 + full_body (raw)",
    "builder": "seq", "pose_mode": "full_body_raw",
    "models": ["mlp", "gru", "cnn"],
}

# ── GROUP C: Insight-driven combinations ────────────────────────────────────

EXPERIMENTS["C1_core3_head_raw"] = {
    "group": "C", "label": "Core-3 + head (raw)",
    "builder": "seq_subset",
    "traj_cols": _CORE3_NAMES, "pose_features": _HEAD_RAW,
    "models": ["mlp", "gru", "cnn"],
}

EXPERIMENTS["C2_core3_torso_head_raw"] = {
    "group": "C", "label": "Core-3 + torso_head (raw)",
    "builder": "seq_subset",
    "traj_cols": _CORE3_NAMES, "pose_features": _TORSO_HEAD_RAW,
    "models": ["mlp", "gru", "cnn"],
}

EXPERIMENTS["C3_core4_head_raw"] = {
    "group": "C", "label": "Core-4 + head (raw)",
    "builder": "seq_subset",
    "traj_cols": _CORE4_NAMES, "pose_features": _HEAD_RAW,
    "models": ["mlp", "gru", "cnn"],
}

EXPERIMENTS["C4_core4_torso_head_raw"] = {
    "group": "C", "label": "Core-4 + torso_head (raw)",
    "builder": "seq_subset",
    "traj_cols": _CORE4_NAMES, "pose_features": _TORSO_HEAD_RAW,
    "models": ["mlp", "gru", "cnn"],
}


# ──────────────────────────────────────────────────────────────────────────────
# SEQUENCE BUILDERS
# ──────────────────────────────────────────────────────────────────────────────


def build_hc30_sequences(samples, tte_seconds, window_seconds):
    """
    Build the 30-dimensional handcrafted feature vector used by the logistic
    regression baseline, reshaped to (N, 1, 30) for the neural network API.

    This is a single-frame input: the GRU/CNN receive a length-1 sequence, so
    only the MLP comparison is truly apples-to-apples with logistic regression.
    Included for completeness and to quantify the nonlinearity gap.
    """
    from utils.feature_extractor import extract_features, is_sample_usable

    seqs, labels = [], []
    dropped = {"enter": 0, "pass": 0}

    for sample in samples:
        if not is_sample_usable(sample, tte_seconds, window_seconds):
            lbl = str(sample.get("label", "")).lower()
            if lbl in dropped:
                dropped[lbl] += 1
            continue

        vec = extract_features(sample, tte_seconds, window_seconds)
        label = str(sample.get("label", "")).lower()
        seqs.append(np.array(vec, dtype=np.float32).reshape(1, -1))  # (1, 30)
        labels.append(1 if label == "enter" else 0)

    if dropped["enter"] > 0 or dropped["pass"] > 0:
        print(
            f"  Dropped (too short for TTE={tte_seconds:.1f}s): "
            f"enter={dropped['enter']}, pass={dropped['pass']}"
        )

    X = np.array(seqs, dtype=np.float32)   # (N, 1, 30)
    y = np.array(labels, dtype=np.int64)
    return X, y


# Column indices in the 6-feature trajectory matrix for subset extraction
_TRAJ_COL_INDEX = {name: i for i, name in enumerate(TRAJ_FEATURE_NAMES)}


def build_subset_sequences(samples, tte_seconds, window_seconds,
                           traj_cols, pose_feature_names):
    """
    Build sequences with a custom subset of trajectory features and optional
    pose features.

    This enables the core-3/core-4 experiments (fewer trajectory features than
    the full 6) and the insight-driven combinations (core features + pose).

    Steps:
      1. Build the full 6-feature trajectory sequence using the existing
         sequence_dataset machinery.
      2. Keep only the requested trajectory columns.
      3. If pose features are requested, build a matching pose mode and
         extract those columns.

    Parameters
    ----------
    traj_cols          : list of str — which of the 6 trajectory features to keep
    pose_feature_names : list of str — pose features to append (can be empty)
    """
    # Determine which trajectory column indices to keep
    traj_indices = [_TRAJ_COL_INDEX[name] for name in traj_cols]
    n_traj = len(traj_indices)

    use_pose = len(pose_feature_names) > 0
    n_pose   = len(pose_feature_names)
    n_total  = n_traj + n_pose

    seq_len = get_seq_len(window_seconds)
    seqs, labels = [], []
    dropped = {"enter": 0, "pass": 0}

    for sample in samples:
        if not is_sample_usable(sample, tte_seconds, window_seconds):
            lbl = str(sample.get("label", "")).lower()
            if lbl in dropped:
                dropped[lbl] += 1
            continue

        fps   = float(sample.get("fps", DEFAULT_FPS) or DEFAULT_FPS)
        label = str(sample.get("label", "")).lower()
        k     = max(2, int(round(window_seconds * fps)))
        tte_frames = int(round(tte_seconds * fps))

        # Door position
        door = sample.get("door_center")
        if door is not None and len(door) == 2:
            door_x, door_y = float(door[0]), float(door[1])
        else:
            door_x = float(sample.get("frame_width",  640)) / 2.0
            door_y = float(sample.get("frame_height", 480)) - 1.0

        # Parse trajectory
        frames = sample.get("frames", [])
        x, y_coord, x1, y1, x2, y2 = get_coordinates(frames)
        x      = np.array(x,      dtype=np.float32)
        y_coord = np.array(y_coord, dtype=np.float32)
        x1     = np.array(x1,     dtype=np.float32)
        y1     = np.array(y1,     dtype=np.float32)
        x2     = np.array(x2,     dtype=np.float32)
        y2     = np.array(y2,     dtype=np.float32)

        t_event   = find_event_frame(label, x, y_coord, door_x, door_y)
        t_predict = t_event - tte_frames

        # Compute all 6 trajectory features per frame
        d_door       = compute_distance_to_door(x, y_coord, door_x, door_y)
        closure      = compute_closure_rate(d_door, fps, k=1)
        vx, vy       = compute_velocity(x, y_coord, fps, k=1)
        _, rel_angle = compute_heading_and_relative_angle(
            vx, vy, x, y_coord, door_x, door_y
        )
        bbox_h = (y2 - y1).astype(np.float32)

        # Stack full 6 features → (n_frames, 6)
        traj_full = np.column_stack(
            [d_door, closure, vx, vy, rel_angle, bbox_h]
        ).astype(np.float32)

        # Select only the requested columns → (n_frames, n_traj)
        traj_subset = traj_full[:, traj_indices]

        # Pose features (if requested)
        pose_lookup = load_pose_sidecar(sample) if use_pose else {}
        frame_idx_list = [f.get("frame_index", i) for i, f in enumerate(frames)]
        frame_w = float(sample.get("frame_width",  640))
        frame_h = float(sample.get("frame_height", 480))

        # Build raw sequence over the observation window
        w_start = t_predict - k + 1
        raw_seq = np.zeros((k, n_total), dtype=np.float32)

        for i, t in enumerate(range(w_start, t_predict + 1)):
            if 0 <= t < len(x):
                row = list(traj_subset[t])

                if use_pose:
                    frame_idx = frame_idx_list[t] if t < len(frame_idx_list) else t
                    landmarks = pose_lookup.get(frame_idx)

                    if landmarks:
                        all_feats = compute_all_pose_features(
                            landmarks, door_x, door_y, frame_w, frame_h
                        )
                        pose_vals = [all_feats.get(name, 0.0)
                                     for name in pose_feature_names]
                    else:
                        pose_vals = [0.0] * n_pose

                    row.extend(pose_vals)

                raw_seq[i] = row

        # Smooth pose columns (causal moving average)
        if use_pose:
            raw_seq = smooth_pose_features(raw_seq, n_traj=n_traj)

        # Pad or trim to fixed seq_len
        seq = np.zeros((seq_len, n_total), dtype=np.float32)
        if k >= seq_len:
            seq[:] = raw_seq[-seq_len:]
        else:
            seq[-k:] = raw_seq

        np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        seqs.append(seq)
        labels.append(1 if label == "enter" else 0)

    if dropped["enter"] > 0 or dropped["pass"] > 0:
        print(
            f"  Dropped (too short for TTE={tte_seconds:.1f}s): "
            f"enter={dropped['enter']}, pass={dropped['pass']}"
        )

    X = np.array(seqs, dtype=np.float32)
    y_out = np.array(labels, dtype=np.int64)
    return X, y_out


def build_experiment_data(experiment, samples, tte_seconds):
    """
    Route to the correct builder based on the experiment specification.

    Returns (X, y, input_dim, feature_description).
    """
    builder = experiment["builder"]

    if builder == "hc30":
        X, y = build_hc30_sequences(samples, tte_seconds, WINDOW_SECONDS)
        return X, y, 30, get_feature_names()

    elif builder == "seq":
        pose_mode = experiment["pose_mode"]
        X, y = build_sequences(
            samples, tte_seconds, WINDOW_SECONDS, pose_mode=pose_mode
        )
        feat_names = ALL_FEATURE_NAMES[pose_mode]
        return X, y, len(feat_names), feat_names

    elif builder == "seq_subset":
        traj_cols      = experiment["traj_cols"]
        pose_features  = experiment["pose_features"]
        X, y = build_subset_sequences(
            samples, tte_seconds, WINDOW_SECONDS,
            traj_cols, pose_features,
        )
        feat_names = traj_cols + pose_features
        return X, y, len(feat_names), feat_names

    else:
        raise ValueError(f"Unknown builder: {builder}")


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP  (identical to train_models.py for reproducibility)
# ──────────────────────────────────────────────────────────────────────────────

MODEL_CLASSES = {
    "mlp": IntentMLP,
    "gru": IntentGRU,
    "cnn": IntentCNN,
}


def class_weight(y_train):
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)


def train_one_fold(model, X_fold_train, y_fold_train, X_fold_eval, y_fold_eval,
                   fold_seed):
    """Train one fold and return evaluation metrics on the held-out set."""
    X_sub_train, X_sub_val, y_sub_train, y_sub_val = train_test_split(
        X_fold_train, y_fold_train,
        test_size=0.15,
        stratify=y_fold_train,
        random_state=fold_seed,
    )

    X_sub_train_n, X_sub_val_n, mean, std = normalise(X_sub_train, X_sub_val)

    X_fold_eval_n = (X_fold_eval - mean) / std
    np.nan_to_num(X_fold_eval_n, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    shuffle_gen  = torch.Generator().manual_seed(fold_seed)
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

    model     = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight(y_sub_train))

    best_val_loss     = float("inf")
    best_weights      = copy.deepcopy(model.state_dict())
    best_epoch        = 1
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
            best_val_loss     = val_loss
            best_weights      = copy.deepcopy(model.state_dict())
            best_epoch        = epoch
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
    probs  = torch.sigmoid(torch.tensor(logits)).numpy()
    preds  = (probs >= 0.5).astype(int)

    return {
        "bal_acc":       balanced_accuracy_score(labels, preds),
        "roc_auc":       roc_auc_score(labels, probs),
        "f1":            f1_score(labels, preds, zero_division=0),
        "precision":     precision_score(labels, preds, zero_division=0),
        "recall":        recall_score(labels, preds, zero_division=0),
        "best_epoch":    best_epoch,
        "best_val_loss": best_val_loss,
    }


def run_cv(X, y, model_class, model_name, input_dim):
    """Repeated stratified k-fold cross-validation."""
    n_evals = CV_REPEATS * CV_FOLDS
    print(f"\n  --- {CV_REPEATS}x{CV_FOLDS}-fold CV  [{model_name}]  "
          f"({n_evals} evals) ---")

    bal_accs, roc_aucs, f1_scores = [], [], []
    precisions, recalls, best_epochs = [], [], []

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

            model   = model_class(input_dim, hidden_size=HIDDEN_SIZE, dropout=DROPOUT)
            metrics = train_one_fold(model, X_train, y_train, X_val, y_val,
                                     fold_seed)

            bal_accs.append(metrics["bal_acc"])
            roc_aucs.append(metrics["roc_auc"])
            f1_scores.append(metrics["f1"])
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])
            best_epochs.append(metrics["best_epoch"])

            print(
                f"    Rep {repeat+1}/{CV_REPEATS} Fold {fold}/{CV_FOLDS}  "
                f"bal_acc={metrics['bal_acc']:.4f}  "
                f"roc_auc={metrics['roc_auc']:.4f}  "
                f"f1={metrics['f1']:.4f}  "
                f"best_epoch={metrics['best_epoch']:>3}  "
                f"(eval n={len(y_val)})"
            )

    return {
        "bal_acc_mean":   float(np.mean(bal_accs)),
        "bal_acc_std":    float(np.std(bal_accs)),
        "roc_auc_mean":   float(np.mean(roc_aucs)),
        "roc_auc_std":    float(np.std(roc_aucs)),
        "f1_mean":        float(np.mean(f1_scores)),
        "f1_std":         float(np.std(f1_scores)),
        "precision_mean": float(np.mean(precisions)),
        "precision_std":  float(np.std(precisions)),
        "recall_mean":    float(np.mean(recalls)),
        "recall_std":     float(np.std(recalls)),
        "avg_best_epoch": float(np.mean(best_epochs)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CSV OUTPUT
# ──────────────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "experiment_key", "group", "label",
    "tte_s", "model",
    "input_dim", "n_traj_features", "n_pose_features",
    "feature_list",
    "bal_acc", "bal_acc_std",
    "roc_auc", "roc_auc_std",
    "f1", "f1_std",
    "precision", "precision_std",
    "recall", "recall_std",
    "avg_epoch", "coverage_pct",
]


def save_results_csv(rows, out_path=None):
    if out_path is None:
        out_path = str(_RESULTS / "results_all.csv")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Results saved to: {out_path}  ({len(rows)} rows)")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    n_experiments = len(EXPERIMENTS)
    n_total_cells = sum(
        len(exp["models"]) * len(TTE_VALUES) for exp in EXPERIMENTS.values()
    )

    print("=" * 72)
    print("  Pedestrian Intention – Structured Ablation Study")
    print(f"  Device       : {DEVICE}")
    print(f"  Data root    : {DATA_ROOT}")
    print(f"  Window       : {WINDOW_SECONDS}s (fixed)")
    print(f"  TTE values   : {TTE_VALUES}")
    print(f"  Experiments  : {n_experiments}  ({n_total_cells} model×TTE cells)")
    print(f"  Epochs       : {EPOCHS}  |  hidden={HIDDEN_SIZE}  "
          f"dropout={DROPOUT}")
    print(f"  CV           : {CV_REPEATS}x{CV_FOLDS}-fold  "
          f"|  patience={EARLY_STOPPING_PATIENCE}")
    print("=" * 72)

    # ── Load data ────────────────────────────────────────────────────────────
    samples = load_all_samples(DATA_ROOT)
    n_enter = sum(1 for s in samples if s.get("label") == "enter")
    n_pass  = sum(1 for s in samples if s.get("label") == "pass")
    print(f"\nLoaded {len(samples)} samples  (enter={n_enter}, pass={n_pass})")

    if len(samples) == 0:
        print("No samples found – check DATA_ROOT.")
        return

    print_sample_counts_per_tte(samples, TTE_VALUES, WINDOW_SECONDS)

    # ── Run experiments ──────────────────────────────────────────────────────
    all_rows = []

    for exp_idx, (exp_key, exp) in enumerate(EXPERIMENTS.items(), start=1):
        group = exp["group"]
        label = exp["label"]

        print("\n" + "=" * 72)
        print(f"  [{exp_idx}/{n_experiments}]  EXPERIMENT {exp_key}")
        print(f"  Group {group}: {label}")
        print("=" * 72)

        for tte in TTE_VALUES:
            print("\n" + "-" * 72)
            print(f"  TTE = {tte:.1f}s  |  {exp_key}")

            X, y, input_dim, feat_names = build_experiment_data(
                exp, samples, tte
            )

            n_total     = len(samples)
            n_remaining = len(y)
            coverage    = 100.0 * n_remaining / max(n_total, 1)

            # Determine feature counts
            if exp["builder"] == "hc30":
                n_traj = 30
                n_pose = 0
            elif exp["builder"] == "seq":
                n_traj = len(TRAJ_FEATURE_NAMES)
                n_pose = input_dim - n_traj
            else:  # seq_subset
                n_traj = len(exp["traj_cols"])
                n_pose = len(exp.get("pose_features", []))

            feat_str = ", ".join(feat_names) if isinstance(feat_names, list) else str(feat_names)

            print(f"  Features ({input_dim}): {feat_str[:100]}{'...' if len(feat_str)>100 else ''}")
            print(f"  Samples: {n_remaining}/{n_total}  ({coverage:.1f}%)  "
                  f"pass={int((y==0).sum())}  enter={int((y==1).sum())}")

            if n_remaining == 0 or len(np.unique(y)) < 2:
                print("  Skipping: insufficient class diversity.")
                continue

            for model_key in exp["models"]:
                model_class = MODEL_CLASSES[model_key]

                _tmp = model_class(input_dim, hidden_size=HIDDEN_SIZE,
                                   dropout=DROPOUT)
                n_params = sum(p.numel() for p in _tmp.parameters()
                               if p.requires_grad)
                print(f"\n  {model_key.upper()} params: {n_params}  "
                      f"(input_dim={input_dim})")
                del _tmp

                run_label = (f"{model_key.upper()} [{exp_key}] "
                             f"TTE={tte:.1f}s")
                res = run_cv(X, y, model_class, run_label, input_dim)

                all_rows.append({
                    "experiment_key":  exp_key,
                    "group":           group,
                    "label":           label,
                    "tte_s":           tte,
                    "model":           model_key,
                    "input_dim":       input_dim,
                    "n_traj_features": n_traj,
                    "n_pose_features": n_pose,
                    "feature_list":    "|".join(feat_names) if isinstance(feat_names, list) else "",
                    "bal_acc":         round(res["bal_acc_mean"], 4),
                    "bal_acc_std":     round(res["bal_acc_std"], 4),
                    "roc_auc":        round(res["roc_auc_mean"], 4),
                    "roc_auc_std":    round(res["roc_auc_std"], 4),
                    "f1":              round(res["f1_mean"], 4),
                    "f1_std":          round(res["f1_std"], 4),
                    "precision":       round(res["precision_mean"], 4),
                    "precision_std":   round(res["precision_std"], 4),
                    "recall":          round(res["recall_mean"], 4),
                    "recall_std":      round(res["recall_std"], 4),
                    "avg_epoch":       round(res["avg_best_epoch"], 1),
                    "coverage_pct":    round(coverage, 1),
                })

                # Save incrementally so we don't lose results on crash
                save_results_csv(all_rows)

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n\n" + "=" * 72)
    print(f"  ABLATION SUMMARY  –  balanced accuracy "
          f"({CV_REPEATS}x{CV_FOLDS}-fold CV)")
    print("=" * 72)

    for group in ["A", "B", "C"]:
        group_exps = [(k, e) for k, e in EXPERIMENTS.items()
                      if e["group"] == group]
        if not group_exps:
            continue

        group_label = {
            "A": "Cross-model anchoring",
            "B": "Progressive pose ablation",
            "C": "Insight-driven combinations",
        }[group]
        print(f"\n  GROUP {group}: {group_label}")

        for model_key in ["mlp", "gru", "cnn"]:
            print(f"\n    Model: {model_key.upper()}")
            header = f"    {'TTE':>5}"
            for exp_key, _ in group_exps:
                short = exp_key.split("_", 1)[1][:12]
                header += f"  {short:>12}"
            print(header)
            print("    " + "-" * (6 + 14 * len(group_exps)))

            for tte in TTE_VALUES:
                row_str = f"    {tte:>4.1f}s"
                for exp_key, _ in group_exps:
                    # Find matching result
                    match = [r for r in all_rows
                             if r["experiment_key"] == exp_key
                             and r["model"] == model_key
                             and r["tte_s"] == tte]
                    if match:
                        row_str += f"  {match[0]['bal_acc']:>12.4f}"
                    else:
                        row_str += f"  {'---':>12}"
                print(row_str)

    save_results_csv(all_rows)

    print("\n" + "=" * 72)
    print("  Structured ablation complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
