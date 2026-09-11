#!/usr/bin/env python3
"""
train_demo_model.py  —  Standalone training script for the open-house demo.

Trains a Core-3 + head (raw) GRU model at TTE=1.5 s and exports:
    demo_open_house/weights/gru_core3head.pt  — model state_dict
    demo_open_house/weights/norm_stats.npz    — z-score mean & std (shape (7,) each)

This script is entirely self-contained — it does NOT import from
the pipeline's utils/ module.  All required logic is inlined below.

Data: cleaned trajectories in MasterData (gitignored; produced by
0_data_cleaning/build_masterdata.py). Layout:

    MasterData/
      enter/   …/*.json   trajectory samples (label enter)
      pass/    …/*.json   trajectory samples (label pass)

Optional *_pose.json sidecars next to each trajectory improve head features.

Run from the repository root (recommended):

    python demo_open_house/train_demo_model.py

That uses MasterData/ at the repo root and writes weights under demo_open_house/weights/.
Override data location if needed:

    python demo_open_house/train_demo_model.py --data /path/to/MasterData

You can also run inside demo_open_house/; defaults still resolve to ../MasterData and ./weights/.
"""

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset


# ──────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────

DEFAULT_FPS = 13.0
WINDOW_SECONDS = 0.5
TTE_SECONDS = 1.5            # prediction horizon for the demo

# Core-3 trajectory features + head raw pose features
TRAJ_FEATURE_NAMES = [
    "dist_to_door", "closure_rate", "vx", "vy",
    "rel_angle_to_door", "bbox_h",
]
CORE3_NAMES = ["dist_to_door", "closure_rate", "rel_angle_to_door"]
HEAD_RAW_NAMES = ["left_ear_x", "left_ear_y", "right_ear_x", "right_ear_y"]

# Feature column indices within the full 6-feature trajectory matrix
_TRAJ_COL_INDEX = {name: i for i, name in enumerate(TRAJ_FEATURE_NAMES)}
CORE3_INDICES = [_TRAJ_COL_INDEX[n] for n in CORE3_NAMES]

INPUT_DIM = len(CORE3_NAMES) + len(HEAD_RAW_NAMES)  # 3 + 4 = 7

# Hyperparameters (matched to train_ablation.py)
HIDDEN_SIZE = 16
DROPOUT = 0.1
EPOCHS = 150
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 15
CV_FOLDS = 5
CV_REPEATS = 3
RANDOM_STATE = 42

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# Repo layout: demo_open_house/train_demo_model.py → parent.parent == pipeline root
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = _REPO_ROOT / "MasterData"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "weights"


# ──────────────────────────────────────────────────────────────────
# MODEL (identical to the utils/models.py IntentGRU)
# ──────────────────────────────────────────────────────────────────

class IntentGRU(nn.Module):
    def __init__(self, input_dim, hidden_size=32, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, final_hidden = self.gru(x)
        h = final_hidden.squeeze(0)
        h = self.dropout(h)
        return self.classifier(h).squeeze(1)


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ──────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────

def is_valid_sample(sample):
    if not isinstance(sample, dict):
        return False
    if "frames" not in sample or not sample["frames"]:
        return False
    label = str(sample.get("label", "")).lower()
    return label in ("enter", "pass")


def load_all_samples(data_root):
    samples = []
    for split_folder in ("enter", "pass"):
        folder_path = os.path.join(data_root, split_folder)
        if not os.path.exists(folder_path):
            print(f"  Warning: folder not found: {folder_path}")
            continue
        for root, dirs, files in os.walk(folder_path):
            dirs[:] = [
                d for d in dirs
                if d.lower() not in ("exit", "removed", "remove", "corrupt")
            ]
            for filename in sorted(files):
                if not filename.endswith(".json"):
                    continue
                if filename.endswith("_pose.json"):
                    continue
                json_path = os.path.join(root, filename)
                try:
                    with open(json_path, "r") as f:
                        sample = json.load(f)
                except Exception as e:
                    print(f"  Could not read {json_path}: {e}")
                    continue
                if is_valid_sample(sample):
                    sample["_source_path"] = json_path
                    samples.append(sample)
    return samples


# ──────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION (inlined from feature_extractor.py)
# ──────────────────────────────────────────────────────────────────

def get_coordinates(frames):
    x, y = [], []
    x1, y1, x2, y2 = [], [], [], []
    for frame in frames:
        center = frame.get("center")
        bbox = frame.get("bbox")
        if center is not None and len(center) == 2:
            x.append(float(center[0]))
            y.append(float(center[1]))
        else:
            x.append(float("nan"))
            y.append(float("nan"))
        if bbox is not None and len(bbox) == 4:
            x1.append(float(bbox[0]))
            y1.append(float(bbox[1]))
            x2.append(float(bbox[2]))
            y2.append(float(bbox[3]))
        else:
            x1.append(float("nan"))
            y1.append(float("nan"))
            x2.append(float("nan"))
            y2.append(float("nan"))
    return x, y, x1, y1, x2, y2


def find_event_frame(label, x, y, door_x, door_y):
    valid = [
        i for i in range(len(x))
        if not math.isnan(x[i]) and not math.isnan(y[i])
    ]
    if not valid:
        return -1
    if label == "pass":
        return valid[-1]
    if label == "enter":
        best_idx = valid[0]
        best_dist = float("inf")
        for i in valid:
            dist = math.sqrt((x[i] - door_x) ** 2 + (y[i] - door_y) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx
    return -1


def compute_distance_to_door(x, y, door_x, door_y):
    return [
        float("nan") if math.isnan(x[i]) or math.isnan(y[i])
        else math.sqrt((x[i] - door_x) ** 2 + (y[i] - door_y) ** 2)
        for i in range(len(x))
    ]


def compute_closure_rate(d_door, fps, k=1):
    dt_k = k / fps
    result = []
    for i in range(len(d_door)):
        prev = i - k
        if prev < 0 or math.isnan(d_door[i]) or math.isnan(d_door[prev]):
            result.append(float("nan"))
        else:
            result.append((d_door[i] - d_door[prev]) / dt_k)
    return result


def compute_velocity(x, y, fps, k=1):
    dt_k = k / fps
    vx, vy = [], []
    for i in range(len(x)):
        prev = i - k
        if prev < 0 or math.isnan(x[i]) or math.isnan(x[prev]):
            vx.append(float("nan"))
        else:
            vx.append((x[i] - x[prev]) / dt_k)
        if prev < 0 or math.isnan(y[i]) or math.isnan(y[prev]):
            vy.append(float("nan"))
        else:
            vy.append((y[i] - y[prev]) / dt_k)
    return vx, vy


def compute_heading_and_relative_angle(vx, vy, x, y, door_x, door_y):
    heading, rel_angle = [], []
    for i in range(len(vx)):
        if math.isnan(vx[i]) or math.isnan(vy[i]):
            heading.append(float("nan"))
            rel_angle.append(float("nan"))
            continue
        theta = math.atan2(vy[i], vx[i])
        heading.append(theta)
        if math.isnan(x[i]) or math.isnan(y[i]):
            rel_angle.append(float("nan"))
        else:
            theta_door = math.atan2(door_y - y[i], door_x - x[i])
            diff = theta - theta_door
            diff_wrapped = math.atan2(math.sin(diff), math.cos(diff))
            rel_angle.append(abs(diff_wrapped))
    return heading, rel_angle


def is_sample_usable(sample, tte_seconds, window_seconds):
    fps = float(sample.get("fps", DEFAULT_FPS) or DEFAULT_FPS)
    frames = sample.get("frames", [])
    if not frames:
        return False
    label = str(sample.get("label", "")).lower()
    if label not in ("enter", "pass"):
        return False
    k = max(2, int(round(window_seconds * fps)))
    tte_frames = int(round(tte_seconds * fps))
    door = sample.get("door_center")
    if door is not None and len(door) == 2:
        door_x, door_y = float(door[0]), float(door[1])
    else:
        door_x = float(sample.get("frame_width", 640)) / 2.0
        door_y = float(sample.get("frame_height", 480)) - 1.0
    x, y, *_ = get_coordinates(frames)
    t_event = find_event_frame(label, x, y, door_x, door_y)
    if t_event < 0:
        return False
    t_predict = t_event - tte_frames
    return t_predict >= k - 1


# ──────────────────────────────────────────────────────────────────
# POSE SIDECAR LOADING
# ──────────────────────────────────────────────────────────────────

def load_pose_sidecar(sample):
    source_path = sample.get("_source_path")
    if not source_path:
        return {}
    pose_path = Path(source_path).parent / (
        Path(source_path).stem + "_pose.json"
    )
    if not pose_path.exists():
        return {}
    with open(pose_path, "r") as f:
        pose_data = json.load(f)
    lookup = {}
    for frame in pose_data.get("frames", []):
        fi = frame["frame_index"]
        lookup[fi] = frame["landmarks"] if frame.get("detected") else None
    return lookup


def compute_raw_ear_features(landmarks):
    """Extract normalised left_ear and right_ear x/y from pose sidecar landmarks.

    Sidecars from extract_pose.py store landmarks as a dict keyed by name
    (e.g. left_ear, right_ear), matching utils/sequence_dataset.py — not
    as a MediaPipe index list.
    """
    zero = {
        "left_ear_x": 0.0, "left_ear_y": 0.0,
        "right_ear_x": 0.0, "right_ear_y": 0.0,
    }
    if not landmarks:
        return zero
    try:
        le = landmarks["left_ear"]
        re = landmarks["right_ear"]
        return {
            "left_ear_x": float(le.get("x", 0.0)),
            "left_ear_y": float(le.get("y", 0.0)),
            "right_ear_x": float(re.get("x", 0.0)),
            "right_ear_y": float(re.get("y", 0.0)),
        }
    except (KeyError, TypeError, AttributeError):
        return zero


def smooth_pose_features(seq, n_traj=3, kernel_size=3):
    """Causal moving average on pose columns only."""
    if seq.shape[1] <= n_traj or seq.shape[0] < 2 or kernel_size < 2:
        return seq
    smoothed = seq.copy()
    for col in range(n_traj, seq.shape[1]):
        for t in range(seq.shape[0]):
            start = max(0, t - kernel_size + 1)
            smoothed[t, col] = seq[start: t + 1, col].mean()
    return smoothed


# ──────────────────────────────────────────────────────────────────
# SEQUENCE BUILDING (Core-3 + head raw)
# ──────────────────────────────────────────────────────────────────

def get_seq_len(window_seconds):
    return max(2, int(round(window_seconds * DEFAULT_FPS)))


def build_sequences(samples, tte_seconds, window_seconds):
    """
    Build sequences for Core-3 + head (raw) — 7 features:
        dist_to_door, closure_rate, rel_angle_to_door,
        left_ear_x, left_ear_y, right_ear_x, right_ear_y
    """
    seq_len = get_seq_len(window_seconds)
    n_total = INPUT_DIM  # 7
    n_traj = len(CORE3_NAMES)  # 3

    seqs, labels = [], []
    dropped = {"enter": 0, "pass": 0}

    for sample in samples:
        if not is_sample_usable(sample, tte_seconds, window_seconds):
            lbl = str(sample.get("label", "")).lower()
            if lbl in dropped:
                dropped[lbl] += 1
            continue

        fps = float(sample.get("fps", DEFAULT_FPS) or DEFAULT_FPS)
        label = str(sample.get("label", "")).lower()
        k = max(2, int(round(window_seconds * fps)))
        tte_frames = int(round(tte_seconds * fps))

        # Door position
        door = sample.get("door_center")
        if door is not None and len(door) == 2:
            door_x, door_y = float(door[0]), float(door[1])
        else:
            door_x = float(sample.get("frame_width", 640)) / 2.0
            door_y = float(sample.get("frame_height", 480)) - 1.0

        # Parse trajectory
        frames = sample.get("frames", [])
        x, y_coord, x1, y1, x2, y2 = get_coordinates(frames)
        x = np.array(x, dtype=np.float32)
        y_coord = np.array(y_coord, dtype=np.float32)
        y1_arr = np.array(y1, dtype=np.float32)
        y2_arr = np.array(y2, dtype=np.float32)

        t_event = find_event_frame(label, x, y_coord, door_x, door_y)
        t_predict = t_event - tte_frames

        # Compute all 6 trajectory features per frame
        d_door = compute_distance_to_door(x, y_coord, door_x, door_y)
        closure = compute_closure_rate(d_door, fps, k=1)
        vx, vy = compute_velocity(x, y_coord, fps, k=1)
        _, rel_angle = compute_heading_and_relative_angle(
            vx, vy, x, y_coord, door_x, door_y
        )
        bbox_h = (y2_arr - y1_arr).astype(np.float32)

        # Stack full 6 → (n_frames, 6), select Core-3
        traj_full = np.column_stack(
            [d_door, closure, vx, vy, rel_angle, bbox_h]
        ).astype(np.float32)
        traj_subset = traj_full[:, CORE3_INDICES]  # (n_frames, 3)

        # Load pose sidecar
        pose_lookup = load_pose_sidecar(sample)
        frame_idx_list = [
            f.get("frame_index", i) for i, f in enumerate(frames)
        ]
        frame_w = float(sample.get("frame_width", 640))
        frame_h = float(sample.get("frame_height", 480))

        # Build raw sequence over observation window
        w_start = t_predict - k + 1
        raw_seq = np.zeros((k, n_total), dtype=np.float32)

        for i, t in enumerate(range(w_start, t_predict + 1)):
            if 0 <= t < len(x):
                row = list(traj_subset[t])

                # Pose features
                frame_idx = frame_idx_list[t] if t < len(frame_idx_list) else t
                landmarks = pose_lookup.get(frame_idx)
                ear_feats = compute_raw_ear_features(landmarks)
                row.extend([
                    ear_feats["left_ear_x"],
                    ear_feats["left_ear_y"],
                    ear_feats["right_ear_x"],
                    ear_feats["right_ear_y"],
                ])

                raw_seq[i] = row

        # Smooth pose columns
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


# ──────────────────────────────────────────────────────────────────
# NORMALISATION
# ──────────────────────────────────────────────────────────────────

def normalise(X_train, X_val):
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8

    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std

    np.nan_to_num(X_train_n, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.nan_to_num(X_val_n, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    return X_train_n, X_val_n, mean, std


# ──────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────

def class_weight(y_train):
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)


def train_one_fold(model, X_train, y_train, X_eval, y_eval, fold_seed):
    """Train one fold, return metrics + best weights + normalisation stats."""

    X_sub_train, X_sub_val, y_sub_train, y_sub_val = train_test_split(
        X_train, y_train,
        test_size=0.15,
        stratify=y_train,
        random_state=fold_seed,
    )

    X_sub_train_n, X_sub_val_n, mean, std = normalise(X_sub_train, X_sub_val)

    X_eval_n = (X_eval - mean) / std
    np.nan_to_num(X_eval_n, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    shuffle_gen = torch.Generator().manual_seed(fold_seed)
    train_loader = DataLoader(
        SequenceDataset(X_sub_train_n, y_sub_train),
        batch_size=BATCH_SIZE, shuffle=True, generator=shuffle_gen,
    )
    val_loader = DataLoader(
        SequenceDataset(X_sub_val_n, y_sub_val),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    eval_loader = DataLoader(
        SequenceDataset(X_eval_n, y_eval),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
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
            for X_batch, y_batch in val_loader:
                val_loss += criterion(
                    model(X_batch.to(DEVICE)), y_batch.to(DEVICE)
                ).item()
        val_loss /= max(len(val_loader), 1)

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
    labels_np = torch.cat(all_labels).numpy().astype(int)
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    return {
        "bal_acc": balanced_accuracy_score(labels_np, preds),
        "roc_auc": roc_auc_score(labels_np, probs),
        "f1": f1_score(labels_np, preds, zero_division=0),
        "best_epoch": best_epoch,
        "weights": best_weights,
        "norm_mean": mean,
        "norm_std": std,
    }


# ──────────────────────────────────────────────────────────────────
# MAIN: CV evaluation + export best model
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train Core-3+head GRU for demo"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        metavar="DIR",
        help=(
            "MasterData directory with enter/ and pass/ (default: "
            "<repo>/MasterData, next to demo_open_house/)"
        ),
    )
    parser.add_argument(
        "--tte", type=float, default=TTE_SECONDS,
        help=f"Time-to-event in seconds (default: {TTE_SECONDS})"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help=(
            "Directory for gru_core3head.pt and norm_stats.npz "
            "(default: demo_open_house/weights next to this script)"
        ),
    )
    args = parser.parse_args()

    data_root = args.data.expanduser().resolve()
    if not data_root.is_dir():
        print(
            f"ERROR: Data directory not found: {data_root}\n"
            "Place cleaned data in MasterData/ at the repo root, or pass "
            "an explicit path:\n"
            "  python demo_open_house/train_demo_model.py --data /path/to/MasterData"
        )
        sys.exit(1)

    out_dir = args.output.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Demo model training — Core-3 + head (raw) GRU")
    print(f"  Device      : {DEVICE}")
    print(f"  Data root   : {data_root}")
    print(f"  TTE         : {args.tte}s")
    print(f"  Window      : {WINDOW_SECONDS}s")
    print(f"  Seq len     : {get_seq_len(WINDOW_SECONDS)}")
    print(f"  Input dim   : {INPUT_DIM}")
    print(f"  Features    : {CORE3_NAMES + HEAD_RAW_NAMES}")
    print(f"  Output dir  : {out_dir}")
    print("=" * 60)

    # Load data
    samples = load_all_samples(str(data_root))
    n_enter = sum(1 for s in samples if s.get("label") == "enter")
    n_pass = sum(1 for s in samples if s.get("label") == "pass")
    print(f"\nLoaded {len(samples)} samples (enter={n_enter}, pass={n_pass})")

    if not samples:
        print("ERROR: No samples found. Check --data path.")
        sys.exit(1)

    # Build sequences
    print(f"\nBuilding sequences for TTE={args.tte}s ...")
    X, y = build_sequences(samples, args.tte, WINDOW_SECONDS)
    print(f"  Sequences: {X.shape[0]}  (enter={int((y==1).sum())}, "
          f"pass={int((y==0).sum())})")
    print(f"  Shape: {X.shape}")

    if len(X) == 0 or len(np.unique(y)) < 2:
        print("ERROR: Not enough data.")
        sys.exit(1)

    # ── Cross-validation (to report metrics) ──
    print(f"\n{'='*60}")
    print(f"  Running {CV_REPEATS}x{CV_FOLDS}-fold CV ...")
    print(f"{'='*60}")

    all_bal_acc = []
    all_roc_auc = []
    all_f1 = []

    # Track the best single fold to export
    best_export_score = -1
    best_export_weights = None
    best_export_mean = None
    best_export_std = None

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

            model = IntentGRU(
                INPUT_DIM, hidden_size=HIDDEN_SIZE, dropout=DROPOUT
            )
            metrics = train_one_fold(
                model, X_train, y_train, X_val, y_val, fold_seed
            )

            all_bal_acc.append(metrics["bal_acc"])
            all_roc_auc.append(metrics["roc_auc"])
            all_f1.append(metrics["f1"])

            print(
                f"  Rep {repeat+1}/{CV_REPEATS} Fold {fold}/{CV_FOLDS}  "
                f"bal_acc={metrics['bal_acc']:.4f}  "
                f"roc_auc={metrics['roc_auc']:.4f}  "
                f"f1={metrics['f1']:.4f}  "
                f"epoch={metrics['best_epoch']}"
            )

            # Keep best fold for export
            if metrics["bal_acc"] > best_export_score:
                best_export_score = metrics["bal_acc"]
                best_export_weights = metrics["weights"]
                best_export_mean = metrics["norm_mean"]
                best_export_std = metrics["norm_std"]

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  CV Results ({CV_REPEATS}x{CV_FOLDS} = "
          f"{len(all_bal_acc)} folds)")
    print(f"  bal_acc : {np.mean(all_bal_acc):.4f} ± {np.std(all_bal_acc):.4f}")
    print(f"  roc_auc : {np.mean(all_roc_auc):.4f} ± {np.std(all_roc_auc):.4f}")
    print(f"  f1      : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"{'='*60}")

    # ── Train final model on ALL data for export ──
    print(f"\n  Training final model on ALL {len(X)} samples for export ...")

    # Use all data for training, with a small held-out for early stopping
    X_final_train, X_final_val, y_final_train, y_final_val = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=RANDOM_STATE
    )

    torch.manual_seed(RANDOM_STATE)
    final_model = IntentGRU(
        INPUT_DIM, hidden_size=HIDDEN_SIZE, dropout=DROPOUT
    )

    # Normalise using ALL training data
    X_final_train_n, X_final_val_n, final_mean, final_std = normalise(
        X_final_train, X_final_val
    )

    shuffle_gen = torch.Generator().manual_seed(RANDOM_STATE)
    train_loader = DataLoader(
        SequenceDataset(X_final_train_n, y_final_train),
        batch_size=BATCH_SIZE, shuffle=True, generator=shuffle_gen,
    )
    val_loader = DataLoader(
        SequenceDataset(X_final_val_n, y_final_val),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    final_model = final_model.to(DEVICE)
    optimizer = torch.optim.Adam(
        final_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight(y_final_train))

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(final_model.state_dict())
    best_epoch = 1
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        final_model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(final_model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        final_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_loss += criterion(
                    final_model(X_batch.to(DEVICE)), y_batch.to(DEVICE)
                ).item()
        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(final_model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            break

    final_model.load_state_dict(best_weights)
    print(f"  Final model: best epoch = {best_epoch}, val_loss = {best_val_loss:.4f}")

    # ── Export ──
    model_path = out_dir / "gru_core3head.pt"
    norm_path = out_dir / "norm_stats.npz"

    # Save to CPU for portability (loads on Jetson regardless of training device)
    torch.save(
        {k: v.cpu() for k, v in final_model.state_dict().items()},
        model_path,
    )
    np.savez(norm_path, mean=final_mean, std=final_std)

    print(f"\n  Saved model:  {model_path}")
    print(f"  Saved norms:  {norm_path}")
    print(f"  norm mean:    {final_mean}")
    print(f"  norm std:     {final_std}")

    # Verify loading
    test_model = IntentGRU(INPUT_DIM, hidden_size=HIDDEN_SIZE, dropout=DROPOUT)
    test_model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    test_model.eval()
    dummy = torch.randn(1, get_seq_len(WINDOW_SECONDS), INPUT_DIM)
    with torch.no_grad():
        p = torch.sigmoid(test_model(dummy)).item()
    print(f"\n  Verification: loaded model, dummy P(enter) = {p:.3f}")
    n_params = sum(p.numel() for p in test_model.parameters())
    print(f"  Model params: {n_params}")

    print(f"\n{'='*60}")
    print("  Done! Copy demo_open_house/weights/ to Jetson (or use paths below) and run:")
    print("  cd demo && python demo_live.py --model weights/gru_core3head.pt "
          "--norm weights/norm_stats.npz")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
