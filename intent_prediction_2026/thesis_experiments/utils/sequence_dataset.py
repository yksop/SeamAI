#!/usr/bin/env python3
"""
sequence_dataset.py

Builds per-frame feature sequences from trajectory JSON samples for the
GRU and MLP models.

The key difference from trajectory_feature_extractor.py:
  - trajectory_feature_extractor.py  ->  one 30-dim feature vector per sample
                                         (mean, variance, latest aggregation)
  - sequence_dataset.py              ->  one (T x D) time-series per sample
                                         (raw per-frame values, no aggregation)

Pose modes control which features are appended to the 6 trajectory features at
each timestep.  Two parallel tracks are available:

  ANGLE track  – one hand-crafted angle per body part (apex = door centre).
  RAW track    – raw (x, y) coordinates per landmark pair; the network learns
                 its own representation.  Each pair contributes 4 values:
                 left_x, left_y, right_x, right_y.

Available modes
---------------
Baseline:
    "trajectory-only"   ->  6 traj features only

Single body part – angle (7 features) vs raw (10 features):
    "hips"              ->  + hip_angle
    "hips_raw"          ->  + left/right hip (x,y)
    "shoulders"         ->  + shoulder_angle
    "shoulders_raw"     ->  + left/right shoulder (x,y)
    "head"              ->  + head_angle  (gaze / ear orientation)
    "head_raw"          ->  + left/right ear (x,y)
    "knees"             ->  + knee_angle
    "knees_raw"         ->  + left/right knee (x,y)

Progressive upper body – angle vs raw:
    "hips_shoulders"        ->  + hip + shoulder angles              (8)
    "hips_shoulders_raw"    ->  + hip + shoulder raw (x,y)           (14)
    "torso_head"            ->  + hip + shoulder + head angles        (9)
    "torso_head_raw"        ->  + hip + shoulder + head raw (x,y)    (18)
    "torso_head_arms"       ->  + hip+shoulder+head+elbow+wrist (11)
    "torso_head_arms_raw"   ->  + same five pairs raw (x,y)          (26)

Lower body – angle vs raw:
    "lower_body"            ->  + knee + ankle angles                 (8)
    "lower_body_raw"        ->  + knee + ankle raw (x,y)             (14)

Full body – angle vs raw:
    "full_body"             ->  all 7 angles                         (13)
    "full_body_raw"         ->  all 7 landmark pairs raw (x,y)       (34)

Run extract_pose.py first to generate the _pose.json sidecar files
needed for any pose mode other than trajectory-only / none.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.feature_extractor import (
    DEFAULT_FPS,
    compute_closure_rate,
    compute_distance_to_door,
    compute_heading_and_relative_angle,
    compute_velocity,
    find_event_frame,
    get_coordinates,
    is_sample_usable,
)

# --- Feature name registries ---

TRAJ_FEATURE_NAMES = [
    "dist_to_door",       # distance from person centre to door (pixels)
    "closure_rate",       # how fast the person is approaching (pixels/s)
    "vx",                 # horizontal velocity (pixels/frame)
    "vy",                 # vertical velocity (pixels/frame)
    "rel_angle_to_door",  # angle between heading direction and door (radians)
    "bbox_h",             # bounding box height (proxy for distance to camera)
]

# All available pose-angle feature names.
# Each is the angle at the door apex subtended by the named bilateral landmark pair.
POSE_ANGLE_FEATURE_NAMES = [
    "hip_angle",       # left_hip  – right_hip
    "shoulder_angle",  # left_shoulder – right_shoulder
    "elbow_angle",     # left_elbow – right_elbow
    "wrist_angle",     # left_wrist – right_wrist
    "knee_angle",      # left_knee – right_knee
    "ankle_angle",     # left_ankle – right_ankle
    "head_angle",      # left_ear  – right_ear
]

# Raw landmark coordinate feature names (normalised [0,1]).
# The network learns spatial relationships instead of hand-crafted angles.
_RAW_LANDMARKS = [
    "nose", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


def _raw_xy(*landmark_names):
    """Expand landmark names into [name_x, name_y, ...] feature list."""
    return [f"{n}_{c}" for n in landmark_names for c in ("x", "y")]


RAW_FEATURE_NAMES = _raw_xy(*_RAW_LANDMARKS)

# Backward-compatible aliases kept for any code that imports them
POSE_FEATURE_NAMES = ["hip_angle", "shoulder_angle"]
HEAD_FEATURE_NAMES = ["head_angle"]

_T = TRAJ_FEATURE_NAMES  # shorthand

ALL_FEATURE_NAMES = {
    # ------------------------------------------------------------------ #
    # 0. Baseline                                                          #
    # ------------------------------------------------------------------ #
    "trajectory-only":      _T,
    "none":                 _T,  # backward-compatible alias

    # ------------------------------------------------------------------ #
    # 1. Single body part  (angle: +1 feature | raw: +4 features)         #
    # ------------------------------------------------------------------ #
    "hips":                 _T + ["hip_angle"],
    "hips_raw":             _T + _raw_xy("left_hip",      "right_hip"),
    "shoulders":            _T + ["shoulder_angle"],
    "shoulders_raw":        _T + _raw_xy("left_shoulder", "right_shoulder"),
    "head":                 _T + ["head_angle"],   # gaze / ear orientation
    "head_raw":             _T + _raw_xy("left_ear",      "right_ear"),
    "knees":                _T + ["knee_angle"],
    "knees_raw":            _T + _raw_xy("left_knee",     "right_knee"),

    # ------------------------------------------------------------------ #
    # 2. Progressive upper body  (angle vs raw)                            #
    # ------------------------------------------------------------------ #
    "hips_shoulders":       _T + ["hip_angle",      "shoulder_angle"],
    "hips_shoulders_raw":   _T + _raw_xy("left_hip", "right_hip",
                                         "left_shoulder", "right_shoulder"),
    "torso_head":           _T + ["hip_angle",      "shoulder_angle", "head_angle"],
    "torso_head_raw":       _T + _raw_xy("left_hip", "right_hip",
                                         "left_shoulder", "right_shoulder",
                                         "left_ear", "right_ear"),
    "torso_head_arms":      _T + ["hip_angle",      "shoulder_angle", "head_angle",
                                   "elbow_angle",    "wrist_angle"],
    "torso_head_arms_raw":  _T + _raw_xy("left_hip", "right_hip",
                                         "left_shoulder", "right_shoulder",
                                         "left_ear", "right_ear",
                                         "left_elbow", "right_elbow",
                                         "left_wrist", "right_wrist"),

    # ------------------------------------------------------------------ #
    # 3. Lower body  (angle vs raw)                                        #
    # ------------------------------------------------------------------ #
    "lower_body":           _T + ["knee_angle",     "ankle_angle"],
    "lower_body_raw":       _T + _raw_xy("left_knee", "right_knee",
                                         "left_ankle", "right_ankle"),

    # ------------------------------------------------------------------ #
    # 4. Full body  (angle vs raw)                                         #
    # ------------------------------------------------------------------ #
    "full_body":            _T + ["hip_angle",      "shoulder_angle", "head_angle",
                                   "elbow_angle",    "wrist_angle",
                                   "knee_angle",     "ankle_angle"],
    "full_body_raw":        _T + _raw_xy("left_hip", "right_hip",
                                         "left_shoulder", "right_shoulder",
                                         "left_ear", "right_ear",
                                         "left_elbow", "right_elbow",
                                         "left_wrist", "right_wrist",
                                         "left_knee", "right_knee",
                                         "left_ankle", "right_ankle"),
}

# Baseline channel count (trajectory-only); tune_hyperparams.py imports this.
INPUT_DIM = len(TRAJ_FEATURE_NAMES)


def get_seq_len(window_seconds):
    """
    How many frames does the observation window correspond to?

    Computed using DEFAULT_FPS so that all samples produce sequences of the
    same length, regardless of the actual fps of each recording.
    Examples: 0.5 s -> 7 frames, 1.0 s -> 13 frames, 1.5 s -> 20 frames.
    """
    return max(2, int(round(window_seconds * DEFAULT_FPS)))


def get_input_dim(pose_mode):
    """Number of features per timestep for the given pose_mode."""
    return len(ALL_FEATURE_NAMES[pose_mode])


# --- Temporal smoothing ---

def smooth_pose_features(seq, n_traj=6, kernel_size=3):
    """
    Apply a causal moving average to the pose feature columns.

    Causal means we only look at past and current frames, not future ones –
    this matches real-time conditions. The trajectory features (first n_traj
    columns) are not changed.
    """
    if seq.shape[1] <= n_traj or seq.shape[0] < 2 or kernel_size < 2:
        return seq

    smoothed = seq.copy()
    for col in range(n_traj, seq.shape[1]):
        for t in range(seq.shape[0]):
            start = max(0, t - kernel_size + 1)
            smoothed[t, col] = seq[start : t + 1, col].mean()

    return smoothed


# --- Pose angle computation ---

def _angle_at_apex(ax, ay, bx, by, apex_x, apex_y):
    r"""
    Compute the angle at the apex of the triangle formed by A, apex, and B.

        A ----------- B
         \           /
          \         /
           [apex]    <- angle measured here

    A large angle means A and B are close to the apex and/or spread wide
    (person is near the door and facing it). A small angle means they are
    far away or viewed side-on.
    """
    dx1, dy1 = ax - apex_x, ay - apex_y
    dx2, dy2 = bx - apex_x, by - apex_y
    dot  = dx1 * dx2 + dy1 * dy2
    mag1 = math.hypot(dx1, dy1)
    mag2 = math.hypot(dx2, dy2)
    if mag1 < 1e-8 or mag2 < 1e-8:
        return 0.0
    cos_val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.acos(cos_val)


def compute_all_pose_angles(landmarks, door_x, door_y, frame_w, frame_h):
    """
    Compute all available body-angle features from MediaPipe landmarks.

    Each angle is measured at the door position (apex) using the bilateral
    landmark pair named by the key.  Landmark coords are normalised [0, 1]
    and converted to pixels to preserve aspect-ratio geometry.

    Returns a dict mapping feature name -> float (0.0 on missing landmark).
    """
    def to_px(name):
        lm = landmarks[name]
        return lm["x"] * frame_w, lm["y"] * frame_h

    def safe_angle(left_name, right_name):
        try:
            lx, ly = to_px(left_name)
            rx, ry = to_px(right_name)
            return _angle_at_apex(lx, ly, rx, ry, door_x, door_y)
        except (KeyError, ZeroDivisionError):
            return 0.0

    return {
        "hip_angle":      safe_angle("left_hip",       "right_hip"),
        "shoulder_angle": safe_angle("left_shoulder",  "right_shoulder"),
        "elbow_angle":    safe_angle("left_elbow",     "right_elbow"),
        "wrist_angle":    safe_angle("left_wrist",     "right_wrist"),
        "knee_angle":     safe_angle("left_knee",      "right_knee"),
        "ankle_angle":    safe_angle("left_ankle",     "right_ankle"),
        "head_angle":     safe_angle("left_ear",       "right_ear"),
    }


def compute_raw_pose_features(landmarks):
    """
    Extract normalised (x, y) coordinates for all landmarks.

    Returns a dict mapping "{landmark}_x" / "{landmark}_y" -> float.
    Uses the normalised [0, 1] values directly so the network receives
    position information without any hand-crafted transformations.
    """
    result = {}
    for name in _RAW_LANDMARKS:
        try:
            lm = landmarks[name]
            result[f"{name}_x"] = lm["x"]
            result[f"{name}_y"] = lm["y"]
        except (KeyError, TypeError):
            result[f"{name}_x"] = 0.0
            result[f"{name}_y"] = 0.0
    return result


def compute_all_pose_features(landmarks, door_x, door_y, frame_w, frame_h):
    """
    Compute both hand-crafted angles and raw coordinates.

    Returns a single dict containing all possible pose features so that
    extract_sequence can pick whichever subset the current pose_mode needs.
    """
    features = compute_all_pose_angles(landmarks, door_x, door_y, frame_w, frame_h)
    features.update(compute_raw_pose_features(landmarks))
    return features


def compute_pose_angles(landmarks, include_head, door_x, door_y, frame_w, frame_h):
    """
    Legacy wrapper kept for backward compatibility.

    Prefer compute_all_pose_features for new code.
    """
    angles_dict = compute_all_pose_angles(landmarks, door_x, door_y, frame_w, frame_h)
    result = [angles_dict["hip_angle"], angles_dict["shoulder_angle"]]
    if include_head:
        result.append(angles_dict["head_angle"])
    return result


def load_pose_sidecar(sample):
    """
    Load the _pose.json sidecar file for a sample.

    Returns a dict mapping frame_index -> landmarks dict (or None if not detected).
    Returns an empty dict if no sidecar file exists.
    """
    source_path = sample.get("_source_path")
    if not source_path:
        return {}

    # The sidecar has the same name as the trajectory JSON but with _pose.json
    pose_path = Path(source_path).parent / (Path(source_path).stem + "_pose.json")
    if not pose_path.exists():
        return {}

    with open(pose_path, "r") as f:
        pose_data = json.load(f)

    lookup = {}
    for frame in pose_data.get("frames", []):
        fi = frame["frame_index"]
        lookup[fi] = frame["landmarks"] if frame.get("detected") else None

    return lookup


# Alias kept for backward compatibility
load_pose_lookup = load_pose_sidecar


# --- Sequence extraction ---

def extract_sequence(sample, tte_seconds, window_seconds, pose_mode="trajectory-only"):
    """
    Extract a fixed-length (seq_len x D) feature sequence from one sample.

    The observation window is the last `window_seconds` of trajectory data
    before the prediction point (TTE seconds before the event).
    Short samples are left-padded with zeros; long ones are trimmed to seq_len.

    Returns (seq, label_int), or (None, None) if the sample is too short.
    """
    if not is_sample_usable(sample, tte_seconds, window_seconds):
        return None, None

    fps   = float(sample.get("fps", DEFAULT_FPS) or DEFAULT_FPS)
    label = str(sample.get("label", "")).lower()

    # Window length and prediction offset in frames
    k          = max(2, int(round(window_seconds * fps)))
    tte_frames = int(round(tte_seconds * fps))

    # Door position (used for both trajectory and pose features)
    door = sample.get("door_center")
    if door is not None and len(door) == 2:
        door_x, door_y = float(door[0]), float(door[1])
    else:
        door_x = float(sample.get("frame_width",  640)) / 2.0
        door_y = float(sample.get("frame_height", 480)) - 1.0

    # Parse trajectory arrays
    frames = sample.get("frames", [])
    x, y, x1, y1, x2, y2 = get_coordinates(frames)
    x  = np.array(x,  dtype=np.float32)
    y  = np.array(y,  dtype=np.float32)
    x1 = np.array(x1, dtype=np.float32)
    y1 = np.array(y1, dtype=np.float32)
    x2 = np.array(x2, dtype=np.float32)
    y2 = np.array(y2, dtype=np.float32)

    t_event   = find_event_frame(label, x, y, door_x, door_y)
    t_predict = t_event - tte_frames

    # Compute per-frame trajectory features (using 1-frame finite differences)
    d_door       = compute_distance_to_door(x, y, door_x, door_y)
    closure      = compute_closure_rate(d_door, fps, k=1)
    vx, vy       = compute_velocity(x, y, fps, k=1)
    _, rel_angle = compute_heading_and_relative_angle(vx, vy, x, y, door_x, door_y)
    bbox_h       = (y2 - y1).astype(np.float32)

    # Stack into (n_frames, 6) matrix
    traj_matrix = np.column_stack(
        [d_door, closure, vx, vy, rel_angle, bbox_h]
    ).astype(np.float32)

    # Determine which pose-angle features this mode requires
    all_feature_names = ALL_FEATURE_NAMES[pose_mode]
    pose_feature_names = [f for f in all_feature_names if f not in TRAJ_FEATURE_NAMES]
    use_pose = len(pose_feature_names) > 0

    pose_lookup = load_pose_sidecar(sample) if use_pose else {}

    frame_idx_list = [f.get("frame_index", i) for i, f in enumerate(frames)]
    frame_w = float(sample.get("frame_width",  640))
    frame_h = float(sample.get("frame_height", 480))

    n_features = len(all_feature_names)
    n_pose     = len(pose_feature_names)
    w_start    = t_predict - k + 1

    # Build the raw sequence (k frames)
    raw_seq = np.zeros((k, n_features), dtype=np.float32)
    for i, t in enumerate(range(w_start, t_predict + 1)):
        if 0 <= t < len(x):
            row = list(traj_matrix[t])

            if use_pose:
                frame_idx = frame_idx_list[t] if t < len(frame_idx_list) else t
                landmarks = pose_lookup.get(frame_idx)

                if landmarks:
                    all_feats = compute_all_pose_features(
                        landmarks, door_x, door_y, frame_w, frame_h
                    )
                    pose_vals = [all_feats.get(name, 0.0) for name in pose_feature_names]
                else:
                    pose_vals = [0.0] * n_pose  # zero-impute missing detections

                row.extend(pose_vals)

            raw_seq[i] = row

    # Smooth pose columns to reduce frame-to-frame noise in landmark estimates
    if use_pose:
        raw_seq = smooth_pose_features(raw_seq, n_traj=len(TRAJ_FEATURE_NAMES))

    # Pad or trim to a fixed length so all samples have the same shape
    seq_len = get_seq_len(window_seconds)
    seq = np.zeros((seq_len, n_features), dtype=np.float32)
    if k >= seq_len:
        seq[:] = raw_seq[-seq_len:]
    else:
        seq[-k:] = raw_seq

    np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    label_int = 1 if label == "enter" else 0
    return seq, label_int


def build_sequences(samples, tte_seconds, window_seconds, pose_mode="trajectory-only"):
    """
    Build (X, y) arrays from all usable samples for a given TTE.

    Returns:
        X : float array, shape (n_samples, seq_len, n_features)
        y : int array,   shape (n_samples,)
    """
    seqs   = []
    labels = []
    dropped = {"enter": 0, "pass": 0}

    for sample in samples:
        seq, label_int = extract_sequence(
            sample, tte_seconds, window_seconds, pose_mode
        )
        if seq is None:
            lbl = str(sample.get("label", "")).lower()
            if lbl in dropped:
                dropped[lbl] += 1
            continue
        seqs.append(seq)
        labels.append(label_int)

    if dropped["enter"] > 0 or dropped["pass"] > 0:
        print(
            f"  Dropped (too short for TTE={tte_seconds:.1f}s): "
            f"enter={dropped['enter']}, pass={dropped['pass']}"
        )

    X = np.array(seqs,   dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y


# --- PyTorch Dataset and normalisation ---

class SequenceDataset(Dataset):
    """Simple PyTorch Dataset wrapping (X, y) sequence arrays."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def normalise(X_train, X_val):
    """
    Z-score normalise features using training set statistics only.

    One mean and std per feature dimension. Applied to both train and val,
    but computed only from the training set to avoid data leakage.
    """
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std  = flat.std(axis=0) + 1e-8

    X_train_norm = (X_train - mean) / std
    X_val_norm   = (X_val   - mean) / std

    np.nan_to_num(X_train_norm, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.nan_to_num(X_val_norm,   nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    return X_train_norm, X_val_norm, mean, std
