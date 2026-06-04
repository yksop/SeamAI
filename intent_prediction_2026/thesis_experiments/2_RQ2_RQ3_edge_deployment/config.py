"""
config.py — Shared configuration for the RQ2/RQ3 edge deployment pipeline.

All step scripts import from here so changes propagate everywhere.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # pipeline root

# Make utils/ importable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Make the RQ1 ablation directory importable (for train_ablation builders)
ABLATION_DIR = ROOT / "1_RQ1_feature_importance" / "neural_network_ablation"
if str(ABLATION_DIR) not in sys.path:
    sys.path.insert(0, str(ABLATION_DIR))

# Data
DATA_ROOT = str(ROOT / "MasterData")

# RQ1 finalists (produced by select_gru_finalists.py)
RQ1_FINALISTS_CSV = ABLATION_DIR / "results" / "gru_finalists.csv"

# RQ1 structured ablation results (produced by train_ablation.py)
RQ1_RESULTS_CSV = ABLATION_DIR / "results" / "results_all.csv"

# Output directories — everything generated lands under outputs/
OUTPUTS_DIR = HERE / "outputs"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"
ONNX_DIR = OUTPUTS_DIR / "onnx"
RESULTS_DIR = OUTPUTS_DIR / "results"
FIG_DIR = OUTPUTS_DIR / "figures"

for _d in [CHECKPOINT_DIR, ONNX_DIR, RESULTS_DIR, FIG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
TTE_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5]
WINDOW_S = 0.5

# Model hyperparameters (identical to RQ1 ablation — train_ablation.py)
HIDDEN_SIZE = 16
DROPOUT = 0.1
EPOCHS = 150
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 15
VAL_FRACTION = 0.15
RANDOM_STATE = 42

# Benchmark parameters
WARMUP_RUNS = 50
TIMED_RUNS = 500

# RQ2 target: 30 FPS = 33.3 ms per frame budget
TARGET_FPS = 30
TARGET_LATENCY_MS = 1000.0 / TARGET_FPS

# ---------------------------------------------------------------------------
# Perception frontend configurations to benchmark (step5)
# ---------------------------------------------------------------------------
# Each entry: (name, model_id, provides_pose, notes)
# model_id is what gets passed to the YOLO/MediaPipe loader.
#
# Entries with "[TRT]" in the name use TensorRT FP16 on the Jetson GPU.
# TensorRT export happens automatically on first run and is cached as .engine.
# If TensorRT is unavailable, these fall back to PyTorch.

PERCEPTION_CONFIGS = [
    # --- PyTorch baselines (no hardware-specific optimization) ---

    # Detection only (bounding box) — sufficient for Tiers 1-2
    ("YOLOv8n", "yolov8n.pt", False,
     "Ultralytics YOLOv8 nano — detection only (PyTorch)"),
    ("YOLOv11n", "yolo11n.pt", False,
     "Ultralytics YOLOv11 nano — detection only (PyTorch)"),

    # Single-stage pose (detection + keypoints) — sufficient for all tiers
    ("YOLOv8n-pose", "yolov8n-pose.pt", True,
     "YOLOv8 nano with pose — 17 keypoints (PyTorch)"),
    ("YOLOv11n-pose", "yolo11n-pose.pt", True,
     "YOLOv11 nano with pose — 17 keypoints (PyTorch)"),

    # Two-stage: YOLO detection + MediaPipe pose
    ("YOLOv8n + MediaPipe", "yolov8n.pt + mediapipe", True,
     "Two-stage: YOLOv8n detection then MediaPipe Pose on crop"),

    # --- TensorRT FP16 (optimised for Jetson GPU) ---
    # These use model.export(format='engine', half=True) for FP16 inference.
    # Export is slow (~2-5 min) on first run but cached as .engine files.

    ("YOLOv8n [TRT]", "yolov8n.pt", False,
     "YOLOv8 nano — detection only (TensorRT FP16)"),
    ("YOLOv11n [TRT]", "yolo11n.pt", False,
     "YOLOv11 nano — detection only (TensorRT FP16)"),
    ("YOLOv8n-pose [TRT]", "yolov8n-pose.pt", True,
     "YOLOv8 nano with pose — 17 keypoints (TensorRT FP16)"),
    ("YOLOv11n-pose [TRT]", "yolo11n-pose.pt", True,
     "YOLOv11 nano with pose — 17 keypoints (TensorRT FP16)"),
]

# ---------------------------------------------------------------------------
# Complexity tiers (must match select_gru_finalists.py)
# ---------------------------------------------------------------------------
COMPLEXITY_TIERS = [
    ("Tier 1: RF-minimal", 1, 4),
    ("Tier 2: Trajectory baseline", 5, 6),
    ("Tier 3: Compact + pose", 7, 10),
    ("Tier 4: Medium pose", 11, 20),
    ("Tier 5: Full body", 21, 100),
]

TIER_COLORS = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd", "#d62728"]

# Which tiers need pose estimation from the perception frontend?
# Tiers 1-2 only need bounding box; Tiers 3-5 need keypoints.
TIER_NEEDS_POSE = {0: False, 1: False, 2: True, 3: True, 4: True}


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
        "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
    })
