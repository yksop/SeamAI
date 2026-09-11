#!/usr/bin/env python3
"""
Step 5 — Benchmark perception frontends on edge hardware.

Measures the inference latency of different perception pipelines that
extract visual features from video frames:

  1. YOLOv8n       — detection only (bounding box)
  2. YOLOv11n      — detection only (bounding box)
  3. YOLOv8n-pose  — single-stage detection + 17 COCO keypoints
  4. YOLOv11n-pose — single-stage detection + 17 COCO keypoints
  5. YOLOv8n + MediaPipe Pose — two-stage: detect then pose on crop

For each frontend, the script measures:
  - Per-frame inference latency (mean, std, p50, p95, p99)
  - Throughput (FPS)
  - Whether it provides bounding box only or full pose keypoints

Benchmarking protocol (following NVIDIA TensorRT + MLPerf guidelines):
  - Frames are loaded from multiple diverse video clips (--input-dir)
    to capture variance in detection difficulty, crop sizes, and poses.
  - Warmup: 50 frames discarded (GPU init, cache warming, thermal ramp)
  - Measurement: >= 500 frames in steady state across all clips
  - Report: mean, std, p50, p95, p99 latency; FPS; detection rate

Run on the Jetson:
    python step5_benchmark_perception.py --input-dir path/to/clips/
    python step5_benchmark_perception.py --input path/to/video.mp4

Quick check without video (no camera or clips needed):
    python step5_benchmark_perception.py --synthetic

Dependencies:
    pip install ultralytics mediapipe opencv-python
"""

import argparse
import random
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

from config import (
    RESULTS_DIR, FIG_DIR,
    TARGET_FPS, TARGET_LATENCY_MS,
    PERCEPTION_CONFIGS,
    style,
)

# ---------------------------------------------------------------------------
# Defaults — aligned with NVIDIA TensorRT / MLPerf best practices
# ---------------------------------------------------------------------------
DEFAULT_WARMUP = 50          # frames discarded before measurement
DEFAULT_TIMED = 500          # minimum steady-state frames
DEFAULT_MAX_FRAMES_PER_CLIP = 150   # cap per clip so no single clip dominates
DEFAULT_MIN_CLIPS = 20       # recommended minimum for diverse benchmarks


# ---------------------------------------------------------------------------
# Perception backends
# ---------------------------------------------------------------------------

class YOLOBackend:
    """YOLO detection or pose via Ultralytics (PyTorch or TensorRT)."""

    def __init__(self, model_id, provides_pose=False, use_tensorrt=False,
                 infer_device="cpu"):
        from ultralytics import YOLO
        self.provides_pose = provides_pose
        self.use_tensorrt = use_tensorrt
        self.infer_device = infer_device

        if use_tensorrt:
            if str(infer_device).startswith("cpu"):
                raise RuntimeError(
                    "TensorRT requires GPU device. Use --device auto/cuda or --skip-trt."
                )
            # Export to TensorRT FP16 engine if not already done.
            # The .engine file is cached so subsequent runs skip export.
            self.model_id = model_id
            engine_path = Path(model_id).with_suffix(".engine")
            if engine_path.exists():
                print(f"    Loading cached TensorRT engine: {engine_path}")
                self.model = YOLO(str(engine_path))
            else:
                print(f"    Exporting {model_id} -> TensorRT FP16 ...")
                base_model = YOLO(model_id)
                try:
                    trt_device = 0
                    if isinstance(infer_device, str) and infer_device.startswith("cuda:"):
                        try:
                            trt_device = int(infer_device.split(":", 1)[1])
                        except (ValueError, IndexError):
                            trt_device = 0

                    exported = base_model.export(
                        format="engine",
                        half=True,       # FP16 precision
                        device=trt_device,
                        verbose=False,
                    )
                    print(f"    Exported: {exported}")
                    self.model = YOLO(exported)
                except Exception as e:
                    print(f"    TensorRT export failed: {e}")
                    print(f"    Falling back to PyTorch model.")
                    self.model = base_model
                    self.use_tensorrt = False
        else:
            self.model = YOLO(model_id)
            self.model_id = model_id

    def warmup(self, frame):
        """Run a few dummy inferences to warm up the model."""
        for _ in range(3):
            self.model(frame, verbose=False, device=self.infer_device)

    def __call__(self, frame):
        """
        Run inference on a single frame.

        Returns:
            bbox: (x1, y1, x2, y2) of the first detected person, or None
            keypoints: (N, 3) array of (x, y, conf) if pose model, else None
        """
        results = self.model(
            frame,
            verbose=False,
            classes=[0],
            device=self.infer_device,
        )  # class 0 = person

        if len(results) == 0 or len(results[0].boxes) == 0:
            return None, None

        # Take the highest-confidence person
        boxes = results[0].boxes
        best_idx = boxes.conf.argmax().item()
        bbox = boxes.xyxy[best_idx].cpu().numpy()

        keypoints = None
        if self.provides_pose and results[0].keypoints is not None:
            kps = results[0].keypoints
            if kps.xy is not None and len(kps.xy) > best_idx:
                keypoints = kps.data[best_idx].cpu().numpy()  # (17, 3)

        return bbox, keypoints


class YOLOMediaPipeBackend:
    """Two-stage: YOLO detection + MediaPipe Pose on the cropped region."""

    def __init__(self, yolo_model_id="yolov8n.pt", infer_device="cpu"):
        from ultralytics import YOLO
        import mediapipe as mp

        self.yolo = YOLO(yolo_model_id)
        self.infer_device = infer_device
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.provides_pose = True

    def warmup(self, frame):
        for _ in range(3):
            self.yolo(frame, verbose=False, device=self.infer_device)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.pose.process(rgb)

    def __call__(self, frame):
        # Stage 1: YOLO detection
        results = self.yolo(
            frame,
            verbose=False,
            classes=[0],
            device=self.infer_device,
        )

        if len(results) == 0 or len(results[0].boxes) == 0:
            return None, None

        boxes = results[0].boxes
        best_idx = boxes.conf.argmax().item()
        bbox = boxes.xyxy[best_idx].cpu().numpy()
        x1, y1, x2, y2 = map(int, bbox)

        # Stage 2: MediaPipe Pose on cropped + padded region
        h, w = frame.shape[:2]
        pad = 20
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)
        crop = frame[cy1:cy2, cx1:cx2]

        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_result = self.pose.process(rgb_crop)

        keypoints = None
        if mp_result.pose_landmarks:
            lms = mp_result.pose_landmarks.landmark
            keypoints = np.array([
                [lm.x, lm.y, lm.visibility] for lm in lms
            ], dtype=np.float32)

        return bbox, keypoints


def create_backend(name, model_id, provides_pose, use_tensorrt=False,
                   infer_device="cpu"):
    """Factory function for perception backends."""
    if "mediapipe" in model_id.lower():
        yolo_id = model_id.split("+")[0].strip()
        return YOLOMediaPipeBackend(yolo_id, infer_device=infer_device)
    else:
        return YOLOBackend(
            model_id,
            provides_pose,
            use_tensorrt=use_tensorrt,
            infer_device=infer_device,
        )


def resolve_inference_device(device_arg):
    """Resolve requested device to a value accepted by Ultralytics."""
    if device_arg != "auto":
        return device_arg

    try:
        import torch

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"CUDA initialization: The NVIDIA driver on your system is too old.*",
                category=UserWarning,
            )
            if torch.cuda.is_available():
                return "cuda:0"
    except Exception:
        pass

    return "cpu"


# ---------------------------------------------------------------------------
# Benchmark function
# ---------------------------------------------------------------------------

def benchmark_backend(backend, frames, warmup_runs, timed_runs):
    """
    Benchmark a perception backend on a list of frames.

    Protocol (NVIDIA TensorRT + MLPerf aligned):
      1. Model warmup (3 dummy inferences)
      2. Warmup phase: run warmup_runs frames (discarded)
      3. Measurement phase: run timed_runs frames, record per-frame latency
      4. Report: mean, std, p50, p95, p99, min, max, FPS

    Returns dict with latency statistics.
    """
    # Model-level warmup
    backend.warmup(frames[0])

    # Frame-level warmup (discarded — thermal equilibration)
    for i in range(min(warmup_runs, len(frames))):
        backend(frames[i % len(frames)])

    # Timed runs — iterate through frames in order (preserving clip diversity)
    latencies = []
    detections = 0
    pose_extractions = 0

    for i in range(timed_runs):
        frame = frames[i % len(frames)]

        t0 = time.perf_counter()
        bbox, keypoints = backend(frame)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)  # ms

        if bbox is not None:
            detections += 1
        if keypoints is not None:
            pose_extractions += 1

    lat = np.array(latencies)
    return {
        "mean_ms": lat.mean(),
        "std_ms": lat.std(),
        "p50_ms": np.median(lat),
        "p95_ms": np.percentile(lat, 95),
        "p99_ms": np.percentile(lat, 99),
        "min_ms": lat.min(),
        "max_ms": lat.max(),
        "fps": 1000.0 / lat.mean(),
        "n_frames": len(lat),
        "detection_rate": detections / timed_runs,
        "pose_rate": pose_extractions / timed_runs,
    }


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------

def load_frames_from_video(video_path, max_frames=150):
    """Load frames from a single video file."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def load_frames_from_dir(dir_path, max_frames_per_clip=150, max_clips=None,
                         shuffle_clips=True, seed=42):
    """
    Load frames from all .mp4 files in a directory.

    This is the recommended mode for benchmarking: using diverse clips
    ensures the latency measurements capture variance in detection
    difficulty, person distance, crop sizes, and pose complexity.

    Args:
        dir_path: Directory containing .mp4 files
        max_frames_per_clip: Cap per clip so no single clip dominates
        max_clips: Limit total clips (None = use all)
        shuffle_clips: Shuffle clip order for randomized frame sequence
        seed: Random seed for reproducibility

    Returns:
        frames: List of frames interleaved from all clips
        clip_info: Dict with per-clip statistics
    """
    dir_path = Path(dir_path)
    mp4_files = sorted(dir_path.glob("*.mp4"))

    if not mp4_files:
        # Also check subdirectories (e.g., enter/ and pass/)
        mp4_files = sorted(dir_path.rglob("*.mp4"))

    if not mp4_files:
        print(f"  WARNING: No .mp4 files found in {dir_path}")
        return [], {}

    if shuffle_clips:
        rng = random.Random(seed)
        rng.shuffle(mp4_files)

    if max_clips:
        mp4_files = mp4_files[:max_clips]

    all_frames = []
    clip_info = {"n_clips": 0, "clips": []}

    for mp4 in mp4_files:
        clip_frames = load_frames_from_video(mp4, max_frames=max_frames_per_clip)
        if clip_frames:
            all_frames.extend(clip_frames)
            clip_info["clips"].append({
                "path": str(mp4.name),
                "n_frames": len(clip_frames),
            })
            clip_info["n_clips"] += 1

    clip_info["total_frames"] = len(all_frames)
    return all_frames, clip_info


def generate_synthetic_frames(n=50, h=480, w=640):
    """Generate synthetic frames with a rectangle (simulating a person)."""
    frames = []
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8) + 128
        # Draw a rectangle that moves slightly each frame
        cx = w // 2 + int(50 * np.sin(i * 0.1))
        cy = h // 2
        cv2.rectangle(frame, (cx - 30, cy - 80), (cx + 30, cy + 80),
                       (200, 180, 160), -1)
        frames.append(frame)
    print(f"  Generated {n} synthetic frames ({w}x{h})")
    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 5: Benchmark perception frontends"
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--input-dir", type=str, default=None,
        help="Directory with .mp4 clips (recommended: 20-30 diverse clips). "
             "Loads frames from all clips for representative benchmarking.",
    )
    grp.add_argument(
        "--input", type=str, default=None,
        help="Path to a single video file (less robust than --input-dir).",
    )
    grp.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic frames (for development/testing only).",
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help=f"Warm-up frames to discard (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_TIMED,
        help=f"Minimum timed frames (default: {DEFAULT_TIMED})",
    )
    parser.add_argument(
        "--max-clips", type=int, default=None,
        help="Max clips to load from --input-dir (default: all)",
    )
    parser.add_argument(
        "--max-frames-per-clip", type=int, default=DEFAULT_MAX_FRAMES_PER_CLIP,
        help=f"Max frames per clip (default: {DEFAULT_MAX_FRAMES_PER_CLIP})",
    )
    parser.add_argument(
        "--resolution", type=str, default="640x480",
        help="Input resolution WxH for synthetic frames (default: 640x480)",
    )
    parser.add_argument(
        "--skip", type=str, nargs="*", default=[],
        help="Names of configs to skip (e.g., 'YOLOv11n-pose')",
    )
    parser.add_argument(
        "--skip-trt", action="store_true",
        help="Skip all TensorRT [TRT] configs (use on machines without TensorRT)",
    )
    parser.add_argument(
        "--only-trt", action="store_true",
        help="Run only TensorRT [TRT] configs (skip PyTorch baselines)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for clip shuffling (default: 42)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"],
        help="Inference device for YOLO backends (default: auto)",
    )
    args = parser.parse_args()

    infer_device = resolve_inference_device(args.device)

    style()

    # -------------------------------------------------------------------
    # Load frames
    # -------------------------------------------------------------------
    clip_info = {}

    if args.input_dir:
        frames, clip_info = load_frames_from_dir(
            args.input_dir,
            max_frames_per_clip=args.max_frames_per_clip,
            max_clips=args.max_clips,
            seed=args.seed,
        )
        if clip_info.get("n_clips", 0) < DEFAULT_MIN_CLIPS:
            print(f"\n  NOTE: Only {clip_info.get('n_clips', 0)} clips loaded. "
                  f"For robust benchmarking, use >= {DEFAULT_MIN_CLIPS} clips.")
    elif args.input:
        frames = load_frames_from_video(args.input, max_frames=9999)
        clip_info = {"n_clips": 1, "total_frames": len(frames),
                     "clips": [{"path": args.input, "n_frames": len(frames)}]}
    else:
        w, h = map(int, args.resolution.split("x"))
        frames = generate_synthetic_frames(n=max(100, args.runs), h=h, w=w)
        clip_info = {"n_clips": 0, "total_frames": len(frames),
                     "source": "synthetic"}

    if not frames:
        print("  ERROR: No frames available.")
        return

    # Ensure we have enough frames for the requested runs
    timed_runs = max(args.runs, DEFAULT_TIMED)

    print("=" * 72)
    print("  STEP 5: PERCEPTION FRONTEND BENCHMARK")
    print(f"  Input source : {'directory' if args.input_dir else 'single file' if args.input else 'synthetic'}")
    if args.input_dir:
        print(f"  Clips loaded : {clip_info.get('n_clips', '?')}")
    print(f"  Total frames : {len(frames)}")
    print(f"  Warm-up      : {args.warmup} frames (discarded)")
    print(f"  Timed runs   : {timed_runs} frames (measured)")
    print(f"  Target       : {TARGET_FPS} FPS ({TARGET_LATENCY_MS:.1f} ms)")
    print(f"  YOLO device  : {infer_device}")
    print(f"  Protocol     : NVIDIA TensorRT / MLPerf aligned")
    print("=" * 72)

    # -------------------------------------------------------------------
    # Run benchmarks
    # -------------------------------------------------------------------
    all_results = []
    skip_set = set(args.skip) if args.skip else set()

    for name, model_id, provides_pose, notes in PERCEPTION_CONFIGS:
        if name in skip_set:
            print(f"\n  Skipping {name}")
            continue
        if args.skip_trt and "[TRT]" in name:
            print(f"\n  Skipping {name} (--skip-trt)")
            continue
        if args.only_trt and "[TRT]" not in name:
            continue

        print(f"\n{'=' * 72}")
        print(f"  {name}")
        print(f"  {notes}")
        print(f"  Provides pose: {provides_pose}")
        print(f"{'=' * 72}")

        try:
            use_trt = "[TRT]" in name
            backend = create_backend(name, model_id, provides_pose,
                                     use_tensorrt=use_trt,
                                     infer_device=infer_device)

            stats = benchmark_backend(
                backend, frames,
                warmup_runs=args.warmup,
                timed_runs=timed_runs,
            )

            result = {
                "name": name,
                "model_id": model_id,
                "provides_pose": provides_pose,
                "notes": notes,
                "n_clips": clip_info.get("n_clips", 0),
                "n_source_frames": len(frames),
                **stats,
            }
            all_results.append(result)

            meets_target = "OK" if stats["fps"] >= TARGET_FPS else "BELOW"
            print(f"  mean={stats['mean_ms']:.2f}ms  "
                  f"std={stats['std_ms']:.2f}ms  "
                  f"p50={stats['p50_ms']:.2f}ms  "
                  f"p95={stats['p95_ms']:.2f}ms  "
                  f"p99={stats['p99_ms']:.2f}ms")
            print(f"  FPS={stats['fps']:.1f}  [{meets_target}]  "
                  f"det_rate={stats['detection_rate']:.2f}  "
                  f"pose_rate={stats['pose_rate']:.2f}")

        except Exception as e:
            print(f"  FAILED: {e}")
            print(f"  (Install required packages: pip install ultralytics mediapipe)")
            all_results.append({
                "name": name,
                "model_id": model_id,
                "provides_pose": provides_pose,
                "notes": notes,
                "mean_ms": float("nan"),
                "fps": 0,
                "error": str(e),
            })

    if not all_results:
        print("\n  No results to save.")
        return

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    import pandas as pd

    results_df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / "perception_benchmark.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to {csv_path}")

    # Save clip info for reproducibility
    if clip_info.get("clips"):
        clip_df = pd.DataFrame(clip_info["clips"])
        clip_df.to_csv(RESULTS_DIR / "perception_benchmark_clips.csv", index=False)

    # --- Plot: FPS comparison ---
    try:
        import matplotlib.pyplot as plt

        valid = results_df.dropna(subset=["mean_ms"])
        if not valid.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Left: FPS bar chart
            ax = axes[0]
            colors = []
            for _, r in valid.iterrows():
                if r.get("provides_pose"):
                    colors.append("#9467bd")  # purple for pose
                else:
                    colors.append("#1f77b4")  # blue for detection
            y_pos = range(len(valid))

            ax.barh(y_pos, valid["fps"], color=colors,
                    edgecolor="white", linewidth=0.6)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(valid["name"], fontsize=10)
            ax.invert_yaxis()

            ax.axvline(x=TARGET_FPS, color="red", linestyle="--",
                       alpha=0.7, linewidth=1.5)
            ax.text(TARGET_FPS + 1, len(valid) - 0.5,
                    f"{TARGET_FPS} FPS target",
                    color="red", fontsize=9, va="center")

            ax.set_xlabel("Throughput (FPS)")
            ax.set_title("Perception Frontend: Throughput")

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#1f77b4", label="Detection only"),
                Patch(facecolor="#9467bd", label="Detection + Pose"),
            ]
            ax.legend(handles=legend_elements, loc="lower right")

            # Right: Latency distribution (p50, p95, p99)
            ax2 = axes[1]
            x_pos = range(len(valid))
            names = valid["name"].tolist()

            ax2.bar([x - 0.2 for x in x_pos], valid["p50_ms"], width=0.2,
                    label="p50", color="#2ca02c", alpha=0.8)
            ax2.bar(list(x_pos), valid["p95_ms"], width=0.2,
                    label="p95", color="#ff7f0e", alpha=0.8)
            ax2.bar([x + 0.2 for x in x_pos], valid["p99_ms"], width=0.2,
                    label="p99", color="#d62728", alpha=0.8)

            ax2.axhline(y=TARGET_LATENCY_MS, color="red", linestyle="--",
                        alpha=0.7, linewidth=1.5)
            ax2.text(len(valid) - 0.5, TARGET_LATENCY_MS + 1,
                     f"{TARGET_LATENCY_MS:.0f}ms budget",
                     color="red", fontsize=9, ha="right")

            ax2.set_xticks(list(x_pos))
            ax2.set_xticklabels(names, fontsize=9, rotation=30, ha="right")
            ax2.set_ylabel("Latency (ms)")
            ax2.set_title("Perception Frontend: Latency Percentiles")
            ax2.legend()

            fig.tight_layout()
            fig.savefig(FIG_DIR / "perception_benchmark.png")
            plt.close()

            print(f"  Plot saved to {FIG_DIR / 'perception_benchmark.png'}")

    except ImportError:
        print("  matplotlib not available, skipping plots")

    # Console summary
    print(f"\n{'=' * 72}")
    print("  PERCEPTION BENCHMARK SUMMARY")
    if clip_info.get("n_clips", 0) > 0:
        print(f"  Input: {clip_info['n_clips']} video clips, "
              f"{clip_info.get('total_frames', '?')} total source frames")
    print(f"  Protocol: {args.warmup} warmup + {timed_runs} measured frames")
    print(f"{'=' * 72}")
    for _, r in results_df.iterrows():
        fps = r.get("fps", 0)
        if np.isnan(fps):
            print(f"  {r['name']:30s}  FAILED: {r.get('error', 'unknown')}")
        else:
            ok = "OK" if fps >= TARGET_FPS else "BELOW"
            pose_str = "pose" if r["provides_pose"] else "bbox"
            print(f"  {r['name']:30s}  FPS={fps:7.1f}  "
                  f"mean={r['mean_ms']:6.2f}ms  "
                  f"p50={r['p50_ms']:6.2f}ms  "
                  f"p95={r['p95_ms']:6.2f}ms  "
                  f"p99={r['p99_ms']:6.2f}ms  "
                  f"[{ok}]  ({pose_str})")


if __name__ == "__main__":
    main()

