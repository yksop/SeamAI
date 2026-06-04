#!/usr/bin/env python3
"""
Step 6 — End-to-end pipeline benchmark: perception + features + GRU.

This is the core RQ2 experiment. It measures the full inference pipeline
as it would run in production:

    Camera frame  ->  Perception (YOLO/MediaPipe)
                  ->  Feature extraction (trajectory + optional pose)
                  ->  GRU classifier
                  ->  Intent prediction (enter/pass)

The script tests all valid (perception frontend, GRU tier) combinations
and produces a Pareto frontier of accuracy vs. pipeline FPS.

Valid combinations:
    - Tiers 1-2 (bbox-only features): any detection-only frontend
    - Tiers 3-5 (pose features): only pose-capable frontends

Benchmarking protocol (NVIDIA TensorRT + MLPerf aligned):
    - Input: multiple diverse video clips via --input-dir (recommended
      20-30 clips from the RISE dataset, mix of enter/pass)
    - Warmup: 50 frames discarded per (frontend x tier) combination
    - Measurement: >= 500 frames in steady state
    - Per-stage timing: perception, feature extraction, GRU separately
    - Report: mean, std, p50, p95, p99 for total and per-stage latency

For each combination, we measure:
    - Total pipeline latency per frame (ms)
    - Pipeline throughput (FPS)
    - Breakdown: perception time + feature extraction time + GRU time
    - Whether the 30 FPS target is met

Output:
    outputs/results/pipeline_benchmark.csv      — one row per (frontend x tier)
    outputs/results/pipeline_pareto.csv         — Pareto-optimal combinations
    outputs/figures/pipeline_fps_vs_accuracy.png — the main RQ2 figure

Run on the Jetson:
    python step6_benchmark_pipeline.py --input-dir path/to/clips/
    python step6_benchmark_pipeline.py --input path/to/video.mp4

Quick check without video (no camera or clips needed):
    python step6_benchmark_pipeline.py --synthetic

Dependencies:
    pip install ultralytics mediapipe opencv-python torch onnxruntime
"""

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from config import (
    CHECKPOINT_DIR, RESULTS_DIR, FIG_DIR,
    WARMUP_RUNS, TIMED_RUNS,
    TARGET_FPS, TARGET_LATENCY_MS,
    TIER_COLORS, TIER_NEEDS_POSE,
    style,
)

# Defaults
DEFAULT_WARMUP = 50
DEFAULT_TIMED = 500


# ---------------------------------------------------------------------------
# Feature extraction from perception output
# ---------------------------------------------------------------------------

def extract_trajectory_features(bbox, door_center, prev_bbox=None,
                                 fps=25.0, frame_h=480):
    """
    Extract the 6 per-frame trajectory features from a bounding box.

    Features:
        dist_to_door       — Euclidean distance from person centre to door
        closure_rate       — rate of change of distance (pixels/s)
        vx                 — horizontal velocity (pixels/frame)
        vy                 — vertical velocity (pixels/frame)
        rel_angle_to_door  — angle between heading direction and door
        bbox_h             — bounding box height
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bh = y2 - y1

    door_x, door_y = door_center
    dist = math.hypot(cx - door_x, cy - door_y)

    if prev_bbox is not None:
        px1, py1, px2, py2 = prev_bbox
        pcx = (px1 + px2) / 2.0
        pcy = (py1 + py2) / 2.0
        prev_dist = math.hypot(pcx - door_x, pcy - door_y)
        closure = (prev_dist - dist) * fps
        vx = cx - pcx
        vy = cy - pcy
        speed = math.hypot(vx, vy)
        if speed > 1e-6:
            dx_door = door_x - cx
            dy_door = door_y - cy
            cos_angle = (vx * dx_door + vy * dy_door) / (speed * max(dist, 1e-6))
            rel_angle = math.acos(max(-1.0, min(1.0, cos_angle)))
        else:
            rel_angle = 0.0
    else:
        closure = 0.0
        vx = 0.0
        vy = 0.0
        rel_angle = 0.0

    return np.array([dist, closure, vx, vy, rel_angle, bh], dtype=np.float32)


def extract_pose_features_from_keypoints(keypoints, door_center,
                                          frame_w, frame_h, pose_mode="head_angle"):
    """
    Extract pose features from YOLO-pose or MediaPipe keypoints.

    Adapts to whatever pose_mode the GRU tier requires:
      - "head_angle": 1 angle feature
      - "torso_head_raw": 12 raw (x,y) features
      - "full_body_raw": 28 raw (x,y) features
    """
    if keypoints is None:
        return None

    # COCO keypoint indices (used by YOLO-pose):
    # 0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear,
    # 5:left_shoulder, 6:right_shoulder, 7:left_elbow, 8:right_elbow,
    # 9:left_wrist, 10:right_wrist, 11:left_hip, 12:right_hip,
    # 13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle

    COCO_MAP = {
        "nose": 0, "left_ear": 3, "right_ear": 4,
        "left_shoulder": 5, "right_shoulder": 6,
        "left_elbow": 7, "right_elbow": 8,
        "left_wrist": 9, "right_wrist": 10,
        "left_hip": 11, "right_hip": 12,
        "left_knee": 13, "right_knee": 14,
        "left_ankle": 15, "right_ankle": 16,
    }

    door_x, door_y = door_center

    def get_px(name):
        idx = COCO_MAP.get(name)
        if idx is None or idx >= len(keypoints):
            return 0.0, 0.0
        kp = keypoints[idx]
        x, y = float(kp[0]), float(kp[1])
        if x <= 1.0 and y <= 1.0:  # likely normalized
            x *= frame_w
            y *= frame_h
        return x, y

    def angle_at_door(left_name, right_name):
        lx, ly = get_px(left_name)
        rx, ry = get_px(right_name)
        dx1, dy1 = lx - door_x, ly - door_y
        dx2, dy2 = rx - door_x, ry - door_y
        dot = dx1 * dx2 + dy1 * dy2
        m1 = math.hypot(dx1, dy1)
        m2 = math.hypot(dx2, dy2)
        if m1 < 1e-8 or m2 < 1e-8:
            return 0.0
        cos_val = max(-1.0, min(1.0, dot / (m1 * m2)))
        return math.acos(cos_val)

    def raw_xy(name):
        idx = COCO_MAP.get(name)
        if idx is None or idx >= len(keypoints):
            return [0.0, 0.0]
        kp = keypoints[idx]
        x, y = float(kp[0]), float(kp[1])
        if x > 1.0:
            x /= frame_w
        if y > 1.0:
            y /= frame_h
        return [x, y]

    return {
        "angle_at_door": angle_at_door,
        "raw_xy": raw_xy,
        "COCO_MAP": COCO_MAP,
    }


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

class PipelineBenchmark:
    """Runs the full perception + feature extraction + GRU pipeline."""

    def __init__(self, perception_backend, gru_checkpoint_path,
                 door_center=(320, 470), frame_w=640, frame_h=480):
        self.backend = perception_backend
        self.door_center = door_center
        self.frame_w = frame_w
        self.frame_h = frame_h

        # Load GRU checkpoint
        ckpt = torch.load(gru_checkpoint_path, map_location="cpu",
                          weights_only=False)
        self.meta = ckpt["metadata"]
        self.mean = ckpt["mean"]
        self.std = ckpt["std"]
        self.feature_names = self.meta["feature_names"]
        self.input_dim = self.meta["input_dim"]
        self.seq_len = self.meta["seq_len"]

        from utils.models import IntentGRU
        self.model = IntentGRU(
            self.input_dim,
            hidden_size=self.meta["hidden_size"],
            dropout=self.meta["dropout"],
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # Determine which features are trajectory vs pose
        traj_names = {"dist_to_door", "closure_rate", "vx", "vy",
                      "rel_angle_to_door", "bbox_h"}
        self.pose_feature_names = [
            f for f in self.feature_names if f not in traj_names
        ]
        self.needs_pose = len(self.pose_feature_names) > 0

        # Rolling buffer for sequence
        self.buffer = np.zeros((self.seq_len, self.input_dim), dtype=np.float32)
        self.prev_bbox = None

    def reset(self):
        self.buffer = np.zeros((self.seq_len, self.input_dim), dtype=np.float32)
        self.prev_bbox = None

    def process_frame(self, frame):
        """
        Full pipeline: frame -> perception -> features -> GRU -> logit.

        Returns (logit, timing_dict) where timing_dict breaks down latency.
        """
        timings = {}

        # 1. Perception
        t0 = time.perf_counter()
        bbox, keypoints = self.backend(frame)
        t1 = time.perf_counter()
        timings["perception_ms"] = (t1 - t0) * 1000

        if bbox is None:
            return None, timings

        # 2. Feature extraction
        t2 = time.perf_counter()

        # Trajectory features (always 6D)
        traj = extract_trajectory_features(
            bbox, self.door_center, self.prev_bbox,
            fps=25.0, frame_h=self.frame_h,
        )
        self.prev_bbox = bbox

        # Build full feature vector
        if self.needs_pose and keypoints is not None:
            helpers = extract_pose_features_from_keypoints(
                keypoints, self.door_center,
                self.frame_w, self.frame_h,
            )
            pose_vals = []
            for fname in self.pose_feature_names:
                if fname.endswith("_angle") or fname in {
                    "hip_angle", "shoulder_angle", "head_angle",
                    "elbow_angle", "wrist_angle", "knee_angle", "ankle_angle"
                }:
                    pair_map = {
                        "hip_angle": ("left_hip", "right_hip"),
                        "shoulder_angle": ("left_shoulder", "right_shoulder"),
                        "head_angle": ("left_ear", "right_ear"),
                        "elbow_angle": ("left_elbow", "right_elbow"),
                        "wrist_angle": ("left_wrist", "right_wrist"),
                        "knee_angle": ("left_knee", "right_knee"),
                        "ankle_angle": ("left_ankle", "right_ankle"),
                    }
                    if fname in pair_map:
                        left, right = pair_map[fname]
                        pose_vals.append(helpers["angle_at_door"](left, right))
                    else:
                        pose_vals.append(0.0)
                elif fname.endswith("_x") or fname.endswith("_y"):
                    lm_name = fname[:-2]
                    xy = helpers["raw_xy"](lm_name)
                    if fname.endswith("_x"):
                        pose_vals.append(xy[0])
                    else:
                        pose_vals.append(xy[1])
                else:
                    pose_vals.append(0.0)

            features = np.concatenate([traj, np.array(pose_vals, dtype=np.float32)])
        elif self.needs_pose:
            features = np.concatenate([
                traj, np.zeros(len(self.pose_feature_names), dtype=np.float32)
            ])
        else:
            features = traj[:self.input_dim]

        t3 = time.perf_counter()
        timings["features_ms"] = (t3 - t2) * 1000

        # 3. Update rolling buffer and run GRU
        t4 = time.perf_counter()
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = features

        x = (self.buffer - self.mean) / (self.std + 1e-8)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logit = self.model(x_tensor).item()

        t5 = time.perf_counter()
        timings["gru_ms"] = (t5 - t4) * 1000
        timings["total_ms"] = (t5 - t0) * 1000

        return logit, timings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 6: End-to-end pipeline benchmark"
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--input-dir", type=str, default=None,
        help="Directory with .mp4 clips (recommended: 20-30 diverse clips).",
    )
    grp.add_argument(
        "--input", type=str, default=None,
        help="Path to a single video file.",
    )
    grp.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic frames for development/testing.",
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help=f"Warm-up frames to discard per combo (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_TIMED,
        help=f"Minimum timed frames per combo (default: {DEFAULT_TIMED})",
    )
    parser.add_argument(
        "--max-clips", type=int, default=None,
        help="Max clips to load from --input-dir (default: all)",
    )
    parser.add_argument(
        "--max-frames-per-clip", type=int, default=150,
        help="Max frames per clip (default: 150)",
    )
    parser.add_argument(
        "--tte", type=float, default=1.0,
        help="TTE of checkpoints to use (default: 1.0)",
    )
    parser.add_argument(
        "--door", type=str, default="320,470",
        help="Door center position x,y (default: 320,470)",
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
    parser.add_argument(
        "--skip-trt", action="store_true",
        help="Skip all TensorRT [TRT] configs",
    )
    args = parser.parse_args()

    from step5_benchmark_perception import resolve_inference_device
    infer_device = resolve_inference_device(args.device)

    style()

    door_x, door_y = map(float, args.door.split(","))
    door_center = (door_x, door_y)

    # -------------------------------------------------------------------
    # Load frames
    # -------------------------------------------------------------------
    from step5_benchmark_perception import (
        load_frames_from_video, load_frames_from_dir,
        generate_synthetic_frames,
    )

    clip_info = {}

    if args.input_dir:
        frames, clip_info = load_frames_from_dir(
            args.input_dir,
            max_frames_per_clip=args.max_frames_per_clip,
            max_clips=args.max_clips,
            seed=args.seed,
        )
    elif args.input:
        frames = load_frames_from_video(args.input, max_frames=9999)
        clip_info = {"n_clips": 1, "total_frames": len(frames)}
    else:
        frames = generate_synthetic_frames(n=max(100, args.runs))
        clip_info = {"n_clips": 0, "total_frames": len(frames),
                     "source": "synthetic"}

    if not frames:
        print("  ERROR: No frames.")
        return

    frame_h, frame_w = frames[0].shape[:2]
    timed_runs = max(args.runs, DEFAULT_TIMED)

    print("=" * 72)
    print("  STEP 6: END-TO-END PIPELINE BENCHMARK")
    print(f"  Input source : {'directory' if args.input_dir else 'single file' if args.input else 'synthetic'}")
    if args.input_dir:
        print(f"  Clips loaded : {clip_info.get('n_clips', '?')}")
    print(f"  Total frames : {len(frames)}")
    print(f"  Door center  : {door_center}")
    print(f"  TTE          : {args.tte}s")
    print(f"  Warm-up      : {args.warmup} frames (discarded per combo)")
    print(f"  Timed runs   : {timed_runs} frames (measured per combo)")
    print(f"  Target       : {TARGET_FPS} FPS ({TARGET_LATENCY_MS:.1f} ms)")
    print(f"  YOLO device  : {infer_device}")
    print(f"  Protocol     : NVIDIA TensorRT / MLPerf aligned")
    print("=" * 72)

    # -------------------------------------------------------------------
    # Find checkpoints
    # -------------------------------------------------------------------
    tte_str = f"tte{args.tte:.1f}"
    checkpoints = sorted(CHECKPOINT_DIR.glob(f"gru_*_{tte_str}.pt"))

    if not checkpoints:
        print(f"  ERROR: No checkpoints matching *_{tte_str}.pt in {CHECKPOINT_DIR}")
        print(f"  Run step2_train_finalists.py first.")
        return

    print(f"\n  Found {len(checkpoints)} GRU checkpoints")

    # -------------------------------------------------------------------
    # Run benchmarks
    # -------------------------------------------------------------------
    from config import PERCEPTION_CONFIGS, COMPLEXITY_TIERS
    from step5_benchmark_perception import create_backend

    all_results = []

    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        meta = ckpt["metadata"]
        exp_key = meta["experiment_key"]
        label = meta["label"]
        input_dim = meta["input_dim"]

        # Determine tier
        tier_idx = -1
        for i, (_, lo, hi) in enumerate(COMPLEXITY_TIERS):
            if lo <= input_dim <= hi:
                tier_idx = i
                break

        needs_pose = TIER_NEEDS_POSE.get(tier_idx, True)

        print(f"\n{'=' * 72}")
        print(f"  GRU: {label} [{exp_key}]  dim={input_dim}  "
              f"tier={tier_idx}  needs_pose={needs_pose}")
        print(f"{'=' * 72}")

        # Test each compatible perception frontend
        for pname, pid, provides_pose, pnotes in PERCEPTION_CONFIGS:
            if needs_pose and not provides_pose:
                print(f"\n  SKIP: {pname} (no pose) x {label} (needs pose)")
                continue
            if args.skip_trt and "[TRT]" in pname:
                print(f"\n  SKIP: {pname} (--skip-trt)")
                continue

            print(f"\n  Testing: {pname} x {label}")

            try:
                use_trt = "[TRT]" in pname
                backend = create_backend(pname, pid, provides_pose,
                                         use_tensorrt=use_trt,
                                         infer_device=infer_device)
                pipeline = PipelineBenchmark(
                    backend, ckpt_path,
                    door_center=door_center,
                    frame_w=frame_w, frame_h=frame_h,
                )

                # Warm up: model warmup + frame-level warmup (discarded)
                backend.warmup(frames[0])
                for i in range(args.warmup):
                    pipeline.process_frame(frames[i % len(frames)])

                # Reset buffer for clean measurement
                pipeline.reset()

                # Timed runs
                all_timings = []
                for i in range(timed_runs):
                    _, timings = pipeline.process_frame(frames[i % len(frames)])
                    all_timings.append(timings)

                # Aggregate — include all frames (detection + no-detection)
                # For frames with no detection, only perception time is measured
                valid_full = [t for t in all_timings if "total_ms" in t]
                valid_perc = [t for t in all_timings if "perception_ms" in t]

                if not valid_full:
                    print(f"    No valid detections in {timed_runs} frames!")
                    continue

                total = np.array([t["total_ms"] for t in valid_full])
                perc = np.array([t["perception_ms"] for t in valid_perc])
                feat = np.array([t.get("features_ms", 0) for t in valid_full])
                gru = np.array([t.get("gru_ms", 0) for t in valid_full])

                result = {
                    "perception": pname,
                    "gru_model": label,
                    "experiment_key": exp_key,
                    "input_dim": input_dim,
                    "tier_idx": tier_idx,
                    "needs_pose": needs_pose,
                    "n_clips": clip_info.get("n_clips", 0),
                    "n_source_frames": len(frames),
                    "n_measured_frames": len(valid_full),
                    # Total pipeline
                    "total_mean_ms": total.mean(),
                    "total_std_ms": total.std(),
                    "total_p50_ms": np.median(total),
                    "total_p95_ms": np.percentile(total, 95),
                    "total_p99_ms": np.percentile(total, 99),
                    "total_fps": 1000.0 / total.mean(),
                    # Per-stage breakdown
                    "perception_mean_ms": perc.mean(),
                    "perception_p95_ms": np.percentile(perc, 95),
                    "features_mean_ms": feat.mean(),
                    "gru_mean_ms": gru.mean(),
                    # Detection rate
                    "detection_rate": len(valid_full) / timed_runs,
                    "meets_target": total.mean() <= TARGET_LATENCY_MS,
                }
                all_results.append(result)

                ok = "OK" if result["meets_target"] else "BELOW"
                print(f"    total={result['total_mean_ms']:.1f}ms "
                      f"(p50={result['total_p50_ms']:.1f} "
                      f"p95={result['total_p95_ms']:.1f} "
                      f"p99={result['total_p99_ms']:.1f})")
                print(f"    breakdown: perc={result['perception_mean_ms']:.1f} + "
                      f"feat={result['features_mean_ms']:.1f} + "
                      f"gru={result['gru_mean_ms']:.1f} ms")
                print(f"    FPS={result['total_fps']:.1f} [{ok}]  "
                      f"det_rate={result['detection_rate']:.2f}")

            except Exception as e:
                print(f"    FAILED: {e}")
                import traceback
                traceback.print_exc()

    if not all_results:
        print("\n  No results.")
        return

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    import pandas as pd
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / "pipeline_benchmark.csv", index=False)
    print(f"\n  Results saved to {RESULTS_DIR / 'pipeline_benchmark.csv'}")

    # --- Pareto frontier: accuracy vs FPS ---
    finalists_csv = RESULTS_DIR / "finalists.csv"
    if finalists_csv.exists():
        fin_df = pd.read_csv(finalists_csv)
        acc_map = dict(zip(fin_df["experiment_key"], fin_df["mean_bal_acc"]))
        results_df["mean_bal_acc"] = results_df["experiment_key"].map(acc_map)

        # Pareto-optimal: no other point has both higher FPS AND higher accuracy
        pareto = []
        sorted_df = results_df.dropna(subset=["mean_bal_acc"]).sort_values(
            "mean_bal_acc", ascending=False)
        best_fps = -1
        for _, r in sorted_df.iterrows():
            if r["total_fps"] > best_fps:
                pareto.append(r.to_dict())
                best_fps = r["total_fps"]
        pareto_df = pd.DataFrame(pareto)
        pareto_df.to_csv(RESULTS_DIR / "pipeline_pareto.csv", index=False)
        print(f"  Pareto frontier saved to {RESULTS_DIR / 'pipeline_pareto.csv'}")

    # --- Main RQ2 figure: accuracy vs pipeline FPS ---
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Left panel: Accuracy vs FPS scatter
        ax = axes[0]
        for _, r in results_df.iterrows():
            tidx = int(r["tier_idx"])
            color = TIER_COLORS[tidx] if 0 <= tidx < len(TIER_COLORS) else "#999"
            marker = "o" if r["meets_target"] else "x"
            ax.scatter(
                r["total_fps"], r.get("mean_bal_acc", 0.87),
                c=color, marker=marker, s=80, zorder=5,
            )
            ax.annotate(
                f"{r['perception'][:10]}\n(dim={int(r['input_dim'])})",
                (r["total_fps"], r.get("mean_bal_acc", 0.87)),
                fontsize=7, ha="center", va="bottom",
                textcoords="offset points", xytext=(0, 8),
            )

        ax.axvline(x=TARGET_FPS, color="red", linestyle="--",
                   alpha=0.7, linewidth=1.5, label=f"{TARGET_FPS} FPS target")

        ax.set_xlabel("Pipeline Throughput (FPS)")
        ax.set_ylabel("Mean Balanced Accuracy (from RQ1 CV)")
        ax.set_title("RQ2: Accuracy vs. Pipeline FPS")
        ax.legend(loc="lower right")

        # Right panel: Latency breakdown stacked bar
        ax2 = axes[1]
        combo_labels = [
            f"{r['perception'][:8]}\nx {r['gru_model'][:15]}"
            for _, r in results_df.iterrows()
        ]
        x_pos = range(len(results_df))

        perc_vals = results_df["perception_mean_ms"].values
        feat_vals = results_df["features_mean_ms"].values
        gru_vals = results_df["gru_mean_ms"].values

        ax2.bar(x_pos, perc_vals, label="Perception", color="#1f77b4")
        ax2.bar(x_pos, feat_vals, bottom=perc_vals,
                label="Features", color="#ff7f0e")
        ax2.bar(x_pos, gru_vals, bottom=perc_vals + feat_vals,
                label="GRU", color="#2ca02c")

        ax2.axhline(y=TARGET_LATENCY_MS, color="red", linestyle="--",
                    alpha=0.7, linewidth=1.5, label=f"{TARGET_LATENCY_MS:.0f}ms budget")

        ax2.set_xticks(list(x_pos))
        ax2.set_xticklabels(combo_labels, fontsize=7, rotation=45, ha="right")
        ax2.set_ylabel("Latency (ms)")
        ax2.set_title("Pipeline Latency Breakdown")
        ax2.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(FIG_DIR / "pipeline_fps_vs_accuracy.png")
        plt.close()

        print(f"  Figure saved to {FIG_DIR / 'pipeline_fps_vs_accuracy.png'}")

    except ImportError:
        print("  matplotlib not available, skipping plot")

    # Console summary
    print(f"\n{'=' * 72}")
    print("  PIPELINE BENCHMARK SUMMARY")
    if clip_info.get("n_clips", 0) > 0:
        print(f"  Input: {clip_info['n_clips']} video clips, "
              f"{clip_info.get('total_frames', '?')} source frames")
    print(f"  Protocol: {args.warmup} warmup + {timed_runs} measured frames per combo")
    print(f"{'=' * 72}")
    for _, r in results_df.iterrows():
        ok = "OK" if r["meets_target"] else "BELOW"
        print(f"  {r['perception']:25s} x {r['gru_model']:30s}  "
              f"FPS={r['total_fps']:7.1f}  "
              f"p95={r['total_p95_ms']:6.1f}ms  [{ok}]")


if __name__ == "__main__":
    main()
