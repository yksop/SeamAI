#!/usr/bin/env python3
"""
Step 9 — End-to-end system latency benchmark with video input (RQ3).

Measures the true end-to-end latency of the full intent prediction
pipeline using a pre-recorded video clip (or live camera). This captures
delays that tensor-only benchmarks miss: video decode / USB capture
latency, frame decoding, and real-world lighting conditions.

The pipeline stages timed per frame are:
    Camera/video capture  ->  Perception (YOLO)
                          ->  Feature extraction
                          ->  GRU classifier
                          ->  Decision output

GDPR compliance: NO frames are saved to disk. All processing is done
in-memory, and only timing measurements are recorded.

Modes:
    --mode quick      5 configs: YOLOv8n-pose FP16 × 5 GRU models
    --mode nano      20 configs: 6 nano variants × 5 GRU models
    --mode standard  28 configs: nano+small FP32/FP16 × 5 GRU (DEFAULT)
    --mode full      60 configs: all 18 perception × 5 GRU (incl INT8+medium)

The 18 perception variants mirror the model-size sweep from step 7:
    YOLOv8 nano/small/medium × detection/pose × FP32/FP16/INT8

The 5 GRU models are the RQ1 finalists:
    A4_core4 (d=4), A2_traj6 (d=6), B1_head_angle (d=7),
    B4_torso_head_raw (d=18), B6_full_body_raw (d=34)

Note: detection-only frontends can only pair with GRU models that use
bounding-box features (A4_core4, A2_traj6). Pose-dependent GRU models
(B1, B4, B6) are automatically skipped for detection-only frontends.

Output:
    outputs/results/rq3_system_benchmark.csv       — per-frame timing for all configs
    outputs/results/rq3_system_summary.csv         — aggregated statistics

Run on the Jetson:
    python step9_benchmark_live_camera.py --video benchmark_videos/rq3_system_clip/rq3_clip.mp4
    python step9_benchmark_live_camera.py --video benchmark_videos/rq3_system_clip/rq3_clip.mp4 --mode quick
    python step9_benchmark_live_camera.py --video benchmark_videos/rq3_system_clip/rq3_clip.mp4 --mode full
    python step9_benchmark_live_camera.py --camera 0 --duration 60
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from config import (
    CHECKPOINT_DIR, RESULTS_DIR, FIG_DIR,
    TARGET_FPS, TARGET_LATENCY_MS,
    TIER_NEEDS_POSE,
    COMPLEXITY_TIERS,
    style,
)


# ---------------------------------------------------------------------------
# GRU finalists from RQ1
# ---------------------------------------------------------------------------
GRU_MODELS = [
    # (label, experiment_key, needs_pose)
    ("Core-4 (d=4)",          "A4_core4",          False),
    ("Traj-6 (d=6)",          "A2_traj6",          False),
    ("Head-angle (d=7)",      "B1_head_angle",     True),
    ("Torso+head (d=18)",     "B4_torso_head_raw", True),
    ("Full-body (d=34)",      "B6_full_body_raw",  True),
]

# ---------------------------------------------------------------------------
# Perception frontend configurations
# ---------------------------------------------------------------------------
# (label, model_id, provides_pose, precision)
# precision: "fp32" = PyTorch, "fp16" = TRT FP16, "int8" = TRT INT8

PERCEPTION_NANO = [
    ("YOLOv8n",              "yolov8n.pt",      False, "fp32"),
    ("YOLOv8n [FP16]",       "yolov8n.pt",      False, "fp16"),
    ("YOLOv8n [INT8]",       "yolov8n.pt",      False, "int8"),
    ("YOLOv8n-pose",         "yolov8n-pose.pt", True,  "fp32"),
    ("YOLOv8n-pose [FP16]",  "yolov8n-pose.pt", True,  "fp16"),
    ("YOLOv8n-pose [INT8]",  "yolov8n-pose.pt", True,  "int8"),
]

PERCEPTION_SMALL = [
    ("YOLOv8s",              "yolov8s.pt",      False, "fp32"),
    ("YOLOv8s [FP16]",       "yolov8s.pt",      False, "fp16"),
    ("YOLOv8s [INT8]",       "yolov8s.pt",      False, "int8"),
    ("YOLOv8s-pose",         "yolov8s-pose.pt", True,  "fp32"),
    ("YOLOv8s-pose [FP16]",  "yolov8s-pose.pt", True,  "fp16"),
    ("YOLOv8s-pose [INT8]",  "yolov8s-pose.pt", True,  "int8"),
]

PERCEPTION_MEDIUM = [
    ("YOLOv8m",              "yolov8m.pt",      False, "fp32"),
    ("YOLOv8m [FP16]",       "yolov8m.pt",      False, "fp16"),
    ("YOLOv8m [INT8]",       "yolov8m.pt",      False, "int8"),
    ("YOLOv8m-pose",         "yolov8m-pose.pt", True,  "fp32"),
    ("YOLOv8m-pose [FP16]",  "yolov8m-pose.pt", True,  "fp16"),
    ("YOLOv8m-pose [INT8]",  "yolov8m-pose.pt", True,  "int8"),
]

PERCEPTION_ALL = PERCEPTION_NANO + PERCEPTION_SMALL + PERCEPTION_MEDIUM

# Quick mode: only the recommended deployment config
PERCEPTION_QUICK = [
    ("YOLOv8n-pose [FP16]",  "yolov8n-pose.pt", True,  "fp16"),
]

# Standard mode: nano + small, FP32 + FP16 (skip INT8 and medium)
# Rationale: RQ2 showed INT8 gives only marginal improvement over FP16,
# and medium FP32 is the only config that fails 30 FPS. This set covers
# the interesting range while keeping runtime reasonable (~28 configs).
PERCEPTION_STANDARD = [
    p for p in PERCEPTION_NANO + PERCEPTION_SMALL
    if p[3] != "int8"
]

# Benchmark settings
DEFAULT_DURATION_S = 30      # seconds of video per config
DEFAULT_WARMUP_S = 5         # seconds of warmup (discarded)
DEFAULT_CAMERA = 0           # default camera index


def detect_camera(camera_arg):
    """Try to open the camera and return a working VideoCapture."""
    sources = [camera_arg]

    if isinstance(camera_arg, str) and camera_arg.isdigit():
        sources = [int(camera_arg), camera_arg]
    elif isinstance(camera_arg, int):
        sources = [camera_arg, str(camera_arg)]

    if camera_arg in (0, "0"):
        sources.extend(["/dev/video0", "/dev/video1"])

    for src in sources:
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"  Camera opened: {src}")
                print(f"  Resolution: {w}x{h}")
                print(f"  Reported FPS: {fps:.1f}")
                return cap
            cap.release()

    return None


def load_trt_model(model_id, precision, infer_device):
    """
    Load or export a TensorRT engine at the given precision.
    Reuses cached engine files (named with precision suffix).
    """
    from ultralytics import YOLO

    stem = Path(model_id).stem
    engine_name = f"{stem}_{precision}.engine"
    engine_path = Path(engine_name)

    if engine_path.exists():
        print(f"    Loading cached {precision.upper()} engine: {engine_path}")
        return YOLO(str(engine_path))

    print(f"    Exporting {model_id} -> TensorRT {precision.upper()} ...")
    base_model = YOLO(model_id)

    trt_device = 0
    if isinstance(infer_device, str) and infer_device.startswith("cuda:"):
        try:
            trt_device = int(infer_device.split(":", 1)[1])
        except (ValueError, IndexError):
            trt_device = 0

    export_kwargs = {
        "format": "engine",
        "device": trt_device,
        "verbose": False,
    }

    if precision == "fp16":
        export_kwargs["half"] = True
    elif precision == "int8":
        export_kwargs["int8"] = True

    try:
        exported = base_model.export(**export_kwargs)
        exported_path = Path(exported)
        if exported_path.exists() and exported_path != engine_path:
            exported_path.rename(engine_path)
        print(f"    Exported: {engine_path}")
        return YOLO(str(engine_path))
    except Exception as e:
        print(f"    TensorRT {precision.upper()} export failed: {e}")
        return None


def create_perception_backend(perc_label, model_id, provides_pose, precision,
                              infer_device):
    """
    Create a YOLO perception backend supporting FP32, FP16, and INT8.
    Returns a YOLOBackend instance from step5.
    """
    from step5_benchmark_perception import YOLOBackend

    if precision == "fp32":
        # Standard PyTorch model
        return YOLOBackend(model_id, provides_pose, use_tensorrt=False,
                           infer_device=infer_device)
    elif precision in ("fp16", "int8"):
        # Load or export TRT engine, then wrap in YOLOBackend-like interface
        model = load_trt_model(model_id, precision, infer_device)
        if model is None:
            raise RuntimeError(f"TRT {precision.upper()} export failed for {model_id}")
        # Create a backend that uses the pre-loaded TRT model
        backend = YOLOBackend.__new__(YOLOBackend)
        backend.provides_pose = provides_pose
        backend.use_tensorrt = True
        backend.infer_device = infer_device
        backend.model_id = model_id
        backend.model = model
        return backend
    else:
        raise ValueError(f"Unknown precision: {precision}")


def find_checkpoint(experiment_key, tte=1.0):
    """Find a GRU checkpoint for the given experiment key and TTE."""
    tte_str = f"tte{tte:.1f}"
    pattern = f"gru_{experiment_key}_{tte_str}.pt"
    ckpt_path = CHECKPOINT_DIR / pattern
    if ckpt_path.exists():
        return ckpt_path

    matches = sorted(CHECKPOINT_DIR.glob(f"gru_{experiment_key}_*.pt"))
    if matches:
        return matches[0]

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Step 9: End-to-end system latency benchmark (RQ3)")
    parser.add_argument(
        "--camera", type=str, default=str(DEFAULT_CAMERA),
        help="Camera index or device path (default: 0)")
    parser.add_argument(
        "--video", type=str, default=None,
        help="Use a pre-recorded video instead of live camera")
    parser.add_argument(
        "--duration", type=int, default=DEFAULT_DURATION_S,
        help=f"Seconds of measurement per config (default: {DEFAULT_DURATION_S})")
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP_S,
        help=f"Seconds of warmup per config (default: {DEFAULT_WARMUP_S})")
    parser.add_argument(
        "--tte", type=float, default=1.0,
        help="TTE of checkpoints to use (default: 1.0)")
    parser.add_argument(
        "--door", type=str, default="320,470",
        help="Door center position x,y in pixels (default: 320,470)")
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"],
        help="Inference device (default: auto)")
    parser.add_argument(
        "--mode", type=str, default="standard",
        choices=["quick", "nano", "standard", "full"],
        help="Benchmark scope: quick (1 perception × 5 GRU = 5), "
             "nano (6 nano × 5 GRU = 20), "
             "standard (8 nano+small FP32/FP16 × 5 GRU = 28), "
             "full (18 all × 5 GRU = 60). Default: standard")
    parser.add_argument(
        "--skip-int8", action="store_true",
        help="Skip INT8 configs (saves time if INT8 engines not yet exported)")
    parser.add_argument(
        "--perception-only", type=str, nargs="+", default=None,
        help="Run only specific perception labels (e.g. 'YOLOv8n [FP16]')")
    parser.add_argument(
        "--gru-only", type=str, nargs="+", default=None,
        help="Run only specific GRU experiment keys (e.g. A4_core4 A2_traj6)")
    args = parser.parse_args()

    from step5_benchmark_perception import resolve_inference_device
    infer_device = resolve_inference_device(args.device)

    door_x, door_y = map(float, args.door.split(","))
    door_center = (door_x, door_y)

    # Select perception configs based on mode
    if args.mode == "quick":
        perception_configs = PERCEPTION_QUICK
    elif args.mode == "nano":
        perception_configs = PERCEPTION_NANO
    elif args.mode == "standard":
        perception_configs = PERCEPTION_STANDARD
    elif args.mode == "full":
        perception_configs = PERCEPTION_ALL

    # Apply filters
    if args.skip_int8:
        perception_configs = [p for p in perception_configs if p[3] != "int8"]

    if args.perception_only:
        perception_configs = [p for p in perception_configs
                              if p[0] in args.perception_only]

    gru_models = GRU_MODELS
    if args.gru_only:
        gru_models = [g for g in gru_models if g[1] in args.gru_only]

    # Build the full config matrix: perception × GRU
    # Skip incompatible pairs (detection-only frontend + pose-dependent GRU)
    configs = []
    for perc_label, model_id, provides_pose, precision in perception_configs:
        for gru_label, exp_key, gru_needs_pose in gru_models:
            if gru_needs_pose and not provides_pose:
                # Skip: this GRU needs pose but frontend is detection-only
                continue
            configs.append((perc_label, model_id, provides_pose, precision,
                            gru_label, exp_key))

    print(f"\n{'='*70}")
    print(f"  STEP 9: END-TO-END SYSTEM LATENCY BENCHMARK (RQ3)")
    print(f"{'='*70}")
    print(f"  Device         : {infer_device}")
    print(f"  Mode           : {args.mode}")
    print(f"  Perception     : {len(perception_configs)} frontends")
    print(f"  GRU models     : {len(gru_models)}")
    print(f"  Total configs  : {len(configs)} (after compatibility filter)")
    print(f"  Duration/config: {args.duration}s measurement + {args.warmup}s warmup")
    print(f"  Door center    : {door_center}")
    print(f"  TTE            : {args.tte}s")
    print(f"  GDPR           : NO frames saved to disk")

    est_minutes = len(configs) * (args.duration + args.warmup) / 60
    print(f"  Est. total time: ~{est_minutes:.0f} minutes")
    print(f"{'='*70}")

    # -------------------------------------------------------------------
    # Open camera or video source
    # -------------------------------------------------------------------
    if args.video:
        print(f"\n  Using pre-recorded video: {args.video}")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"  ERROR: Cannot open video: {args.video}")
            return
        is_live = False
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  Video frames: {total_frames}, FPS: {video_fps:.1f}")
    else:
        print(f"\n  Detecting camera (index={args.camera}) ...")
        cap = detect_camera(args.camera)
        if cap is None:
            print(f"  ERROR: No camera found. Try --camera <index> or --video <path>")
            return
        is_live = True

    ret, test_frame = cap.read()
    if not ret:
        print("  ERROR: Cannot read from camera/video")
        cap.release()
        return
    frame_h, frame_w = test_frame.shape[:2]
    print(f"  Frame size: {frame_w}x{frame_h}")

    # -------------------------------------------------------------------
    # Verify checkpoints exist for all GRU models
    # -------------------------------------------------------------------
    print(f"\n  Checking GRU checkpoints ...")
    gru_checkpoints = {}
    for gru_label, exp_key, gru_needs_pose in gru_models:
        ckpt = find_checkpoint(exp_key, args.tte)
        if ckpt is None:
            print(f"    SKIP {gru_label}: no checkpoint for {exp_key}")
        else:
            gru_checkpoints[exp_key] = ckpt
            print(f"    OK: {gru_label} -> {ckpt.name}")

    # Filter configs to only those with available checkpoints
    configs = [c for c in configs if c[5] in gru_checkpoints]
    if not configs:
        print("  ERROR: No valid configs (missing checkpoints). Run step2 first.")
        cap.release()
        return
    print(f"\n  Valid configs after checkpoint check: {len(configs)}")

    # -------------------------------------------------------------------
    # Import pipeline class from step6
    # -------------------------------------------------------------------
    from step6_benchmark_pipeline import PipelineBenchmark

    # -------------------------------------------------------------------
    # Benchmark each configuration
    # -------------------------------------------------------------------
    import pandas as pd

    all_frame_results = []
    summary_results = []

    # Cache perception backends to avoid reloading when only GRU changes
    backend_cache = {}

    for config_idx, (perc_label, model_id, provides_pose, precision,
                     gru_label, exp_key) in enumerate(configs):

        config_name = f"{perc_label} × {gru_label}"
        ckpt_path = gru_checkpoints[exp_key]

        print(f"\n{'='*70}")
        print(f"  [{config_idx+1}/{len(configs)}] {config_name}")
        print(f"  Perception: {perc_label} ({precision}) | GRU: {exp_key}")
        print(f"{'='*70}")

        # --- Create or reuse perception backend ---
        cache_key = (perc_label, model_id, precision)
        if cache_key in backend_cache:
            backend = backend_cache[cache_key]
            print(f"  Reusing cached backend: {perc_label}")
        else:
            try:
                backend = create_perception_backend(
                    perc_label, model_id, provides_pose, precision,
                    infer_device,
                )
                backend_cache[cache_key] = backend
            except Exception as e:
                print(f"  FAILED to create backend: {e}")
                continue

        # --- Create pipeline ---
        try:
            pipeline = PipelineBenchmark(
                backend, str(ckpt_path),
                door_center=door_center,
                frame_w=frame_w, frame_h=frame_h,
            )
        except Exception as e:
            print(f"  FAILED to create pipeline: {e}")
            continue

        # --- Warmup phase ---
        print(f"  Warming up ({args.warmup}s) ...")
        backend.warmup(test_frame)

        warmup_start = time.time()
        frame_count = 0
        while time.time() - warmup_start < args.warmup:
            if is_live:
                ret, frame = cap.read()
            else:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
            if not ret:
                break
            pipeline.process_frame(frame)
            frame_count += 1

        print(f"    Warmup: {frame_count} frames in {args.warmup}s")

        # Reset pipeline buffer for clean measurement
        pipeline.reset()

        # --- Measurement phase ---
        print(f"  Measuring ({args.duration}s) ...")
        measure_start = time.time()
        frame_idx = 0
        config_timings = []

        while time.time() - measure_start < args.duration:
            # Capture timing includes the actual camera/video read
            t_capture_start = time.perf_counter()
            if is_live:
                ret, frame = cap.read()
            else:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
            t_capture_end = time.perf_counter()

            if not ret:
                print("  WARNING: Frame read failed, stopping")
                break

            capture_ms = (t_capture_end - t_capture_start) * 1000

            # Run the pipeline (perception + features + GRU)
            logit, timings = pipeline.process_frame(frame)

            # Add capture timing
            timings["capture_ms"] = capture_ms

            # Compute end-to-end including capture
            if "total_ms" in timings:
                timings["e2e_ms"] = capture_ms + timings["total_ms"]
            else:
                timings["e2e_ms"] = capture_ms + timings.get("perception_ms", 0)

            timings["frame_idx"] = frame_idx
            timings["config"] = config_name
            timings["perception"] = perc_label
            timings["precision"] = precision
            timings["gru_model"] = gru_label
            timings["experiment_key"] = exp_key
            timings["has_detection"] = logit is not None
            timings["logit"] = logit if logit is not None else float("nan")

            config_timings.append(timings)
            frame_idx += 1

        elapsed = time.time() - measure_start
        print(f"    Captured {frame_idx} frames in {elapsed:.1f}s "
              f"({frame_idx/elapsed:.1f} effective FPS)")

        # --- Aggregate stats for this config ---
        valid = [t for t in config_timings if "total_ms" in t]
        if not valid:
            print("    No valid detections!")
            continue

        capture = np.array([t["capture_ms"] for t in config_timings])
        total_pipeline = np.array([t["total_ms"] for t in valid])
        e2e = np.array([t["e2e_ms"] for t in valid])
        perc = np.array([t["perception_ms"] for t in valid])
        feat = np.array([t.get("features_ms", 0) for t in valid])
        gru = np.array([t.get("gru_ms", 0) for t in valid])

        summary = {
            "config": config_name,
            "perception": perc_label,
            "model_id": model_id,
            "precision": precision,
            "provides_pose": provides_pose,
            "gru_model": gru_label,
            "experiment_key": exp_key,
            "is_live_camera": is_live,
            "n_frames": frame_idx,
            "n_detections": len(valid),
            "detection_rate": len(valid) / max(frame_idx, 1),
            "duration_s": elapsed,
            "effective_fps": frame_idx / elapsed,
            # Camera capture
            "capture_mean_ms": capture.mean(),
            "capture_p50_ms": np.median(capture),
            "capture_p95_ms": np.percentile(capture, 95),
            "capture_p99_ms": np.percentile(capture, 99),
            # Pipeline (perception + features + GRU)
            "pipeline_mean_ms": total_pipeline.mean(),
            "pipeline_p50_ms": np.median(total_pipeline),
            "pipeline_p95_ms": np.percentile(total_pipeline, 95),
            "pipeline_p99_ms": np.percentile(total_pipeline, 99),
            # End-to-end (capture + pipeline)
            "e2e_mean_ms": e2e.mean(),
            "e2e_p50_ms": np.median(e2e),
            "e2e_p95_ms": np.percentile(e2e, 95),
            "e2e_p99_ms": np.percentile(e2e, 99),
            "e2e_fps": 1000.0 / e2e.mean(),
            # Per-stage breakdown
            "perception_mean_ms": perc.mean(),
            "features_mean_ms": feat.mean(),
            "gru_mean_ms": gru.mean(),
            # Target assessment
            "meets_30fps_pipeline": total_pipeline.mean() <= TARGET_LATENCY_MS,
            "meets_30fps_e2e": e2e.mean() <= TARGET_LATENCY_MS,
        }
        summary_results.append(summary)
        all_frame_results.extend(config_timings)

        # Print per-config summary
        print(f"\n    {'Stage':<20s}  {'Mean':>8s}  {'p50':>8s}  {'p95':>8s}  {'p99':>8s}")
        print(f"    {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
        print(f"    {'Camera capture':<20s}  {summary['capture_mean_ms']:>7.2f}  "
              f"{summary['capture_p50_ms']:>7.2f}  "
              f"{summary['capture_p95_ms']:>7.2f}  "
              f"{summary['capture_p99_ms']:>7.2f}")
        print(f"    {'Perception':<20s}  {summary['perception_mean_ms']:>7.2f}  "
              f"{'':>8s}  {'':>8s}  {'':>8s}")
        print(f"    {'Feature extract.':<20s}  {summary['features_mean_ms']:>7.2f}  "
              f"{'':>8s}  {'':>8s}  {'':>8s}")
        print(f"    {'GRU inference':<20s}  {summary['gru_mean_ms']:>7.2f}  "
              f"{'':>8s}  {'':>8s}  {'':>8s}")
        print(f"    {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
        print(f"    {'Pipeline total':<20s}  {summary['pipeline_mean_ms']:>7.2f}  "
              f"{summary['pipeline_p50_ms']:>7.2f}  "
              f"{summary['pipeline_p95_ms']:>7.2f}  "
              f"{summary['pipeline_p99_ms']:>7.2f}")
        print(f"    {'End-to-end':<20s}  {summary['e2e_mean_ms']:>7.2f}  "
              f"{summary['e2e_p50_ms']:>7.2f}  "
              f"{summary['e2e_p95_ms']:>7.2f}  "
              f"{summary['e2e_p99_ms']:>7.2f}")

        e2e_ok = "YES" if summary["meets_30fps_e2e"] else "NO"
        print(f"\n    End-to-end FPS: {summary['e2e_fps']:.1f}  "
              f"(meets 30 FPS: {e2e_ok})")

        # Reset video to start for next config
        if not is_live:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cap.release()

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    if not summary_results:
        print("\n  No results to save.")
        return

    df_summary = pd.DataFrame(summary_results)
    df_frames = pd.DataFrame(all_frame_results)

    csv_summary = RESULTS_DIR / "rq3_system_summary.csv"
    csv_frames = RESULTS_DIR / "rq3_system_benchmark.csv"

    df_summary.to_csv(csv_summary, index=False)
    df_frames.to_csv(csv_frames, index=False)

    print(f"\n  Summary saved to: {csv_summary}")
    print(f"  Per-frame data saved to: {csv_frames}")

    # -------------------------------------------------------------------
    # Final summary table
    # -------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"  RQ3: SYSTEM LATENCY SUMMARY")
    print(f"{'='*80}")
    print(f"  Source: {'Live camera' if is_live else 'Pre-recorded video'}")
    print(f"  Frame size: {frame_w}x{frame_h}")
    print(f"{'='*80}")
    print(f"  {'Config':<50s}  {'Prec':>5s}  {'Capt':>6s}  {'Pipe':>6s}  "
          f"{'E2E':>6s}  {'FPS':>6s}  {'30?':>4s}")
    print(f"  {'-'*50}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*4}")
    for s in summary_results:
        ok = "YES" if s["meets_30fps_e2e"] else "NO"
        print(f"  {s['config']:<50s}  {s['precision']:>5s}  "
              f"{s['capture_mean_ms']:>5.1f}  "
              f"{s['pipeline_mean_ms']:>5.1f}  "
              f"{s['e2e_mean_ms']:>5.1f}  "
              f"{s['e2e_fps']:>5.1f}  {ok:>4s}")

    # -------------------------------------------------------------------
    # Group by perception frontend (averaged across GRU models)
    # -------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"  PERCEPTION FRONTEND COMPARISON (averaged across GRU models)")
    print(f"{'='*80}")
    perc_groups = {}
    for s in summary_results:
        key = s["perception"]
        if key not in perc_groups:
            perc_groups[key] = []
        perc_groups[key].append(s)

    print(f"  {'Perception':<30s}  {'Prec':>5s}  {'E2E ms':>7s}  {'FPS':>6s}  "
          f"{'Det%':>5s}  {'30?':>4s}")
    print(f"  {'-'*30}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*5}  {'-'*4}")
    for key in perc_groups:
        group = perc_groups[key]
        avg_e2e = np.mean([s["e2e_mean_ms"] for s in group])
        avg_fps = 1000.0 / avg_e2e
        avg_det = np.mean([s["detection_rate"] for s in group])
        prec = group[0]["precision"]
        ok = "YES" if avg_e2e <= TARGET_LATENCY_MS else "NO"
        print(f"  {key:<30s}  {prec:>5s}  {avg_e2e:>6.1f}  {avg_fps:>5.1f}  "
              f"{avg_det:>4.0%}  {ok:>4s}")

    # -------------------------------------------------------------------
    # Note about camera latency
    # -------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"  NOTE: Camera/video decode adds to latency. For live camera,")
    print(f"  USB cameras typically add 5-30ms; CSI cameras add 1-5ms.")
    print(f"  EN 16005 requires total system response < 500ms for safety.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
