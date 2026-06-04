#!/usr/bin/env python3
"""
Run all RQ2/RQ3 edge deployment steps in sequence, on the Jetson.

Everything runs on one machine. Steps 1-3 train the GRU finalists and write them
to checkpoints/; steps 4-9 benchmark on that same Jetson and load the checkpoints
straight from checkpoints/, so nothing is copied between machines.

Usage:
    python run_all.py            # full run (auto-uses benchmark_videos/ if populated)
    python run_all.py --from 4   # resume at step 4
    python run_all.py --only 5 6 # run just these steps
    python run_all.py --dry-run  # list the steps without running

Video inputs live under benchmark_videos/ and are auto-detected when present:
    Steps 5-7 (RQ2) : clips in benchmark_videos/rq2_perception_clips/
                      (fill it with that folder's select_clips.py, or use --synthetic).
    Step 9   (RQ3)  : benchmark_videos/rq3_system_clip/rq3_clip.mp4
                      (record it with that folder's record_clip.py, or use --camera).
Override the auto-detected paths with --input-dir / --rq3-video. Steps 1-3 and 8
need the cleaned dataset in MasterData/.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    (1, "step1_select_finalists.py", "Select GRU finalists from RQ1"),
    (2, "step2_train_finalists.py",  "Retrain finalists on full data"),
    (3, "step3_validate_hyperparams.py", "Hyperparameter sensitivity check"),
    (4, "step4_benchmark_classifier.py", "Benchmark GRU classifier inference"),
    (5, "step5_benchmark_perception.py", "Benchmark perception frontends"),
    (6, "step6_benchmark_pipeline.py",   "End-to-end pipeline benchmark"),
    (7, "step7_benchmark_model_sizes.py", "YOLO model size comparison"),
    (8, "step8_benchmark_gru_sizes.py",   "GRU hidden size comparison"),
    (9, "step9_benchmark_live_camera.py", "Live camera system latency (RQ3)"),
]


def main():
    parser = argparse.ArgumentParser(description="Run the RQ2/RQ3 pipeline on the Jetson")
    parser.add_argument(
        "--from", type=int, default=1, dest="start",
        help="Start from this step number (default: 1)",
    )
    parser.add_argument(
        "--only", type=int, nargs="+", default=None,
        help="Run only these step numbers",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print steps without executing",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Pass --synthetic to steps 5-7 (random frames; quick check without video)",
    )
    parser.add_argument(
        "--input-dir", type=str, default=None,
        help="Folder of .mp4 clips for steps 5-7 (default: benchmark_videos/rq2_perception_clips/)",
    )
    parser.add_argument(
        "--rq3-video", type=str, default=None, dest="rq3_video",
        help="Recorded reference clip for step 9 (RQ3); passed to step9 as --video",
    )
    args = parser.parse_args()

    HERE = Path(__file__).resolve().parent
    DEFAULT_CLIPS = HERE / "benchmark_videos" / "rq2_perception_clips"
    DEFAULT_RQ3 = HERE / "benchmark_videos" / "rq3_system_clip" / "rq3_clip.mp4"
    clips_available = any(DEFAULT_CLIPS.glob("*.mp4"))

    selected = []
    for num, script, desc in STEPS:
        if args.only:
            if num in args.only:
                selected.append((num, script, desc))
        elif num >= args.start:
            selected.append((num, script, desc))

    if not selected:
        print("No steps selected.")
        return

    print("=" * 72)
    print("  RQ2/RQ3 EDGE DEPLOYMENT PIPELINE")
    print("=" * 72)
    for num, script, desc in selected:
        print(f"  Step {num}: {desc}  ({script})")
    print("=" * 72)

    # Up-front input check, so missing video is caught before any step runs.
    nums = {n for n, _, _ in selected}
    if (nums & {5, 6, 7}) or (9 in nums):
        n_clips = len(list(DEFAULT_CLIPS.glob("*.mp4")))
        print("  Inputs needed for the camera / perception steps:")
        if nums & {5, 6, 7}:
            clip_status = (f"{n_clips} clip(s) found" if n_clips
                           else "none — run benchmark_videos/rq2_perception_clips/select_clips.py (or use --synthetic)")
            print(f"    steps 5-7 : {clip_status}")
        if 9 in nums:
            rq3_status = ("clip found" if (DEFAULT_RQ3.exists() or args.rq3_video)
                          else "none — run benchmark_videos/rq3_system_clip/record_clip.py --duration 90")
            print(f"    step 9    : {rq3_status}")
        print("=" * 72)

    if args.dry_run:
        print("\n  (dry run — nothing executed)")
        return

    for num, script, desc in selected:
        print(f"\n{'=' * 72}")
        print(f"  STEP {num}: {desc}")
        print(f"{'=' * 72}\n")

        cmd = [sys.executable, script]
        if num in (5, 6, 7):
            if args.input_dir:
                cmd.extend(["--input-dir", args.input_dir])
            elif args.synthetic:
                cmd.append("--synthetic")
            elif clips_available:
                cmd.extend(["--input-dir", str(DEFAULT_CLIPS)])   # auto-detected clips
            else:
                print("  SKIP: this step needs video clips. Fill\n"
                      "        benchmark_videos/rq2_perception_clips/ with its select_clips.py\n"
                      "        (or pass --synthetic), then re-run.")
                continue
        elif num == 9:
            video = args.rq3_video or (str(DEFAULT_RQ3) if DEFAULT_RQ3.exists() else None)
            if video:
                cmd.extend(["--video", video])
            else:
                print("  SKIP: step 9 (RQ3) needs a recorded clip. Record one with\n"
                      "        benchmark_videos/rq3_system_clip/record_clip.py --duration 90,\n"
                      "        then re-run (or run step9 directly with --camera).")
                continue

        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"\n  STEP {num} FAILED (exit code {result.returncode})")
            print(f"  Stopping pipeline.")
            sys.exit(1)

        print(f"\n  Step {num} completed in {elapsed:.1f}s")

    print(f"\n{'=' * 72}")
    print("  ALL STEPS COMPLETED")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
