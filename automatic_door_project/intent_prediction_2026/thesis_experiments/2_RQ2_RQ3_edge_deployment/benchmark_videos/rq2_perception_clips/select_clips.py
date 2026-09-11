#!/usr/bin/env python3
"""
Select a diverse subset of video clips for the perception benchmarks.

Picks N clips randomly (stratified by enter/pass) from MasterData and
copies them into this folder, ready to pass to steps 5-7 as --input-dir.

Usage (from the 2_RQ2_RQ3_edge_deployment/ folder):
    python benchmark_videos/rq2_perception_clips/select_clips.py         # 30 clips
    python benchmark_videos/rq2_perception_clips/select_clips.py --n 50  # more clips
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

# This script sits two levels below the RQ2/RQ3 folder; add that folder to the
# import path so `config` resolves no matter where the script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DATA_ROOT


def main():
    parser = argparse.ArgumentParser(
        description="Select diverse test clips for Jetson benchmarking"
    )
    parser.add_argument(
        "--n", type=int, default=30,
        help="Total number of clips to select (default: 30)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: this folder)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    data_root = Path(DATA_ROOT)
    output_dir = Path(args.output) if args.output else Path(__file__).resolve().parent
    output_dir.mkdir(exist_ok=True)

    # Find all mp4 files
    enter_clips = sorted((data_root / "enter").glob("*.mp4"))
    pass_clips = sorted((data_root / "pass").glob("*.mp4"))

    print(f"  Available: {len(enter_clips)} enter, {len(pass_clips)} pass")

    # Stratified sampling: roughly proportional to class distribution
    n_enter = round(args.n * len(enter_clips) / (len(enter_clips) + len(pass_clips)))
    n_pass = args.n - n_enter

    rng = random.Random(args.seed)
    selected_enter = rng.sample(enter_clips, min(n_enter, len(enter_clips)))
    selected_pass = rng.sample(pass_clips, min(n_pass, len(pass_clips)))

    selected = selected_enter + selected_pass
    rng.shuffle(selected)

    print(f"  Selected: {len(selected_enter)} enter + {len(selected_pass)} pass "
          f"= {len(selected)} clips")

    # Copy
    for clip in selected:
        dest = output_dir / clip.name
        if not dest.exists():
            shutil.copy2(clip, dest)

    print(f"  Copied to: {output_dir}")
    print(f"\n  Then run:")
    print(f"    python step5_benchmark_perception.py --input-dir benchmark_videos/rq2_perception_clips/")
    print(f"    python step6_benchmark_pipeline.py --input-dir benchmark_videos/rq2_perception_clips/")


if __name__ == "__main__":
    main()
