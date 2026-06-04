#!/usr/bin/env python3
"""
Verify that benchmark_videos/rq3_system_clip/rq3_clip.mp4 is suitable for RQ3 benchmarking.

Runs YOLOv8n detection on EVERY frame and reports:
  - Video properties (resolution, FPS, duration, total frames)
  - Detection rate (fraction of frames with at least one person)
  - Frame-by-frame detection count
  - Segments with no detections (potential gaps)

Usage (on the Jetson):
    python benchmark_videos/rq3_system_clip/verify_clip.py
    python benchmark_videos/rq3_system_clip/verify_clip.py --video my_clip.mp4
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Verify RQ3 clip")
    default_clip = str(Path(__file__).resolve().parent / "rq3_clip.mp4")
    parser.add_argument("--video", default=default_clip,
                        help="Path to the video clip (default: rq3_clip.mp4 in this folder)")
    args = parser.parse_args()

    # --- Open video ---
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Could not open {args.video}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"\n  Video: {args.video}")
    print(f"  Resolution: {w}x{h}")
    print(f"  FPS: {fps:.1f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.1f}s")

    # --- Check basic requirements ---
    issues = []
    if w != 640 or h != 480:
        issues.append(f"Resolution is {w}x{h}, expected 640x480")
    if fps < 20 or fps > 35:
        issues.append(f"FPS is {fps:.1f}, expected 25-30")
    if duration < 30:
        issues.append(f"Duration is only {duration:.1f}s, recommended >= 60s")

    if issues:
        print("\n  WARNINGS:")
        for issue in issues:
            print(f"    - {issue}")

    # --- Run YOLO detection on every frame ---
    print(f"\n  Running YOLOv8n detection on all {total_frames} frames...")
    model = YOLO("yolov8n.pt")

    frames_read = 0
    frames_with_person = 0
    no_detection_segments = []  # list of (start_sec, end_sec)
    current_gap_start = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, classes=[0], verbose=False)
        n_persons = len(results[0].boxes)
        frames_read += 1

        t_sec = frames_read / fps

        if n_persons > 0:
            frames_with_person += 1
            if current_gap_start is not None:
                gap_end = (frames_read - 1) / fps
                gap_duration = gap_end - current_gap_start
                if gap_duration >= 1.0:  # only report gaps >= 1 second
                    no_detection_segments.append(
                        (current_gap_start, gap_end, gap_duration)
                    )
                current_gap_start = None
        else:
            if current_gap_start is None:
                current_gap_start = t_sec

        # Progress every 5 seconds of video
        if frames_read % (int(fps) * 5) == 0:
            pct = 100 * frames_read / total_frames
            det_rate = frames_with_person / frames_read
            print(f"    {t_sec:.0f}s / {duration:.0f}s "
                  f"({pct:.0f}%, det rate so far: {det_rate:.2f})")

    # Close any trailing gap
    if current_gap_start is not None:
        gap_end = frames_read / fps
        gap_duration = gap_end - current_gap_start
        if gap_duration >= 1.0:
            no_detection_segments.append(
                (current_gap_start, gap_end, gap_duration)
            )

    cap.release()

    # --- Report ---
    det_rate = frames_with_person / frames_read if frames_read > 0 else 0
    print(f"\n  === RESULTS ===")
    print(f"  Frames analysed: {frames_read}")
    print(f"  Frames with person: {frames_with_person}")
    print(f"  Detection rate: {det_rate:.2%}")

    if no_detection_segments:
        print(f"\n  Gaps without detection (>= 1s):")
        for start, end, dur in no_detection_segments:
            print(f"    {start:.1f}s - {end:.1f}s  ({dur:.1f}s)")
    else:
        print(f"\n  No gaps longer than 1 second without detection.")

    # --- Verdict ---
    print()
    if det_rate >= 0.50:
        print(f"  VERDICT: Clip is USABLE for RQ3 benchmarking.")
        print(f"  Detection rate {det_rate:.0%} means the full pipeline")
        print(f"  (YOLO + features + GRU) will execute on most frames.")
    else:
        print(f"  VERDICT: Detection rate is low ({det_rate:.0%}).")
        print(f"  Consider re-recording with more time in front of camera.")

    if det_rate >= 0.70:
        print(f"\n  Quality: GOOD (>70% detection)")
    elif det_rate >= 0.50:
        print(f"\n  Quality: ACCEPTABLE (50-70% detection)")
    else:
        print(f"\n  Quality: POOR (<50% detection)")


if __name__ == "__main__":
    main()
