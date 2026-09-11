#!/usr/bin/env python3
"""
Record a video clip for RQ3 benchmarking.

Records a single MP4 clip from the USB camera at 640x480.
The clip is used as input for step9_benchmark_live_camera.py
so that all 5 GRU configurations are tested on the exact same
video sequence.

Setup:
    1. Connect the USB camera to the Jetson.
    2. If using a screen, run with --preview first to check the angle.
    3. Position the camera so it looks down toward the ground,
       similar to how a door-mounted camera would be installed.
       The red "door" dot in preview mode shows the assumed door
       location (bottom center of the frame).
    4. Press 'r' in preview to start recording, or 'q' to quit.

Usage:
    python benchmark_videos/rq3_system_clip/record_clip.py --preview           # preview only (set up angle)
    python benchmark_videos/rq3_system_clip/record_clip.py                     # record 60s, camera 0
    python benchmark_videos/rq3_system_clip/record_clip.py --duration 90       # 90 seconds
    python benchmark_videos/rq3_system_clip/record_clip.py --camera 1          # use camera index 1

    # If running headless (no screen), skip preview:
    python benchmark_videos/rq3_system_clip/record_clip.py --duration 60

Recording instructions:
    Walk naturally toward and away from the camera at various
    distances and angles for the full duration. The content does
    not need to be labeled; only the timing of each pipeline stage
    is measured in step 9. The important thing is that YOLO detects
    a person in the frame so the full pipeline executes.

After recording:
    python step9_benchmark_live_camera.py --video benchmark_videos/rq3_system_clip/rq3_clip.mp4
"""

import argparse
import os
import time

import cv2


def main():
    parser = argparse.ArgumentParser(description="Record a clip for RQ3")
    parser.add_argument("--duration", type=int, default=60,
                        help="Recording duration in seconds (default: 60)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--output", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "rq3_clip.mp4"),
                        help="Output path (default: rq3_clip.mp4 in this folder)")
    parser.add_argument("--preview", action="store_true",
                        help="Show live preview only (no recording). "
                             "Use to set up camera angle. Press 'q' to quit.")
    args = parser.parse_args()

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {args.camera}")
        return

    # Set resolution to 640x480 (matches training data)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Read one frame to confirm settings
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read from camera")
        cap.release()
        return

    h, w = frame.shape[:2]
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Camera: index {args.camera}")
    print(f"  Resolution: {w}x{h}")
    print(f"  Reported FPS: {reported_fps:.1f}")

    # --- Preview mode ---
    if args.preview:
        print(f"\n  PREVIEW MODE — press 'q' to quit, 'r' to start recording.\n")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Draw door center reference
            door_x, door_y = w // 2, h - 10
            cv2.circle(frame, (door_x, door_y), 8, (0, 0, 255), -1)
            cv2.putText(frame, "door", (door_x + 12, door_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # Draw crosshair
            cv2.line(frame, (w // 2, 0), (w // 2, h), (100, 100, 100), 1)
            cv2.line(frame, (0, h // 2), (w, h // 2), (100, 100, 100), 1)
            cv2.putText(frame, "PREVIEW - 'q' quit, 'r' record",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Camera Preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("  Preview ended.")
                return
            if key == ord('r'):
                cv2.destroyAllWindows()
                print("  Switching to recording mode...")
                break
        # If user pressed 'r', fall through to recording below

    print(f"  Duration: {args.duration}s")
    print(f"  Output: {args.output}")

    # Use actual FPS for the writer (fallback to 30 if reported is 0)
    write_fps = reported_fps if reported_fps > 0 else 30.0

    # Set up video writer (create the output folder if needed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, write_fps, (w, h))

    print(f"\n  Recording starts in 3 seconds...")
    time.sleep(3)
    print(f"  RECORDING — walk in front of the camera!")
    print(f"  Press Ctrl+C to stop early.\n")

    frame_count = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("  Camera read failed, stopping.")
                break

            writer.write(frame)
            frame_count += 1

            elapsed = time.time() - t_start
            if elapsed >= args.duration:
                break

            # Print progress every 10 seconds
            if frame_count % (int(write_fps) * 10) == 0:
                actual_fps = frame_count / elapsed
                print(f"  {elapsed:.0f}s / {args.duration}s "
                      f"({frame_count} frames, {actual_fps:.1f} FPS)")

    except KeyboardInterrupt:
        print("\n  Stopped early by user.")

    elapsed = time.time() - t_start
    actual_fps = frame_count / elapsed if elapsed > 0 else 0

    writer.release()
    cap.release()

    print(f"\n  Done! Recorded {frame_count} frames in {elapsed:.1f}s")
    print(f"  Actual FPS: {actual_fps:.1f}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
