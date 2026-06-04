"""
Quick camera test — run this FIRST on the Jetson to verify what
your camera actually delivers (resolution, FPS).

Usage:
    python test_camera.py              # default camera 0
    python test_camera.py --source 1   # camera index 1
    python test_camera.py --source /dev/video0
"""

import cv2
import time
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=0, help="Camera index or device path")
    args = p.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {source}")
        return

    # Print what the camera reports
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera reports: {w}x{h} @ {reported_fps} FPS")

    # Try requesting specific settings
    print("\n--- Testing different resolutions ---")
    for res_w, res_h in [(640, 480), (1280, 720), (1920, 1080)]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  Requested {res_w}x{res_h} → Got {actual_w}x{actual_h} @ {actual_fps} FPS")

    # Reset to 640x480 for FPS test
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Measure ACTUAL capture FPS (read 100 frames, time it)
    print("\n--- Measuring actual capture FPS (100 frames) ---")
    # Warm up
    for _ in range(10):
        cap.read()

    t0 = time.time()
    count = 0
    for _ in range(100):
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame")
            break
        count += 1
    elapsed = time.time() - t0
    actual_fps = count / elapsed
    print(f"  Read {count} frames in {elapsed:.2f}s = {actual_fps:.1f} FPS (actual)")

    # Show live preview with FPS counter
    print("\n--- Live preview (press 'q' to quit) ---")
    window_name = "Camera Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    prev_time = time.time()
    frame_count = 0
    display_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        now = time.time()
        if now - prev_time >= 1.0:
            display_fps = frame_count / (now - prev_time)
            frame_count = 0
            prev_time = now

        cv2.putText(frame, f"Actual FPS: {display_fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        cv2.putText(frame, f"Resolution: {frame.shape[1]}x{frame.shape[0]}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
