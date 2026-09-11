"""
Author: Hanna Norberg
e-mail: hanna.gjelstrup.norberg@gmail.com
"""

import cv2
import time

def measure_camera_fps(camera_index=0, measure_duration=5):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Cannot open camera {camera_index}")
        return

    print(f"Measuring FPS for {measure_duration} seconds...")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        elapsed_time = time.time() - start_time

        if elapsed_time >= measure_duration:
            break

    cap.release()

    fps = frame_count / elapsed_time
    print(f"\n✅ Estimated FPS: {fps:.2f}")

if __name__ == "__main__":
    measure_camera_fps()
