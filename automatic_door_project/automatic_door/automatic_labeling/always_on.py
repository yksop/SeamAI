"""
Author: Hanna Norberg
e-mail: hanna.gjelstrup.norberg@gmail.com
"""

import cv2
import time
import numpy as np
import os
from ultralytics import YOLO


class PeopleTracker:
    def __init__(self,
                 yolo_model="yolo11s.pt",
                 ellipse_axes=(280, 140),
                 camera_index=0,
                 computer="Jetson",
                 distort=False,
                 fullscreen=True,
                 debug=False,
                 record=True,
                 output_dir="recordings",
                 vanish_threshold=30,
                 extra_seconds=2):

        # --- Params ---
        self.ellipse_axes = ellipse_axes
        self.camera_index = camera_index
        self.distort = distort
        self.fullscreen = fullscreen
        self.debug = debug
        self.record = record
        self.output_dir = output_dir
        self.vanish_threshold = vanish_threshold
        self.extra_seconds = extra_seconds

        os.makedirs(self.output_dir, exist_ok=True)

        # --- Camera calibration ---
        if computer == "ASUS":
            self.mtx = np.array([[882.97786113, 0, 969.24568304],
                                 [0, 863.01861637, 504.95560309],
                                 [0, 0, 1]])
            self.dist = np.array([-0.24485893, 0.04001458, 0.0040836,
                                  0.00174867, -0.00255292])
        elif computer == "Jetson":
            self.mtx = np.array([[307.8047385, 0, 355.19676862],
                                 [0, 302.44366762, 233.22849986],
                                 [0, 0, 1]])
            self.dist = np.array([-0.290496, 0.07539763,
                                  -0.00075077, -0.00159761, -0.00811828])

        # --- YOLO ---
        self.yolo_model = YOLO(yolo_model)

        # --- Capture ---
        self.cap = cv2.VideoCapture(self.camera_index)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS)) or 25

        # --- Frame handling ---
        if self.distort:
            self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist,
                (self.frame_width, self.frame_height), 1,
                (self.frame_width, self.frame_height)
            )
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.mtx, self.dist, None, self.newcameramtx,
                (self.frame_width, self.frame_height), cv2.CV_32FC1
            )
            self.clean_width, self.clean_height = self.roi[2], self.roi[3]
            self.door_center = (self.clean_width // 2, self.clean_height)
        else:
            self.clean_width, self.clean_height = self.frame_width, self.frame_height
            self.door_center = (self.clean_width // 2, self.clean_height)

        # --- FPS tracking ---
        self.fps = self.frame_rate
        self._prev_fps_time = time.time()
        self.frame_count_for_fps = 0

        # --- Recording dict ---
        self.recordings = {}

    def run(self):
        print("Starting live tracking. Press 'q' to quit.")

        # --- Window ---
        cv2.namedWindow("Live Tracking", cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty("Live Tracking",
                                  cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # --- Undistort ---
            if self.distort:
                frame = cv2.remap(frame, self.map1, self.map2,
                                  interpolation=cv2.INTER_LINEAR)
                x, y, w, h = self.roi
                frame = frame[y:y + h, x:x + w]
            else:
                frame = cv2.undistort(frame, self.mtx, self.dist, None, self.mtx)

            # --- YOLO ---
            results = self.yolo_model.track(
                frame,
                persist=True,
                classes=[0],  # persons
                tracker="botsort.yaml",
                verbose=False
            )

            boxes = results[0].boxes
            current_ids = set()

            if boxes.id is not None:
                for box, track_id in zip(boxes.xyxy.cpu().numpy(),
                                         boxes.id.cpu().numpy()):
                    track_id = int(track_id)
                    current_ids.add(track_id)
                    x1, y1, x2, y2 = map(int, box)

                    # Draw bbox + ID
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 1)

                    # Start recording
                    if self.record and track_id not in self.recordings:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"{timestamp}_id{track_id}.mp4"
                        filepath = os.path.join(self.output_dir, filename)
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(filepath, fourcc,
                                                 self.frame_rate,
                                                 (self.clean_width, self.clean_height))
                        self.recordings[track_id] = {
                            "writer": writer,
                            "frames": [],
                            "vanish_frames": 0,
                            "filepath": filepath,
                        }
                        print(f"[START]Tracking ID {track_id}, file: {filename}")

                    # Save frame
                    if self.record:
                        if track_id in self.recordings:
                            self.recordings[track_id]["writer"].write(frame)

            # --- Vanish logic ---
            if self.record:
                vanished_ids = set(self.recordings.keys()) - current_ids
                finished_ids = set()
                for vid in vanished_ids:
                    self.recordings[vid]["vanish_frames"] += 1

                    if self.recordings[vid]["vanish_frames"] == self.vanish_threshold:
                        print(f"[VANISH] ID {vid} vanished, adding {self.extra_seconds}s buffer...")

                    if self.recordings[vid]["vanish_frames"] >= self.vanish_threshold + self.frame_rate * self.extra_seconds:
                        finished_ids.add(vid)

                for fid in finished_ids:
                    self.finish_recording(fid)

            # --- Ellipse ---
            cv2.ellipse(frame, self.door_center, self.ellipse_axes,
                        0, 180, 360, (255, 0, 0), 2)

            # --- FPS ---
            self.frame_count_for_fps += 1
            now = time.time()
            if now - self._prev_fps_time >= 1.0:
                self.fps = self.frame_count_for_fps
                self.frame_count_for_fps = 0
                self._prev_fps_time = now

            cv2.putText(frame, f"FPS: {self.fps}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
            
            # --- Overlay "Not Recording" if recording disabled ---
            if not self.record:
                text = "NOT RECORDING"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8   # smaller, less bold
                thickness = 2      # thinner line
                color = (0, 0, 255)

                # Get text size to center it
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                x = (frame.shape[1] - text_w) // 2
                y = 40  # top padding

                cv2.putText(frame, text, (x, y),
                            font, font_scale, color, thickness, cv2.LINE_AA)

            # --- Show ---
            #display_frame = frame if self.fullscreen else cv2.resize(frame, (640, 360))
            display_frame = cv2.resize(frame, (1920, 1200))
            cv2.imshow("Live Tracking", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cleanup()

    def finish_recording(self, track_id):
        """Stop writing for one track"""
        if track_id in self.recordings:
            self.recordings[track_id]["writer"].release()
            print(f"[FINISH] Saved recording for ID {track_id} -> {self.recordings[track_id]['filepath']}")
            del self.recordings[track_id]

    def cleanup(self):
        if self.record:
            for track_id in list(self.recordings.keys()):
                self.finish_recording(track_id)
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    tracker = PeopleTracker(
        yolo_model="yolo11s.pt",
        ellipse_axes=(250, 120),
        camera_index=0,
        computer="Jetson",
        distort=False,
        fullscreen=True,
        debug=False,
        record=False  
    )
    tracker.run()


if __name__ == "__main__":
    main()
