"""
Author: Hanna Norberg
e-mail: hanna.gjelstrup.norberg@gmail.com
"""

import cv2
import json
import os
import time
import datetime
import math
import numpy as np

from ultralytics import YOLO
from collections import defaultdict

class LiveTracker:
    def __init__(self,
                 yolo_model="yolo11s.pt",
                 ellipse_axes=(280, 140),
                 vanish_threshold=90,
                 extra_seconds=1,
                 camera_index=0,
                 computer="Jetson",
                 distort=False,
                 fullscreen=True,
                 debug=False,
                 test_rec=False):
        
        # --- Output dirs ---
        if test_rec == True:
            self.OUTPUT_DIR = "live_output_test"
        elif test_rec == False:
            self.OUTPUT_DIR = "live_output"
        self.TEMP_DIR = os.path.join(self.OUTPUT_DIR, "temp")
        self.ENTER_DIR = os.path.join(self.OUTPUT_DIR, "enter")
        self.PASS_DIR = os.path.join(self.OUTPUT_DIR, "pass")
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        os.makedirs(self.ENTER_DIR, exist_ok=True)
        os.makedirs(self.PASS_DIR, exist_ok=True)

        # --- Params ---
        self.ellipse_axes = ellipse_axes
        self.vanish_threshold = vanish_threshold
        self.extra_seconds = extra_seconds
        self.camera_index = camera_index
        self.distort = distort
        self.fullscreen = fullscreen
        self.debug = debug

        # --- Camera calibration ---
        if computer=="ASUS":
            self.mtx = np.array([[882.97786113, 0, 969.24568304],
                                 [0, 863.01861637, 504.95560309],
                                 [0, 0, 1]])
            self.dist = np.array([-0.24485893, 0.04001458, 0.0040836, 0.00174867, -0.00255292])
        elif computer=="Jetson":
            self.mtx = np.array([[307.8047385, 0, 355.19676862],
                                 [0, 302.44366762, 233.22849986],
                                 [0, 0, 1]])
            self.dist = np.array([-0.290496, 0.07539763, -0.00075077, -0.00159761, -0.00811828])
        
        # --- YOLO ---
        self.yolo_model = YOLO(yolo_model)

        # --- Capture ---
        self.cap = cv2.VideoCapture(self.camera_index)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS)) or 25

        # --- Frame --- 
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
            self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist,
                (self.frame_width, self.frame_height), 1,
                (self.frame_width, self.frame_height)
            )
            self.clean_width, self.clean_height = self.frame_width, self.frame_height
            self.door_center = (self.clean_width // 2, self.clean_height)

        # --- Tracking memory ---
        self.tracked_ids = set()
        self.recordings = {}
        self.vanish_counter = {}
        self.vanish_points = []
        
        # --- FPS tracking ---
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 25  # fallback FPS
        self._prev_fps_time = time.time()
        self.frame_count_for_fps = 0


    def is_inside_ellipse(self, x, y):
        dx = x - self.door_center[0]
        dy = y - self.door_center[1]
        if dy > 0:
            return False
        return ((dx*dx)/(self.ellipse_axes[0]**2))+((dy*dy)/(self.ellipse_axes[1]**2)) <= 1


    def run(self):
        print("Starting live tracking. Press 'q' to quit.")
        prev_time = time.time()
        frame_count_for_fps = 0
        fps = 0

        # --- Window setting ---
        cv2.namedWindow("Live Tracking", cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(
                "Live Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if self.distort:
                frame = cv2.remap(
                    frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR
                )
                x, y, w, h = self.roi
                frame = frame[y:y+h, x:x+w]
            else:
                frame = cv2.undistort(frame, self.mtx, self.dist, None, self.mtx)

            clean_frame = frame.copy()

            results = self.yolo_model.track(
                frame,
                persist=True,
                classes=[0],
                tracker='botsort.yaml',
                verbose=False,
            ) # botsort

            boxes = results[0].boxes
            current_ids = set()

            if boxes.id is not None:
                for box, track_id in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy()):
                    track_id = int(track_id)
                    x1, y1, x2, y2 = box
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    current_ids.add(track_id)

                    if track_id not in self.tracked_ids:
                        self.start_recording(track_id)
                    
                    self.recordings[track_id]["frames"].append({
                        "frame_index": len(self.recordings[track_id]["frames"]),
                        "center": [int(center[0]), int(center[1])],
                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                    })
                    
                    self.recordings[track_id]["writer"].write(clean_frame)

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {track_id}",
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            self.handle_vanished(current_ids, clean_frame, frame)

            # --- Draw ellipse for door ---
            cv2.ellipse(frame, self.door_center, self.ellipse_axes, 0, 180, 360, (255, 0, 0), 2)
            
            if self.debug:
                for vid, (vx, vy) in self.vanish_points:
                    cv2.circle(frame, (vx, vy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"ID {vid}", (vx + 5, vy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # --- FPS ---
            self.frame_count_for_fps += 1
            now = time.time()
            if now - self._prev_fps_time >= 1.0:
                self.fps = self.frame_count_for_fps
                self.frame_count_for_fps = 0
                self._prev_fps_time = now

            cv2.putText(frame, f"FPS: {self.fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if self.fullscreen:
                display_frame = frame
            else:
                display_frame = cv2.resize(frame, (640, 360))
            
            
            cv2.imshow("Live Tracking", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()


    def start_recording(self, track_id):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = f"{timestamp}_id{track_id}.mp4"
        video_path = os.path.join(self.TEMP_DIR, video_name)
        
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(self.fps),
            (self.clean_width, self.clean_height)
        )

        self.recordings[track_id] = {
            "writer": writer,
            "frames": [],
            "video_path": video_path,
            "video_name": video_name,
            "vanish_frames": 0
        }

        self.tracked_ids.add(track_id)
        print(f"[START] Tracking ID {track_id}, file: {video_name}")


    def handle_vanished(self, current_ids, clean_frame, frame):
        finished_ids = set()
        for track_id in list(self.tracked_ids):
            
            if track_id not in current_ids:
                self.recordings[track_id]["frames"].append({
                    "frame_index": len(self.recordings[track_id]["frames"]),
                    "center": None,
                    "bbox": None
                })
                self.recordings[track_id]["vanish_frames"] += 1
                
                self.recordings[track_id]["writer"].write(clean_frame)
                
                if self.recordings[track_id]["vanish_frames"] == self.vanish_threshold:
                    print(f"[VANISH] ID {track_id} vanished, adding {self.extra_seconds}s buffer...")
                
                if self.recordings[track_id]["vanish_frames"] >= self.vanish_threshold+self.fps*self.extra_seconds:
                    finished_ids.add(track_id)
            else:
                self.recordings[track_id]["vanish_frames"] = 0
        
        for finished_id in finished_ids:
            self.finish_recording(finished_id, clean_frame)


    def finish_recording(self, track_id, clean_frame):
        rec = self.recordings[track_id]
        total_frames = len(rec["frames"])
        
        if self.debug:
            print(f"[INFO] Track ID {track_id} has {total_frames} frames recorded.")
        
        rec["writer"].release()
        
        if len(rec["frames"]) < self.fps:
            print(f"[SKIPPED] ID {track_id}, not enough frames.")
            if os.path.exists(rec["video_path"]):
                os.remove(rec["video_path"]) # Delete temp video
            self.tracked_ids.remove(track_id)
            del self.recordings[track_id]
            return
        
        last_center = None
        for f in reversed(rec["frames"]):
            if f["center"] is not None:
                last_center = f["center"]
                break

        # determine label based on last valid center
        if last_center is not None:
            label = "enter" if self.is_inside_ellipse(*last_center) else "pass"
        else:
            # fallback if all frames were empty
            label = "pass"
        out_dir = self.ENTER_DIR if label == "enter" else self.PASS_DIR
        
        if self.debug:
            self.vanish_points.append((track_id, tuple(last_center)))

        base_name = rec["video_name"].replace(".mp4", f"_{label}.mp4")
        final_video_path = os.path.join(out_dir, base_name)
        os.rename(rec["video_path"], final_video_path)

        json_path = final_video_path.replace(".mp4", ".json")
        json_data = {
            "id": track_id,
            "label": label,
            "frames": rec["frames"]
        }

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)
        
        if self.debug:
            total_frames_video = len(rec["frames"])
            print(f"[INFO] Final video frames: {total_frames_video}")
        
        print(f"[SAVED] {final_video_path}")
        print(f"[SAVED] {json_path}")

        self.tracked_ids.remove(track_id)
        del self.recordings[track_id]


    def cleanup(self):
        for rec in self.recordings.values():
            rec["writer"].release()
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    tracker = LiveTracker(
        yolo_model="yolo11s.pt",
        ellipse_axes=(480, 130),
        vanish_threshold=10,
        extra_seconds=0,
        camera_index=0,
        computer="Jetson",
        distort=False,
        fullscreen=False,
        debug=False,
        test_rec=False
    )
    tracker.run()


if __name__ == "__main__":
    main()
