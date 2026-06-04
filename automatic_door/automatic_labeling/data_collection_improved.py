#!/usr/bin/env python3
"""
data_collection_improved.py

A cleaned-up, refactored version of data_collection.py (original by Hanna
Norberg). It does the same job and writes the SAME OUTPUT, so it is a drop-in
replacement for the data-collection step and stays compatible with the cleaning
pipeline that reads the clips.

Output (identical to the original):

    live_output/                      (live_output_test/ when TEST_REC = True)
      enter/   <timestamp>_id<ID>_enter.mp4  + matching .json
      pass/    <timestamp>_id<ID>_pass.mp4   + matching .json
      temp/    clips still being recorded

JSON saved next to each clip:

    {
      "id": <int>,
      "label": "enter" | "pass",
      "frames": [
        {"frame_index": <int>, "center": [x, y] | null, "bbox": [x1,y1,x2,y2] | null},
        ...
      ]
    }

What changed compared to the original (full notes in
data_collection_EXPLAINED.md):

  * FPS is handled correctly. The original used one variable for both the saved
    video's frame rate and the live on-screen counter, and overwrote it with the
    measured display rate mid-run. That meant clips were saved with an
    inconsistent (often wrong) playback speed and the "too short" / vanish
    timing drifted during a session. Here the timeline FPS is fixed for the
    whole run and the on-screen counter is a separate value.
  * Clear error if the camera cannot be opened.
  * The camera and video writers are always released, even on Ctrl-C
    (try/finally), so you don't get corrupt temp clips.
  * Dead code/imports removed, settings collected in one block at the top,
    short docstrings, and the calibration stored as plain data.

The detection, tracking, labeling rule and file format are unchanged.
"""

import cv2
import json
import os
import time
import datetime

import numpy as np
from ultralytics import YOLO


# ============================== SETTINGS ==============================
# Edit these to change a run. There are no command-line flags, same as the
# original — just change a value here and run `python data_collection_improved.py`.

YOLO_MODEL       = "yolo11s.pt"   # downloaded automatically on first use
CAMERA_INDEX     = 0              # webcam index (/dev/video0 on the Jetson)
COMPUTER         = "Jetson"       # which lens calibration to use: "Jetson" or "ASUS"

ELLIPSE_AXES     = (480, 130)     # door-zone half-axes in pixels (horizontal, vertical)
VANISH_THRESHOLD = 10             # frames a person may be missing before the clip is closed
EXTRA_SECONDS    = 0              # extra footage kept after the vanish threshold is hit

OUTPUT_FPS       = None           # timeline/playback FPS for saved clips.
                                  # None = use the camera's reported rate (fallback 25).
DISTORT          = False          # apply lens undistortion + crop to the valid region
FULLSCREEN       = False          # show the preview window fullscreen
DEBUG            = False          # print extra info and draw the vanish points
TEST_REC         = False          # True -> write to live_output_test/ for a throwaway run
# =====================================================================

# Lens calibration per machine: camera matrix (intrinsics) + distortion coeffs.
CALIBRATION = {
    "ASUS": {
        "mtx":  [[882.97786113, 0, 969.24568304],
                 [0, 863.01861637, 504.95560309],
                 [0, 0, 1]],
        "dist": [-0.24485893, 0.04001458, 0.0040836, 0.00174867, -0.00255292],
    },
    "Jetson": {
        "mtx":  [[307.8047385, 0, 355.19676862],
                 [0, 302.44366762, 233.22849986],
                 [0, 0, 1]],
        "dist": [-0.290496, 0.07539763, -0.00075077, -0.00159761, -0.00811828],
    },
}

PERSON_CLASS = 0   # "person" class id in the COCO-trained YOLO model


class LiveTracker:
    """Tracks people from a live camera and saves one labeled clip per person."""

    def __init__(self, yolo_model=YOLO_MODEL, ellipse_axes=ELLIPSE_AXES,
                 vanish_threshold=VANISH_THRESHOLD, extra_seconds=EXTRA_SECONDS,
                 camera_index=CAMERA_INDEX, computer=COMPUTER, output_fps=OUTPUT_FPS,
                 distort=DISTORT, fullscreen=FULLSCREEN, debug=DEBUG, test_rec=TEST_REC):

        # --- Output folders ---
        self.output_dir = "live_output_test" if test_rec else "live_output"
        self.temp_dir  = os.path.join(self.output_dir, "temp")
        self.enter_dir = os.path.join(self.output_dir, "enter")
        self.pass_dir  = os.path.join(self.output_dir, "pass")
        for d in (self.temp_dir, self.enter_dir, self.pass_dir):
            os.makedirs(d, exist_ok=True)

        # --- Parameters ---
        self.ellipse_axes = ellipse_axes
        self.vanish_threshold = vanish_threshold
        self.extra_seconds = extra_seconds
        self.distort = distort
        self.fullscreen = fullscreen
        self.debug = debug

        # --- Lens calibration ---
        if computer not in CALIBRATION:
            raise ValueError(f"Unknown computer '{computer}'. Options: {list(CALIBRATION)}")
        self.mtx  = np.array(CALIBRATION[computer]["mtx"])
        self.dist = np.array(CALIBRATION[computer]["dist"])

        # --- Detector + tracker ---
        self.yolo_model = YOLO(yolo_model)

        # --- Camera ---
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {camera_index}. Check that the camera "
                f"is connected and not in use by another program."
            )
        self.frame_width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # FPS used for the saved video AND for the frames<->seconds timing
        # (vanish buffer, "too short" check). Fixed for the whole run so every
        # clip shares one consistent timeline. Kept separate from the on-screen
        # counter below, which is purely cosmetic.
        camera_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.output_fps = output_fps or camera_fps or 25

        # --- Undistortion maps + cleaned frame size ---
        if self.distort:
            self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist,
                (self.frame_width, self.frame_height), 1,
                (self.frame_width, self.frame_height),
            )
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.mtx, self.dist, None, self.newcameramtx,
                (self.frame_width, self.frame_height), cv2.CV_32FC1,
            )
            self.clean_width, self.clean_height = self.roi[2], self.roi[3]
        else:
            self.clean_width, self.clean_height = self.frame_width, self.frame_height

        # The door is anchored at the bottom-center of the (cleaned) frame.
        self.door_center = (self.clean_width // 2, self.clean_height)

        # --- Tracking memory ---
        self.tracked_ids = set()      # ids we are currently recording
        self.recordings = {}          # id -> {writer, frames, video_path, video_name, vanish_frames}
        self.vanish_points = []       # (id, point) pairs, drawn only in debug mode

        # --- On-screen FPS counter (display only, never used for timing) ---
        self.display_fps = 0
        self._fps_frame_count = 0
        self._fps_prev_time = time.time()

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #
    def is_inside_ellipse(self, x, y):
        """True if (x, y) falls inside the door ellipse around door_center.

        Points below the door line (dy > 0) count as outside — the door zone is
        the upper half of the ellipse only.
        """
        dx = x - self.door_center[0]
        dy = y - self.door_center[1]
        if dy > 0:
            return False
        return (dx * dx) / (self.ellipse_axes[0] ** 2) + \
               (dy * dy) / (self.ellipse_axes[1] ** 2) <= 1

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self):
        print("Starting live tracking. Press 'q' to quit.")
        cv2.namedWindow("Live Tracking", cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty("Live Tracking", cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = self._undistort(frame)
                clean_frame = frame.copy()   # saved to disk WITHOUT any overlays

                results = self.yolo_model.track(
                    frame, persist=True, classes=[PERSON_CLASS],
                    tracker="botsort.yaml", verbose=False,
                )
                boxes = results[0].boxes
                current_ids = set()

                if boxes.id is not None:
                    for box, track_id in zip(boxes.xyxy.cpu().numpy(),
                                             boxes.id.cpu().numpy()):
                        track_id = int(track_id)
                        x1, y1, x2, y2 = box
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        current_ids.add(track_id)

                        if track_id not in self.tracked_ids:
                            self.start_recording(track_id)

                        self.recordings[track_id]["frames"].append({
                            "frame_index": len(self.recordings[track_id]["frames"]),
                            "center": [int(center[0]), int(center[1])],
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        })
                        self.recordings[track_id]["writer"].write(clean_frame)

                        self._draw_box(frame, track_id, x1, y1, x2, y2)

                self.handle_vanished(current_ids, clean_frame)
                self._draw_overlays(frame)
                self._update_display_fps()
                cv2.putText(frame, f"FPS: {self.display_fps}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                display_frame = frame if self.fullscreen else cv2.resize(frame, (640, 360))
                cv2.imshow("Live Tracking", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            # Always release the camera and any open writers, even on Ctrl-C.
            self.cleanup()

    # ------------------------------------------------------------------ #
    # Recording lifecycle
    # ------------------------------------------------------------------ #
    def start_recording(self, track_id):
        """Open a temp video + an empty frame list for a newly seen person."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = f"{timestamp}_id{track_id}.mp4"
        video_path = os.path.join(self.temp_dir, video_name)

        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(self.output_fps),
            (self.clean_width, self.clean_height),
        )

        self.recordings[track_id] = {
            "writer": writer,
            "frames": [],
            "video_path": video_path,
            "video_name": video_name,
            "vanish_frames": 0,
        }
        self.tracked_ids.add(track_id)
        print(f"[START] Tracking ID {track_id}, file: {video_name}")

    def handle_vanished(self, current_ids, clean_frame):
        """Advance the 'missing' counter for ids not seen this frame and close
        any whose person has been gone long enough."""
        finished_ids = set()
        for track_id in list(self.tracked_ids):
            rec = self.recordings[track_id]

            if track_id not in current_ids:
                # Person not detected this frame: log an empty frame, keep
                # recording the (person-less) view, and count toward the timeout.
                rec["frames"].append({
                    "frame_index": len(rec["frames"]),
                    "center": None,
                    "bbox": None,
                })
                rec["vanish_frames"] += 1
                rec["writer"].write(clean_frame)

                if rec["vanish_frames"] == self.vanish_threshold:
                    print(f"[VANISH] ID {track_id} vanished, "
                          f"adding {self.extra_seconds}s buffer...")

                if rec["vanish_frames"] >= self.vanish_threshold + self.output_fps * self.extra_seconds:
                    finished_ids.add(track_id)
            else:
                rec["vanish_frames"] = 0

        for finished_id in finished_ids:
            self.finish_recording(finished_id)

    def finish_recording(self, track_id):
        """Close a clip: drop it if too short, otherwise label it enter/pass,
        move it into the right folder and write the JSON sidecar."""
        rec = self.recordings[track_id]
        total_frames = len(rec["frames"])

        if self.debug:
            print(f"[INFO] Track ID {track_id} has {total_frames} frames recorded.")

        rec["writer"].release()

        # Too short to be useful (less than ~1 second of footage): discard.
        if total_frames < self.output_fps:
            print(f"[SKIPPED] ID {track_id}, not enough frames.")
            if os.path.exists(rec["video_path"]):
                os.remove(rec["video_path"])
            self.tracked_ids.remove(track_id)
            del self.recordings[track_id]
            return

        # Label from the last frame where the person was actually detected.
        last_center = None
        for f in reversed(rec["frames"]):
            if f["center"] is not None:
                last_center = f["center"]
                break

        if last_center is not None:
            label = "enter" if self.is_inside_ellipse(*last_center) else "pass"
        else:
            label = "pass"   # fallback: never had a valid detection
        out_dir = self.enter_dir if label == "enter" else self.pass_dir

        if self.debug and last_center is not None:
            self.vanish_points.append((track_id, tuple(last_center)))

        # Move temp video -> enter/ or pass/, with the label in the filename.
        base_name = rec["video_name"].replace(".mp4", f"_{label}.mp4")
        final_video_path = os.path.join(out_dir, base_name)
        os.rename(rec["video_path"], final_video_path)

        json_path = final_video_path.replace(".mp4", ".json")
        json_data = {
            "id": track_id,
            "label": label,
            "frames": rec["frames"],
        }
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)

        print(f"[SAVED] {final_video_path}")
        print(f"[SAVED] {json_path}")

        self.tracked_ids.remove(track_id)
        del self.recordings[track_id]

    def cleanup(self):
        """Release every open writer and the camera."""
        for rec in self.recordings.values():
            rec["writer"].release()
        self.cap.release()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------ #
    # Small helpers (frame processing + drawing)
    # ------------------------------------------------------------------ #
    def _undistort(self, frame):
        """Apply the lens correction. With DISTORT, remap + crop to the valid
        region; otherwise just undistort in place (same size)."""
        if self.distort:
            frame = cv2.remap(frame, self.map1, self.map2,
                              interpolation=cv2.INTER_LINEAR)
            x, y, w, h = self.roi
            return frame[y:y + h, x:x + w]
        return cv2.undistort(frame, self.mtx, self.dist, None, self.mtx)

    def _draw_box(self, frame, track_id, x1, y1, x2, y2):
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def _draw_overlays(self, frame):
        # Door zone (upper half of the ellipse) in blue.
        cv2.ellipse(frame, self.door_center, self.ellipse_axes, 0, 180, 360,
                    (255, 0, 0), 2)
        if self.debug:
            for vid, (vx, vy) in self.vanish_points:
                cv2.circle(frame, (vx, vy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"ID {vid}", (vx + 5, vy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    def _update_display_fps(self):
        """Recompute the cosmetic on-screen FPS once per second."""
        self._fps_frame_count += 1
        now = time.time()
        if now - self._fps_prev_time >= 1.0:
            self.display_fps = self._fps_frame_count
            self._fps_frame_count = 0
            self._fps_prev_time = now


def main():
    tracker = LiveTracker(
        yolo_model=YOLO_MODEL,
        ellipse_axes=ELLIPSE_AXES,
        vanish_threshold=VANISH_THRESHOLD,
        extra_seconds=EXTRA_SECONDS,
        camera_index=CAMERA_INDEX,
        computer=COMPUTER,
        output_fps=OUTPUT_FPS,
        distort=DISTORT,
        fullscreen=FULLSCREEN,
        debug=DEBUG,
        test_rec=TEST_REC,
    )
    tracker.run()


if __name__ == "__main__":
    main()
