#!/usr/bin/env python3
"""
RISE Open-House Demo — Real-time pedestrian intent prediction.

Pipeline:
    Camera → YOLOv8n-pose (BoT-SORT) → Feature extraction → GRU → Visualisation

Features (Core-3 + head raw, 7 dims):
    dist_to_door, closure_rate, rel_angle_to_door,
    left_ear_x, left_ear_y, right_ear_x, right_ear_y

Usage:
    # With trained model:
    python demo_live.py --model weights/gru_core3head.pt --norm weights/norm_stats.npz

    # Demo mode (no trained model, shows features + skeleton only):
    python demo_live.py

    # Use a video file instead of camera:
    python demo_live.py --source path/to/video.mp4
"""

import argparse
import csv
import cv2
import math
import time
import datetime
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

from ultralytics import YOLO
from features import FeatureComputer, KP_LEFT_EAR, KP_RIGHT_EAR
from model import IntentGRU, load_model, load_normalisation_stats


# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────

DEFAULT_CFG = dict(
    # YOLO
    yolo_model="yolov8n-pose.pt",
    tracker="botsort.yaml",
    conf_thresh=0.4,

    # Camera / source
    source=0,
    frame_w=640,
    frame_h=480,

    # Door position (pixel coords — adjust for your setup)
    door_x=320,          # center-x of the frame by default
    door_y=480,          # bottom of the frame

    # GRU
    seq_len=6,           # round(0.5s * 13fps) = 6 frames
    training_fps=13.0,   # FPS the training recordings were captured at
    input_dim=7,

    # Decision logic
    open_threshold=0.75,         # P(enter) must exceed this
    confirm_frames=5,            # … for this many consecutive frames
    max_open_distance=0.35,      # … and person must be within this dist (normalised)

    # Display
    fullscreen=True,
    show_features=True,          # show feature values on screen
    show_skeleton=True,          # draw COCO keypoints

    # Logging
    log_latency=False,
    log_dir="logs",
)


# ──────────────────────────────────────────────────────────────────
# Colour helpers
# ──────────────────────────────────────────────────────────────────

def intent_colour(p: float, threshold: float = 0.75):
    """
    Simple 3-state colour based on P(enter):
      green  — above threshold (likely entering)
      yellow — uncertain zone (0.4 .. threshold)
      white  — low probability (passing by)
    """
    if p >= threshold:
        return (0, 230, 0)       # BGR green
    elif p >= 0.4:
        return (0, 230, 230)     # BGR yellow
    else:
        return (220, 220, 220)   # BGR white/grey


# COCO skeleton connections (for drawing)
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6),                                     # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),           # arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]


# ──────────────────────────────────────────────────────────────────
# Latency logger
# ──────────────────────────────────────────────────────────────────

class LatencyLogger:
    """Logs per-frame timing breakdown to CSV for RQ2/RQ3 analysis."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.log_dir / f"latency_{timestamp}.csv"
        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "frame_idx",
            "t_capture_ms",          # timestamp of frame capture
            "dt_yolo_ms",            # YOLO inference duration
            "dt_features_ms",        # feature computation duration
            "dt_gru_ms",             # GRU inference duration
            "dt_viz_ms",             # visualisation drawing duration
            "dt_total_ms",           # total frame pipeline
            "n_persons",             # number of tracked persons
            "fps_measured",          # actual measured FPS
        ])
        self.frame_idx = 0

    def log(self, t_capture, dt_yolo, dt_feat, dt_gru, dt_viz, dt_total,
            n_persons, fps):
        self.writer.writerow([
            self.frame_idx,
            f"{t_capture:.3f}",
            f"{dt_yolo:.2f}",
            f"{dt_feat:.2f}",
            f"{dt_gru:.2f}",
            f"{dt_viz:.2f}",
            f"{dt_total:.2f}",
            n_persons,
            f"{fps:.1f}",
        ])
        self.frame_idx += 1

    def close(self):
        self.file.close()
        print(f"[LOG] Latency data saved to {self.path}")


# ──────────────────────────────────────────────────────────────────
# Per-person feature logger (CSV)
# ──────────────────────────────────────────────────────────────────

class FeatureLogger:
    """Logs per-person per-frame features to CSV (no video needed)."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.log_dir / f"features_{timestamp}.csv"
        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "frame_idx", "timestamp_ms", "track_id",
            "center_x", "center_y",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "dist_to_door", "closure_rate", "rel_angle_to_door",
            "left_ear_x", "left_ear_y", "right_ear_x", "right_ear_y",
            "p_enter", "door_decision",
        ])
        self.frame_idx = 0

    def log(self, track_id, bbox, state, p_enter, door_open):
        x1, y1, x2, y2 = bbox
        cx, cy = state.last_center
        self.writer.writerow([
            self.frame_idx,
            f"{time.time() * 1000:.1f}",
            track_id,
            f"{cx:.1f}", f"{cy:.1f}",
            f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
            f"{state.last_dist:.4f}",
            f"{state.last_closure:.4f}",
            f"{state.last_angle:.4f}",
            f"{state.last_ear_l[0]:.1f}", f"{state.last_ear_l[1]:.1f}",
            f"{state.last_ear_r[0]:.1f}", f"{state.last_ear_r[1]:.1f}",
            f"{p_enter:.4f}",
            int(door_open),
        ])

    def tick(self):
        self.frame_idx += 1

    def close(self):
        self.file.close()
        print(f"[LOG] Feature data saved to {self.path}")


# ──────────────────────────────────────────────────────────────────
# Visualisation overlay
# ──────────────────────────────────────────────────────────────────

def draw_skeleton(frame, keypoints, colour=(200, 200, 200), thickness=1):
    """Draw COCO pose skeleton on frame."""
    kps = keypoints  # shape (17, 3)
    for i, j in SKELETON:
        if kps[i][2] > 0.3 and kps[j][2] > 0.3:
            pt1 = (int(kps[i][0]), int(kps[i][1]))
            pt2 = (int(kps[j][0]), int(kps[j][1]))
            cv2.line(frame, pt1, pt2, colour, thickness, cv2.LINE_AA)
    for k in range(17):
        if kps[k][2] > 0.3:
            cv2.circle(frame, (int(kps[k][0]), int(kps[k][1])),
                       3, (0, 255, 255), -1, cv2.LINE_AA)


def draw_head_direction(frame, keypoints):
    """Draw arrow showing head orientation from ear midpoint."""
    le = keypoints[KP_LEFT_EAR]
    re = keypoints[KP_RIGHT_EAR]
    if le[2] < 0.3 or re[2] < 0.3:
        return
    mid_x = (le[0] + re[0]) / 2
    mid_y = (le[1] + re[1]) / 2
    # Direction perpendicular to ear-to-ear line (rough head facing)
    dx = re[0] - le[0]
    dy = re[1] - le[1]
    # Perpendicular (pointing "forward" from the face)
    perp_x, perp_y = -dy, dx
    length = math.sqrt(perp_x ** 2 + perp_y ** 2)
    if length < 1:
        return
    scale = 30  # arrow length in pixels
    perp_x, perp_y = perp_x / length * scale, perp_y / length * scale
    pt1 = (int(mid_x), int(mid_y))
    pt2 = (int(mid_x + perp_x), int(mid_y + perp_y))
    cv2.arrowedLine(frame, pt1, pt2, (0, 200, 255), 2, tipLength=0.35)


def draw_door_zone(frame, door_center, radius=80):
    """Draw a semi-transparent door zone at the bottom."""
    overlay = frame.copy()
    ellipse_center = (door_center[0], door_center[1] + 10)
    axes = (int(frame.shape[1] * 0.46), int(frame.shape[0] * 0.10))
    cv2.ellipse(overlay, ellipse_center, axes,
                0, 180, 360, (255, 100, 0), -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.ellipse(frame, ellipse_center, axes,
                0, 180, 360, (255, 100, 0), 2)


def draw_distance_line(frame, person_center, door_center, dist_val):
    """Draw a dashed line from person to door with distance label."""
    pc = (int(person_center[0]), int(person_center[1]))
    cv2.line(frame, pc, door_center, (180, 180, 180), 1, cv2.LINE_AA)
    mid = ((pc[0] + door_center[0]) // 2, (pc[1] + door_center[1]) // 2)
    cv2.putText(frame, f"d={dist_val:.2f}", mid,
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def draw_pedestrian_panel(frame, rows):
    """Draw a compact side panel that scales to crowded scenes."""
    if not rows:
        return

    rows = sorted(rows, key=lambda r: r[1], reverse=True)
    max_rows = 10
    visible = rows[:max_rows]
    overflow = len(rows) - len(visible)

    frame_h, frame_w = frame.shape[:2]
    panel_w = min(460, max(320, int(frame_w * 0.38)))
    x0 = 10
    y0 = 8
    row_h = 24
    header_h = 34
    footer_h = 22 if overflow > 0 else 8
    panel_h = header_h + len(visible) * row_h + footer_h
    y1 = min(frame_h - 10, y0 + panel_h)
    x1 = x0 + panel_w

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (22, 22, 22), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (110, 110, 110), 1)

    cv2.putText(frame, "PEDESTRIANS", (x0 + 10, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(frame, f"count: {len(rows)}", (x0 + panel_w - 120, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    y = y0 + header_h
    for track_id, p_enter, dist, closure, decision in visible:
        colour = intent_colour(p_enter)
        state_txt = "OPEN" if decision else "WAIT"
        line = f"#{track_id:02d}  p={p_enter:0.2f}  d={dist:0.2f}  c={closure:+0.3f}  {state_txt}"
        cv2.putText(frame, line, (x0 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, colour, 1, cv2.LINE_AA)
        y += row_h

    if overflow > 0:
        cv2.putText(frame, f"+{overflow} more", (x0 + 10, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, cv2.LINE_AA)


def draw_door_status(frame, is_open, frame_w):
    """Draw a door status banner at the top-right."""
    text = "DOOR: OPEN" if is_open else "DOOR: CLOSED"
    colour = (0, 220, 0) if is_open else (0, 0, 220)
    bg_colour = (0, 60, 0) if is_open else (0, 0, 60)

    # Background rectangle
    tw, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0:2]
    x = frame_w - tw[0] - 30 if isinstance(tw, tuple) else frame_w - tw - 30
    # Simpler approach
    x_start = frame_w - 220
    cv2.rectangle(frame, (x_start, 8), (frame_w - 10, 48), bg_colour, -1)
    cv2.rectangle(frame, (x_start, 8), (frame_w - 10, 48), colour, 2)
    cv2.putText(frame, text, (x_start + 10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)


def draw_probability_bar(frame, p_enter, bbox, colour):
    """Draw a small horizontal probability bar above the bounding box."""
    x1, y1, x2, _ = [int(v) for v in bbox]
    bar_w = x2 - x1
    bar_h = 8
    bar_y = y1 - bar_h - 4
    if bar_y < 0:
        bar_y = y1 + 2

    # Background
    cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_h), (50, 50, 50), -1)
    # Fill
    fill_w = int(bar_w * p_enter)
    cv2.rectangle(frame, (x1, bar_y), (x1 + fill_w, bar_y + bar_h), colour, -1)
    # Border
    cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_h), (200, 200, 200), 1)


def draw_person_tag(frame, bbox, track_id, p_enter, colour):
    """Draw a compact, readable label above each person box."""
    x1, y1, _, _ = [int(v) for v in bbox]
    text = f"#{track_id} {p_enter:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_top = max(0, y1 - th - 10)
    cv2.rectangle(frame, (x1, y_top), (x1 + tw + 10, y_top + th + 8), (30, 30, 30), -1)
    cv2.rectangle(frame, (x1, y_top), (x1 + tw + 10, y_top + th + 8), colour, 1)
    cv2.putText(frame, text, (x1 + 5, y_top + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────
# Main demo loop
# ──────────────────────────────────────────────────────────────────

class DemoRunner:
    def __init__(self, cfg: dict):
        self.cfg = cfg

        # --- YOLO ---
        print(f"[YOLO] Loading {cfg['yolo_model']}...")
        self.yolo = YOLO(cfg["yolo_model"])

        # --- Camera ---
        self.cap = cv2.VideoCapture(cfg["source"])
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {cfg['source']}")

        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or cfg["frame_w"]
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or cfg["frame_h"]
        self.camera_fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 25

        # --- Frame skipping (match training FPS) ---
        #
        # The GRU was trained on data collected at 13 FPS with seq_len=7
        # (= 0.5 s observation window).  If the camera runs faster, we
        # must subsample so that 7 consecutive feature frames still span
        # ~0.5 s.  Only feature computation is skipped; YOLO + tracking
        # + visualisation still run every frame for smooth video.
        #
        self.training_fps = cfg["training_fps"]
        self.skip_n = max(1, round(self.camera_fps / self.training_fps))
        self._global_frame_idx = 0
        print(f"[FPS] Camera={self.camera_fps}, training={self.training_fps}, "
              f"skip_n={self.skip_n} (feature update every {self.skip_n} frames)")

        door_center = (cfg["door_x"], cfg["door_y"])

        # --- GRU model ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        norm_mean, norm_std = None, None
        if cfg.get("norm_path"):
            norm_mean, norm_std = load_normalisation_stats(cfg["norm_path"])
            print(f"[NORM] Loaded normalisation stats from {cfg['norm_path']}")

        self.model = load_model(
            path=cfg.get("model_path"),
            input_dim=cfg["input_dim"],
            device=self.device,
        )
        self.has_model = cfg.get("model_path") is not None

        # --- Feature computer ---
        self.feat = FeatureComputer(
            door_center=door_center,
            frame_w=self.frame_w,
            frame_h=self.frame_h,
            seq_len=cfg["seq_len"],
            fps=self.training_fps,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        # --- Decision state per person ---
        self.confirm_counter: dict[int, int] = defaultdict(int)
        # Cache last prediction per person (for frames we skip)
        self._last_predictions: dict[int, float] = defaultdict(lambda: 0.5)

        # --- Loggers ---
        self.latency_log = None
        self.feature_log = None
        if cfg["log_latency"]:
            self.latency_log = LatencyLogger(cfg["log_dir"])
            self.feature_log = FeatureLogger(cfg["log_dir"])

        # --- FPS measurement ---
        self._prev_time = time.time()
        self._frame_count = 0
        self._measured_pipeline_fps = 0.0    # actual end-to-end pipeline FPS
        self._feature_update_count = 0
        self._measured_feature_fps = 0.0     # how often features are computed

    def run(self):
        print(f"[DEMO] Starting — resolution {self.frame_w}x{self.frame_h}, "
              f"device={self.device}")
        print("[DEMO] Press 'q' or Esc to quit, 'f' to toggle fullscreen")

        win_name = "RISE Demo — Pedestrian Intent Prediction"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        if self.cfg["fullscreen"]:
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)

        door_is_open = False

        while True:
            t0 = time.time()

            ret, frame = self.cap.read()
            if not ret:
                break
            t_capture = time.time()

            # ── YOLO-Pose + tracking ──
            results = self.yolo.track(
                frame,
                persist=True,
                classes=[0],
                tracker=self.cfg["tracker"],
                conf=self.cfg["conf_thresh"],
                verbose=False,
            )
            t_yolo = time.time()

            boxes = results[0].boxes
            keypoints_all = results[0].keypoints  # Keypoints object

            current_ids = set()
            person_data = []  # (track_id, bbox, kps, state, p_enter)

            # Should we update features this frame?
            is_feature_frame = (self._global_frame_idx % self.skip_n == 0)
            self._global_frame_idx += 1

            # ── Feature extraction ──
            if boxes.id is not None and keypoints_all is not None:
                bboxes = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)
                kps_data = keypoints_all.data.cpu().numpy()  # (N, 17, 3)

                for bbox, track_id, kps in zip(bboxes, ids, kps_data):
                    current_ids.add(track_id)

                    if is_feature_frame:
                        # Update features + run GRU (subsampled to ~training FPS)
                        state = self.feat.update(track_id, bbox, kps)

                        p_enter = 0.5  # default
                        if self.has_model:
                            seq = self.feat.get_sequence(track_id)
                            if seq is not None:
                                x = torch.from_numpy(seq).to(self.device)
                                p_enter = self.model.predict_proba(x)

                        self._last_predictions[track_id] = p_enter
                    else:
                        # Non-feature frame: reuse last prediction, keep
                        # state reference for visualisation
                        state = self.feat.persons.get(track_id)
                        if state is None:
                            # First time seeing this person on a skip frame
                            state = self.feat.update(track_id, bbox, kps)
                        p_enter = self._last_predictions[track_id]

                    person_data.append((track_id, bbox, kps, state, p_enter))
            t_feat = time.time()

            # ── Decision logic ──
            door_is_open = False
            decisions = {}
            for track_id, bbox, kps, state, p_enter in person_data:
                if (p_enter >= self.cfg["open_threshold"]
                        and state.last_dist_norm <= self.cfg["max_open_distance"]):
                    self.confirm_counter[track_id] += 1
                else:
                    self.confirm_counter[track_id] = max(
                        0, self.confirm_counter[track_id] - 1
                    )

                person_triggers = (
                    self.confirm_counter[track_id] >= self.cfg["confirm_frames"]
                )
                decisions[track_id] = person_triggers
                if person_triggers:
                    door_is_open = True
            t_gru = time.time()

            # ── Cleanup vanished tracks ──
            vanished = set(self.feat.persons.keys()) - current_ids
            for vid in vanished:
                self.feat.remove(vid)
                self.confirm_counter.pop(vid, None)
                self._last_predictions.pop(vid, None)

            # ── Visualisation ──
            display = frame.copy()

            # Door zone
            door_center = (self.cfg["door_x"], self.cfg["door_y"])
            draw_door_zone(display, door_center)

            # Per-person overlays
            panel_rows = []
            for track_id, bbox, kps, state, p_enter in person_data:
                colour = intent_colour(p_enter, self.cfg["open_threshold"])
                x1, y1, x2, y2 = [int(v) for v in bbox]

                # Bounding box
                cv2.rectangle(display, (x1, y1), (x2, y2), colour, 2)

                # Probability bar + label
                draw_probability_bar(display, p_enter, bbox, colour)
                draw_person_tag(display, bbox, track_id, p_enter, colour)

                # Distance line to door
                draw_distance_line(display, state.last_center, door_center,
                                   state.last_dist)

                # Skeleton
                if self.cfg["show_skeleton"]:
                    draw_skeleton(display, kps, colour)
                    draw_head_direction(display, kps)

                # Ear keypoints highlighted
                for ear_pt in [state.last_ear_l, state.last_ear_r]:
                    cv2.circle(display, (int(ear_pt[0]), int(ear_pt[1])),
                               5, (0, 0, 255), -1)

                panel_rows.append(
                    (track_id, p_enter, state.last_dist, state.last_closure,
                     decisions.get(track_id, False))
                )

                # Log features
                if self.feature_log:
                    self.feature_log.log(
                        track_id, bbox, state, p_enter,
                        decisions.get(track_id, False)
                    )

            # Door status banner
            draw_door_status(display, door_is_open, self.frame_w)

            # Compact side panel for many simultaneous pedestrians
            if self.cfg["show_features"]:
                draw_pedestrian_panel(display, panel_rows)

            t_viz = time.time()

            # ── FPS counters ──
            self._frame_count += 1
            if is_feature_frame:
                self._feature_update_count += 1
            elapsed = time.time() - self._prev_time
            if elapsed >= 1.0:
                self._measured_pipeline_fps = self._frame_count / elapsed
                self._measured_feature_fps = self._feature_update_count / elapsed
                self._frame_count = 0
                self._feature_update_count = 0
                self._prev_time = time.time()

            # ── Latency logging ──
            dt_total = (t_viz - t0) * 1000
            if self.latency_log:
                self.latency_log.log(
                    t_capture=t_capture * 1000,
                    dt_yolo=(t_yolo - t_capture) * 1000,
                    dt_feat=(t_feat - t_yolo) * 1000,
                    dt_gru=(t_gru - t_feat) * 1000,
                    dt_viz=(t_viz - t_gru) * 1000,
                    dt_total=dt_total,
                    n_persons=len(person_data),
                    fps=self._measured_pipeline_fps,
                )
            if self.feature_log:
                self.feature_log.tick()

            # ── Show ──
            cv2.imshow(win_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("f"):
                # Toggle fullscreen
                prop = cv2.getWindowProperty(
                    win_name, cv2.WND_PROP_FULLSCREEN
                )
                if prop == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty(
                        win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
                    )
                else:
                    cv2.setWindowProperty(
                        win_name, cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN,
                    )

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        if self.latency_log:
            self.latency_log.close()
        if self.feature_log:
            self.feature_log.close()
        print("[DEMO] Finished.")


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="RISE Demo — Intent Prediction")
    p.add_argument("--source", default=0,
                   help="Camera index or video path (default: 0)")
    p.add_argument("--yolo", default="yolov8n-pose.pt",
                   help="YOLO-Pose model file")
    p.add_argument("--model", default=None,
                   help="Trained GRU weights (.pt)")
    p.add_argument("--norm", default=None,
                   help="Normalisation stats (.npz)")
    p.add_argument("--door-x", type=int, default=None,
                   help="Door center X (default: frame center)")
    p.add_argument("--door-y", type=int, default=None,
                   help="Door center Y (default: frame bottom)")
    p.add_argument("--threshold", type=float, default=0.75,
                   help="P(enter) threshold for door opening")
    p.add_argument("--confirm", type=int, default=5,
                   help="Consecutive frames above threshold to confirm")
    p.add_argument("--fullscreen", action="store_true", default=True)
    p.add_argument("--no-features", action="store_true",
                   help="Hide feature panel")
    p.add_argument("--no-skeleton", action="store_true",
                   help="Hide skeleton overlay")
    p.add_argument("--no-log", action="store_true",
                   help="Disable CSV logging")
    p.add_argument("--log-dir", default="logs",
                   help="Directory for log files")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = DEFAULT_CFG.copy()

    # Source
    try:
        cfg["source"] = int(args.source)
    except (ValueError, TypeError):
        cfg["source"] = args.source

    cfg["yolo_model"] = args.yolo

    # Model
    cfg["model_path"] = args.model
    cfg["norm_path"] = args.norm

    # Door position (will be overridden after cap is opened if None)
    if args.door_x is not None:
        cfg["door_x"] = args.door_x
    if args.door_y is not None:
        cfg["door_y"] = args.door_y

    # Decision
    cfg["open_threshold"] = args.threshold
    cfg["confirm_frames"] = args.confirm

    # Display
    cfg["fullscreen"] = args.fullscreen
    cfg["show_features"] = not args.no_features
    cfg["show_skeleton"] = not args.no_skeleton

    # Logging
    cfg["log_latency"] = False
    cfg["log_dir"] = args.log_dir

    # Handle default door position (frame center bottom)
    # We need to open the camera first to know the resolution
    cap_temp = cv2.VideoCapture(cfg["source"])
    if cap_temp.isOpened():
        w = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if args.door_x is None:
            cfg["door_x"] = w // 2
        if args.door_y is None:
            cfg["door_y"] = h
        cap_temp.release()

    demo = DemoRunner(cfg)
    demo.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[DEMO] Interrupted by user. Exiting cleanly.")
