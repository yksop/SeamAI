#!/usr/bin/env python3
"""
extract_pose.py

One-time preprocessing script.

For every trajectory sample in MasterData it:
  1. Opens the paired .mp4 video file.
  2. For each frame listed in the trajectory JSON, crops a region around the
     person's bounding box (with a margin) and runs MediaPipe PoseLandmarker
     on that crop.  Cropping focuses the model on the person, which improves
     detection for small/distant pedestrians compared to running on the full
     frame.
  3. Converts the crop-relative landmark coordinates back to full-frame
     normalised coordinates and saves them to a sidecar file:
         <original_name>_pose.json

Running this script is the prerequisite for the "Trajectory + Pose" and
"Trajectory + Pose + Head" ablation conditions.

--- Landmarks saved per frame (normalised [0,1] relative to full image size) ---

    nose (0), left_ear (7), right_ear (8),
    left_shoulder (11), right_shoulder (12),
    left_elbow (13), right_elbow (14),
    left_wrist (15), right_wrist (16),
    left_hip (23), right_hip (24),
    left_knee (25), right_knee (26),
    left_ankle (27), right_ankle (28)

Each landmark stores (x, y, visibility).
A detected flag indicates whether MediaPipe found a person in that frame.

--- Usage ---

    python extract_pose.py                # skip already-processed files
    python extract_pose.py --overwrite    # re-process everything

Needs the MediaPipe model `pose_landmarker_lite.task` in this folder
(0_data_cleaning/) — a one-time download, see README.md. It is resolved next
to this script, so it is found no matter which folder you run from.

~2-5 minutes for the full dataset on CPU.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# MediaPipe landmark indices we care about.
# Upper body (for shoulder/torso/head features) + lower body (for stride/gait).
LANDMARK_IDS = {
    # Head
    "nose":              0,
    "left_ear":          7,
    "right_ear":         8,
    # Upper body
    "left_shoulder":    11,
    "right_shoulder":   12,
    "left_elbow":       13,
    "right_elbow":      14,
    "left_wrist":       15,
    "right_wrist":      16,
    # Core
    "left_hip":         23,
    "right_hip":        24,
    # Lower body
    "left_knee":        25,
    "right_knee":       26,
    "left_ankle":       27,
    "right_ankle":      28,
}

SKIP_FOLDERS   = {"corrupt", "exit", "removed"}
DATA_ROOT      = Path(__file__).resolve().parents[1] / "MasterData"
# MediaPipe pose model — resolved next to this script, so it is found no matter
# which folder you launch from. One-time download (see README.md).
MODEL_PATH     = Path(__file__).resolve().parent / "pose_landmarker_lite.task"


def find_pairs(data_root: Path):
    """
    Return (json_path, mp4_path) for every valid sample.
    Skips folders in SKIP_FOLDERS and samples without a paired .mp4.
    """
    pairs = []
    for folder in sorted(data_root.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name.lower() in SKIP_FOLDERS:
            continue
        for json_path in sorted(folder.glob("*.json")):
            if json_path.stem.endswith("_pose"):
                continue
            mp4_path = json_path.with_suffix(".mp4")
            if not mp4_path.exists():
                print(f"  [SKIP] No .mp4 found for {json_path.name}")
                continue
            pairs.append((json_path, mp4_path))
    return pairs


def _crop_with_margin(bgr, bbox, margin_x_frac=0.40, margin_y_frac=0.60):
    """
    Crop the BGR frame around a bounding box, adding a proportional margin.

    A generous vertical margin (default 60 %) captures the head above the box
    and the feet below, which is important for full-body pose estimation.

    Returns (crop, cx1, cy1, crop_w, crop_h).
    Falls back to the full frame if the crop would be degenerate.
    """
    H, W = bgr.shape[:2]
    x1b, y1b, x2b, y2b = [int(v) for v in bbox]
    bw = max(x2b - x1b, 1)
    bh = max(y2b - y1b, 1)

    mx  = max(int(bw * margin_x_frac), 10)
    my  = max(int(bh * margin_y_frac), 20)
    cx1 = max(0, x1b - mx)
    cy1 = max(0, y1b - my)
    cx2 = min(W, x2b + mx)
    cy2 = min(H, y2b + my)

    crop_w = cx2 - cx1
    crop_h = cy2 - cy1

    if crop_w < 2 or crop_h < 2:
        return bgr, 0, 0, W, H

    return bgr[cy1:cy2, cx1:cx2], cx1, cy1, crop_w, crop_h


def extract_pose_for_sample(json_path: Path, mp4_path: Path, landmarker) -> dict:
    """
    Run PoseLandmarker on all frames listed in the JSON trajectory file.

    For each frame, the person's bounding box from the trajectory JSON is used
    to crop the video frame before running MediaPipe.  This focuses the model
    on the pedestrian rather than the entire scene.

    Landmark x/y coordinates are converted from crop-relative normalised coords
    back to full-frame normalised coords before saving.

    Returns a dict ready to be written as _pose.json.
    """
    with open(json_path, "r") as f:
        traj = json.load(f)

    frames_data = traj.get("frames", [])

    needed_frame_indices = {f["frame_index"] for f in frames_data}

    # Map frame_index → bbox so we can crop each frame to the person
    bbox_lookup: dict = {}
    for frame in frames_data:
        fi   = frame.get("frame_index")
        bbox = frame.get("bbox")
        if bbox and len(bbox) == 4:
            bbox_lookup[fi] = bbox

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or traj.get("fps", 13.0)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose_frames   = []
    video_frame_i = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if video_frame_i in needed_frame_indices:
            bbox = bbox_lookup.get(video_frame_i)

            if bbox is not None and W > 0 and H > 0:
                crop, cx1, cy1, crop_w, crop_h = _crop_with_margin(bgr, bbox)
            else:
                # No bbox available – run on the full frame
                crop, cx1, cy1, crop_w, crop_h = bgr, 0, 0, W, H

            rgb_input = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_input)

            # IMAGE mode: each frame is independent, no timestamp needed
            result = landmarker.detect(mp_image)

            frame_data = {"frame_index": video_frame_i, "detected": False}

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]   # most prominent person
                frame_data["detected"] = True

                landmarks = {}
                for name, idx in LANDMARK_IDS.items():
                    pt = lm[idx]
                    # MediaPipe landmark coords are normalised relative to the
                    # crop.  Convert back to full-frame normalised coords.
                    full_x = (cx1 + pt.x * crop_w) / W if W > 0 else pt.x
                    full_y = (cy1 + pt.y * crop_h) / H if H > 0 else pt.y
                    landmarks[name] = {
                        "x":          round(max(0.0, min(1.0, full_x)), 5),
                        "y":          round(max(0.0, min(1.0, full_y)), 5),
                        "visibility": round(pt.visibility, 4),
                    }
                frame_data["landmarks"] = landmarks

            pose_frames.append(frame_data)

        video_frame_i += 1

    cap.release()

    return {
        "source":     mp4_path.name,
        "label":      traj.get("label", ""),
        "fps":        fps,
        "n_detected": sum(1 for f in pose_frames if f["detected"]),
        "n_frames":   len(pose_frames),
        "frames":     pose_frames,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process samples that already have a _pose.json")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        print("Download it once with:")
        print(f"  curl -L -o '{MODEL_PATH}' \\")
        print("    https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
              "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
        sys.exit(1)

    pairs = find_pairs(DATA_ROOT)
    total = len(pairs)
    print(f"Found {total} sample pairs in {DATA_ROOT}/")

    # IMAGE mode: each frame is processed independently (no temporal state).
    # This lets us reuse one landmarker instance across all samples and frames,
    # and allows per-frame cropping without confusing the tracker.
    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )

    skipped = done = failed = 0

    # One landmarker reused for all samples (IMAGE mode is stateless)
    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        for i, (json_path, mp4_path) in enumerate(pairs, start=1):
            pose_path = json_path.parent / (json_path.stem + "_pose.json")

            if pose_path.exists() and not args.overwrite:
                skipped += 1
                continue

            try:
                pose_data = extract_pose_for_sample(
                    json_path, mp4_path, landmarker
                )
                with open(pose_path, "w") as f:
                    json.dump(pose_data, f)

                det_rate = pose_data["n_detected"] / max(pose_data["n_frames"], 1)
                print(
                    f"  [{i:3d}/{total}]  {json_path.name:<45}  "
                    f"frames={pose_data['n_frames']:3d}  "
                    f"detected={pose_data['n_detected']:3d}  "
                    f"({det_rate:.0%})"
                )
                done += 1

            except Exception as e:
                print(f"  [{i:3d}/{total}]  ERROR  {json_path.name}: {e}",
                      file=sys.stderr)
                failed += 1

    print(f"\nDone.  processed={done}  skipped={skipped}  failed={failed}")
    if skipped > 0:
        print("  (run with --overwrite to re-process existing pose files)")


if __name__ == "__main__":
    main()
