import os
import json
import math
from typing import List, Dict, Tuple, Optional, Any

import cv2
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathways import pathways


RAW_DATA_PATHS = pathways.office_videos()
MASTER_JSON_DIR = "master_jsons"
os.makedirs(MASTER_JSON_DIR, exist_ok=True)


def data_packager(repos):
    if isinstance(repos, str):
        repos = [repos]

    pairs = []

    for repo in repos:
        repo = os.path.normpath(repo)
        for scenario in os.listdir(repo):
            scenario_path = os.path.join(repo, scenario)
            if not os.path.isdir(scenario_path):
                continue

            for label in ("enter", "pass"):
                label_path = os.path.join(scenario_path, label)
                if not os.path.isdir(label_path):
                    continue

                videos, jsons = {}, {}

                for f in os.listdir(label_path):
                    full_path = os.path.join(label_path, f)
                    key = os.path.splitext(f)[0]
                    if f.lower().endswith(".mp4"):
                        videos[key] = full_path
                    elif f.lower().endswith(".json"):
                        jsons[key] = full_path

                for key, vpath in videos.items():
                    if key in jsons:
                        pairs.append({
                            "scenario": scenario,
                            "label": label,
                            "video": vpath,
                            "json": jsons[key]
                        })
    return pairs


def run_mediapipe(video_path: str, frames: List[dict]) -> List[dict]:
    """
    Run Mediapipe pose on the video, cropping each frame to bbox, and store
    pose landmarks in image coordinates in each frame dict under 'pose_landmarks'.
    Also attach 'frame_size' (width, height) onto the top-level data in skeleton().
    """
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
    out_frames = []

    # We will read frames in sync with frames list — if the video has different
    # number of frames the original code handled that, keep same behavior.
    frame_idx = 0
    frame_w = None
    frame_h = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # store frame size once
        if frame_w is None or frame_h is None:
            frame_h, frame_w = frame.shape[:2]

        # If frames list shorter than video, we still should process until frames length
        if frame_idx >= len(frames):
            break

        f_info = frames[frame_idx].copy()
        bbox = f_info.get('bbox')
        if bbox is None:
            f_info['pose_landmarks'] = None
            out_frames.append(f_info)
            frame_idx += 1
            continue

        x_min, y_min, x_max, y_max = bbox
        h, w = frame.shape[:2]

        x_min = int(max(0, min(x_min, w-1)))
        y_min = int(max(0, min(y_min, h-1)))
        x_max = int(max(0, min(x_max, w-1)))
        y_max = int(max(0, min(y_max, h-1)))

        if x_max <= x_min or y_max <= y_min:
            f_info['pose_landmarks'] = None
            out_frames.append(f_info)
            frame_idx += 1
            continue

        crop = frame[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            f_info['pose_landmarks'] = None
            out_frames.append(f_info)
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            landmarks = []
            for lm in res.pose_landmarks.landmark:
                # convert landmark back to original image coordinates
                x = lm.x * (x_max - x_min) + x_min
                y = lm.y * (y_max - y_min) + y_min
                z = lm.z
                visibility = getattr(lm, 'visibility', 0.0)
                landmarks.append([float(x), float(y), float(z), float(visibility)])
            f_info['pose_landmarks'] = landmarks
        else:
            f_info['pose_landmarks'] = None
        out_frames.append(f_info)
        frame_idx += 1

    cap.release()
    pose.close()

    # we will return out_frames; the caller (skeleton) will insert frame_size into the data dict
    return out_frames

def skeleton(video: str, video_json: str) -> dict:
    with open(video_json, "r") as f:
        data = json.load(f)

    # Check if the video can be opened
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video}. Skipping.")
        cap.release()
        return None
    cap.release()

    frames = data.get('frames', [])
    augmented_frames = run_mediapipe(video, frames)

    # Skip if run_mediapipe failed to extract any frames
    if not augmented_frames:
        print(f"[WARN] No frames processed for video: {video}. Skipping.")
        return None

    # Add reduced landmarks
    REDUCED_LANDMARKS = [0, 11, 12, 23, 24]
    for frame in augmented_frames:
        lm = frame.get('pose_landmarks')
        if lm is not None:
            frame['pose_landmarks_reduced'] = [lm[i] for i in REDUCED_LANDMARKS if i < len(lm)]
        else:
            frame['pose_landmarks_reduced'] = None

    data['frames'] = augmented_frames

    try:
        cap = cv2.VideoCapture(video)
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            data['frame_size'] = [int(w), int(h)]
        cap.release()
    except Exception:
        data['frame_size'] = [0, 0]

    # fallback if still zero
    if data.get('frame_size', [0, 0]) == [0, 0]:
        max_w = 0
        max_h = 0
        for f in frames:
            bb = f.get('bbox')
            if bb:
                max_w = max(max_w, bb[2])
                max_h = max(max_h, bb[3])
        if max_w > 0 and max_h > 0:
            data['frame_size'] = [int(max_w), int(max_h)]

    return data

def skeleton_angles(video_json: dict, selected_landmarks=[0,11,12,23,24]) -> dict:
    data = dict(video_json)
    w, h = data['frame_size']
    for frame in data.get('frames', []):
        lm = frame.get('pose_landmarks')
        if lm is None:
            frame['pose_angles'] = None
            continue
        try:
            arr = np.array(lm)[:, :3]  # x, y, z
            door_point = np.array([w/2, h, 0])
            angles = {}
            for idx in selected_landmarks:
                joint = arr[idx] if idx < len(arr) else np.array([np.nan, np.nan, np.nan])
                vec = door_point - joint
                angle = float(np.degrees(np.arctan2(vec[1], vec[0])))
                angles[f"joint{idx}_to_door"] = angle
            frame['pose_angles'] = angles
        except Exception:
            frame['pose_angles'] = None
    return data


def build_master_json(repo_path, out_dir=MASTER_JSON_DIR):
    print(f"Processing videos in {repo_path}...")
    pairs = data_packager(repo_path)
    master_data = {}

    for p in pairs:
        video_name = os.path.splitext(os.path.basename(p['video']))[0]
        print(f"Processing video {video_name}...")
        try:
            data_with_skeleton = skeleton(p['video'], p['json'])
            if data_with_skeleton is None:
                print(f"Skipping {video_name} (no skeleton data).")
                continue
            master_data[video_name] = skeleton_angles(data_with_skeleton)
        except Exception as e:
            print(f"Failed processing {video_name}: {e}")
            continue

    out_path = os.path.join(out_dir, "master_dataset.json")
    with open(out_path, "w") as f:
        json.dump(master_data, f, indent=4)

    print(f"Master JSON saved to {out_path}")
    return out_path


if __name__ == "__main__":
    build_master_json(RAW_DATA_PATHS)