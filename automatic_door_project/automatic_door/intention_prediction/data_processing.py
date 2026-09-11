"""
Author: Hanna Norberg
e-mail: hanna.gjelstrup.norberg@gmail.com
"""

import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_master_json(json_path):
    """Load a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def calculate_frames_from_seconds(fps, seconds):
    """Calculate the number of frames corresponding to a time horizon in seconds."""
    return int(round(fps * seconds))

def get_video_fps(video_data):
    return video_data.get("fps", 12) # Default to 12 fps if not specified

def pad_sequences(sequence, target_length):
    """Pad a sequence to target length."""
    current_length = len(sequence)

    if current_length == target_length:
        return sequence
    elif current_length < target_length:
        last_frame = sequence[-1]
        padding_count = target_length - current_length
        if sequence.ndim == 1:
            padding = np.tile(last_frame, padding_count)
        else:
            padding = np.tile(last_frame[np.newaxis, :], (padding_count, 1))
        return np.concatenate([sequence, padding], axis=0)

def extract_features(video_data, use_bbox=True, use_skeleton=True, use_angles=True, use_reduced_skeleton=True):
    """Extract features from video data (excluding centroids)"""
    frames = video_data.get("frames", [])
    features = []

    for frame in frames:
        frame_feats = []

        # BBox features
        if use_bbox:
            bbox = frame.get("bbox")
            if bbox:
                x_min, y_min, x_max, y_max = map(float, bbox)
                # area = max(1.0, (x_max - x_min) * (y_max - y_min))
                frame_feats.extend([x_min, y_min, x_max, y_max])
            else:
                frame_feats.extend([0.0] * 4)  # Placeholder for missing bbox

        # Skeleton features
        if use_skeleton:
            lm_key = "pose_landmarks_reduced" if use_reduced_skeleton else "pose_landmarks"
            lm = frame.get(lm_key)
            if lm:
                for point in lm:
                    frame_feats.extend([float(point[0]), float(point[1]), float(point[2]), float(point[3])])
            else:
                n_landmarks = 5 if use_reduced_skeleton else 33
                frame_feats.extend([0.0] * (n_landmarks * 4))
        
        # Angle features
        if use_angles:
            angles = frame.get("pose_angles")
            if angles:
                frame_feats.extend([float(v) for v in angles.values()])
            else:
                frame_feats.extend([0.0] * 3)
        
        features.append(frame_feats)
    
    return np.array(features, dtype=np.float32)

def extract_centroids(video_data):
    """Extract centroid coordinates from video data"""
    frames = video_data.get("frames", [])
    centroids = []

    for frame in frames:
        center = frame.get("center", [0.0, 0.0])
        centroids.append([float(center[0]), float(center[1])])

    return np.array(centroids, dtype=np.float32)

def prepare_classification_data(X_sequences, labels, test_size=0.2):
    """Prepare data for classification."""
    
    X = np.array(X_sequences)
    y = np.array(labels)

    # Check if stratification is possible
    unique, counts = np.unique(y, return_counts=True)
    min_class_count = min(counts)
    
    # Use stratify only if each class has at least 2 samples
    stratify_param = y if min_class_count >= 2 else None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_param
    )
    
    # Scale features
    feature_scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    feature_scaler.fit(X_train_flat)
    
    X_train_scaled = feature_scaler.transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = feature_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_scaler

def build_intention_dataset(master_json, scenario_config, past_seconds=2.0, distance_from_end=None):
    """Build dataset for label prediction with flexible windowing like TP."""
    X_sequences, labels = [], []
    max_past_frames = 0

    for video_name, video_data in master_json.items():
        fps = get_video_fps(video_data)
        past_frames = calculate_frames_from_seconds(fps, past_seconds)
        max_past_frames = max(max_past_frames, past_frames)

        # Handle centroid_only case
        if scenario_config.get("use_centroid_only", False):
            features = extract_centroids(video_data)
        else:
            features = extract_features(video_data, **scenario_config)
        
        label = video_data.get("label", "unknown")

        if distance_from_end is not None:
            distance_frames = calculate_frames_from_seconds(fps, distance_from_end)
            total_frames_needed = past_frames + distance_frames
            
            # Use 75% threshold like TP
            min_past = int(past_frames * 0.75)
            min_distance = int(distance_frames * 0.75)
            min_frames_threshold = min_past + min_distance

            if len(features) < min_frames_threshold:
                continue

            # Adaptive windowing like TP
            if len(features) >= total_frames_needed:
                # Video long enough, use exact frames
                start_idx = len(features) - distance_frames - past_frames
                end_idx = len(features) - distance_frames
                X_seq = features[start_idx:end_idx]
            else:
                # Video too short, use adaptive logic
                n_frames = len(features)
                past = max(1, min_past)
                distance = max(1, min_distance)
                
                # Distribute excess frames
                excess = n_frames - min_past - min_distance
                if excess > 0:
                    distance_boost = min(excess, distance_frames - min_distance)
                    distance += distance_boost
                    
                    remaining_excess = excess - distance_boost
                    if remaining_excess > 0:
                        past_boost = min(remaining_excess, past_frames - min_past)
                        past += past_boost
                
                start_idx = max(0, n_frames - distance - past)
                end_idx = start_idx + past
                X_seq = features[start_idx:end_idx]
        else:
            # Original logic for non-distance cases with 75% threshold
            min_past = int(past_frames * 0.75)
            
            if len(features) < min_past:
                continue

            if len(features) >= past_frames:
                X_seq = features[-past_frames:]
            else:
                X_seq = features

        X_sequences.append(X_seq)
        labels.append(1 if label == "enter" else 0)

    X_padded = [pad_sequences(seq, max_past_frames) for seq in X_sequences]
    return X_padded, labels
