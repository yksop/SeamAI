#!/usr/bin/env python3
"""
dataset_utils.py

Helper functions shared by train_logreg.py and
rf_feature_importance.py.

Both scripts need to load samples and build feature matrices the same way,
so the logic lives here to avoid copy-pasting.
"""

import json
import os

import numpy as np

from utils.feature_extractor import extract_features, is_sample_usable


def is_valid_sample(sample):
    """
    Return True if the sample should be included in experiments.

    We keep only samples labelled "enter" or "pass" that have not been
    flagged as bad data by the cleaning pipeline.
    """
    label = str(sample.get("label", "")).lower()
    if label not in ("enter", "pass"):
        return False

    # Skip anything the cleaning step marked as bad
    cleaning = str(sample.get("cleaning_result", "keep")).lower()
    if cleaning in ("corrupt", "corrup", "exit", "removed", "remove"):
        return False

    return True


def load_all_samples(data_root):
    """
    Load all valid trajectory JSON files from MasterData.

    Folder structure:
        MasterData/
            enter/   <- pedestrians who entered the door
            pass/    <- pedestrians who walked past

    Returns a list of dicts, one per sample.
    """
    samples = []

    for split_folder in ("enter", "pass"):
        folder_path = os.path.join(data_root, split_folder)

        if not os.path.exists(folder_path):
            print(f"  Warning: folder not found: {folder_path}")
            continue

        for root, dirs, files in os.walk(folder_path):
            # Don't recurse into rejected sub-folders
            dirs[:] = [
                d for d in dirs
                if d.lower() not in ("exit", "removed", "remove", "corrupt")
            ]

            for filename in sorted(files):
                if not filename.endswith(".json"):
                    continue
                # Skip pose sidecar files created by extract_pose.py
                if filename.endswith("_pose.json"):
                    continue

                json_path = os.path.join(root, filename)

                try:
                    with open(json_path, "r") as f:
                        sample = json.load(f)
                except Exception as e:
                    print(f"  Could not read {json_path}: {e}")
                    continue

                if isinstance(sample, dict) and is_valid_sample(sample):
                    # Store the path so sequence_dataset can find the paired _pose.json
                    sample["_source_path"] = json_path
                    samples.append(sample)

    return samples


def build_feature_matrix(samples, tte_seconds, window_seconds):
    """
    Build a feature matrix X and label vector y from a list of samples.

    Samples that are too short for the given TTE + window are dropped.
    Without this check, t_predict would be clipped to frame 0 and the
    features would not correspond to the intended prediction time.

    Returns:
        X : float array, shape (n_usable, 30)
        y : int array,   shape (n_usable,)
    """
    label_map = {"pass": 0, "enter": 1}
    X_rows  = []
    y_labels = []
    dropped = {"enter": 0, "pass": 0}

    for sample in samples:
        label = str(sample.get("label", "")).lower()

        if not is_sample_usable(sample, tte_seconds, window_seconds):
            if label in dropped:
                dropped[label] += 1
            continue

        try:
            features = extract_features(
                sample,
                tte_seconds=tte_seconds,
                window_seconds=window_seconds,
            )
        except Exception as e:
            print(f"  Skipping sample due to error: {e}")
            continue

        X_rows.append(features)
        y_labels.append(label_map[label])

    if dropped["enter"] > 0 or dropped["pass"] > 0:
        print(
            f"  Dropped (too short for TTE={tte_seconds:.1f}s): "
            f"enter={dropped['enter']}, pass={dropped['pass']}"
        )

    X = np.array(X_rows,   dtype=float)
    y = np.array(y_labels, dtype=int)
    return X, y


def build_hc_sequences(samples, tte_seconds, window_seconds):
    """
    Build the 30 handcrafted features as a (N, 1, 30) array ready for the
    neural network pipeline.

    This uses the same features as train_logreg.py (mean + variance
    + latest value for each of the 10 trajectory features).  Reshaping to
    (N, 1, 30) means the sequence length is 1, so MLP, GRU and CNN all
    receive the same information – there is no temporal structure to exploit.
    For this mode only MLP is meaningful; GRU and CNN reduce to MLP.

    Returns:
        X : float32 array, shape (n_usable, 1, 30)
        y : int array,     shape (n_usable,)
    """
    X_flat, y = build_feature_matrix(samples, tte_seconds, window_seconds)
    # Insert a dummy time-step dimension so the shape matches (batch, T, D)
    X_seq = X_flat[:, np.newaxis, :].astype(np.float32)
    return X_seq, y


def print_sample_counts_per_tte(samples, tte_values, window_seconds):
    """
    Print how many enter/pass samples are usable at each TTE value.

    Useful for spotting data loss when TTE is large – some trajectories
    are simply not long enough to provide a valid observation window.
    """
    total_enter = sum(1 for s in samples if s.get("label") == "enter")
    total_pass  = sum(1 for s in samples if s.get("label") == "pass")
    total_all   = total_enter + total_pass

    print(f"\n  Sample counts per TTE (window = {window_seconds:.1f}s):")
    print(f"  {'TTE (s)':>8} | {'enter':>6} | {'pass':>6} | {'total':>6} | {'dropped':>8}")
    print("  " + "-" * 48)

    for tte in tte_values:
        n_enter = sum(
            1 for s in samples
            if s.get("label") == "enter" and is_sample_usable(s, tte, window_seconds)
        )
        n_pass = sum(
            1 for s in samples
            if s.get("label") == "pass" and is_sample_usable(s, tte, window_seconds)
        )
        total   = n_enter + n_pass
        dropped = total_all - total
        print(f"  {tte:>8.1f} | {n_enter:>6} | {n_pass:>6} | {total:>6} | {dropped:>8}")

    print()
