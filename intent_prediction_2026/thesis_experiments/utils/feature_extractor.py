#!/usr/bin/env python3
"""
feature_extractor.py

Extracts handcrafted motion features from one pedestrian trajectory sample
and returns a fixed-size feature vector for use with a classifier.

All features are based on:
    Method section: "Baseline Model: Logistic Regression and Handcrafted Features"

--- What is a trajectory sample? ---

Each sample is a JSON file that contains:
  - "frames"      : a list of per-frame measurements, one dict per frame
                    Each frame has:
                        "center": [x, y]           -- person center in pixels
                        "bbox":   [x1, y1, x2, y2] -- bounding box corners
  - "label"       : "enter" or "pass"
  - "fps"         : frames per second of the recording
  - "door_center" : [x, y] position of the entrance in the image

--- How features are extracted ---

Step 1: Parse the frame data into coordinate arrays.
Step 2: Find T_event (the frame where the key event happens).
Step 3: Compute T_predict = T_event - tte_seconds * fps
        This is the last frame we are "allowed" to see before making a prediction.
Step 4: Define the observation window: the K frames just before T_predict.
Step 5: Compute 10 feature time-series across the full recording.
Step 6: For each feature, extract its values in the observation window
        and aggregate them into [mean, variance, latest] = 3 numbers.
Step 7: Stack all 30 numbers into one vector (10 features × 3 statistics).
"""

import json
import math


# Fallback FPS used when the JSON does not contain an fps value
DEFAULT_FPS = 13.0

# Step 1 – Parse raw frame data

def get_coordinates(frames):
    """
    Convert the list of frame dicts into plain Python lists of floats.

    Missing detections (None values) are replaced with float("nan") so we
    can do math on the arrays later without crashing on None comparisons.

    Returns six lists, all the same length as `frames`:
        x, y         -- center point (pixels)
        x1, y1       -- top-left corner of bounding box (pixels)
        x2, y2       -- bottom-right corner of bounding box (pixels)
    """
    x,  y  = [], []
    x1, y1 = [], []
    x2, y2 = [], []

    for frame in frames:
        center = frame.get("center")
        bbox   = frame.get("bbox")

        # --- center point ---
        if center is not None and len(center) == 2:
            x.append(float(center[0]))
            y.append(float(center[1]))
        else:
            x.append(float("nan"))  # detector missed this frame
            y.append(float("nan"))

        # --- bounding box ---
        if bbox is not None and len(bbox) == 4:
            x1.append(float(bbox[0]))
            y1.append(float(bbox[1]))
            x2.append(float(bbox[2]))
            y2.append(float(bbox[3]))
        else:
            x1.append(float("nan"))
            y1.append(float("nan"))
            x2.append(float("nan"))
            y2.append(float("nan"))

    return x, y, x1, y1, x2, y2

# Step 2 – Locate the event frame (T_event)

def find_event_frame(label, x, y, door_x, door_y):
    """
    Find the frame index T_event where the key event happens.

    "enter" -> the frame where the person is closest to the door center.
               This represents the moment they step through the entrance.

    "pass"  -> the last frame where the person is still visible (has a
               valid center point). This represents when they leave the scene.

    Returns the index as an integer, or -1 if no valid frames exist.
    """
    # Collect all frame indices that have a valid detected center
    valid = [i for i in range(len(x)) if not math.isnan(x[i]) and not math.isnan(y[i])]

    if len(valid) == 0:
        return -1  # completely empty trajectory

    if label == "pass":
        # For "pass" the event is simply the last frame the person is visible
        return valid[-1]

    if label == "enter":
        # For "enter" we look for the frame closest to the door center
        best_idx  = valid[0]
        best_dist = float("inf")

        for i in valid:
            dist = math.sqrt((x[i] - door_x)**2 + (y[i] - door_y)**2)
            if dist < best_dist:
                best_dist = dist
                best_idx  = i

        return best_idx

    return -1


# Step 5 – Compute each feature time-series

def compute_distance_to_door(x, y, door_x, door_y):
    """
    Goal-Oriented Feature 1: Distance to door.

    D_door = sqrt( (x_center - x_door)^2 + (y_center - y_door)^2 )

    A decreasing distance over time strongly suggests the person intends
    to enter the door.
    """
    d_door = []
    for i in range(len(x)):
        if math.isnan(x[i]) or math.isnan(y[i]):
            d_door.append(float("nan"))
        else:
            d = math.sqrt((x[i] - door_x)**2 + (y[i] - door_y)**2)
            d_door.append(d)
    return d_door


def compute_closure_rate(d_door, fps, k):
    """
    Goal-Oriented Feature 2: Rate of closure.

    Closure Rate = ( D_door(t) - D_door(t - K) ) / ( K / fps )

    Negative = approaching the door (closing the distance).
    Positive = moving away from the door.

    We use a lag of K frames (the full observation window width) so the
    rate is averaged over that period rather than just one frame.
    """
    dt_k = k / fps  # time in seconds for K frames
    closure_rate = []

    for i in range(len(d_door)):
        prev_i = i - k
        if prev_i < 0 or math.isnan(d_door[i]) or math.isnan(d_door[prev_i]):
            closure_rate.append(float("nan"))
        else:
            rate = (d_door[i] - d_door[prev_i]) / dt_k
            closure_rate.append(rate)

    return closure_rate


def compute_velocity(x, y, fps, k):
    """
    Kinematic Features 3 & 4: Velocity components (vx, vy).

    vx = ( x_center(t) - x_center(t-K) ) / ( K / fps )
    vy = ( y_center(t) - y_center(t-K) ) / ( K / fps )

    Using a K-frame lag smooths out the noise from individual frame detections.
    Note: in image coordinates, positive vy points downward.
    """
    dt_k = k / fps
    vx = []
    vy = []

    for i in range(len(x)):
        prev_i = i - k

        if prev_i < 0 or math.isnan(x[i]) or math.isnan(x[prev_i]):
            vx.append(float("nan"))
        else:
            vx.append((x[i] - x[prev_i]) / dt_k)

        if prev_i < 0 or math.isnan(y[i]) or math.isnan(y[prev_i]):
            vy.append(float("nan"))
        else:
            vy.append((y[i] - y[prev_i]) / dt_k)

    return vx, vy


def compute_absolute_speed(vx, vy):
    """
    Kinematic Feature 5: Absolute speed |v|.

    |v| = sqrt( vx^2 + vy^2 )

    Direction-independent measure of how fast the person is moving.
    """
    abs_speed = []
    for i in range(len(vx)):
        if math.isnan(vx[i]) or math.isnan(vy[i]):
            abs_speed.append(float("nan"))
        else:
            abs_speed.append(math.sqrt(vx[i]**2 + vy[i]**2))
    return abs_speed


def compute_step_displacement(x, y):
    """
    Kinematic Feature 6: Step displacement.

    s = sqrt( (x(t) - x(t-1))^2 + (y(t) - y(t-1))^2 )

    This is the raw pixel distance moved between consecutive frames.
    Unlike velocity (which uses a K-frame lag), this captures fine-grained
    movement, including short pauses or sudden changes in direction.

    The first frame always gets NaN because there is no previous frame.
    """
    step_disp = [float("nan")]  # frame 0 has no previous frame

    for i in range(1, len(x)):
        if math.isnan(x[i]) or math.isnan(y[i]) or math.isnan(x[i-1]) or math.isnan(y[i-1]):
            step_disp.append(float("nan"))
        else:
            d = math.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
            step_disp.append(d)

    return step_disp


def compute_heading_and_relative_angle(vx, vy, x, y, door_x, door_y):
    """
    Trajectory & Orientation Features 7 & 8.

    Feature 7 – Heading angle (theta):
        theta = arctan2(vy, vx)
        The compass direction the person is currently walking toward.

    Feature 8 – Relative angle to door (delta_theta):
        theta_door = arctan2(y_door - y_center, x_door - x_center)
        delta_theta = |theta - theta_door|

        Small delta_theta means the person is walking directly toward the door.
        Large delta_theta means they are walking away or perpendicular to it.

    The difference is computed using arctan2(sin, cos) to correctly handle
    the angle wrap-around (e.g., so that the difference between 170° and -170°
    comes out as 20°, not 340°).
    """
    heading   = []
    rel_angle = []

    for i in range(len(vx)):
        if math.isnan(vx[i]) or math.isnan(vy[i]):
            heading.append(float("nan"))
            rel_angle.append(float("nan"))
            continue

        theta = math.atan2(vy[i], vx[i])
        heading.append(theta)

        if math.isnan(x[i]) or math.isnan(y[i]):
            rel_angle.append(float("nan"))
        else:
            theta_door = math.atan2(door_y - y[i], door_x - x[i])
            diff = theta - theta_door
            # Wrap to [-pi, pi] before taking absolute value
            diff_wrapped = math.atan2(math.sin(diff), math.cos(diff))
            rel_angle.append(abs(diff_wrapped))

    return heading, rel_angle


def compute_bbox_features(x1, y1, x2, y2, fps, k):
    """
    Bounding Box Features 9 & 10.

    Feature 9 – Aspect ratio (lambda):
        lambda = (x2 - x1) / (y2 - y1)   i.e. width / height
        Changes in shape reflect changes in viewing angle as the person
        turns toward or away from the camera.

    Feature 10 – Scale change rate (delta_H):
        H = y2 - y1   (bounding box height in pixels)
        delta_H = ( H(t) - H(t-K) ) / ( K / fps )
        A growing box means the person is walking toward the camera.
        A shrinking box means they are walking away.
    """
    dt_k = k / fps
    aspect_ratio  = []
    scale_change  = []

    for i in range(len(x1)):
        # --- Aspect ratio ---
        if any(math.isnan(v) for v in [x1[i], y1[i], x2[i], y2[i]]):
            aspect_ratio.append(float("nan"))
        else:
            width  = x2[i] - x1[i]
            height = y2[i] - y1[i]
            if height == 0:
                aspect_ratio.append(float("nan"))  # avoid divide by zero
            else:
                aspect_ratio.append(width / height)

        # --- Scale change rate ---
        prev_i = i - k
        if prev_i < 0:
            scale_change.append(float("nan"))
        else:
            h_now  = (y2[i]       - y1[i])       if not (math.isnan(y1[i])       or math.isnan(y2[i]))       else float("nan")
            h_prev = (y2[prev_i]  - y1[prev_i])  if not (math.isnan(y1[prev_i])  or math.isnan(y2[prev_i]))  else float("nan")

            if math.isnan(h_now) or math.isnan(h_prev):
                scale_change.append(float("nan"))
            else:
                scale_change.append((h_now - h_prev) / dt_k)

    return aspect_ratio, scale_change


# Step 6 – Extract window and aggregate into [mean, variance, latest]

def get_window_values(feature_series, w_start, w_end):
    """
    Extract the values of a feature at frame indices w_start … w_end (inclusive).

    If w_start is negative (window starts before the recording began), those
    positions are padded with NaN. This keeps the window length consistent even
    for samples that are shorter than the requested window.
    """
    window = []
    n = len(feature_series)

    for i in range(w_start, w_end + 1):
        if i < 0 or i >= n:
            window.append(float("nan"))  # outside recording range
        else:
            window.append(feature_series[i])

    return window


def aggregate_window(window, latest_value):
    """
    Compress one feature's observation window into three statistics.

    Returns: (mean, variance, latest)

    mean      -- average value over the window (what is typical?)
    variance  -- spread of values over the window (is it stable or changing?)
    latest    -- value at the very last frame T_predict (what is the state right now?)

    NaN values inside the window are ignored. If all values are NaN we fall
    back to 0.0 so the output is always a valid float.
    """
    valid_values = [v for v in window if not math.isnan(v)]

    if len(valid_values) == 0:
        mean_val = 0.0
        var_val  = 0.0
    else:
        mean_val = sum(valid_values) / len(valid_values)
        # Population variance: average squared deviation from the mean
        var_val  = sum((v - mean_val)**2 for v in valid_values) / len(valid_values)

    latest_val = latest_value if not math.isnan(latest_value) else 0.0

    return mean_val, var_val, latest_val

# Main entry point

def extract_features(sample, tte_seconds=2.0, window_seconds=0.5):
    """
    Extract a 30-dimensional handcrafted feature vector from one JSON sample.

    Parameters
    ----------
    sample         : dict loaded from a trajectory JSON file
    tte_seconds    : Time-To-Event in seconds.
                     How many seconds before T_event the prediction is made.
                     Larger TTE = earlier prediction = harder task.
    window_seconds : Observation window length in seconds.
                     How much history the model gets to see.

    Returns
    -------
    A list of 30 floats structured as:
        [mean, var, latest] for each of 10 features, in this order:
         1. Distance to door
         2. Closure rate
         3. vx (x-velocity)
         4. vy (y-velocity)
         5. Absolute speed |v|
         6. Step displacement
         7. Heading angle
         8. Relative angle to door
         9. Bounding box aspect ratio
        10. Bounding box scale change rate
    """
    # ---- Read metadata ----
    fps   = float(sample.get("fps", DEFAULT_FPS) or DEFAULT_FPS)
    label = str(sample.get("label", "")).lower()

    # K = number of frames in the observation window
    k = max(2, int(round(window_seconds * fps)))

    # Door position in image coordinates
    door = sample.get("door_center")
    if door is not None and len(door) == 2:
        door_x, door_y = float(door[0]), float(door[1])
    else:
        # If not provided, assume the door is at the bottom-center of the frame
        door_x = float(sample.get("frame_width",  640)) / 2.0
        door_y = float(sample.get("frame_height", 480)) - 1.0

    # ---- Parse frame data ----
    frames = sample.get("frames", [])
    if len(frames) == 0:
        return [0.0] * 30  # nothing to work with

    x, y, x1, y1, x2, y2 = get_coordinates(frames)
    n = len(x)

    # ---- Locate T_event and T_predict ----
    t_event = find_event_frame(label, x, y, door_x, door_y)
    if t_event < 0:
        return [0.0] * 30  # no valid frames found

    tte_frames = int(round(tte_seconds * fps))
    t_predict  = t_event - tte_frames
    t_predict  = max(0, min(t_predict, n - 1))  # keep within valid range

    # ---- Define the observation window ----
    # K frames ending at T_predict
    w_start = t_predict - k + 1
    w_end   = t_predict

    # ---- Compute all 10 feature time-series ----
    d_door       = compute_distance_to_door(x, y, door_x, door_y)
    closure_rate = compute_closure_rate(d_door, fps, k)
    vx, vy       = compute_velocity(x, y, fps, k)
    abs_speed    = compute_absolute_speed(vx, vy)
    step_disp    = compute_step_displacement(x, y)
    heading, rel_angle = compute_heading_and_relative_angle(vx, vy, x, y, door_x, door_y)
    aspect_ratio, scale_change = compute_bbox_features(x1, y1, x2, y2, fps, k)

    # The order here determines which columns are which in the feature matrix.
    all_features = [
        d_door,
        closure_rate,
        vx,
        vy,
        abs_speed,
        step_disp,
        heading,
        rel_angle,
        aspect_ratio,
        scale_change,
    ]

    # ---- Aggregate each series into [mean, variance, latest] ----
    feature_vector = []

    for series in all_features:
        window      = get_window_values(series, w_start, w_end)
        latest      = series[t_predict] if 0 <= t_predict < len(series) else float("nan")
        m, v, l     = aggregate_window(window, latest)
        feature_vector.extend([m, v, l])

    return feature_vector  # length = 10 features × 3 statistics = 30


def is_sample_usable(sample, tte_seconds, window_seconds=0.5):
    """
    Check whether a sample has enough frames to extract features correctly.

    The core requirement is that T_predict = T_event - tte_frames must be >= 0.
    If this does not hold, T_predict would be clipped to frame 0, meaning
    features would be extracted from the wrong part of the trajectory and
    would NOT represent the intended prediction time. Such samples must be
    excluded from the dataset.

    Returns True if the sample is long enough, False otherwise.
    """
    fps = float(sample.get("fps", DEFAULT_FPS) or DEFAULT_FPS)

    frames = sample.get("frames", [])
    if not frames:
        return False

    label = str(sample.get("label", "")).lower()
    if label not in ("enter", "pass"):
        return False

    k          = max(2, int(round(window_seconds * fps)))
    tte_frames = int(round(tte_seconds * fps))

    # Re-derive door position exactly as extract_features does
    door = sample.get("door_center")
    if door is not None and len(door) == 2:
        door_x, door_y = float(door[0]), float(door[1])
    else:
        door_x = float(sample.get("frame_width",  640)) / 2.0
        door_y = float(sample.get("frame_height", 480)) - 1.0

    x, y, *_ = get_coordinates(frames)
    t_event   = find_event_frame(label, x, y, door_x, door_y)

    if t_event < 0:
        return False

    # t_predict must be >= 0 (TTE fits) and the observation window should
    # start at or after frame 0 (full window available without NaN padding)
    t_predict = t_event - tte_frames
    return t_predict >= k - 1


def get_feature_families():
    """
    Return a mapping from feature family name to the list of feature indices
    in the 30-dimensional vector produced by extract_features().

    The four families match the categories described in the Method section:

        Goal-Oriented        – features that measure proximity to the door
        Kinematic            – features that describe speed and motion magnitude
        Trajectory & Orient. – features that describe direction of movement
        Bounding Box         – features derived from the detected bounding box

    Each base feature contributes 3 entries (mean, var, latest), so the
    total number of indices per family is 3 × (number of base features).

    Example usage
    -------------
    from trajectory_feature_extractor import get_feature_families
    import numpy as np

    families  = get_feature_families()
    family_importance = {
        name: np.mean(perm_means[indices])
        for name, indices in families.items()
    }
    """
    # The base features are ordered exactly as in get_feature_names():
    #   0: dist_to_door   (indices 0, 1, 2)
    #   1: closure_rate   (indices 3, 4, 5)
    #   2: vx             (indices 6, 7, 8)
    #   3: vy             (indices 9, 10, 11)
    #   4: abs_speed      (indices 12, 13, 14)
    #   5: step_displace. (indices 15, 16, 17)
    #   6: heading_angle  (indices 18, 19, 20)
    #   7: rel_angle_door (indices 21, 22, 23)
    #   8: aspect_ratio   (indices 24, 25, 26)
    #   9: scale_change   (indices 27, 28, 29)

    def _indices(base_feature_numbers):
        """Convert base-feature numbers (0-9) to the 3 flat indices each occupies."""
        result = []
        for n in base_feature_numbers:
            start = n * 3
            result.extend([start, start + 1, start + 2])
        return result

    return {
        "Goal-Oriented":        _indices([0, 1]),        # dist_to_door, closure_rate
        "Kinematic":            _indices([2, 3, 4, 5]),  # vx, vy, abs_speed, step_displacement
        "Trajectory & Orient.": _indices([6, 7]),        # heading_angle, rel_angle_to_door
        "Bounding Box":         _indices([8, 9]),         # aspect_ratio, scale_change_rate
    }


def get_feature_names():
    """
    Return the names of the 30 features in the same order as extract_features().

    The naming convention is:  <feature>_<statistic>
    where statistic is one of:
        mean   -- average value over the observation window
        var    -- variance over the observation window
        latest -- value at T_predict (last frame of the window)

    Use this whenever you need to label columns, e.g. for a Random Forest
    feature importance plot or a pandas DataFrame.

    Example
    -------
    import pandas as pd
    from trajectory_feature_extractor import extract_features, get_feature_names

    X = np.vstack([extract_features(s) for s in samples])
    df = pd.DataFrame(X, columns=get_feature_names())
    """
    base_features = [
        "dist_to_door",       # Euclidean distance to door center
        "closure_rate",       # rate of change of distance to door (negative = approaching)
        "vx",                 # horizontal velocity component
        "vy",                 # vertical velocity component
        "abs_speed",          # scalar speed |v|
        "step_displacement",  # frame-to-frame displacement
        "heading_angle",      # direction of motion (radians)
        "rel_angle_to_door",  # angular difference between heading and door direction
        "aspect_ratio",       # bounding box width / height
        "scale_change_rate",  # rate of change of bounding box height
    ]

    stats = ["mean", "var", "latest"]

    # Build the full list: [dist_to_door_mean, dist_to_door_var, dist_to_door_latest, ...]
    names = []
    for feature in base_features:
        for stat in stats:
            names.append(f"{feature}_{stat}")

    return names  # always length 30
