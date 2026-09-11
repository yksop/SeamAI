"""
Feature computation for Core-3 + head (raw) model.

Features per frame (7 total):
  Trajectory (3): dist_to_door, closure_rate, rel_angle_to_door
  Pose (4):       left_ear_x, left_ear_y, right_ear_x, right_ear_y

All spatial values are normalised to frame dimensions so that they are
resolution-independent (matching the training data convention).
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# COCO-pose keypoint indices (YOLOv8-pose / YOLOv11-pose)
# ---------------------------------------------------------------------------
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4


@dataclass
class PersonState:
    """Tracks per-frame state for a single person (track_id)."""
    track_id: int
    # Ring buffer of recent feature vectors (each shape (7,))
    history: List[np.ndarray] = field(default_factory=list)
    # Previous distance to door (for closure_rate)
    prev_dist: Optional[float] = None
    # Previous center (for velocity-based angle)
    prev_center: Optional[Tuple[float, float]] = None
    # Raw latest values (for visualisation / decision logic)
    last_dist: float = 0.0
    last_dist_norm: float = 0.0   # diagonal-normalised, for decision threshold
    last_closure: float = 0.0
    last_angle: float = 0.0
    last_ear_l: Tuple[float, float] = (0.0, 0.0)
    last_ear_r: Tuple[float, float] = (0.0, 0.0)
    last_center: Tuple[float, float] = (0.0, 0.0)


class FeatureComputer:
    """
    Computes the 7 Core-3+head features from YOLO-Pose detections.

    Parameters
    ----------
    door_center : tuple (x, y) in pixel coordinates
    frame_w, frame_h : frame dimensions (for normalisation)
    seq_len : number of frames kept per person (GRU window)
    norm_mean, norm_std : arrays of shape (7,) for z-score normalisation.
        If None, raw (resolution-normalised) features are returned.
    """

    def __init__(
        self,
        door_center: Tuple[int, int],
        frame_w: int,
        frame_h: int,
        seq_len: int = 7,
        fps: float = 13.0,
        norm_mean: Optional[np.ndarray] = None,
        norm_std: Optional[np.ndarray] = None,
    ):
        self.door_x, self.door_y = door_center
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.seq_len = seq_len
        self.fps = fps
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        # Diagonal for distance normalisation (consistent with training)
        self.diag = math.sqrt(frame_w ** 2 + frame_h ** 2)

        # Active person states keyed by track_id
        self.persons: dict[int, PersonState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        keypoints: np.ndarray,
    ) -> PersonState:
        """
        Update features for one person in the current frame.

        Parameters
        ----------
        track_id : int
        bbox : (x1, y1, x2, y2)
        keypoints : array of shape (17, 3) — x, y, confidence per keypoint

        Returns
        -------
        PersonState with updated history.
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # --- Trajectory features ---
        dist = self._dist_to_door(cx, cy)
        state = self.persons.get(track_id)

        if state is None:
            state = PersonState(track_id=track_id)
            self.persons[track_id] = state

        closure = self._closure_rate(dist, state.prev_dist)
        angle = self._rel_angle(cx, cy, state.prev_center)

        # --- Pose features (ear keypoints) ---
        le_x, le_y, le_conf = keypoints[KP_LEFT_EAR]
        re_x, re_y, re_conf = keypoints[KP_RIGHT_EAR]

        # Normalise to [0, 1]
        le_x_n = le_x / self.frame_w
        le_y_n = le_y / self.frame_h
        re_x_n = re_x / self.frame_w
        re_y_n = re_y / self.frame_h

        # If keypoint confidence is very low, use bbox center as fallback
        if le_conf < 0.3:
            le_x_n = cx / self.frame_w
            le_y_n = cy / self.frame_h
        if re_conf < 0.3:
            re_x_n = cx / self.frame_w
            re_y_n = cy / self.frame_h

        # --- Assemble feature vector ---
        feat = np.array([
            dist,       # dist_to_door (normalised by diagonal)
            closure,    # closure_rate (normalised by diagonal)
            angle,      # rel_angle_to_door (radians / pi → [-1, 1])
            le_x_n,     # left_ear_x  (normalised 0-1)
            le_y_n,     # left_ear_y  (normalised 0-1)
            re_x_n,     # right_ear_x (normalised 0-1)
            re_y_n,     # right_ear_y (normalised 0-1)
        ], dtype=np.float32)

        # --- Update state ---
        state.prev_dist = dist
        state.prev_center = (cx, cy)
        state.last_dist = dist
        state.last_dist_norm = self._dist_to_door_norm(cx, cy)
        state.last_closure = closure
        state.last_angle = angle
        state.last_ear_l = (le_x, le_y)
        state.last_ear_r = (re_x, re_y)
        state.last_center = (cx, cy)

        state.history.append(feat)
        if len(state.history) > self.seq_len:
            state.history = state.history[-self.seq_len:]

        return state

    def get_sequence(self, track_id: int) -> Optional[np.ndarray]:
        """
        Return the z-score normalised sequence for a person, ready for
        the GRU model. Shape: (1, seq_len, 7).

        Left-pads with zeros if fewer than seq_len frames are available.
        Applies causal moving-average smoothing to pose columns (matching
        the training pipeline) before z-score normalisation.
        Returns None if the track_id is unknown.
        """
        state = self.persons.get(track_id)
        if state is None or len(state.history) == 0:
            return None

        n = len(state.history)
        seq = np.zeros((self.seq_len, 7), dtype=np.float32)

        if n >= self.seq_len:
            seq[:] = np.array(state.history[-self.seq_len:])
        else:
            seq[self.seq_len - n:] = np.array(state.history)

        # Causal moving-average smoothing on pose columns (cols 3-6)
        # to match train_demo_model.py smooth_pose_features(n_traj=3, kernel_size=3)
        seq = self._smooth_pose(seq, n_traj=3, kernel_size=3)

        # Z-score normalisation
        if self.norm_mean is not None and self.norm_std is not None:
            seq = (seq - self.norm_mean) / (self.norm_std + 1e-8)
            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

        return seq[np.newaxis, ...]  # (1, seq_len, 7)

    def remove(self, track_id: int):
        """Remove a person who is no longer tracked."""
        self.persons.pop(track_id, None)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _dist_to_door(self, cx: float, cy: float) -> float:
        """Raw pixel distance (matches training convention)."""
        dx = cx - self.door_x
        dy = cy - self.door_y
        return math.sqrt(dx * dx + dy * dy)

    def _dist_to_door_norm(self, cx: float, cy: float) -> float:
        """Diagonal-normalised distance for decision logic only."""
        return self._dist_to_door(cx, cy) / self.diag

    def _closure_rate(self, dist_now: float, dist_prev: Optional[float]) -> float:
        """(dist_now - dist_prev) * fps — negative means approaching."""
        if dist_prev is None:
            return 0.0
        return (dist_now - dist_prev) * self.fps

    def _rel_angle(
        self, cx: float, cy: float, prev_center: Optional[Tuple[float, float]]
    ) -> float:
        """Absolute angle between movement and door direction, in radians [0, pi]."""
        if prev_center is None:
            return 0.0

        mvx = cx - prev_center[0]
        mvy = cy - prev_center[1]
        mv_len = math.sqrt(mvx * mvx + mvy * mvy)
        if mv_len < 1e-6:
            return 0.0

        dvx = self.door_x - cx
        dvy = self.door_y - cy
        dv_len = math.sqrt(dvx * dvx + dvy * dvy)
        if dv_len < 1e-6:
            return 0.0

        cos_a = (mvx * dvx + mvy * dvy) / (mv_len * dv_len)
        cos_a = max(-1.0, min(1.0, cos_a))
        return math.acos(cos_a)

    @staticmethod
    def _smooth_pose(seq: np.ndarray, n_traj: int = 3,
                     kernel_size: int = 3) -> np.ndarray:
        """Causal moving average on pose columns, matching training."""
        if seq.shape[1] <= n_traj or seq.shape[0] < 2 or kernel_size < 2:
            return seq
        smoothed = seq.copy()
        for col in range(n_traj, seq.shape[1]):
            for t in range(seq.shape[0]):
                start = max(0, t - kernel_size + 1)
                smoothed[t, col] = seq[start: t + 1, col].mean()
        return smoothed
