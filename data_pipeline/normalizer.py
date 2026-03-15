"""Normalize and smooth extracted skeleton clips.

Normalization centers landmarks around the hip midpoint and scales by body height.
Smoothing uses Savitzky-Golay where possible, with safe fallback behavior.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

POSE_JOINTS = 33
HAND_JOINTS = 21


def _frame_center_and_scale(frame: dict[str, Any]) -> tuple[np.ndarray, float]:
    pose = frame.get("pose")
    if not pose or len(pose) < 25:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32), 1.0

    left_hip = np.array(pose[23], dtype=np.float32)
    right_hip = np.array(pose[24], dtype=np.float32)
    center = (left_hip + right_hip) / 2.0

    pose_arr = np.array(pose, dtype=np.float32)
    y_min = float(np.nanmin(pose_arr[:, 1]))
    y_max = float(np.nanmax(pose_arr[:, 1]))
    scale = max(y_max - y_min, 1e-6)
    return center, scale


def normalize_clip(clip: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(clip)
    normalized_frames: list[dict[str, Any]] = []

    for frame in out.get("frames", []):
        center, scale = _frame_center_and_scale(frame)
        norm_frame: dict[str, Any] = {}

        for key in ("pose", "left_hand", "right_hand"):
            lms = frame.get(key)
            if lms is None:
                norm_frame[key] = None
                continue

            arr = np.array(lms, dtype=np.float32)
            arr = (arr - center) / scale
            norm_frame[key] = arr.tolist()

        normalized_frames.append(norm_frame)

    out["frames"] = normalized_frames
    return out


def _interpolate_nans(values: np.ndarray) -> np.ndarray:
    idx = np.arange(values.size)
    valid = ~np.isnan(values)
    if not np.any(valid):
        return np.zeros_like(values)
    if np.sum(valid) == 1:
        return np.full_like(values, values[valid][0])
    return np.interp(idx, idx[valid], values[valid])


def _smooth_landmark_track(track: np.ndarray, window: int, poly: int) -> np.ndarray:
    out = track.copy()
    t, joints, axes = out.shape
    if t < 3:
        return out

    if window > t:
        window = t if t % 2 == 1 else t - 1
    if window < 3:
        return out
    if poly >= window:
        poly = max(1, window - 1)

    for j in range(joints):
        for a in range(axes):
            series = _interpolate_nans(out[:, j, a])
            out[:, j, a] = savgol_filter(series, window_length=window, polyorder=poly)
    return out


def _frames_to_track(
    frames: list[dict[str, Any]], key: str, joints: int
) -> tuple[np.ndarray, np.ndarray]:
    t = len(frames)
    track = np.full((t, joints, 3), np.nan, dtype=np.float32)
    mask = np.zeros((t,), dtype=bool)

    for i, frame in enumerate(frames):
        lms = frame.get(key)
        if lms is None:
            continue
        arr = np.array(lms, dtype=np.float32)
        if arr.shape == (joints, 3):
            track[i] = arr
            mask[i] = True

    return track, mask


def smooth_clip(clip: dict[str, Any], window: int = 5, poly: int = 2) -> dict[str, Any]:
    out = deepcopy(clip)
    frames = out.get("frames", [])
    if not frames:
        return out

    specs = {
        "pose": POSE_JOINTS,
        "left_hand": HAND_JOINTS,
        "right_hand": HAND_JOINTS,
    }

    smoothed: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for key, joints in specs.items():
        track, mask = _frames_to_track(frames, key, joints)
        smoothed[key] = _smooth_landmark_track(track, window=window, poly=poly)
        masks[key] = mask

    new_frames: list[dict[str, Any]] = []
    for i in range(len(frames)):
        item: dict[str, Any] = {}
        for key, _ in specs.items():
            if not masks[key][i]:
                item[key] = None
            else:
                item[key] = smoothed[key][i].tolist()
        new_frames.append(item)

    out["frames"] = new_frames
    return out


def normalize_and_smooth_clip(
    clip: dict[str, Any], window: int = 5, poly: int = 2
) -> dict[str, Any]:
    return smooth_clip(normalize_clip(clip), window=window, poly=poly)
