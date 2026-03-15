"""Video to skeleton clip extraction using MediaPipe Tasks API (mediapipe>=0.10).

This module is Phase 1 (post-download) groundwork: given a signer video,
extract per-frame pose and hand landmarks and serialize to JSON.

Uses PoseLandmarker + HandLandmarker (VIDEO mode) instead of the deprecated
Holistic solution that was removed in mediapipe 0.10.x.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
from mediapipe.tasks.python.core import base_options as _base_options_lib
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode as _RunningMode

_BaseOptions = _base_options_lib.BaseOptions

# Model bundle files are downloaded once and cached in data_pipeline/models/.
_MODEL_DIR = Path(__file__).parent / "models"
_POSE_MODEL_PATH = _MODEL_DIR / "pose_landmarker_lite.task"
_HAND_MODEL_PATH = _MODEL_DIR / "hand_landmarker.task"
_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def _ensure_models() -> None:
    """Download MediaPipe model bundles if not already present."""
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for url, path in [(_POSE_MODEL_URL, _POSE_MODEL_PATH), (_HAND_MODEL_URL, _HAND_MODEL_PATH)]:
        if not path.exists():
            print(f"Downloading {path.name} …", flush=True)
            urllib.request.urlretrieve(url, path)
            print(f"  Saved → {path}", flush=True)


def _lm_list_to_xyz(landmarks: list[Any]) -> list[list[float]]:
    return [[float(lm.x), float(lm.y), float(lm.z)] for lm in landmarks]


def extract_landmarks_from_results(
    pose_result: Any,
    hand_result: Any,
) -> dict[str, list[list[float]] | None]:
    pose: list[list[float]] | None = None
    if pose_result.pose_landmarks:
        pose = _lm_list_to_xyz(pose_result.pose_landmarks[0])

    left_hand: list[list[float]] | None = None
    right_hand: list[list[float]] | None = None
    for i, handedness_list in enumerate(hand_result.handedness):
        if not handedness_list:
            continue
        label = handedness_list[0].category_name  # "Left" or "Right"
        lms = _lm_list_to_xyz(hand_result.hand_landmarks[i])
        if label == "Left":
            left_hand = lms
        else:
            right_hand = lms

    return {"pose": pose, "left_hand": left_hand, "right_hand": right_hand}


def extract_clip(
    video_path: Path,
    gloss: str,
    source: str = "WLASL-v0.3",
    video_id: str | None = None,
    frame_start: int = 1,
    frame_end: int = -1,
    model_complexity: int = 1,  # kept for API compatibility; lite model always used
) -> dict[str, Any]:
    _ensure_models()

    pose_options = PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=str(_POSE_MODEL_PATH)),
        running_mode=_RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_options = HandLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=str(_HAND_MODEL_PATH)),
        running_mode=_RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    start_idx = max(frame_start - 1, 0)
    end_idx = frame_end - 1 if frame_end and frame_end > 0 else -1
    frames: list[dict[str, Any]] = []

    with (
        PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
        HandLandmarker.create_from_options(hand_options) as hand_landmarker,
    ):
        frame_idx = -1
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            if frame_idx < start_idx:
                continue
            if end_idx >= 0 and frame_idx > end_idx:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000 / fps)

            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            frames.append(extract_landmarks_from_results(pose_result, hand_result))

    cap.release()

    pose_detected = sum(1 for f in frames if f["pose"] is not None)
    hand_detected = sum(
        1
        for f in frames
        if f["left_hand"] is not None or f["right_hand"] is not None
    )

    return {
        "gloss": gloss.upper(),
        "source": source,
        "video_id": video_id or video_path.stem,
        "fps": fps,
        "num_frames": len(frames),
        "frames": frames,
        "quality": {
            "pose_frame_ratio": (pose_detected / len(frames)) if frames else 0.0,
            "hand_frame_ratio": (hand_detected / len(frames)) if frames else 0.0,
        },
    }


def save_clip(clip: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(clip, f, ensure_ascii=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract skeleton clip from one video")
    parser.add_argument("--video", required=True, type=Path, help="Input .mp4 video")
    parser.add_argument("--out", required=True, type=Path, help="Output .json path")
    parser.add_argument("--gloss", required=True, help="Gloss token for this clip")
    parser.add_argument("--source", default="WLASL-v0.3")
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int, default=-1)
    parser.add_argument("--model-complexity", type=int, choices=[0, 1, 2], default=1)
    args = parser.parse_args()

    clip = extract_clip(
        video_path=args.video,
        gloss=args.gloss,
        source=args.source,
        video_id=args.video_id,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        model_complexity=args.model_complexity,
    )
    save_clip(clip, args.out)

    print(
        f"Saved {args.out} | frames={clip['num_frames']} "
        f"pose_ratio={clip['quality']['pose_frame_ratio']:.2f} "
        f"hand_ratio={clip['quality']['hand_frame_ratio']:.2f}"
    )


if __name__ == "__main__":
    main()
