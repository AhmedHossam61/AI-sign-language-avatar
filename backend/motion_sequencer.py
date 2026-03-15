"""Motion sequencer — stitches ASL gloss clips into a single animation.

Usage:
    from motion_sequencer import sequence_glosses

    animation = sequence_glosses(["HELLO", "HOW", "YOU"])
    # animation = {"fps": 30, "num_frames": N, "frames": [...]}

Unknown glosses fall back to per-letter fingerspelling if fingerspell clips
are available, otherwise the token is silently skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Paths (resolved relative to this file's location).
_BACKEND_DIR = Path(__file__).parent
MOTION_LIBRARY_DIR = _BACKEND_DIR / "motion_library"
FINGERSPELL_DIR = _BACKEND_DIR / "fingerspell"

# Number of frames used for the cross-fade between consecutive clips.
BLEND_FRAMES = 5

# Neutral/rest pose returned when a clip has no usable frames.
# 33 pose joints + 21 left-hand + 21 right-hand, all zeroed.
_ZERO_FRAME: dict[str, Any] = {
    "pose": [[0.0, 0.0, 0.0]] * 33,
    "left_hand": None,
    "right_hand": None,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_clip(path: Path) -> dict[str, Any] | None:
    """Load a clip JSON from disk, returning None on any error."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _get_clip(gloss: str) -> dict[str, Any] | None:
    """Look up a gloss in the motion library. Returns None if not found."""
    path = MOTION_LIBRARY_DIR / f"{gloss.upper()}.json"
    return _load_clip(path) if path.exists() else None


def _get_fingerspell_clips(letter: str) -> dict[str, Any] | None:
    """Return the fingerspell clip for a single letter, or None."""
    path = FINGERSPELL_DIR / f"{letter.upper()}.json"
    return _load_clip(path) if path.exists() else None


def _fingerspell_token(token: str) -> list[dict[str, Any]]:
    """Expand a token into per-letter fingerspell frames.

    Returns an empty list if no fingerspell clips are available yet.
    """
    frames: list[dict[str, Any]] = []
    for letter in token:
        if not letter.isalpha():
            continue
        clip = _get_fingerspell_clips(letter)
        if clip and clip.get("frames"):
            frames.extend(clip["frames"])
    return frames


def _frame_to_array(frame: dict[str, Any], n_pose: int = 33, n_hand: int = 21) -> np.ndarray:
    """Flatten a frame dict to a 1-D numpy array for interpolation."""
    pose = np.array(frame.get("pose") or [[0.0, 0.0, 0.0]] * n_pose, dtype=float)
    lh = np.array(frame.get("left_hand") or [[0.0, 0.0, 0.0]] * n_hand, dtype=float)
    rh = np.array(frame.get("right_hand") or [[0.0, 0.0, 0.0]] * n_hand, dtype=float)
    return np.concatenate([pose.ravel(), lh.ravel(), rh.ravel()])


def _array_to_frame(
    arr: np.ndarray,
    had_left: bool,
    had_right: bool,
    n_pose: int = 33,
    n_hand: int = 21,
) -> dict[str, Any]:
    """Rebuild a frame dict from a 1-D numpy array."""
    p = n_pose * 3
    h = n_hand * 3
    pose = arr[:p].reshape(n_pose, 3).tolist()
    lh_arr = arr[p : p + h].reshape(n_hand, 3)
    rh_arr = arr[p + h :].reshape(n_hand, 3)
    return {
        "pose": pose,
        "left_hand": lh_arr.tolist() if had_left else None,
        "right_hand": rh_arr.tolist() if had_right else None,
    }


def _blend_transition(
    frame_a: dict[str, Any],
    frame_b: dict[str, Any],
    n: int = BLEND_FRAMES,
) -> list[dict[str, Any]]:
    """Generate `n` linearly-interpolated frames between frame_a and frame_b."""
    arr_a = _frame_to_array(frame_a)
    arr_b = _frame_to_array(frame_b)
    had_left = frame_a.get("left_hand") is not None or frame_b.get("left_hand") is not None
    had_right = frame_a.get("right_hand") is not None or frame_b.get("right_hand") is not None
    blended = []
    for i in range(1, n + 1):
        alpha = i / (n + 1)
        arr = (1.0 - alpha) * arr_a + alpha * arr_b
        blended.append(_array_to_frame(arr, had_left, had_right))
    return blended


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sequence_glosses(
    gloss_tokens: list[str],
    fps: float = 30.0,
    blend_frames: int = BLEND_FRAMES,
    skip_unknown: bool = False,
) -> dict[str, Any]:
    """Stitch a list of ASL gloss tokens into a single animation dict.

    Args:
        gloss_tokens: Uppercase gloss tokens, e.g. ["HELLO", "HOW", "YOU"].
        fps: Output frame rate (defaults to 30).
        blend_frames: Number of interpolation frames inserted between clips.
        skip_unknown: If True, tokens without a clip AND without fingerspell
            clips are silently dropped. If False (default), unknown tokens are
            included as a flat zero-pose pause (1 frame).

    Returns:
        Dict with keys: fps, num_frames, frames, glosses_used, glosses_unknown.
    """
    all_frames: list[dict[str, Any]] = []
    last_frame: dict[str, Any] | None = None
    glosses_used: list[str] = []
    glosses_unknown: list[str] = []

    for token in gloss_tokens:
        clip = _get_clip(token)

        if clip is not None and clip.get("frames"):
            clip_frames = clip["frames"]
        else:
            # Fallback: try fingerspelling letter-by-letter.
            clip_frames = _fingerspell_token(token)
            if not clip_frames:
                glosses_unknown.append(token)
                if not skip_unknown:
                    # Insert a brief pause so the sequencer doesn't silently skip.
                    clip_frames = [_ZERO_FRAME]
                else:
                    continue

        glosses_used.append(token)

        if last_frame is not None and blend_frames > 0:
            all_frames.extend(_blend_transition(last_frame, clip_frames[0], n=blend_frames))

        all_frames.extend(clip_frames)
        last_frame = clip_frames[-1]

    return {
        "fps": fps,
        "num_frames": len(all_frames),
        "frames": all_frames,
        "glosses_used": glosses_used,
        "glosses_unknown": glosses_unknown,
    }


def sequence_text(
    sentence: str,
    fps: float = 30.0,
    blend_frames: int = BLEND_FRAMES,
    skip_unknown: bool = False,
) -> dict[str, Any]:
    """Convenience wrapper: English sentence → animation dict.

    Calls the gloss converter internally.
    """
    from gloss_converter import text_to_gloss  # local import to keep modules loosely coupled

    gloss_tokens = text_to_gloss(sentence)
    result = sequence_glosses(gloss_tokens, fps=fps, blend_frames=blend_frames, skip_unknown=skip_unknown)
    result["input_text"] = sentence
    result["gloss_tokens"] = gloss_tokens
    return result


# ---------------------------------------------------------------------------
# CLI for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sentence = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello how are you today"
    print(f"Input:   {sentence!r}")

    result = sequence_text(sentence)
    print(f"Gloss:   {result['gloss_tokens']}")
    print(f"Used:    {result['glosses_used']}")
    print(f"Unknown: {result['glosses_unknown']}")
    print(f"Frames:  {result['num_frames']}  (fps={result['fps']})")
    duration = result["num_frames"] / result["fps"]
    print(f"Duration: {duration:.1f}s")
