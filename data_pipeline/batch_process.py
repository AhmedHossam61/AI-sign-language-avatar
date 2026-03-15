"""Batch build Phase 1 motion library from downloaded WLASL videos.

Input:
- WLASL_v0.3.json index
- start_kit/raw_videos/*.mp4 downloaded IDs

Output:
- backend/motion_library/<GLOSS>.json (best clip per gloss)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mediapipe_extractor import extract_clip
from normalizer import normalize_and_smooth_clip


@dataclass
class Candidate:
    gloss: str
    video_id: str
    source: str
    frame_start: int
    frame_end: int
    video_path: Path


def _sanitize_gloss(gloss: str) -> str:
    token = re.sub(r"[^A-Z0-9]+", "_", gloss.upper()).strip("_")
    return token or "UNKNOWN"


def _load_index(index_path: Path) -> list[dict[str, Any]]:
    with index_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_candidates(index_data: list[dict[str, Any]], raw_videos_dir: Path) -> dict[str, list[Candidate]]:
    by_gloss: dict[str, list[Candidate]] = {}
    for entry in index_data:
        gloss = str(entry.get("gloss", "")).strip().upper()
        if not gloss:
            continue

        for instance in entry.get("instances", []):
            video_id = str(instance.get("video_id", "")).strip()
            if not video_id:
                continue

            video_path = raw_videos_dir / f"{video_id}.mp4"
            if not video_path.exists():
                continue

            candidate = Candidate(
                gloss=gloss,
                video_id=video_id,
                source=str(instance.get("source", "WLASL")),
                frame_start=int(instance.get("frame_start", 1) or 1),
                frame_end=int(instance.get("frame_end", -1) or -1),
                video_path=video_path,
            )
            by_gloss.setdefault(gloss, []).append(candidate)

    return by_gloss


def _score_clip(clip: dict[str, Any]) -> float:
    frames = max(int(clip.get("num_frames", 0)), 1)
    quality = clip.get("quality", {})
    pose_ratio = float(quality.get("pose_frame_ratio", 0.0))
    hand_ratio = float(quality.get("hand_frame_ratio", 0.0))
    return (0.6 * pose_ratio + 0.4 * hand_ratio) + min(frames / 200.0, 1.0)


def build_motion_library(
    index_path: Path,
    raw_videos_dir: Path,
    out_dir: Path,
    max_per_gloss: int = 5,
    min_frames: int = 8,
    smooth_window: int = 5,
    smooth_poly: int = 2,
    dry_run: bool = False,
) -> dict[str, Any]:
    index_data = _load_index(index_path)
    candidates_by_gloss = _build_candidates(index_data, raw_videos_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "gloss_total": len(candidates_by_gloss),
        "gloss_written": 0,
        "gloss_skipped": [],
        "written_files": [],
        "extraction_error_count": 0,
        "extraction_error_examples": [],
    }

    for gloss, candidates in sorted(candidates_by_gloss.items()):
        best_clip: dict[str, Any] | None = None
        best_score = -1.0

        for candidate in candidates[:max_per_gloss]:
            try:
                raw_clip = extract_clip(
                    video_path=candidate.video_path,
                    gloss=gloss,
                    source=f"WLASL-{candidate.source}",
                    video_id=candidate.video_id,
                    frame_start=candidate.frame_start,
                    frame_end=candidate.frame_end,
                )
            except Exception as exc:
                report["extraction_error_count"] += 1
                if len(report["extraction_error_examples"]) < 10:
                    report["extraction_error_examples"].append(
                        {
                            "gloss": gloss,
                            "video_id": candidate.video_id,
                            "error": str(exc),
                        }
                    )
                continue

            if int(raw_clip.get("num_frames", 0)) < min_frames:
                continue

            clip = normalize_and_smooth_clip(raw_clip, window=smooth_window, poly=smooth_poly)
            score = _score_clip(clip)
            if score > best_score:
                best_score = score
                best_clip = clip

        if best_clip is None:
            report["gloss_skipped"].append(gloss)
            continue

        gloss_file = f"{_sanitize_gloss(gloss)}.json"
        output_path = out_dir / gloss_file
        if not dry_run:
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(best_clip, f, ensure_ascii=True)

        report["gloss_written"] += 1
        report["written_files"].append(str(output_path))

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build motion library from downloaded WLASL videos")
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("WLASL-master/start_kit/WLASL_v0.3.json"),
        help="Path to WLASL_v0.3.json",
    )
    parser.add_argument(
        "--raw-videos",
        type=Path,
        default=Path("WLASL-master/start_kit/raw_videos"),
        help="Directory containing downloaded *.mp4 videos",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("backend/motion_library"),
        help="Output motion library directory",
    )
    parser.add_argument("--max-per-gloss", type=int, default=5)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--smooth-poly", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("backend/motion_library/report.json"),
        help="Output report path",
    )
    args = parser.parse_args()

    report = build_motion_library(
        index_path=args.index,
        raw_videos_dir=args.raw_videos,
        out_dir=args.out_dir,
        max_per_gloss=args.max_per_gloss,
        min_frames=args.min_frames,
        smooth_window=args.smooth_window,
        smooth_poly=args.smooth_poly,
        dry_run=args.dry_run,
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)

    print(
        f"Done. gloss_total={report['gloss_total']} "
        f"gloss_written={report['gloss_written']} skipped={len(report['gloss_skipped'])}"
    )
    if report["extraction_error_count"]:
        print(
            "Extraction errors: "
            f"{report['extraction_error_count']} (see extraction_error_examples in report)"
        )
    print(f"Report: {args.report}")

    if report["gloss_written"] == 0 and report["extraction_error_count"] > 0:
        raise SystemExit(
            "No clips were written because extraction failed for all tried candidates. "
            "Fix the environment/runtime error shown in report.json and rerun."
        )


if __name__ == "__main__":
    main()
