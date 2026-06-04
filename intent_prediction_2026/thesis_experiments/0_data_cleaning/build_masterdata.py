#!/usr/bin/env python3
from __future__ import annotations

"""
build_masterdata.py

Create a cleaned dataset in:
  MasterData/

from raw recordings in:
  DataOriginal/  (or override with --src)

Cleaning rules:
1) Remove pairs where video is shorter than 0.5 seconds.
2) Remove pairs with corrupted mp4 files.
3) Remove pairs that are likely "exit" behavior:
   - first appears in lower part of the image
   - moves away from the door/ellipse over time
4) Remove pairs where only a partial head/body is visible near the camera bottom.
5) Remove duplicate pairs globally across all source folders:
   - duplicate pair = same mp4 content (SHA1) + same tracking signature
     (hash of label + frames trajectory)

Output folder structure:
  MasterData/
    Corrup/
    enter/
      exit/
      removed/
    pass/
      exit/
      remove/

Notes:
- The script keeps source data unchanged and copies pairs into MasterData.
- JSON output keeps the original frame-level data and ensures metadata keys exist:
  id, label, frames, fps, frame_width, frame_height, door_center, ellipse_axes
- For thesis/report readability, each JSON also gets:
  cleaning_result, cleaning_reason
- Duplicate handling is deterministic: first occurrence is kept, later matches are removed.
- A duplicate-report CSV (duplicate_report.csv) is written inside MasterData/.
"""

import argparse
import csv
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2


VALID_SPLITS = {"enter", "pass"}


@dataclass(frozen=True)
class CleanerConfig:
    min_duration_s: float = 0.5
    # Exit heuristic
    exit_start_y_min_norm: float = 0.68
    exit_min_delta_dist_norm: float = 0.10
    exit_min_end_farther_ratio: float = 1.10
    # "Head at bottom" / near-camera heuristic
    bottom_presence_y_min_norm: float = 0.72
    bottom_presence_start_y_min_norm: float = 0.82
    bottom_presence_max_net_disp_norm: float = 0.18
    bottom_presence_max_y_span_norm: float = 0.26
    bottom_presence_min_bottom_touch_ratio: float = 0.45
    # Geometry defaults from the data-collection setup
    ellipse_a: int = 480
    ellipse_b: int = 130


@dataclass(frozen=True)
class PairPaths:
    split: str
    json_path: Path
    mp4_path: Path


@dataclass(frozen=True)
class EvaluatedPair:
    pair: PairPaths
    raw_json: Optional[dict[str, Any]]
    fps: float
    width: int
    height: int
    result: str
    reason: str


def repo_root_from_this_file() -> Path:
    # .../thesis_experiments/0_data_cleaning/build_masterdata.py -> root = ../
    return Path(__file__).resolve().parents[1]


def default_paths(repo_root: Path) -> tuple[Path, Path]:
    src = repo_root / "DataOriginal"
    dst = repo_root / "MasterData"
    return src, dst


def ensure_output_folders(dst_root: Path) -> None:
    required = [
        dst_root / "Corrup",
        dst_root / "enter",
        dst_root / "enter" / "exit",
        dst_root / "enter" / "removed",
        dst_root / "pass",
        dst_root / "pass" / "exit",
        dst_root / "pass" / "remove",
    ]
    for p in required:
        p.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def find_pairs(src_root: Path) -> list[PairPaths]:
    pairs: list[PairPaths] = []
    for split in VALID_SPLITS:
        for split_dir in src_root.rglob(split):
            if not split_dir.is_dir():
                continue
            for json_path in split_dir.glob("*.json"):
                mp4_path = json_path.with_suffix(".mp4")
                if mp4_path.exists():
                    pairs.append(PairPaths(split=split, json_path=json_path, mp4_path=mp4_path))
    pairs.sort(key=lambda p: str(p.json_path))
    return pairs


def readable_and_properties(mp4_path: Path) -> Optional[tuple[float, int, int, int]]:
    """
    Return (fps, width, height, frame_count) if readable enough for cleaning.
    Return None when video appears corrupted/unreadable.
    """
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        cap.release()
        return None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Basic readability check (first frame).
    ok_first, _ = cap.read()
    if not ok_first:
        cap.release()
        return None

    # Optional late read check if we have enough frames.
    if frame_count > 2:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 2))
        ok_late, _ = cap.read()
        if not ok_late:
            cap.release()
            return None

    cap.release()
    if width <= 0 or height <= 0:
        return None
    if fps <= 1e-6:
        fps = 13.0
    return fps, width, height, frame_count


def valid_centers(sample: dict[str, Any]) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for fr in sample.get("frames", []):
        c = fr.get("center")
        if isinstance(c, list) and len(c) == 2:
            pts.append((float(c[0]), float(c[1])))
    return pts


def bottom_touch_ratio(sample: dict[str, Any], frame_height: int, margin_px: int = 2) -> float:
    if frame_height <= 0:
        return 0.0
    total = 0
    touches = 0
    h1 = frame_height - 1
    for fr in sample.get("frames", []):
        bb = fr.get("bbox")
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        total += 1
        y2 = float(bb[3])
        if y2 >= (h1 - margin_px):
            touches += 1
    if total == 0:
        return 0.0
    return float(touches) / float(total)


def normalize_distance(x: float, y: float, door_x: float, door_y: float, width: int, height: int) -> float:
    dx = (x - door_x) / float(max(1, width))
    dy = (y - door_y) / float(max(1, height))
    return math.sqrt(dx * dx + dy * dy)


def sha1_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def canonical_json_hash(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def tracking_signature_hash(data: dict[str, Any]) -> str:
    """
    Hash only trajectory-related content.

    This avoids removing valid multi-person tracks that share the same source MP4
    but have different frame/center/bbox sequences.
    """
    signature = {
        "label": data.get("label", None),
        "frames": data.get("frames", []),
    }
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def write_duplicate_report_csv(report_path: Path, rows: list[dict[str, Any]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "duplicate_type",
        "split",
        "duplicate_json",
        "duplicate_mp4",
        "original_json",
        "original_mp4",
        "mp4_sha1",
        "tracking_sha1",
        "json_sha1",
    ]
    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def classify_pair(
    sample: dict[str, Any],
    cfg: CleanerConfig,
    width: int,
    height: int,
) -> tuple[str, str]:
    """
    Returns (cleaning_result, cleaning_reason)
    cleaning_result in {"keep", "exit", "removed"}.
    """
    pts = valid_centers(sample)
    if len(pts) < 2:
        return "removed", "too_few_valid_points"

    door_center = sample.get("door_center")
    if isinstance(door_center, list) and len(door_center) == 2:
        door_x = float(door_center[0])
        door_y = float(door_center[1])
    else:
        door_x = width / 2.0
        door_y = height - 1.0

    ys_n = [p[1] / float(max(1, height)) for p in pts]
    start = pts[0]
    end = pts[-1]
    start_y_n = start[1] / float(max(1, height))
    min_y_n = min(ys_n)
    max_y_n = max(ys_n)
    y_span_n = max_y_n - min_y_n

    dists = [normalize_distance(x, y, door_x, door_y, width, height) for x, y in pts]
    d_start = dists[0]
    d_end = dists[-1]
    delta_d = d_end - d_start

    net_disp = normalize_distance(end[0], end[1], start[0], start[1], width, height)

    # 1) EXIT behavior:
    # Starts low (near door/bottom), then gets farther from door center.
    if (
        start_y_n >= cfg.exit_start_y_min_norm
        and delta_d >= cfg.exit_min_delta_dist_norm
        and d_end >= max(1e-6, d_start) * cfg.exit_min_end_farther_ratio
    ):
        return "exit", "starts_low_and_moves_away_from_door"

    # 2) Presence-only near camera bottom (head-only style clips).
    # Stays in lower region with limited movement and often touches bottom border.
    b_touch = bottom_touch_ratio(sample, frame_height=height)
    if (
        start_y_n >= cfg.bottom_presence_start_y_min_norm
        and min_y_n >= cfg.bottom_presence_y_min_norm
        and net_disp <= cfg.bottom_presence_max_net_disp_norm
        and y_span_n <= cfg.bottom_presence_max_y_span_norm
        and b_touch >= cfg.bottom_presence_min_bottom_touch_ratio
    ):
        return "removed", "partial_person_bottom_presence"

    return "keep", "accepted"


def build_json_output(
    original: dict[str, Any],
    split_label: str,
    fps: float,
    width: int,
    height: int,
    cfg: CleanerConfig,
    cleaning_result: str,
    cleaning_reason: str,
) -> dict[str, Any]:
    out = dict(original)

    # Keep key structure aligned with sample JSON.
    if "id" not in out:
        out["id"] = -1
    out["label"] = split_label
    out.setdefault("frames", [])
    out["fps"] = float(fps)
    out["frame_width"] = int(width)
    out["frame_height"] = int(height)
    out.setdefault("door_center", [int(width // 2), int(height - 1)])
    out.setdefault("ellipse_axes", [int(cfg.ellipse_a), int(cfg.ellipse_b)])

    # Added for transparency in report/debugging.
    out["cleaning_result"] = cleaning_result
    out["cleaning_reason"] = cleaning_reason
    return out


def unique_dst_pair(dst_dir: Path, base_stem: str) -> tuple[Path, Path]:
    candidate_json = dst_dir / f"{base_stem}.json"
    candidate_mp4 = dst_dir / f"{base_stem}.mp4"
    if not candidate_json.exists() and not candidate_mp4.exists():
        return candidate_json, candidate_mp4

    i = 2
    while True:
        candidate_json = dst_dir / f"{base_stem}_dup{i}.json"
        candidate_mp4 = dst_dir / f"{base_stem}_dup{i}.mp4"
        if not candidate_json.exists() and not candidate_mp4.exists():
            return candidate_json, candidate_mp4
        i += 1


def write_pair(
    *,
    dst_dir: Path,
    source_mp4: Path,
    output_json: dict[str, Any],
    source_json_name: str,
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(source_json_name).stem
    dst_json, dst_mp4 = unique_dst_pair(dst_dir, stem)
    shutil.copy2(source_mp4, dst_mp4)
    with dst_json.open("w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=4)


def destination_dir_for(split: str, result: str, dst_root: Path) -> Path:
    if result == "corrupt":
        return dst_root / "Corrup"
    if split == "enter":
        if result == "exit":
            return dst_root / "enter" / "exit"
        if result == "removed":
            return dst_root / "enter" / "removed"
        return dst_root / "enter"
    if split == "pass":
        if result == "exit":
            return dst_root / "pass" / "exit"
        if result == "removed":
            return dst_root / "pass" / "remove"
        return dst_root / "pass"
    raise ValueError(f"Unknown split/result combination: {split=} {result=}")


def clean_dataset(
    src_root: Path,
    dst_root: Path,
    cfg: CleanerConfig,
    summary_only: bool = False,
    duplicate_report_csv: Optional[Path] = None,
) -> dict[str, int]:
    stats = {
        "pairs_found": 0,
        "pairs_found_enter": 0,
        "pairs_found_pass": 0,
        "pairs_processed": 0,
        "projected_keep_enter": 0,
        "projected_keep_pass": 0,
        "projected_exit": 0,
        "projected_removed": 0,
        "projected_corrup": 0,
        "written_keep_enter": 0,
        "written_keep_pass": 0,
        "written_exit": 0,
        "written_removed": 0,
        "written_corrup": 0,
        "skipped_missing_json": 0,
        "removed_short_duration": 0,
        "removed_duplicate_pair": 0,
        "shared_mp4_multiple_tracks": 0,
        "removed_partial_person_bottom_presence": 0,
        "removed_too_few_valid_points": 0,
        "exit_starts_low_and_moves_away_from_door": 0,
        "corrup_mp4_unreadable_or_corrupt": 0,
    }

    pairs = find_pairs(src_root)
    stats["pairs_found"] = len(pairs)
    stats["pairs_found_enter"] = sum(1 for p in pairs if p.split == "enter")
    stats["pairs_found_pass"] = sum(1 for p in pairs if p.split == "pass")
    if not pairs:
        return stats

    raw_json_cache: dict[Path, Optional[dict[str, Any]]] = {}
    duplicate_reason_by_json_path: dict[Path, str] = {}
    seen_pair_key_to_path: dict[tuple[str, str], Path] = {}
    seen_mp4_hash_to_track_hashes: dict[str, set[str]] = {}
    duplicate_rows: list[dict[str, Any]] = []

    # Duplicate detection applies globally across all source folders.
    for pair in pairs:
        raw_json = load_json(pair.json_path)
        raw_json_cache[pair.json_path] = raw_json

        if raw_json is None:
            continue

        mp4_hash = sha1_of_file(pair.mp4_path)
        track_hash = tracking_signature_hash(raw_json)
        json_hash = canonical_json_hash(raw_json)

        tracks_for_mp4 = seen_mp4_hash_to_track_hashes.setdefault(mp4_hash, set())
        if tracks_for_mp4 and track_hash not in tracks_for_mp4:
            # Same recording MP4 but another tracked person/object.
            stats["shared_mp4_multiple_tracks"] += 1
        tracks_for_mp4.add(track_hash)

        pair_key = (mp4_hash, track_hash)
        if pair_key in seen_pair_key_to_path:
            first_json = seen_pair_key_to_path[pair_key]
            duplicate_reason_by_json_path[pair.json_path] = f"duplicate_pair_of_{first_json.name}"
            duplicate_rows.append(
                {
                    "duplicate_type": "pair_mp4+track",
                    "split": pair.split,
                    "duplicate_json": str(pair.json_path),
                    "duplicate_mp4": str(pair.mp4_path),
                    "original_json": str(first_json),
                    "original_mp4": str(first_json.with_suffix(".mp4")),
                    "mp4_sha1": mp4_hash,
                    "tracking_sha1": track_hash,
                    "json_sha1": json_hash,
                }
            )
            continue
        seen_pair_key_to_path[pair_key] = pair.json_path

    evaluations: list[EvaluatedPair] = []

    for pair in pairs:
        raw_json = raw_json_cache.get(pair.json_path)
        if raw_json is None:
            stats["skipped_missing_json"] += 1
            continue

        if pair.json_path in duplicate_reason_by_json_path:
            duplicate_reason = duplicate_reason_by_json_path[pair.json_path]
            stats["removed_duplicate_pair"] += 1
            stats["projected_removed"] += 1
            evaluations.append(
                EvaluatedPair(
                    pair=pair,
                    raw_json=raw_json,
                    fps=float(raw_json.get("fps", 13.0) or 13.0),
                    width=int(raw_json.get("frame_width", 640) or 640),
                    height=int(raw_json.get("frame_height", 480) or 480),
                    result="removed",
                    reason=duplicate_reason,
                )
            )
            continue

        props = readable_and_properties(pair.mp4_path)
        if props is None:
            stats["corrup_mp4_unreadable_or_corrupt"] += 1
            stats["projected_corrup"] += 1
            evaluations.append(
                EvaluatedPair(
                    pair=pair,
                    raw_json=raw_json,
                    fps=13.0,
                    width=int(raw_json.get("frame_width", 640) or 640),
                    height=int(raw_json.get("frame_height", 480) or 480),
                    result="corrupt",
                    reason="mp4_unreadable_or_corrupt",
                )
            )
            continue

        fps, width, height, frame_count = props
        duration_s = float(frame_count) / float(max(1e-6, fps))

        if duration_s < cfg.min_duration_s:
            stats["removed_short_duration"] += 1
            stats["projected_removed"] += 1
            evaluations.append(
                EvaluatedPair(
                    pair=pair,
                    raw_json=raw_json,
                    fps=fps,
                    width=width,
                    height=height,
                    result="removed",
                    reason="video_shorter_than_0_5_seconds",
                )
            )
            continue

        enriched = build_json_output(
            original=raw_json,
            split_label=pair.split,
            fps=fps,
            width=width,
            height=height,
            cfg=cfg,
            cleaning_result="keep",
            cleaning_reason="accepted",
        )
        result, reason = classify_pair(enriched, cfg=cfg, width=width, height=height)

        if result == "exit":
            stats["exit_starts_low_and_moves_away_from_door"] += 1
            stats["projected_exit"] += 1
        elif reason == "partial_person_bottom_presence":
            stats["removed_partial_person_bottom_presence"] += 1
            stats["projected_removed"] += 1
        elif reason == "too_few_valid_points":
            stats["removed_too_few_valid_points"] += 1
            stats["projected_removed"] += 1
        elif result == "removed":
            stats["projected_removed"] += 1
        elif result == "keep" and pair.split == "enter":
            stats["projected_keep_enter"] += 1
        elif result == "keep" and pair.split == "pass":
            stats["projected_keep_pass"] += 1

        evaluations.append(
            EvaluatedPair(
                pair=pair,
                raw_json=raw_json,
                fps=fps,
                width=width,
                height=height,
                result=result,
                reason=reason,
            )
        )

    stats["pairs_processed"] = len(evaluations)

    # Print summary before writing output data.
    print("Pre-run summary")
    print("----------------")
    print(f"Original pairs found (json+mp4): {stats['pairs_found']}")
    print(f"  - enter before cleaning:        {stats['pairs_found_enter']}")
    print(f"  - pass before cleaning:         {stats['pairs_found_pass']}")
    print(f"Pairs with valid JSON:           {stats['pairs_processed']}")
    print(f"Pairs skipped (invalid JSON):    {stats['skipped_missing_json']}")
    print()
    print("Removed / filtered by reason")
    print(f"  - short duration (< {cfg.min_duration_s:.2f}s):      {stats['removed_short_duration']}")
    print(f"  - duplicate pair (same mp4 + same track): {stats['removed_duplicate_pair']}")
    print(f"  - shared mp4 with different tracks (kept): {stats['shared_mp4_multiple_tracks']}")
    print(f"  - bottom/head-only presence:          {stats['removed_partial_person_bottom_presence']}")
    print(f"  - too few valid tracking points:      {stats['removed_too_few_valid_points']}")
    print(f"  - exit behavior:                      {stats['exit_starts_low_and_moves_away_from_door']}")
    print(f"  - corrupted mp4:                      {stats['corrup_mp4_unreadable_or_corrupt']}")
    print()
    print("Projected output counts")
    print(f"  - keep enter:   {stats['projected_keep_enter']}")
    print(f"  - keep pass:    {stats['projected_keep_pass']}")
    print(f"  - move to exit: {stats['projected_exit']}")
    print(f"  - move removed: {stats['projected_removed']}")
    print(f"  - move Corrup:  {stats['projected_corrup']}")
    print()

    if duplicate_report_csv is not None:
        write_duplicate_report_csv(duplicate_report_csv, duplicate_rows)
        print(f"Duplicate report CSV: {duplicate_report_csv}")
        print()

    if summary_only:
        return stats

    ensure_output_folders(dst_root)
    for ev in evaluations:
        out_json = build_json_output(
            original=ev.raw_json if ev.raw_json is not None else {},
            split_label=ev.pair.split,
            fps=ev.fps,
            width=ev.width,
            height=ev.height,
            cfg=cfg,
            cleaning_result=ev.result,
            cleaning_reason=ev.reason,
        )
        dst_dir = destination_dir_for(ev.pair.split, ev.result, dst_root)
        write_pair(
            dst_dir=dst_dir,
            source_mp4=ev.pair.mp4_path,
            output_json=out_json,
            source_json_name=ev.pair.json_path.name,
        )

        if ev.result == "corrupt":
            stats["written_corrup"] += 1
        elif ev.result == "removed":
            stats["written_removed"] += 1
        elif ev.result == "exit":
            stats["written_exit"] += 1
        elif ev.pair.split == "enter":
            stats["written_keep_enter"] += 1
        else:
            stats["written_keep_pass"] += 1

    return stats


def parse_args() -> argparse.Namespace:
    repo_root = repo_root_from_this_file()
    default_src, default_dst = default_paths(repo_root)

    p = argparse.ArgumentParser(description="Clean raw data into MasterData.")
    p.add_argument("--src", type=Path, default=default_src, help="Source root (default: %(default)s)")
    p.add_argument("--dst", type=Path, default=default_dst, help="Destination root (default: %(default)s)")
    p.add_argument("--min-duration-s", type=float, default=CleanerConfig.min_duration_s, help="Min clip duration")
    p.add_argument("--ellipse-a", type=int, default=CleanerConfig.ellipse_a, help="Ellipse axis a")
    p.add_argument("--ellipse-b", type=int, default=CleanerConfig.ellipse_b, help="Ellipse axis b")
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary counts; do not copy or write any output files.",
    )
    p.add_argument(
        "--duplicate-report-csv",
        type=Path,
        default=default_dst / "duplicate_report.csv",
        help=(
            "Path to write the duplicate-report CSV. "
            "Default: duplicate_report.csv inside MasterData/."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = CleanerConfig(
        min_duration_s=float(args.min_duration_s),
        ellipse_a=int(args.ellipse_a),
        ellipse_b=int(args.ellipse_b),
    )

    if not args.src.exists():
        print(f"Source folder not found: {args.src}")
        return 2

    stats = clean_dataset(
        args.src,
        args.dst,
        cfg=cfg,
        summary_only=bool(args.summary_only),
        duplicate_report_csv=args.duplicate_report_csv,
    )

    print("Done.")
    print(f"Source:      {args.src}")
    print(f"Destination: {args.dst}")
    print()
    print(f"Pairs found:            {stats['pairs_found']}")
    print(f"Kept in enter/:         {stats['written_keep_enter']}")
    print(f"Kept in pass/:          {stats['written_keep_pass']}")
    print(f"Moved to */exit/:       {stats['written_exit']}")
    print(f"Moved to removed/remove:{stats['written_removed']}")
    print(f"Moved to Corrup/:       {stats['written_corrup']}")
    print(f"Skipped invalid JSON:   {stats['skipped_missing_json']}")
    if args.summary_only:
        print()
        print("Summary-only mode: no files were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
