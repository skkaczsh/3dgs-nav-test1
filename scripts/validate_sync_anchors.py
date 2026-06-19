#!/usr/bin/env python3
"""Validate exported manual sync anchors before running the remote solver."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solve_sync_path_from_candidates import apply_timestamp_phase, load_frame_timestamps, percentile
from sync_frame_map import read_jsonl, selected_video_idx


def parse_cams(values: list[int]) -> list[int]:
    return sorted(set(int(v) for v in values))


def accepted_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted = []
    invalid = []
    for row in rows:
        if str(row.get("anchor_status", "")).lower() != "accepted":
            continue
        try:
            video_idx = selected_video_idx(row)
        except Exception as exc:  # noqa: BLE001 - converted to report row
            invalid.append({"row": row, "error": str(exc)})
            continue
        out = dict(row)
        out["resolved_video_idx"] = int(video_idx)
        accepted.append(out)
    return accepted, invalid


def summarize_cam(
    rows: list[dict[str, Any]],
    timestamps: dict[int, float],
    expected_fps: float | None,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (int(row["frame_id"]), int(row["resolved_video_idx"])))
    intervals = []
    negative_steps = 0
    duplicate_frames = 0
    seen_frames = set()
    for row in ordered:
        frame_id = int(row["frame_id"])
        if frame_id in seen_frames:
            duplicate_frames += 1
        seen_frames.add(frame_id)
    for prev, cur in zip(ordered, ordered[1:]):
        frame_delta = int(cur["frame_id"]) - int(prev["frame_id"])
        video_delta = int(cur["resolved_video_idx"]) - int(prev["resolved_video_idx"])
        if video_delta < 0:
            negative_steps += 1
        time_delta = None
        implied_fps = None
        if int(prev["frame_id"]) in timestamps and int(cur["frame_id"]) in timestamps:
            time_delta = float(timestamps[int(cur["frame_id"])]) - float(timestamps[int(prev["frame_id"])])
            if abs(time_delta) > 1e-9:
                implied_fps = video_delta / time_delta
        intervals.append({
            "from_frame": int(prev["frame_id"]),
            "to_frame": int(cur["frame_id"]),
            "frame_delta": int(frame_delta),
            "video_delta": int(video_delta),
            "time_delta": time_delta,
            "implied_fps": implied_fps,
            "fps_error": None if implied_fps is None or expected_fps is None else implied_fps - expected_fps,
        })
    fps_values = [float(item["implied_fps"]) for item in intervals if item["implied_fps"] is not None and math.isfinite(float(item["implied_fps"]))]
    fps_errors = [abs(float(item["fps_error"])) for item in intervals if item["fps_error"] is not None and math.isfinite(float(item["fps_error"]))]
    return {
        "accepted_count": len(ordered),
        "frame_range": [int(ordered[0]["frame_id"]), int(ordered[-1]["frame_id"])] if ordered else None,
        "video_range": [int(ordered[0]["resolved_video_idx"]), int(ordered[-1]["resolved_video_idx"])] if ordered else None,
        "duplicate_frame_count": int(duplicate_frames),
        "negative_step_count": int(negative_steps),
        "interval_count": len(intervals),
        "implied_fps": {
            "count": len(fps_values),
            "p50": float(percentile(fps_values, 50)) if fps_values else None,
            "mean": float(statistics.fmean(fps_values)) if fps_values else None,
            "min": float(min(fps_values)) if fps_values else None,
            "max": float(max(fps_values)) if fps_values else None,
        },
        "fps_error_abs": {
            "count": len(fps_errors),
            "p50": float(percentile(fps_errors, 50)) if fps_errors else None,
            "mean": float(statistics.fmean(fps_errors)) if fps_errors else None,
            "max": float(max(fps_errors)) if fps_errors else None,
        },
        "intervals_sample": intervals[:20],
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.anchors_jsonl)
    accepted, invalid = accepted_rows(rows)
    cams = parse_cams(args.cams)
    raw_timestamps = load_frame_timestamps(args.img_pos_file) if args.img_pos_file else {}
    timestamps = apply_timestamp_phase(raw_timestamps, args.timestamp_phase_fraction)
    by_cam: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        by_cam[int(row["cam_id"])].append(row)

    errors = []
    warnings = []
    if invalid:
        errors.append(f"invalid_accepted_rows={len(invalid)}")
    for cam_id in cams:
        count = len(by_cam.get(cam_id, []))
        if count < int(args.min_accepted_per_cam):
            errors.append(f"accepted_anchors_cam{cam_id}={count}<min{args.min_accepted_per_cam}")

    cam_reports = {}
    for cam_id in cams:
        cam_report = summarize_cam(by_cam.get(cam_id, []), timestamps, args.expected_fps)
        cam_reports[str(cam_id)] = cam_report
        if cam_report["negative_step_count"] > 0:
            errors.append(f"negative_video_steps_cam{cam_id}={cam_report['negative_step_count']}")
        if cam_report["duplicate_frame_count"] > 0:
            warnings.append(f"duplicate_anchor_frames_cam{cam_id}={cam_report['duplicate_frame_count']}")
        max_fps_error = cam_report["fps_error_abs"]["max"]
        if max_fps_error is not None and max_fps_error > args.max_fps_error:
            warnings.append(f"large_fps_error_cam{cam_id}={max_fps_error:.3f}>max{args.max_fps_error}")

    status_counts = Counter(str(row.get("anchor_status", "unknown")) for row in rows)
    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "anchors_jsonl": str(args.anchors_jsonl),
        "img_pos_file": str(args.img_pos_file) if args.img_pos_file else None,
        "timestamp_phase_fraction": float(args.timestamp_phase_fraction),
        "expected_fps": args.expected_fps,
        "min_accepted_per_cam": int(args.min_accepted_per_cam),
        "row_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "accepted_count": len(accepted),
        "invalid_accepted_rows": invalid[:20],
        "cam_reports": cam_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors-jsonl", type=Path, required=True)
    parser.add_argument("--img-pos-file", type=Path)
    parser.add_argument("--timestamp-phase-fraction", type=float, default=0.0)
    parser.add_argument("--expected-fps", type=float, default=6.0)
    parser.add_argument("--max-fps-error", type=float, default=2.0)
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--min-accepted-per-cam", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = build_report(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
