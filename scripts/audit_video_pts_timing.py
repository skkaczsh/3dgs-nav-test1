#!/usr/bin/env python3
"""Audit video PTS/keyframe timing for sync-sensitive scan routes.

This is a diagnostic tool.  It answers whether a camera video can be treated as
uniform frame-index time, or whether downstream extraction must use an explicit
frame map/PTS model.  It does not modify data.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any


def parse_ffprobe_frame_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if parts and parts[0] == "frame":
            parts = parts[1:]
        key_frame = None
        pict_type = None
        timestamps: list[float] = []
        for part in parts:
            value = part.strip()
            if not value or value.upper() == "N/A":
                continue
            if value in {"0", "1"} and key_frame is None:
                key_frame = int(value)
                continue
            if len(value) == 1 and value.isalpha():
                pict_type = value
                continue
            try:
                timestamps.append(float(value))
            except ValueError:
                continue
        pts = timestamps[0] if timestamps else None
        rows.append({
            "index": len(rows),
            "pts": pts,
            "key_frame": key_frame,
            "pict_type": pict_type,
        })
    return rows


def ffprobe_frames(video_path: Path) -> list[dict[str, Any]]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=key_frame,best_effort_timestamp_time,pts_time,pkt_pts_time,pict_type",
        "-of",
        "csv=p=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffprobe failed: {video_path}")
    return parse_ffprobe_frame_rows(result.stdout)


def percentile(values: list[float], pct: float) -> float | None:
    finite = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    pos = (len(finite) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return finite[lo]
    frac = pos - lo
    return finite[lo] * (1.0 - frac) + finite[hi] * frac


def summarize_video(video_path: Path, expected_fps: float, jitter_ratio: float) -> dict[str, Any]:
    rows = ffprobe_frames(video_path)
    pts = [float(row["pts"]) for row in rows if row.get("pts") is not None]
    dts = [b - a for a, b in zip(pts, pts[1:])]
    expected_dt = 1.0 / float(expected_fps) if expected_fps > 0 else None
    tolerance = (expected_dt or 0.0) * float(jitter_ratio)
    large_jumps = [
        {"from_index": i, "to_index": i + 1, "dt": float(dt)}
        for i, dt in enumerate(dts)
        if expected_dt is not None and abs(dt - expected_dt) > tolerance
    ]
    nonmonotonic = [
        {"from_index": i, "to_index": i + 1, "dt": float(dt)}
        for i, dt in enumerate(dts)
        if dt <= 0
    ]
    keyframes = [row["index"] for row in rows if row.get("key_frame") == 1]
    return {
        "video_path": str(video_path),
        "frame_rows": len(rows),
        "pts_count": len(pts),
        "missing_pts_count": len(rows) - len(pts),
        "pts_first": pts[0] if pts else None,
        "pts_last": pts[-1] if pts else None,
        "duration_by_pts": (pts[-1] - pts[0]) if len(pts) >= 2 else None,
        "expected_fps": float(expected_fps),
        "expected_dt": expected_dt,
        "dt": {
            "min": min(dts) if dts else None,
            "p50": percentile(dts, 50),
            "mean": (sum(dts) / len(dts)) if dts else None,
            "p95": percentile(dts, 95),
            "max": max(dts) if dts else None,
        },
        "nonmonotonic_count": len(nonmonotonic),
        "large_jump_count": len(large_jumps),
        "large_jump_examples": large_jumps[:20],
        "keyframe_count": len(keyframes),
        "keyframe_first": keyframes[:20],
        "can_assume_uniform_index_time": bool(
            len(rows) == len(pts)
            and not nonmonotonic
            and len(large_jumps) == 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-fps", type=float, default=10.0)
    parser.add_argument("--jitter-ratio", type=float, default=0.2)
    args = parser.parse_args()

    reports = [summarize_video(path, args.expected_fps, args.jitter_ratio) for path in args.videos]
    output = {
        "expected_fps": float(args.expected_fps),
        "jitter_ratio": float(args.jitter_ratio),
        "all_uniform_index_time": all(item["can_assume_uniform_index_time"] for item in reports),
        "videos": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "all_uniform_index_time": output["all_uniform_index_time"],
        "videos": [
            {
                "path": item["video_path"],
                "frame_rows": item["frame_rows"],
                "missing_pts_count": item["missing_pts_count"],
                "large_jump_count": item["large_jump_count"],
                "keyframe_count": item["keyframe_count"],
            }
            for item in reports
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
