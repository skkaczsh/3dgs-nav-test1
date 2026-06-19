#!/usr/bin/env python3
"""Expand an accepted probe sync path into a production frame-map.

The sync solver only operates on sparse review probes.  Image extraction and
colorization need a concrete mapping for every requested section/camera pair.
This script expands the accepted timestamp model per camera:

  video_idx = (img_pos.timestamp - time_origin) * video_fps + intercept

`time_origin` is taken from the solved probe rows for that camera, and
`intercept` is taken from the solver report.  The output JSONL is suitable for
`--frame-map-jsonl --require-frame-map`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from solve_sync_path_from_candidates import apply_timestamp_phase, load_frame_timestamps, load_jsonl


def parse_cams(values: list[int]) -> list[int]:
    return sorted(set(int(v) for v in values))


def frame_ids(start: int, end: int, stride: int) -> list[int]:
    return list(range(int(start), int(end) + 1, max(int(stride), 1)))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cam_model_from_path(
    path_rows: list[dict[str, Any]],
    solver_report: dict[str, Any],
    cam_id: int,
    allow_rejected_solver: bool = False,
) -> dict[str, float]:
    cam_rows = [row for row in path_rows if int(row["cam_id"]) == int(cam_id)]
    if not cam_rows:
        raise ValueError(f"no path rows for cam{cam_id}")
    cam_report = solver_report.get("cam_reports", {}).get(str(cam_id))
    if not cam_report:
        raise ValueError(f"solver report missing cam{cam_id}")
    if not bool(cam_report.get("accepted")) and not allow_rejected_solver:
        raise ValueError(f"cam{cam_id} solver report is not accepted")
    timestamps = [float(row["sync_timestamp"]) for row in cam_rows if row.get("sync_timestamp") is not None]
    if not timestamps:
        raise ValueError(f"path rows for cam{cam_id} do not carry sync_timestamp")
    intercept = cam_report.get("absolute_intercept")
    if intercept is None:
        raise ValueError(f"cam{cam_id} solver report missing absolute_intercept")
    return {
        "time_origin": float(min(timestamps)),
        "video_fps": float(solver_report.get("video_fps", 10.0)),
        "absolute_intercept": float(intercept),
    }


def rounded_video_idx(value: float, video_frame_count: int | None) -> int:
    out = int(round(float(value)))
    if video_frame_count is not None:
        out = max(0, min(int(video_frame_count) - 1, out))
    return out


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    solver_report = read_json(args.solver_report)
    if str(solver_report.get("status")) != "accepted" and not args.allow_rejected_solver:
        raise ValueError(f"solver report is not accepted: {solver_report.get('status')}")
    path_rows = load_jsonl(args.path_jsonl)
    raw_timestamps = load_frame_timestamps(args.img_pos_file)
    phase_arg = getattr(args, "timestamp_phase_fraction", None)
    phase = phase_arg if phase_arg is not None else float(solver_report.get("timestamp_phase_fraction", 0.0))
    timestamps = apply_timestamp_phase(raw_timestamps, phase)
    frames = frame_ids(args.start, args.end, args.stride)
    cams = parse_cams(args.cams)
    missing = [frame_id for frame_id in frames if frame_id not in timestamps]
    if missing:
        raise ValueError(f"img_pos timestamps missing for frame ids: {missing[:10]}")

    models = {
        cam_id: cam_model_from_path(path_rows, solver_report, cam_id, args.allow_rejected_solver)
        for cam_id in cams
    }
    rows: list[dict[str, Any]] = []
    clipped = 0
    for frame_id in frames:
        ts = float(timestamps[frame_id])
        for cam_id in cams:
            model = models[cam_id]
            expected = (ts - model["time_origin"]) * model["video_fps"] + model["absolute_intercept"]
            video_idx = rounded_video_idx(expected, args.video_frame_count)
            if args.video_frame_count is not None and int(round(expected)) != video_idx:
                clipped += 1
            rows.append({
                "frame_id": int(frame_id),
                "cam_id": int(cam_id),
                "video_idx": int(video_idx),
                "cam_path_status": "accepted",
                "status": "ok",
                "source": "expanded_timestamp_absprior",
                "sync_timestamp": ts,
                "raw_sync_timestamp": float(raw_timestamps[frame_id]),
                "absolute_expected_video_idx": float(expected),
                "absolute_prior_error": abs(float(video_idx) - float(expected)),
                "time_origin": model["time_origin"],
                "video_fps": model["video_fps"],
                "absolute_intercept": model["absolute_intercept"],
            })
    report = {
        "path_jsonl": str(args.path_jsonl),
        "solver_report": str(args.solver_report),
        "img_pos_file": str(args.img_pos_file),
        "output_jsonl": str(args.output_jsonl),
        "start": int(args.start),
        "end": int(args.end),
        "stride": int(args.stride),
        "cams": cams,
        "frame_count": len(frames),
        "row_count": len(rows),
        "video_frame_count": args.video_frame_count,
        "timestamp_phase_fraction": float(phase),
        "clipped_count": int(clipped),
        "models": {str(k): v for k, v in models.items()},
    }
    return rows, report


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-jsonl", type=Path, required=True)
    parser.add_argument("--solver-report", type=Path, required=True)
    parser.add_argument("--img-pos-file", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--video-frame-count", type=int)
    parser.add_argument("--timestamp-phase-fraction", type=float,
                        help="Override solver report timestamp phase fraction.")
    parser.add_argument("--allow-rejected-solver", action="store_true")
    args = parser.parse_args()

    rows, report = build_rows(args)
    write_jsonl(args.output_jsonl, rows)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
