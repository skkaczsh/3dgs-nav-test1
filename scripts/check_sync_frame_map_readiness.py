#!/usr/bin/env python3
"""Check whether accepted sync anchors or a sync frame-map are production-ready."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sync_frame_map import load_frame_map, read_jsonl, row_rejection_status


def parse_cams(values: list[int]) -> list[int]:
    return sorted(set(int(v) for v in values))


def parse_frames(args: argparse.Namespace) -> list[int]:
    if args.frames:
        return sorted(set(int(v) for v in args.frames))
    if args.start is None or args.end is None:
        return []
    return list(range(int(args.start), int(args.end) + 1, max(int(args.stride), 1)))


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def accepted_anchor_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if str(row.get("anchor_status", "")).lower() != "accepted":
            continue
        if row.get("selected_video_idx") is None and row.get("video_idx") is None and row.get("selected_option_idx") is None:
            continue
        out.append(row)
    return out


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(row.get("anchor_status", row.get("cam_path_status", row.get("status", "unknown")))) for row in rows)
    rejections = Counter()
    by_cam = defaultdict(int)
    for row in rows:
        rejection = row_rejection_status(row)
        if rejection:
            rejections[rejection] += 1
        if "cam_id" in row:
            by_cam[int(row["cam_id"])] += 1
    return {
        "row_count": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "rejection_counts": dict(sorted(rejections.items())),
        "rows_by_cam": {str(k): int(v) for k, v in sorted(by_cam.items())},
    }


def check_solver_report(report: dict[str, Any] | None, allow_rejected: bool) -> tuple[bool, list[str], dict[str, Any]]:
    if report is None:
        return True, [], {}
    errors = []
    status = str(report.get("status", "unknown"))
    cam_reports = report.get("cam_reports", {})
    if status != "accepted" and not allow_rejected:
        errors.append(f"solver_report_status={status}")
    rejected_cams = []
    for cam, cam_report in cam_reports.items():
        if not bool(cam_report.get("accepted")):
            rejected_cams.append(str(cam))
    if rejected_cams and not allow_rejected:
        errors.append(f"solver_rejected_cams={','.join(rejected_cams)}")
    summary = {
        "solver_status": status,
        "accepted_anchor_count": int(report.get("accepted_anchor_count", 0)),
        "cam_status": {
            str(cam): str(cam_report.get("status", "unknown"))
            for cam, cam_report in sorted(cam_reports.items(), key=lambda item: str(item[0]))
        },
        "rejected_cams": rejected_cams,
    }
    return not errors, errors, summary


def check_frame_map(
    path: Path | None,
    frames: list[int],
    cams: list[int],
    allow_rejected: bool,
) -> tuple[bool, list[str], dict[str, Any]]:
    if path is None:
        return True, [], {}
    if not path.exists():
        return False, [f"frame_map_missing={path}"], {"path": str(path)}
    errors = []
    rows = read_jsonl(path)
    try:
        frame_map = load_frame_map(path, allow_rejected=allow_rejected)
    except ValueError as exc:
        return False, [str(exc)], {"path": str(path), **summarize_rows(rows)}
    missing = []
    if frames and cams:
        for frame_id in frames:
            for cam_id in cams:
                if (int(frame_id), int(cam_id)) not in frame_map:
                    missing.append({"frame_id": int(frame_id), "cam_id": int(cam_id)})
    if missing:
        errors.append(f"missing_frame_map_pairs={len(missing)}")
    mapped_non_direct = sum(1 for (frame_id, _cam_id), video_idx in frame_map.items() if int(frame_id) != int(video_idx))
    summary = {
        "path": str(path),
        "loaded_pairs": len(frame_map),
        "mapped_non_direct_pairs": int(mapped_non_direct),
        "missing_pair_count": len(missing),
        "missing_pairs_sample": missing[:20],
        **summarize_rows(rows),
    }
    return not errors, errors, summary


def check_anchor_manifest(
    path: Path | None,
    cams: list[int],
    min_accepted_per_cam: int,
) -> tuple[bool, list[str], dict[str, Any]]:
    if path is None:
        return True, [], {}
    if not path.exists():
        return False, [f"anchors_missing={path}"], {"path": str(path)}
    rows = read_jsonl(path)
    accepted = accepted_anchor_rows(rows)
    by_cam = Counter(int(row["cam_id"]) for row in accepted if "cam_id" in row)
    errors = []
    for cam_id in cams:
        if by_cam[int(cam_id)] < int(min_accepted_per_cam):
            errors.append(f"accepted_anchors_cam{cam_id}={by_cam[int(cam_id)]}<min{min_accepted_per_cam}")
    summary = {
        "path": str(path),
        "accepted_anchor_count": len(accepted),
        "accepted_by_cam": {str(k): int(v) for k, v in sorted(by_cam.items())},
        "min_accepted_per_cam": int(min_accepted_per_cam),
        **summarize_rows(rows),
    }
    return not errors, errors, summary


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    frames = parse_frames(args)
    cams = parse_cams(args.cams)
    checks = {}
    errors = []

    ok, sub_errors, summary = check_anchor_manifest(args.anchors_jsonl, cams, args.min_accepted_per_cam)
    checks["anchors"] = {"passed": ok, "errors": sub_errors, **summary}
    errors.extend(sub_errors)

    try:
        solver_report = load_json(args.solver_report)
        ok, sub_errors, summary = check_solver_report(solver_report, args.allow_rejected)
    except FileNotFoundError as exc:
        ok, sub_errors, summary = False, [f"solver_report_missing={exc}"], {"path": str(args.solver_report)}
    checks["solver_report"] = {"passed": ok, "errors": sub_errors, **summary}
    errors.extend(sub_errors)

    ok, sub_errors, summary = check_frame_map(args.frame_map_jsonl, frames, cams, args.allow_rejected)
    checks["frame_map"] = {"passed": ok, "errors": sub_errors, **summary}
    errors.extend(sub_errors)

    return {
        "passed": not errors,
        "errors": errors,
        "frames_checked": len(frames),
        "cams": cams,
        "allow_rejected": bool(args.allow_rejected),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors-jsonl", type=Path)
    parser.add_argument("--frame-map-jsonl", type=Path)
    parser.add_argument("--solver-report", type=Path)
    parser.add_argument("--frames", type=int, nargs="*")
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--min-accepted-per-cam", type=int, default=2)
    parser.add_argument("--allow-rejected", action="store_true")
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
