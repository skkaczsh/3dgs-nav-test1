#!/usr/bin/env python3
"""Summarize the current sync-gated parking production state."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"exists": False, "path": None}
    if not path.exists():
        return {"exists": False, "path": str(path)}
    try:
        return {"exists": True, "path": str(path), "data": json.loads(path.read_text(encoding="utf-8"))}
    except json.JSONDecodeError as exc:
        return {"exists": True, "path": str(path), "error": f"json_decode_error: {exc}"}


def count_jsonl(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"exists": False, "path": None, "row_count": 0}
    if not path.exists():
        return {"exists": False, "path": str(path), "row_count": 0}
    rows = 0
    accepted = 0
    by_cam: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("anchor_status", "")).lower() == "accepted":
                accepted += 1
                cam = str(row.get("cam_id", "unknown"))
                by_cam[cam] = by_cam.get(cam, 0) + 1
    return {
        "exists": True,
        "path": str(path),
        "row_count": rows,
        "accepted_count": accepted,
        "accepted_by_cam": dict(sorted(by_cam.items())),
    }


def read_exit_code(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"exists": False, "path": None, "value": None}
    if not path.exists():
        return {"exists": False, "path": str(path), "value": None}
    value = path.read_text(encoding="utf-8").strip()
    return {"exists": True, "path": str(path), "value": value, "passed": value == "0"}


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def derive_next_action(report: dict[str, Any]) -> dict[str, str]:
    downloads = report["downloads"]
    staged = report["staged_anchors"]
    validation = report["anchor_validation"]
    readiness_exit = report["readiness_exit"]
    readiness = report["readiness"]
    preflight = report["preflight"]
    production = report["production_outputs"]

    if not downloads["exists"] and not staged["exists"]:
        return {
            "status": "waiting_for_manual_anchors",
            "command": "Open the review page, export accepted_sync_anchors.jsonl, then run: python3 scripts/stage_accepted_sync_anchors.py --force --run-solver",
        }
    if not staged["exists"]:
        return {
            "status": "stage_latest_anchors",
            "command": "python3 scripts/stage_accepted_sync_anchors.py --force --run-solver",
        }
    if validation["exists"] and nested(validation, "data", "passed") is False:
        return {
            "status": "fix_anchor_validation",
            "command": "Review accepted_sync_anchor_validation.json and add/fix anchors before solving.",
        }
    if not readiness_exit["exists"]:
        return {
            "status": "run_constrained_sync_solver",
            "command": "scripts/run_rtx5070_sync_anchor_solver.sh",
        }
    if not readiness_exit.get("passed"):
        return {
            "status": "sync_readiness_failed",
            "command": "Review sync_frame_map_readiness.json and the regenerated review pack; export better anchors.",
        }
    if readiness["exists"] and nested(readiness, "data", "passed") is not True:
        return {
            "status": "sync_readiness_report_not_passing",
            "command": "Review sync_frame_map_readiness.json before production.",
        }
    if preflight["exists"] and nested(preflight, "data", "passed") is False:
        return {
            "status": "fix_rtx5070_preflight",
            "command": "Review sync_absprior_s10_preflight.json and fix runtime/input problems.",
        }
    if not production["frames_summary"]["exists"] or not production["priority_summary"]["exists"]:
        return {
            "status": "ready_to_build_sync_gated_dataset",
            "command": "RUN=1 scripts/run_rtx5070_sync_gated_parking_dataset.sh",
        }
    return {
        "status": "sync_gated_dataset_assets_present",
        "command": "Run QA on extracted frames and priority masks, then continue target/object build.",
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    downloads = count_jsonl(args.downloads_anchor)
    staged = count_jsonl(args.staged_anchor)
    validation = read_json(args.anchor_validation)
    readiness = read_json(args.readiness_json)
    readiness_exit = read_exit_code(args.readiness_exit)
    expanded_map = count_jsonl(args.expanded_frame_map)
    expanded_report = read_json(args.expanded_frame_map_report)
    preflight = read_json(args.preflight)
    frames_summary = read_json(args.frames_summary)
    priority_summary = read_json(args.priority_summary)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "review_url": args.review_url,
        "downloads": downloads,
        "staged_anchors": staged,
        "anchor_validation": validation,
        "readiness": readiness,
        "readiness_exit": readiness_exit,
        "expanded_frame_map": expanded_map,
        "expanded_frame_map_report": expanded_report,
        "preflight": preflight,
        "production_outputs": {
            "frames_summary": frames_summary,
            "priority_summary": priority_summary,
        },
    }
    report["next_action"] = derive_next_action(report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    next_action = report["next_action"]
    lines = [
        "# Sync-Gated Parking Status",
        "",
        f"- generated at: `{report['generated_at']}`",
        f"- next status: `{next_action['status']}`",
        f"- next command: `{next_action['command']}`",
        f"- review URL: `{report['review_url']}`",
        "",
        "## Anchors",
        "",
        f"- downloads export: `{report['downloads'].get('exists')}` rows `{report['downloads'].get('row_count')}` accepted `{report['downloads'].get('accepted_count')}`",
        f"- staged anchors: `{report['staged_anchors'].get('exists')}` accepted by cam `{report['staged_anchors'].get('accepted_by_cam')}`",
        f"- anchor validation: `{nested(report, 'anchor_validation', 'data', 'passed')}`",
        "",
        "## Sync Gate",
        "",
        f"- readiness exit: `{report['readiness_exit'].get('value')}`",
        f"- readiness passed: `{nested(report, 'readiness', 'data', 'passed')}`",
        f"- expanded frame-map rows: `{report['expanded_frame_map'].get('row_count')}`",
        f"- expanded clipped count: `{nested(report, 'expanded_frame_map_report', 'data', 'clipped_count')}`",
        "",
        "## Production",
        "",
        f"- preflight passed: `{nested(report, 'preflight', 'data', 'passed')}`",
        f"- frames summary exists: `{report['production_outputs']['frames_summary'].get('exists')}`",
        f"- priority summary exists: `{report['production_outputs']['priority_summary'].get('exists')}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_review = "sync_anchor_review_priority_sky_penalty_timestamp_absprior_dot3_20260619"
    default_run = "sync_anchor_constrained_timestamp_absprior_dot3_20260619"
    base = root / "server_parking_priority_s10"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads-anchor", type=Path, default=Path.home() / "Downloads" / "accepted_sync_anchors.jsonl")
    parser.add_argument("--staged-anchor", type=Path, default=base / default_review / "accepted_sync_anchors.jsonl")
    parser.add_argument("--anchor-validation", type=Path, default=base / default_run / "accepted_sync_anchor_validation.json")
    parser.add_argument("--readiness-json", type=Path, default=base / default_run / "sync_frame_map_readiness.json")
    parser.add_argument("--readiness-exit", type=Path, default=base / default_run / "sync_frame_map_readiness.exit_code")
    parser.add_argument("--expanded-frame-map", type=Path, default=base / default_run / "expanded_frame_map.jsonl")
    parser.add_argument("--expanded-frame-map-report", type=Path, default=base / default_run / "expanded_frame_map_report.json")
    parser.add_argument("--preflight", type=Path, default=base / "sync_absprior_s10_preflight.json")
    parser.add_argument("--frames-summary", type=Path, default=base / "frames_jpeg_sync_absprior_s10" / "extract_report.json")
    parser.add_argument("--priority-summary", type=Path, default=base / "priority_surface_mapillary_sync_absprior_s10" / "priority_segmentation_summary.json")
    parser.add_argument("--review-url", default=f"http://127.0.0.1:8765/server_parking_priority_s10/{default_review}/anchor_review_priority.html")
    parser.add_argument("--output-json", type=Path, default=base / "sync_gated_parking_status.json")
    parser.add_argument("--output-md", type=Path, default=base / "sync_gated_parking_status.md")
    args = parser.parse_args()

    report = build_report(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(args.output_json), "markdown": str(args.output_md), "next_action": report["next_action"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
