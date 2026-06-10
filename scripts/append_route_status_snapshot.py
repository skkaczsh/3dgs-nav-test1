#!/usr/bin/env python3
"""Append a compact route status snapshot to a JSONL history."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compact(status: dict, timestamp: str) -> dict:
    connectivity = status.get("connectivity", {})
    main = status.get("main_route", {})
    stage = main.get("stage_status", {})
    workflow = main.get("manual_workflow_pending", {})
    qa = main.get("manual_merge_qa", {})
    delivery = status.get("delivery", {})
    offline_qa = status.get("offline_qa", {})
    resume_plan = status.get("server_resume_command_plan", {})
    resume_outputs = status.get("server_resume_outputs", {})
    return {
        "timestamp": timestamp,
        "offline_qa_passed": offline_qa.get("passed"),
        "offline_qa_git_head": offline_qa.get("git_head"),
        "resume_command_plan_passed": resume_plan.get("passed"),
        "resume_command_plan_error_count": len(resume_plan.get("errors", [])),
        "resume_outputs_passed": resume_outputs.get("passed"),
        "resume_outputs_blocker_count": len(resume_outputs.get("blockers", [])),
        "all_servers_reachable": connectivity.get("all_reachable"),
        "qwen_review_ready": stage.get("qwen_review_ready"),
        "review_pack_ready": stage.get("review_pack_ready"),
        "contact_sheets_ready": stage.get("contact_sheets_ready"),
        "manual_html_ready": stage.get("manual_html_ready"),
        "pending_apply_safe": stage.get("pending_apply_safe"),
        "manual_review_count": workflow.get("manual_review_count"),
        "accepted_merge_count": qa.get("accepted_merge_count", workflow.get("accepted_merge_count")),
        "input_object_count": workflow.get("input_object_count"),
        "output_object_count": workflow.get("output_object_count"),
        "qa_passed": qa.get("passed"),
        "input_point_count": qa.get("input_point_count"),
        "output_point_count": qa.get("output_point_count"),
        "delivery_file_count": delivery.get("file_count"),
        "delivery_missing_count": len(delivery.get("missing", [])),
        "conceptseg_status": status.get("new_model_side_track", {}).get("status"),
        "old_route_status": status.get("old_route_side_track", {}).get("status"),
    }


def append_snapshot(status_path: Path, history_path: Path, timestamp: str) -> dict:
    snapshot = compact(load_json(status_path), timestamp)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
    return snapshot


def write_latest(snapshot: dict, latest_path: Path) -> None:
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--history-jsonl", type=Path, required=True)
    parser.add_argument("--latest-json", type=Path, required=True)
    parser.add_argument("--timestamp", default="")
    args = parser.parse_args()

    timestamp = args.timestamp or datetime.now(timezone.utc).isoformat()
    snapshot = append_snapshot(args.status_json, args.history_jsonl, timestamp)
    write_latest(snapshot, args.latest_json)
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
