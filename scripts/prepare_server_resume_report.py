#!/usr/bin/env python3
"""Prepare a local server-resume readiness report without probing servers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.exists():
        return {"missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def file_state(path: Path) -> dict:
    return {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}


def build_report(args: argparse.Namespace) -> dict:
    route = read_json(args.route_status)
    latest = read_json(args.latest_snapshot)
    offline = read_json(args.offline_qa)
    delivery_zip = file_state(args.delivery_zip)
    manual_csv = file_state(args.manual_csv)
    review_jsonl = file_state(args.review_jsonl)
    long_objects = file_state(args.long_objects)

    blockers = []
    if not offline.get("passed"):
        blockers.append("offline_qa_not_passing")
    if not delivery_zip["exists"]:
        blockers.append("missing_delivery_zip")
    if not manual_csv["exists"]:
        blockers.append("missing_manual_review_csv")
    if not review_jsonl["exists"]:
        blockers.append("missing_review_jsonl")
    if not long_objects["exists"]:
        blockers.append("missing_long_objects")

    return {
        "offline_mode": True,
        "ready_for_server_probe": not blockers,
        "blockers": blockers,
        "offline_qa": {
            "passed": offline.get("passed"),
            "git_head": offline.get("git_head"),
            "timestamp": offline.get("timestamp"),
            "checks": offline.get("checks", []),
        },
        "latest_snapshot": {
            "offline_qa_passed": latest.get("offline_qa_passed"),
            "offline_qa_git_head": latest.get("offline_qa_git_head"),
            "resume_command_plan_passed": latest.get("resume_command_plan_passed"),
            "resume_command_plan_error_count": latest.get("resume_command_plan_error_count"),
            "resume_outputs_passed": latest.get("resume_outputs_passed"),
            "resume_outputs_blocker_count": latest.get("resume_outputs_blocker_count"),
            "review_pack_ready": latest.get("review_pack_ready"),
            "delivery_file_count": latest.get("delivery_file_count"),
            "delivery_missing_count": latest.get("delivery_missing_count"),
            "conceptseg_status": latest.get("conceptseg_status"),
            "old_route_status": latest.get("old_route_status"),
        },
        "artifacts": {
            "route_status": file_state(args.route_status),
            "latest_snapshot": file_state(args.latest_snapshot),
            "delivery_zip": delivery_zip,
            "manual_csv": manual_csv,
            "review_jsonl": review_jsonl,
            "long_objects": long_objects,
        },
        "resume_commands": [
            "python3 scripts/prepare_server_resume_commands.py",
            "python3 scripts/diagnose_server_connectivity.py --output /Users/skkac/Work/SCAN/server_connectivity_diagnosis_20260610_latest.json",
            "BIND_ADDRESS=192.168.0.3 SERVER=scan-train CONCURRENCY=4 bash scripts/resume_server_qwen_review.sh",
            "PATCH_SCENE_PROMPTS=1 SHARDS=4 bash scripts/run_server_semantic_completion_sharded.sh",
            "BIND_ADDRESS=192.168.0.3 SERVER=scan-train bash scripts/run_server_dataset_readiness.sh",
            "MIN_MERGE_CONFIDENCE=0.5 bash scripts/run_server_target_object_fusion.sh",
        ],
        "notes": [
            "Do not run server commands while outside the server LAN.",
            "Run Qwen review before applying automatic object merges.",
            "Keep ConceptSeg-R1 and old route as side tracks until main-route GPU demand is low.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-status", type=Path, default=Path("/Users/skkac/Work/SCAN/route_status_20260610/dense_semantic_route_status.json"))
    parser.add_argument("--latest-snapshot", type=Path, default=Path("/Users/skkac/Work/SCAN/route_status_20260610/route_status_latest.json"))
    parser.add_argument("--offline-qa", type=Path, default=Path("/Users/skkac/Work/SCAN/route_status_20260610/offline_quality_latest.json"))
    parser.add_argument("--delivery-zip", type=Path, default=Path("/Users/skkac/Work/SCAN/server_frame_fine_cross_candidate_review_delivery_v008/cross_candidate_review_delivery.zip"))
    parser.add_argument("--manual-csv", type=Path, default=Path("/Users/skkac/Work/SCAN/server_frame_fine_cross_candidate_review_delivery_v008/cross_candidate_review_delivery/review_html/manual_merge_decisions.csv"))
    parser.add_argument("--review-jsonl", type=Path, default=Path("/Users/skkac/Work/SCAN/server_frame_fine_cross_candidate_review_delivery_v008/cross_candidate_review_delivery/cross_candidate_review_items.jsonl"))
    parser.add_argument("--long-objects", type=Path, default=Path("/Users/skkac/Work/SCAN/server_frame_fine_long_assoc_v008/frame_fine_tracklet_long_assoc_0000_0999_v008_gap60_v2_samecand_loose/long_objects.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("/Users/skkac/Work/SCAN/route_status_20260610/server_resume_readiness.json"))
    args = parser.parse_args()

    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["blockers"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
