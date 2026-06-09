#!/usr/bin/env python3
"""Validate local artifacts after the server resume plan has run."""

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


def pass_fail(name: str, passed: bool, detail: dict) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def validate(args: argparse.Namespace) -> dict:
    qwen_report = read_json(args.qwen_report)
    reviewed_merge_qa = read_json(args.reviewed_merge_qa)
    dataset = read_json(args.dataset_readiness)
    target_object_qa = read_json(args.target_object_qa)
    command_plan_validation = read_json(args.command_plan_validation)

    qwen_passed = "missing" not in qwen_report and int(qwen_report.get("error_count", 0)) == 0
    reviewed_merge_passed = reviewed_merge_qa.get("passed") is True
    dataset_ratios = dataset.get("ratios", {})
    semantic_ratio = float(dataset_ratios.get("completion_semantic_images", 0.0) or 0.0)
    color_ratio = float(dataset_ratios.get("color_ply", 0.0) or 0.0)
    dataset_passed = semantic_ratio >= args.min_semantic_ratio and color_ratio >= args.min_color_ratio
    target_frames = target_object_qa.get("frames", {})
    target_objects = target_object_qa.get("objects", {})
    target_ok_ratio = float(target_frames.get("ok_count", 0) / max(target_frames.get("count", 0), 1))
    ambiguous_ratio = float(target_objects.get("ambiguous_ratio", 1.0) or 0.0)
    target_object_passed = (
        "missing" not in target_object_qa
        and target_ok_ratio >= args.min_target_frame_ok_ratio
        and ambiguous_ratio <= args.max_ambiguous_ratio
    )
    command_plan_passed = command_plan_validation.get("passed") is True

    checks = [
        pass_fail(
            "command_plan_validation",
            command_plan_passed,
            {
                "path": str(args.command_plan_validation),
                "errors": command_plan_validation.get("errors", []),
                "warnings": command_plan_validation.get("warnings", []),
            },
        ),
        pass_fail(
            "qwen_review",
            qwen_passed,
            {
                "path": str(args.qwen_report),
                "exists": args.qwen_report.exists(),
                "error_count": qwen_report.get("error_count"),
                "result_count": qwen_report.get("result_count"),
            },
        ),
        pass_fail(
            "reviewed_merge_qa",
            reviewed_merge_passed,
            {
                "path": str(args.reviewed_merge_qa),
                "exists": args.reviewed_merge_qa.exists(),
                "accepted_merge_count": reviewed_merge_qa.get("accepted_merge_count"),
                "checks": reviewed_merge_qa.get("checks", {}),
            },
        ),
        pass_fail(
            "semantic_dataset",
            dataset_passed,
            {
                "path": str(args.dataset_readiness),
                "exists": args.dataset_readiness.exists(),
                "semantic_ratio": semantic_ratio,
                "min_semantic_ratio": args.min_semantic_ratio,
                "color_ratio": color_ratio,
                "min_color_ratio": args.min_color_ratio,
            },
        ),
        pass_fail(
            "target_object_fusion",
            target_object_passed,
            {
                "path": str(args.target_object_qa),
                "exists": args.target_object_qa.exists(),
                "target_frame_ok_ratio": target_ok_ratio,
                "min_target_frame_ok_ratio": args.min_target_frame_ok_ratio,
                "ambiguous_ratio": ambiguous_ratio,
                "max_ambiguous_ratio": args.max_ambiguous_ratio,
                "object_count": target_objects.get("count"),
            },
        ),
    ]

    blockers = [row["name"] for row in checks if not row["passed"]]
    side_tracks = {
        "conceptseg": file_state(args.conceptseg_report),
        "old_route": file_state(args.old_route_summary),
    }
    return {
        "passed": not blockers,
        "blockers": blockers,
        "checks": checks,
        "thresholds": {
            "min_semantic_ratio": args.min_semantic_ratio,
            "min_color_ratio": args.min_color_ratio,
            "min_target_frame_ok_ratio": args.min_target_frame_ok_ratio,
            "max_ambiguous_ratio": args.max_ambiguous_ratio,
        },
        "side_tracks": side_tracks,
        "next_gate": "dataset_ready_for_model_and_old_route_side_tracks" if not blockers else "finish_main_route_outputs_first",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-report", type=Path, default=Path("/Users/skkac/Work/SCAN/server_frame_fine_cross_candidate_review_pack_v008_v2/vlm_review_qwen_compact/vlm_merge_review_report.json"))
    parser.add_argument("--reviewed-merge-qa", type=Path, default=Path("/Users/skkac/Work/SCAN/server_frame_fine_cross_candidate_review_pack_v008_v2/vlm_review_qwen_compact_applied/qa_reviewed_merge_report.json"))
    parser.add_argument("--dataset-readiness", type=Path, default=Path("/Users/skkac/Work/SCAN/route_status_20260610/server_dataset_readiness_0000_0999.json"))
    parser.add_argument("--target-object-qa", type=Path, default=Path("/Users/skkac/Work/SCAN/server_target_object_existing_completion_0000_0999/target_object_qa.json"))
    parser.add_argument("--command-plan-validation", type=Path, default=Path("/Users/skkac/Work/SCAN/route_status_20260610/server_resume_commands_validation.json"))
    parser.add_argument("--conceptseg-report", type=Path, default=Path("/Users/skkac/Work/SCAN/new_route/docs/model_eval/conceptseg_r1_side_track.md"))
    parser.add_argument("--old-route-summary", type=Path, default=Path("/Users/skkac/Work/SCAN/server_old_route_smoke/world_colorize_summary.json"))
    parser.add_argument("--min-semantic-ratio", type=float, default=0.90)
    parser.add_argument("--min-color-ratio", type=float, default=0.95)
    parser.add_argument("--min-target-frame-ok-ratio", type=float, default=0.95)
    parser.add_argument("--max-ambiguous-ratio", type=float, default=0.35)
    parser.add_argument("--output", type=Path, default=Path("/Users/skkac/Work/SCAN/route_status_20260610/server_resume_output_validation.json"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = validate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
