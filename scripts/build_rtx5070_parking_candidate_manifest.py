#!/usr/bin/env python3
"""Build a lightweight manifest for the RTX 5070Ti parking candidate route."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def file_entry(path: Path, role: str, required: bool = True, remote_path: str = "") -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "remote_path": remote_path,
        "required": required,
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return ""


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def check(name: str, passed: bool, detail: str, value: Any = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail, "value": value}


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    viewer_report = read_json(args.viewer_report)
    qa_report = read_json(args.qa_report)
    compare_report = read_json(args.compare_report)
    object_relabel = read_json(args.object_relabel_report)
    geometry_refine = read_json(args.geometry_refine_report)

    files = [
        file_entry(args.viewer_ply, "candidate_viewer_ply", remote_path=args.remote_viewer_ply),
        file_entry(args.viewer_objects_jsonl, "candidate_viewer_objects_jsonl", remote_path=args.remote_viewer_objects_jsonl),
        file_entry(args.viewer_report, "candidate_viewer_export_report", remote_path=args.remote_viewer_report),
        file_entry(args.qa_report, "candidate_frame_local_qa_report", remote_path=args.remote_qa_report),
        file_entry(args.qa_contact, "candidate_frame_local_qa_contact", required=False, remote_path=args.remote_qa_contact),
        file_entry(args.qa_candidates, "candidate_frame_local_qa_candidates", required=False, remote_path=args.remote_qa_candidates),
        file_entry(args.qa_evidence, "candidate_frame_local_qa_evidence", required=False, remote_path=args.remote_qa_evidence),
        file_entry(args.compare_report, "surface_refinement_full_risk_compare_json", remote_path=args.remote_compare_report),
        file_entry(args.compare_markdown, "surface_refinement_full_risk_compare_markdown", remote_path=args.remote_compare_markdown),
        file_entry(args.object_relabel_report, "candidate_object_relabel_report", remote_path=args.remote_object_relabel_report),
        file_entry(args.geometry_refine_report, "candidate_geometry_refine_summary", remote_path=args.remote_geometry_refine_report),
    ]

    baseline = str(compare_report.get("baseline") or "strict_surface")
    candidate = args.candidate_name
    candidate_delta = nested(compare_report, "all_risk_deltas_from_baseline", candidate, default={}) or {}
    versions = compare_report.get("versions") or {}
    baseline_risk = nested(versions, baseline, "all_candidate_count", default=None)
    candidate_risk = nested(versions, candidate, "all_candidate_count", default=None)

    label_counts = viewer_report.get("label_counts") or {}
    all_risks = qa_report.get("all_risk_reason_counts") or {}
    checks = [
        check("required_files_exist", all(row["exists"] and row["bytes"] > 0 for row in files if row["required"]), "all required manifest artifacts exist"),
        check("viewer_missing_target_points_zero", as_int(viewer_report.get("missing_target_points"), -1) == 0, "viewer export has no missing target points", viewer_report.get("missing_target_points")),
        check("viewer_has_points", as_int(viewer_report.get("output_vertices")) > 0, "viewer PLY contains points", viewer_report.get("output_vertices")),
        check("qa_has_full_risk_counts", bool(all_risks), "QA report includes all_risk_reason_counts", all_risks),
        check("candidate_in_compare", candidate in versions, "candidate version exists in comparison report", sorted(versions)),
        check("candidate_reduces_all_risky_objects", candidate_risk is not None and baseline_risk is not None and int(candidate_risk) <= int(baseline_risk), "candidate all-risk count is no worse than baseline", {"baseline": baseline_risk, "candidate": candidate_risk}),
        check("ground_high_span_not_worse", int(candidate_delta.get("ground_has_large_height_span", 1)) <= 0, "candidate does not increase ground high-span risk", candidate_delta.get("ground_has_large_height_span")),
        check("wall_flat_risk_reduced", int(candidate_delta.get("wall_too_flat_low_height", 0)) < 0, "candidate reduces flat/low wall risk", candidate_delta.get("wall_too_flat_low_height")),
        check("wall_up_normal_risk_reduced", int(candidate_delta.get("wall_normal_too_up", 0)) < 0, "candidate reduces up-normal wall risk", candidate_delta.get("wall_normal_too_up")),
    ]

    manifest = {
        "manifest_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(args.repo),
        "status": "candidate_ready_for_visual_review",
        "passed": all(row["passed"] for row in checks),
        "dataset": {
            "name": "MT20260616-175807_parking_candidate_surface_route",
            "source_host": "scan-rtx5070",
            "route": "guarded_v2 + ground artifact guard + strict surface fusion + object surface relabel",
            "candidate_name": candidate,
            "baseline_name": baseline,
            "source_targets": str(args.source_targets),
        },
        "viewer": {
            "url": args.viewer_url,
            "local_ply": str(args.viewer_ply),
            "local_objects": str(args.viewer_objects_jsonl),
        },
        "commands": {
            "remote_rebuild": "cd /home/zsh/Work/SCAN/new_route && scripts/run_rtx5070_parking_candidate_surface_route.sh",
            "remote_rebuild_check": "cd /home/zsh/Work/SCAN/new_route && CHECK_ONLY=1 scripts/run_rtx5070_parking_candidate_surface_route.sh",
            "local_pull": "cd /Users/skkac/Work/SCAN/new_route && scripts/pull_rtx5070_parking_candidate_surface_route.sh",
            "local_pull_dry_run": "cd /Users/skkac/Work/SCAN/new_route && DRY_RUN=1 scripts/pull_rtx5070_parking_candidate_surface_route.sh",
        },
        "metrics": {
            "viewer": {
                "input_vertices": viewer_report.get("input_vertices"),
                "output_vertices": viewer_report.get("output_vertices"),
                "stride": viewer_report.get("stride"),
                "missing_target_points": viewer_report.get("missing_target_points"),
                "object_count_with_points": viewer_report.get("object_count_with_points"),
                "target_records": viewer_report.get("target_records"),
                "label_counts": label_counts,
            },
            "qa": {
                "objects": qa_report.get("objects"),
                "status_counts": qa_report.get("status_counts"),
                "semantic_label_counts": qa_report.get("semantic_label_counts"),
                "all_candidate_count": qa_report.get("all_candidate_count"),
                "all_candidate_label_counts": qa_report.get("all_candidate_label_counts"),
                "all_risk_reason_counts": all_risks,
            },
            "comparison": {
                "baseline": baseline,
                "candidate": candidate,
                "baseline_all_candidate_count": baseline_risk,
                "candidate_all_candidate_count": candidate_risk,
                "candidate_deltas_from_baseline": candidate_delta,
            },
            "object_relabel": {
                "changed_count": object_relabel.get("changed_count"),
                "changed_ratio": object_relabel.get("changed_ratio"),
                "reason_counts": object_relabel.get("reason_counts"),
                "label_counts_before": object_relabel.get("label_counts_before"),
                "label_counts_after": object_relabel.get("label_counts_after"),
            },
            "geometry_refine": {
                "input_targets": geometry_refine.get("input_targets"),
                "output_targets": geometry_refine.get("output_targets"),
                "missing_target_points": geometry_refine.get("missing_target_points"),
                "split_source_targets": geometry_refine.get("split_source_targets"),
                "relabelled_targets": geometry_refine.get("relabelled_targets"),
                "refinement_reason_counts": geometry_refine.get("refinement_reason_counts"),
            },
        },
        "checks": checks,
        "files": files,
    }
    return manifest


def render_markdown(manifest: dict[str, Any]) -> str:
    metrics = manifest["metrics"]
    checks = manifest["checks"]
    lines = [
        "# RTX 5070Ti Parking Candidate Manifest",
        "",
        f"- status: `{manifest['status']}`",
        f"- passed: `{manifest['passed']}`",
        f"- generated at: `{manifest['generated_at']}`",
        f"- git head: `{manifest['git_head']}`",
        f"- route: `{manifest['dataset']['route']}`",
        "",
        "## Viewer",
        "",
        manifest["viewer"]["url"],
        "",
        "## Key Metrics",
        "",
        f"- viewer output vertices: `{nested(metrics, 'viewer', 'output_vertices')}`",
        f"- object count with points: `{nested(metrics, 'viewer', 'object_count_with_points')}`",
        f"- target records: `{nested(metrics, 'viewer', 'target_records')}`",
        f"- QA all risky objects: `{nested(metrics, 'qa', 'all_candidate_count')}`",
        f"- baseline all risky objects: `{nested(metrics, 'comparison', 'baseline_all_candidate_count')}`",
        f"- candidate all risky objects: `{nested(metrics, 'comparison', 'candidate_all_candidate_count')}`",
        f"- candidate deltas: `{json.dumps(nested(metrics, 'comparison', 'candidate_deltas_from_baseline', default={}), ensure_ascii=False)}`",
        "",
        "## Label Counts",
        "",
    ]
    for label, count in (nested(metrics, "viewer", "label_counts", default={}) or {}).items():
        lines.append(f"- {label}: `{count}`")
    lines.extend(["", "## Checks", ""])
    for row in checks:
        mark = "PASS" if row["passed"] else "FAIL"
        lines.append(f"- `{mark}` {row['name']}: {row['detail']} value=`{row.get('value')}`")
    lines.extend(["", "## Commands", ""])
    for name, command in manifest["commands"].items():
        lines.append(f"- `{name}`: `{command}`")
    lines.extend(["", "## Files", ""])
    for row in manifest["files"]:
        lines.append(f"- `{row['role']}` exists=`{row['exists']}` bytes=`{row['bytes']}` path=`{row['path']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    base = repo / "server_parking_priority_s10"
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--output-json", type=Path, default=base / "parking_candidate_manifest_rtx5070" / "manifest.json")
    parser.add_argument("--output-md", type=Path, default=base / "parking_candidate_manifest_rtx5070" / "manifest.md")
    parser.add_argument("--candidate-name", default="ground_guard_object_relabel")
    parser.add_argument("--source-targets", type=Path, default=base / "frame_object_viewer_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070")
    parser.add_argument("--viewer-ply", type=Path, default=base / "frame_object_viewer_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070" / "frame_object_points_stride10.ply")
    parser.add_argument("--viewer-objects-jsonl", type=Path, default=base / "frame_object_viewer_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070" / "frame_objects_viewer.jsonl")
    parser.add_argument("--viewer-report", type=Path, default=base / "frame_object_viewer_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070" / "frame_object_viewer_export_report.json")
    parser.add_argument("--qa-report", type=Path, default=base / "frame_local_object_qa_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070" / "frame_local_object_qa_report.json")
    parser.add_argument("--qa-contact", type=Path, default=base / "frame_local_object_qa_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070" / "frame_local_object_qa_contact.jpg")
    parser.add_argument("--qa-candidates", type=Path, default=base / "frame_local_object_qa_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070" / "frame_local_object_qa_candidates.jsonl")
    parser.add_argument("--qa-evidence", type=Path, default=base / "frame_local_object_qa_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070" / "frame_local_object_qa_evidence.jsonl")
    parser.add_argument("--compare-report", type=Path, default=base / "guarded_v2_surface_refinement_all_risk_compare" / "qa_compare.json")
    parser.add_argument("--compare-markdown", type=Path, default=base / "guarded_v2_surface_refinement_all_risk_compare" / "qa_compare.md")
    parser.add_argument("--object-relabel-report", type=Path, default=base / "guarded_v2_ground_guard_object_relabel_reports" / "object_relabel_report.json")
    parser.add_argument("--geometry-refine-report", type=Path, default=base / "guarded_v2_ground_artifact_guard_reports" / "geometry_refine_summary.json")
    parser.add_argument("--viewer-url", default="http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/frame_object_viewer_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070/frame_object_points_stride10.ply&objects=/server_parking_priority_s10/frame_object_viewer_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070/frame_objects_viewer.jsonl&mode=semantic&stride=1&pointSize=1.5")
    parser.add_argument("--remote-viewer-ply", default="/home/zsh/Work/SCAN/work_MT20260616-175807/frame_object_viewer_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070/frame_object_points_stride10.ply")
    parser.add_argument("--remote-viewer-objects-jsonl", default="/home/zsh/Work/SCAN/work_MT20260616-175807/frame_object_viewer_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070/frame_objects_viewer.jsonl")
    parser.add_argument("--remote-viewer-report", default="/home/zsh/Work/SCAN/work_MT20260616-175807/frame_object_viewer_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070/frame_object_viewer_export_report.json")
    parser.add_argument("--remote-qa-report", default="/home/zsh/Work/SCAN/work_MT20260616-175807/frame_local_object_qa_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070/frame_local_object_qa_report.json")
    parser.add_argument("--remote-qa-contact", default="/home/zsh/Work/SCAN/work_MT20260616-175807/frame_local_object_qa_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070/frame_local_object_qa_contact.jpg")
    parser.add_argument("--remote-qa-candidates", default="/home/zsh/Work/SCAN/work_MT20260616-175807/frame_local_object_qa_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070/frame_local_object_qa_candidates.jsonl")
    parser.add_argument("--remote-qa-evidence", default="/home/zsh/Work/SCAN/work_MT20260616-175807/frame_local_object_qa_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070/frame_local_object_qa_evidence.jsonl")
    parser.add_argument("--remote-compare-report", default="/home/zsh/Work/SCAN/work_MT20260616-175807/guarded_v2_surface_refinement_all_risk_compare/qa_compare.json")
    parser.add_argument("--remote-compare-markdown", default="/home/zsh/Work/SCAN/work_MT20260616-175807/guarded_v2_surface_refinement_all_risk_compare/qa_compare.md")
    parser.add_argument("--remote-object-relabel-report", default="/home/zsh/Work/SCAN/work_MT20260616-175807/frame_objects_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070/object_relabel_report.json")
    parser.add_argument("--remote-geometry-refine-report", default="/home/zsh/Work/SCAN/work_MT20260616-175807/frame_targets_guarded_v2_full_s10_ground_artifact_guard_rtx5070/geometry_refine_summary.json")
    args = parser.parse_args()

    manifest = build_manifest(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(manifest), encoding="utf-8")
    print(json.dumps({"passed": manifest["passed"], "output_json": str(args.output_json), "output_md": str(args.output_md)}, ensure_ascii=False, indent=2))
    if not manifest["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
