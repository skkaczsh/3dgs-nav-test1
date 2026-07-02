#!/usr/bin/env python3
"""Run or print the validated object semantic evidence pipeline.

Pipeline:
1. fuse object semantic evidence
2. validate fused object metadata
3. export semantic-colored viewer PLY from the validated metadata

The pipeline is blocked by default until the patch experiment promotion gate has
passed.  This keeps unreviewed patch experiments out of promoted semantic
viewer artifacts.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.current_mainline_contract import reject_forbidden_production_input
from scripts.run_object_semantic_evidence_fusion import DEFAULT_PATCH_GATE, patch_gate_status

DEFAULT_MAINLINE_HEALTHCHECK = REPO_ROOT / "scripts" / "validate_current_mainline.py"


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def existing_file(path: Path, name: str) -> None:
    reject_forbidden_production_input(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} missing: {path}")
    if not path.is_file():
        raise ValueError(f"{name} is not a file: {path}")


def build_commands(args: argparse.Namespace) -> list[dict[str, Any]]:
    fuse = [
        args.python,
        "scripts/fuse_object_semantic_evidence.py",
        "--objects-jsonl",
        str(args.objects_jsonl),
        "--output-jsonl",
        str(args.fused_objects_jsonl),
        "--report",
        str(args.fusion_report),
        "--sam-weight",
        str(args.sam_weight),
        "--teacher-weight",
        str(args.teacher_weight),
        "--scene-weight",
        str(args.scene_weight),
        "--min-total-weight",
        str(args.min_total_weight),
        "--min-winner-ratio",
        str(args.min_winner_ratio),
        "--min-scene-supported-ratio",
        str(args.min_scene_supported_ratio),
    ] + (["--allow-scene-only"] if args.allow_scene_only else [])

    validate = [
        args.python,
        "scripts/validate_object_semantic_evidence_fusion.py",
        "--input-objects",
        str(args.objects_jsonl),
        "--output-objects",
        str(args.fused_objects_jsonl),
        "--report",
        str(args.fusion_report),
        "--output-json",
        str(args.fusion_validation),
    ] + (["--allow-scene-only"] if args.allow_scene_only else [])

    export = [
        args.python,
        "scripts/run_validated_semantic_viewer_export.py",
        "--source-ply",
        str(args.source_ply),
        "--objects-jsonl",
        str(args.fused_objects_jsonl),
        "--fusion-validation",
        str(args.fusion_validation),
        "--output-ply",
        str(args.output_ply),
        "--report-json",
        str(args.viewer_report),
        "--plan-json",
        str(args.viewer_export_plan),
        "--run",
    ]
    if args.allow_qa_preview_source:
        export.append("--allow-qa-preview-source")
    if args.allow_unvalidated_export:
        export.append("--allow-unvalidated-export")

    rows = [
        ("fuse_object_semantic_evidence", fuse),
        ("validate_object_semantic_evidence_fusion", validate),
        ("run_validated_semantic_viewer_export", export),
    ]
    return [{"name": name, "argv": command, "shell": shell_join(command)} for name, command in rows]


def build_plan(args: argparse.Namespace, gate: dict[str, Any]) -> dict[str, Any]:
    blocked = not gate["passed"] and not args.allow_unpromoted_patch_experiment
    return {
        "schema": "semantic-evidence-pipeline-plan/v1",
        "status": "blocked" if blocked else "ready",
        "source_ply": str(args.source_ply),
        "objects_jsonl": str(args.objects_jsonl),
        "fused_objects_jsonl": str(args.fused_objects_jsonl),
        "fusion_report": str(args.fusion_report),
        "fusion_validation": str(args.fusion_validation),
        "output_ply": str(args.output_ply),
        "viewer_report": str(args.viewer_report),
        "patch_gate": gate,
        "allow_unpromoted_patch_experiment": bool(args.allow_unpromoted_patch_experiment),
        "commands": build_commands(args),
    }


def run_command(argv: list[str], cwd: Path) -> None:
    subprocess.run(argv, cwd=cwd, check=True)


def run_mainline_healthcheck(args: argparse.Namespace) -> None:
    if args.skip_mainline_healthcheck:
        return
    script = args.mainline_healthcheck
    if not script.exists():
        raise FileNotFoundError(f"mainline healthcheck missing: {script}")
    subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ply", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="semantic_evidence")
    parser.add_argument("--plan-json", type=Path)
    parser.add_argument("--python", default="python")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--patch-gate", type=Path, default=DEFAULT_PATCH_GATE)
    parser.add_argument("--allow-unpromoted-patch-experiment", action="store_true")
    parser.add_argument("--allow-unvalidated-export", action="store_true")
    parser.add_argument(
        "--allow-qa-preview-source",
        action="store_true",
        help="Allow stride-sampled viewer PLY as QA source for the final validated viewer export.",
    )
    parser.add_argument("--mainline-healthcheck", type=Path, default=DEFAULT_MAINLINE_HEALTHCHECK)
    parser.add_argument("--skip-mainline-healthcheck", action="store_true")
    parser.add_argument("--sam-weight", type=float, default=1.0)
    parser.add_argument("--teacher-weight", type=float, default=1.25)
    parser.add_argument("--scene-weight", type=float, default=0.35)
    parser.add_argument("--min-total-weight", type=float, default=3.0)
    parser.add_argument("--min-winner-ratio", type=float, default=0.58)
    parser.add_argument("--min-scene-supported-ratio", type=float, default=0.52)
    parser.add_argument("--allow-scene-only", action="store_true")
    args = parser.parse_args()
    prefix = args.output_prefix
    args.fused_objects_jsonl = args.output_dir / f"{prefix}_objects.jsonl"
    args.fusion_report = args.output_dir / f"{prefix}_fusion_report.json"
    args.fusion_validation = args.output_dir / f"{prefix}_fusion_validation.json"
    args.output_ply = args.output_dir / f"{prefix}_viewer.ply"
    args.viewer_report = args.output_dir / f"{prefix}_viewer_report.json"
    args.viewer_export_plan = args.output_dir / f"{prefix}_viewer_export_plan.json"
    return args


def main() -> int:
    args = parse_args()
    reject_forbidden_production_input(args.source_ply, allow_qa_preview=args.allow_qa_preview_source)
    if not args.source_ply.exists():
        raise FileNotFoundError(f"source ply missing: {args.source_ply}")
    if not args.source_ply.is_file():
        raise ValueError(f"source ply is not a file: {args.source_ply}")
    existing_file(args.objects_jsonl, "objects jsonl")
    reject_forbidden_production_input(args.output_dir)
    for path in (
        args.fused_objects_jsonl,
        args.fusion_report,
        args.fusion_validation,
        args.output_ply,
        args.viewer_report,
        args.viewer_export_plan,
    ):
        reject_forbidden_production_input(path)
    gate = patch_gate_status(args.patch_gate)
    plan = build_plan(args, gate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.plan_json or (args.output_dir / f"{args.output_prefix}_pipeline_plan.json")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if plan["status"] == "blocked":
        return 2 if args.run else 0
    if args.run:
        run_mainline_healthcheck(args)
        cwd = Path.cwd()
        for item in plan["commands"]:
            run_command([str(part) for part in item["argv"]], cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
