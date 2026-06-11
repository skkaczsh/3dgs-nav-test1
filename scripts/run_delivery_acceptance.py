#!/usr/bin/env python3
"""Run local acceptance checks for the dense semantic delivery package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_cmd(cmd: list[str], cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "passed": proc.returncode == 0,
    }


def check_route_decision(path: Path) -> dict[str, Any]:
    data = read_json(path)
    errors: list[str] = []
    main = data.get("main_route", {})
    concept = data.get("conceptseg_side_track", {})
    old = data.get("old_route_side_track", {})
    if main.get("decision") != "continue_as_authoritative_route":
        errors.append(f"unexpected main route decision: {main.get('decision')}")
    if not main.get("dataset_manifest_passed"):
        errors.append("dataset manifest did not pass in route decision")
    if not main.get("output_validation_passed"):
        errors.append("output validation did not pass in route decision")
    if concept.get("decision") != "keep_as_conservative_fine_object_refinement_only":
        errors.append(f"unexpected ConceptSeg decision: {concept.get('decision')}")
    if concept.get("semantically_discriminative_target_count") != 0:
        errors.append("ConceptSeg appears semantically discriminative; decision summary may need review")
    if old.get("decision") != "keep_as_fixed_visual_color_reference_only":
        errors.append(f"unexpected old-route decision: {old.get('decision')}")
    if not old.get("validation_passed"):
        errors.append("old-route reference validation did not pass")
    return {
        "path": str(path),
        "passed": not errors,
        "errors": errors,
        "main_route": main,
        "conceptseg_side_track": concept,
        "old_route_side_track": old,
    }


def check_package_metrics(package_manifest: Path) -> dict[str, Any]:
    data = read_json(package_manifest)
    metrics = data.get("metrics", {})
    errors: list[str] = []
    if not data.get("passed"):
        errors.append("package manifest passed=false")
    if metrics.get("route_decision") != "continue_as_authoritative_route":
        errors.append("package route decision is not authoritative main route")
    if metrics.get("conceptseg_decision") != "keep_as_conservative_fine_object_refinement_only":
        errors.append("package ConceptSeg decision is not conservative refinement")
    if metrics.get("old_route_decision") != "keep_as_fixed_visual_color_reference_only":
        errors.append("package old-route decision is not fixed visual reference")
    if int(metrics.get("conceptseg_instance_accepted_candidates") or 0) <= 0:
        errors.append("package has no accepted ConceptSeg intersection candidates")
    if not metrics.get("old_route_reference_passed"):
        errors.append("package old-route reference did not pass")
    return {
        "path": str(package_manifest),
        "passed": not errors,
        "errors": errors,
        "metrics": {
            key: metrics.get(key)
            for key in [
                "route_decision",
                "conceptseg_decision",
                "conceptseg_instance_accepted_candidates",
                "conceptseg_instance_target_status_counts",
                "old_route_decision",
                "old_route_reference_passed",
                "target_count",
                "object_count",
            ]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("/Users/skkac/Work/SCAN")
    repo = root / "new_route"
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--route-decision", type=Path, default=root / "route_status_20260610/dense_semantic_route_decision_20260611.json")
    parser.add_argument("--delivery-manifest", type=Path, default=root / "route_status_20260610/dataset_delivery_manifest_0000_0999.json")
    parser.add_argument("--manifest-validation", type=Path, default=root / "route_status_20260610/dataset_delivery_manifest_0000_0999_validation.json")
    parser.add_argument("--package-dir", type=Path, default=root / "dataset_delivery_0000_0999")
    parser.add_argument("--package-validation", type=Path, default=root / "dataset_delivery_0000_0999_validation.json")
    parser.add_argument("--output", type=Path, default=root / "route_status_20260610/delivery_acceptance_20260611.json")
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []
    checks.append({"name": "route_decision", **check_route_decision(args.route_decision)})
    checks.append(
        {
            "name": "manifest_validation_command",
            **run_cmd(
                [
                    sys.executable,
                    "scripts/validate_dataset_delivery_manifest.py",
                    "--manifest",
                    str(args.delivery_manifest),
                    "--output",
                    str(args.manifest_validation),
                ],
                args.repo,
            ),
        }
    )
    checks.append(
        {
            "name": "package_validation_command",
            **run_cmd(
                [
                    sys.executable,
                    "scripts/validate_dataset_package.py",
                    "--package-dir",
                    str(args.package_dir),
                    "--output",
                    str(args.package_validation),
                ],
                args.repo,
            ),
        }
    )
    checks.append({"name": "package_metrics", **check_package_metrics(args.package_dir / "package_manifest.json")})

    required_paths = [
        args.package_dir / "qa_index.html",
        args.package_dir / "qa_index.md",
        args.package_dir / "artifacts/dense_semantic_route_decision.json",
        args.package_dir / "artifacts/conceptseg_instance_intersection.json",
        args.package_dir / "artifacts/old_route_reference_validation.json",
    ]
    missing = [str(path) for path in required_paths if not path.exists() or path.stat().st_size <= 0]
    checks.append({"name": "required_qa_artifacts", "passed": not missing, "errors": missing})

    result = {
        "passed": all(check.get("passed") for check in checks),
        "checks": checks,
        "next_manual_gate": "visual_acceptance_in_ply_viewer_or_cloudcompare",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
