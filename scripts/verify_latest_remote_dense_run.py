#!/usr/bin/env python3
"""Verify latest_remote_run metrics against its generated reports.

This is an operator check for remote evidence.  It is intentionally separate
from validate_current_mainline.py because it may use SSH and should not make
local health checks depend on network availability.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = REPO_ROOT / "docs" / "current_dense_patch_state.json"
CANDIDATE_REPORT_REL = (
    "object_merge_candidates_v7_structural_multimaterial/geo_patch_object_merge_candidates_report.json"
)
OBJECT_REPORT_REL = "objects_v7_structural_multimaterial/geo_patch_objects_v7_structural_multimaterial_report.json"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def ssh_read_json(host: str, path: str) -> dict[str, Any]:
    result = subprocess.run(
        ["ssh", host, "cat", path],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ssh read failed for {host}:{path}: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    if not isinstance(data, dict):
        raise ValueError(f"{host}:{path} must contain a JSON object")
    return data


def ssh_file_exists(host: str, path: str) -> bool:
    result = subprocess.run(["ssh", host, "test", "-f", path], check=False)
    return result.returncode == 0


def nested_int(data: dict[str, Any], *keys: str) -> int:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return 0
        cur = cur.get(key)
    try:
        return int(cur)
    except (TypeError, ValueError):
        return 0


def expected_candidate_metrics(report: dict[str, Any]) -> dict[str, int]:
    return {
        "patch_count": nested_int(report, "patch_count"),
        "edge_pair_count": nested_int(report, "edge_pair_count"),
        "candidate_count": nested_int(report, "candidate_count"),
        "same_material_candidates": nested_int(report, "merge_class_counts", "same_material"),
        "structural_multimaterial_candidates": nested_int(report, "merge_class_counts", "structural_multimaterial"),
        "big_mixed_attachment_count": nested_int(report, "big_mixed_attachment_count"),
    }


def expected_object_metrics(report: dict[str, Any]) -> dict[str, int]:
    return {
        "input_patch_count": nested_int(report, "input_patch_count"),
        "input_candidate_count": nested_int(report, "input_candidate_count"),
        "accepted_candidate_rows": nested_int(report, "accepted_candidate_rows"),
        "output_object_count": nested_int(report, "output_object_count"),
        "preview_points_stride10": nested_int(report, "preview_points"),
    }


def int_counts(data: Any) -> dict[str, int]:
    if not isinstance(data, dict):
        return {}
    out: dict[str, int] = {}
    for key, value in data.items():
        try:
            out[str(key)] = int(value)
        except (TypeError, ValueError):
            out[str(key)] = 0
    return out


def object_rejection_counts(report: dict[str, Any]) -> dict[str, int]:
    reason_counts = int_counts(report.get("candidate_reason_counts"))
    return {
        key: value
        for key, value in reason_counts.items()
        if not key.startswith("accepted")
    }


def compare_metrics(prefix: str, expected: dict[str, int], actual: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key, value in expected.items():
        actual_value = nested_int(actual, key)
        if actual_value != value:
            errors.append(f"{prefix}_{key}_mismatch:state={actual_value}:report={value}")
    return errors


def compare_count_dict(prefix: str, expected: dict[str, int], actual: Any) -> list[str]:
    actual_counts = int_counts(actual)
    errors: list[str] = []
    if actual_counts != expected:
        errors.append(f"{prefix}_counts_mismatch:state={actual_counts}:report={expected}")
    return errors


def read_reports(latest: dict[str, Any], *, report_root: Path | None, ssh_host: str | None) -> tuple[dict[str, Any], dict[str, Any], bool]:
    remote_dir = str(latest.get("remote_dir", ""))
    if report_root is not None:
        candidate = read_json(report_root / CANDIDATE_REPORT_REL)
        obj = read_json(report_root / OBJECT_REPORT_REL)
        done = (report_root / "DONE").is_file()
        return candidate, obj, done
    host = ssh_host or str(latest.get("host", ""))
    if not host:
        raise ValueError("ssh host is required when --report-root is not provided")
    candidate = ssh_read_json(host, f"{remote_dir}/{CANDIDATE_REPORT_REL}")
    obj = ssh_read_json(host, f"{remote_dir}/{OBJECT_REPORT_REL}")
    done = ssh_file_exists(host, f"{remote_dir}/DONE")
    return candidate, obj, done


def validate(state_path: Path, *, report_root: Path | None = None, ssh_host: str | None = None) -> dict[str, Any]:
    state = read_json(state_path)
    latest = state.get("latest_remote_run")
    if not isinstance(latest, dict):
        raise ValueError("state missing latest_remote_run object")
    errors: list[str] = []
    candidate_report, object_report, done_exists = read_reports(latest, report_root=report_root, ssh_host=ssh_host)
    if latest.get("status") != "completed":
        errors.append(f"latest_status_not_completed={latest.get('status')!r}")
    if not done_exists:
        errors.append("latest_done_missing")
    errors.extend(
        compare_metrics(
            "candidate",
            expected_candidate_metrics(candidate_report),
            latest.get("candidate_metrics") or {},
        )
    )
    errors.extend(
        compare_metrics(
            "object",
            expected_object_metrics(object_report),
            latest.get("object_metrics") or {},
        )
    )
    candidate_metrics = latest.get("candidate_metrics") or {}
    object_metrics = latest.get("object_metrics") or {}
    errors.extend(
        compare_count_dict(
            "candidate_reject",
            int_counts(candidate_report.get("reject_counts")),
            candidate_metrics.get("reject_counts"),
        )
    )
    errors.extend(
        compare_count_dict(
            "object_rejection",
            object_rejection_counts(object_report),
            object_metrics.get("rejection_counts"),
        )
    )
    return {
        "schema": "latest-remote-dense-run-verification/v1",
        "passed": not errors,
        "state": str(state_path),
        "latest_id": latest.get("id"),
        "remote_dir": latest.get("remote_dir"),
        "report_root": str(report_root) if report_root is not None else None,
        "ssh_host": ssh_host or latest.get("host"),
        "done_exists": done_exists,
        "candidate_report_metrics": expected_candidate_metrics(candidate_report),
        "object_report_metrics": expected_object_metrics(object_report),
        "candidate_report_reject_counts": int_counts(candidate_report.get("reject_counts")),
        "object_report_rejection_counts": object_rejection_counts(object_report),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--report-root", type=Path, help="Local copy of latest_remote_run remote_dir")
    parser.add_argument("--ssh-host", help="Override latest_remote_run.host")
    args = parser.parse_args()
    report = validate(args.state, report_root=args.report_root, ssh_host=args.ssh_host)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
