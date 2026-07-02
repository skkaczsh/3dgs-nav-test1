from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.verify_latest_remote_dense_run import validate


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_latest_remote_dense_run.py"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def make_state(path: Path, remote_dir: str, *, accepted: int = 7, candidate_reject_score: int = 5) -> Path:
    write_json(
        path,
        {
            "latest_remote_run": {
                "id": "run_a",
                "status": "completed",
                "host": "scan-train",
                "remote_dir": remote_dir,
                "candidate_metrics": {
                    "patch_count": 10,
                    "edge_pair_count": 20,
                    "candidate_count": 3,
                    "same_material_candidates": 2,
                    "structural_multimaterial_candidates": 1,
                    "big_mixed_attachment_count": 4,
                    "reject_counts": {
                        "small_patch": 11,
                        "score": candidate_reject_score,
                    },
                },
                "object_metrics": {
                    "input_patch_count": 10,
                    "input_candidate_count": 3,
                    "accepted_candidate_rows": accepted,
                    "output_object_count": 5,
                    "preview_points_stride10": 100,
                    "rejection_counts": {
                        "score": 9,
                        "contact_ratio": 1,
                    },
                },
            }
        },
    )
    return path


def make_report_root(root: Path) -> Path:
    write_json(
        root
        / "object_merge_candidates_v7_structural_multimaterial"
        / "geo_patch_object_merge_candidates_report.json",
        {
            "patch_count": 10,
            "edge_pair_count": 20,
            "candidate_count": 3,
            "merge_class_counts": {"same_material": 2, "structural_multimaterial": 1},
            "big_mixed_attachment_count": 4,
            "reject_counts": {"small_patch": 11, "score": 5},
        },
    )
    write_json(
        root / "objects_v7_structural_multimaterial" / "geo_patch_objects_v7_structural_multimaterial_report.json",
        {
            "input_patch_count": 10,
            "input_candidate_count": 3,
            "accepted_candidate_rows": 7,
            "output_object_count": 5,
            "preview_points": 100,
            "candidate_reason_counts": {
                "accepted": 2,
                "accepted_structural_multimaterial": 5,
                "score": 9,
                "contact_ratio": 1,
            },
        },
    )
    (root / "DONE").write_text("done\n", encoding="utf-8")
    return root


def test_validate_latest_remote_dense_run_from_local_reports(tmp_path: Path) -> None:
    report_root = make_report_root(tmp_path / "run")
    state = make_state(tmp_path / "state.json", str(report_root))

    report = validate(state, report_root=report_root)

    assert report["passed"] is True
    assert report["done_exists"] is True
    assert report["object_report_metrics"]["accepted_candidate_rows"] == 7
    assert report["candidate_report_reject_counts"]["small_patch"] == 11
    assert report["object_report_rejection_counts"]["score"] == 9


def test_validate_latest_remote_dense_run_detects_metric_mismatch(tmp_path: Path) -> None:
    report_root = make_report_root(tmp_path / "run")
    state = make_state(tmp_path / "state.json", str(report_root), accepted=8)

    report = validate(state, report_root=report_root)

    assert report["passed"] is False
    assert "object_accepted_candidate_rows_mismatch:state=8:report=7" in report["errors"]


def test_validate_latest_remote_dense_run_detects_reject_count_mismatch(tmp_path: Path) -> None:
    report_root = make_report_root(tmp_path / "run")
    state = make_state(tmp_path / "state.json", str(report_root), candidate_reject_score=6)

    report = validate(state, report_root=report_root)

    assert report["passed"] is False
    assert any(error.startswith("candidate_reject_counts_mismatch") for error in report["errors"])


def test_cli_report_root_passes(tmp_path: Path) -> None:
    report_root = make_report_root(tmp_path / "run")
    state = make_state(tmp_path / "state.json", str(report_root))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--state", str(state), "--report-root", str(report_root)],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["passed"] is True
