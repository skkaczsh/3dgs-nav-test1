from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.gate_patch_experiment_promotion import evaluate


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gate_patch_experiment_promotion.py"


def write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def visual(status: str = "accepted", candidate: str = "v2_bucket_attach") -> dict:
    return {
        "schema": "patch-experiment-visual-acceptance/v1",
        "status": status,
        "selected_candidate": candidate,
        "candidate_policy": "geometry_input_only",
        "review_index_url": "http://127.0.0.1:8765/docs/patch_experiment_review_index.html",
        "reviewer": "tester",
        "reviewed_at": "2026-07-02T20:00:00+08:00",
        "checks": [
            {"id": "metric_comparison_reviewed", "required": True, "status": "accepted"},
            {"id": "no_major_structure_overmerge", "required": True, "status": "accepted"},
        ],
    }


def args(tmp_path: Path, visual_path: Path):
    return argparse.Namespace(
        visual_acceptance=visual_path,
        output=tmp_path / "gate.json",
    )


def test_gate_fails_without_visual_acceptance(tmp_path: Path) -> None:
    result = evaluate(args(tmp_path, tmp_path / "missing.json"))

    assert result["status"] == "fail"
    assert any("missing_visual_acceptance" in reason for reason in result["reasons"])


def test_gate_passes_with_accepted_visual_acceptance(tmp_path: Path) -> None:
    visual_path = write_json(tmp_path / "visual.json", visual())

    result = evaluate(args(tmp_path, visual_path))

    assert result["status"] == "pass"
    assert result["reasons"] == []


def test_gate_rejects_pending_required_checks(tmp_path: Path) -> None:
    record = visual(status="pending")
    record["checks"][0]["status"] = "pending"
    visual_path = write_json(tmp_path / "visual.json", record)

    result = evaluate(args(tmp_path, visual_path))

    assert result["status"] == "fail"
    assert any("visual_status_not_accepted" in reason for reason in result["reasons"])
    assert any("visual_required_checks_not_accepted" in reason for reason in result["reasons"])


def test_gate_rejects_unknown_candidate(tmp_path: Path) -> None:
    visual_path = write_json(tmp_path / "visual.json", visual(candidate="v1_bucket_split"))

    result = evaluate(args(tmp_path, visual_path))

    assert result["status"] == "fail"
    assert any("candidate_not_allowed" in reason for reason in result["reasons"])


def test_cli_writes_failure_report_when_visual_acceptance_missing(tmp_path: Path) -> None:
    output = tmp_path / "gate.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--visual-acceptance",
            str(tmp_path / "missing.json"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "fail"
