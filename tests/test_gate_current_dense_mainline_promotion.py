from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.gate_current_dense_mainline_promotion import evaluate


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gate_current_dense_mainline_promotion.py"


def write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def qa(nonzero_surface: bool = False, unknown_delta: int = 0) -> dict:
    return {
        "schema": "current-dense-mainline-qa/v1",
        "object_refinement": {
            "metrics": {
                "delta_v8_minus_v7": {
                    "accepted_candidate_rows": 10,
                    "output_object_count": -10,
                    "mixed_object_voxel_ratio_020": -0.01,
                }
            }
        },
        "surface_guard": {
            "label_point_counts": {
                "delta_v17_minus_v9": {"floor": 1 if nonzero_surface else 0, "wall": 0}
            },
            "unknown_point_delta_v17_minus_v9": unknown_delta,
        },
    }


def visual(status: str = "accepted") -> dict:
    return {
        "schema": "current-dense-visual-acceptance/v1",
        "status": status,
        "accepted_candidate": "v8_object_refinement",
        "review_index_url": "http://127.0.0.1:8765/docs/current_dense_review_index.html",
        "reviewer": "tester",
        "reviewed_at": "2026-06-24T18:20:00+08:00",
        "checks": [
            {"id": "object_fragmentation", "required": True, "status": "accepted"},
            {"id": "no_obvious_overmerge", "required": True, "status": "accepted"},
        ],
    }


def args(tmp_path: Path, qa_path: Path, visual_path: Path | None, require_visual: bool = True):
    return argparse.Namespace(
        qa_json=qa_path,
        visual_acceptance=visual_path,
        output=tmp_path / "gate.json",
        min_accepted_delta=1.0,
        max_output_object_delta=0.0,
        max_overlap_delta=0.0,
        max_unknown_point_delta=0.0,
        no_require_visual_acceptance=not require_visual,
    )


def test_gate_fails_without_visual_acceptance(tmp_path: Path) -> None:
    qa_path = write_json(tmp_path / "qa.json", qa())

    result = evaluate(args(tmp_path, qa_path, tmp_path / "missing.json"))

    assert result["status"] == "fail"
    assert any("missing_visual_acceptance" in reason for reason in result["reasons"])


def test_gate_passes_with_visual_acceptance(tmp_path: Path) -> None:
    qa_path = write_json(tmp_path / "qa.json", qa())
    visual_path = write_json(tmp_path / "visual.json", visual())

    result = evaluate(args(tmp_path, qa_path, visual_path))

    assert result["status"] == "pass"
    assert result["reasons"] == []


def test_gate_rejects_surface_guard_label_changes(tmp_path: Path) -> None:
    qa_path = write_json(tmp_path / "qa.json", qa(nonzero_surface=True))
    visual_path = write_json(tmp_path / "visual.json", visual())

    result = evaluate(args(tmp_path, qa_path, visual_path))

    assert result["status"] == "fail"
    assert any("surface_guard_changed_labels" in reason for reason in result["reasons"])


def test_gate_rejects_unknown_point_spike(tmp_path: Path) -> None:
    qa_path = write_json(tmp_path / "qa.json", qa(unknown_delta=100))
    visual_path = write_json(tmp_path / "visual.json", visual())

    result = evaluate(args(tmp_path, qa_path, visual_path))

    assert result["status"] == "fail"
    assert any("unknown_point_delta" in reason for reason in result["reasons"])


def test_cli_writes_failure_report_when_visual_acceptance_missing(tmp_path: Path) -> None:
    qa_path = write_json(tmp_path / "qa.json", qa())
    output = tmp_path / "gate.json"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--qa-json", str(qa_path), "--visual-acceptance", str(tmp_path / "missing.json"), "--output", str(output)],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "fail"
