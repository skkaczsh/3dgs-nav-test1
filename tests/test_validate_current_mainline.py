from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.gate_current_dense_mainline_promotion import evaluate as evaluate_dense_gate
from scripts.validate_current_mainline import validate_promotion_gate, validate_promotion_plan


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_current_mainline.py"


def test_current_mainline_healthcheck_passes_with_visual_pending() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["passed"] is True
    assert report["checks"]["review_artifact_allowlist"]["passed"] is True
    assert report["checks"]["promotion_plan_health"]["passed"] is True
    assert "promotion_gate_health:promotion_candidate_waiting_for_visual_acceptance" in report["warnings"]
    assert "promotion_plan_health:promotion_plan_waiting_for_gate_pass" in report["warnings"]


def test_promotion_gate_health_rejects_unknown_spike(tmp_path: Path) -> None:
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "schema": "current-dense-promotion-gate/v1",
                "status": "fail",
                "candidate": "v8_object_refinement",
                "metrics": {
                    "accepted_delta": 1,
                    "output_object_delta": -1,
                    "overlap_delta": -0.1,
                    "unknown_point_delta": 10,
                    "nonzero_surface_delta": {},
                },
                "reasons": ["visual_status_not_accepted=pending"],
            }
        ),
        encoding="utf-8",
    )

    report = validate_promotion_gate(gate)

    assert report["passed"] is False
    assert "promotion_gate_unknown_spike" in report["errors"]


def test_promotion_gate_health_rejects_stale_cached_metrics(tmp_path: Path) -> None:
    qa = tmp_path / "qa.json"
    visual = tmp_path / "visual.json"
    gate = tmp_path / "gate.json"
    qa.write_text(
        json.dumps(
            {
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
                    "label_point_counts": {"delta_v17_minus_v9": {"floor": 0}},
                    "unknown_point_delta_v17_minus_v9": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    visual.write_text(
        json.dumps(
            {
                "schema": "current-dense-visual-acceptance/v1",
                "status": "accepted",
                "accepted_candidate": "v8_object_refinement",
                "review_index_url": "http://127.0.0.1:8765/docs/current_dense_review_index.html",
                "checks": [{"id": "reviewed", "required": True, "status": "accepted"}],
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        qa_json=qa,
        visual_acceptance=visual,
        output=gate,
        min_accepted_delta=1.0,
        max_output_object_delta=0.0,
        max_overlap_delta=0.0,
        max_unknown_point_delta=0.0,
        no_require_visual_acceptance=False,
    )
    cached = evaluate_dense_gate(args)
    cached["metrics"]["accepted_delta"] = 9.0
    gate.write_text(json.dumps(cached), encoding="utf-8")

    report = validate_promotion_gate(gate)

    assert report["passed"] is False
    assert "promotion_gate_stale_metrics" in report["errors"]


def test_promotion_plan_health_rejects_candidate_mismatch(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    qa = tmp_path / "qa.json"
    gate = tmp_path / "gate.json"
    state.write_text(
        json.dumps(
            {
                "current_object_baseline": {"id": "old"},
                "current_promotion_candidate": {
                    "id": "v8_object_refinement",
                    "qa_candidate_id": "v8_tiny_attach",
                    "source_run_id": "dense_patch_object_refinement_v8_tiny_attach_20260624_170619",
                },
            }
        ),
        encoding="utf-8",
    )
    qa.write_text(
        json.dumps(
            {
                "object_refinement": {
                    "candidate": "wrong",
                    "metrics": {
                        "v8": {
                            "candidate_count": 1,
                            "accepted_candidate_rows": 1,
                            "output_object_count": 1,
                            "mixed_object_voxel_ratio_020": 0,
                            "object_count_in_overlap_preview": 1,
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    gate.write_text(json.dumps({"candidate": "v8_object_refinement", "status": "pass"}), encoding="utf-8")

    report = validate_promotion_plan(state, qa, gate)

    assert report["passed"] is False
    assert "promotion_plan_error=qa_candidate_mismatch=wrong!=v8_tiny_attach" in report["errors"]
