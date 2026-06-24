from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.validate_current_mainline import validate_promotion_gate


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
    assert "promotion_gate_health:promotion_candidate_waiting_for_visual_acceptance" in report["warnings"]


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
