import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gate_sam2_trt_promotion.py"


def write_compare(path: Path, **summary_overrides):
    summary = {
        "images": 10,
        "ok_images": 10,
        "missing_baseline": 0,
        "missing_candidate": 0,
        "mean_coverage_delta": 0.02,
        "mean_matched_iou": 0.95,
        "mean_unmatched_baseline_masks": 2.0,
        "mean_unmatched_candidate_masks": 5.0,
    }
    summary.update(summary_overrides)
    rows = [
        {
            "image_id": f"cam0_{i:06d}",
            "status": "ok",
            "coverage_delta": 0.02,
        }
        for i in range(10)
    ]
    path.write_text(json.dumps({"summary": summary, "rows": rows}), encoding="utf-8")


def test_gate_passes_for_within_thresholds(tmp_path):
    compare = tmp_path / "compare.json"
    output = tmp_path / "gate.json"
    write_compare(compare)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--compare-json", str(compare), "--output", str(output)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"


def test_gate_fails_for_excess_candidate_masks(tmp_path):
    compare = tmp_path / "compare.json"
    output = tmp_path / "gate.json"
    write_compare(compare, mean_unmatched_candidate_masks=12.0)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--compare-json", str(compare), "--output", str(output)],
        check=False,
        text=True,
        capture_output=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert result.returncode == 2
    assert report["status"] == "fail"
    assert "mean_unmatched_candidate_masks" in report["reasons"][0]
