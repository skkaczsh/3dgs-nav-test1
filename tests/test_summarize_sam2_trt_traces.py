import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_sam2_trt_traces.py"


def test_trace_summary_computes_stage_ratios(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "cam0_000001_trace.json").write_text(
        json.dumps(
            {
                "image_name": "cam0_000001",
                "crop_boxes": 5,
                "totals": {
                    "raw_candidates": 100,
                    "after_within_crop_nms": 20,
                    "dropped_near_crop_edge": 5,
                    "after_crop_edge_filter": 15,
                    "before_cross_crop_nms": 15,
                    "after_cross_crop_nms": 10,
                    "after_overlap_resolution": 8,
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "summary.json"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--trace-dir", str(trace_dir), "--json-output", str(output)],
        check=True,
        text=True,
        capture_output=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["images"] == 1
    assert report["summary"]["within_crop_nms_keep_ratio"] == 0.2
    assert report["summary"]["crop_edge_drop_ratio"] == 0.25
    assert report["summary"]["cross_crop_nms_keep_ratio"] == 10 / 15
    assert report["summary"]["overlap_resolution_keep_ratio"] == 0.8
    assert report["rows"][0]["image_id"] == "cam0_000001"
    assert "within_crop_nms_keep_ratio" in result.stdout
