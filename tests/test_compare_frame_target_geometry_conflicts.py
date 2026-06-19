from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "compare_frame_target_geometry_conflicts_for_test",
    SCRIPTS / "compare_frame_target_geometry_conflicts.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def write_report(path: Path, **overrides):
    data = {
        "target_count": 10,
        "finding_count": 2,
        "finding_label_counts": {"railing": 1, "wall": 1},
        "top_windows": [
            {"window": "0000_0100", "cam_id": 1, "finding_points": 30, "score_sum": 70},
        ],
    }
    data.update(overrides)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_parse_named_report():
    name, path = module.parse_named_report("refined=/tmp/report.json")

    assert name == "refined"
    assert path == Path("/tmp/report.json")


def test_build_comparison_reports_deltas(tmp_path: Path):
    baseline = write_report(tmp_path / "baseline.json")
    refined = write_report(
        tmp_path / "refined.json",
        target_count=12,
        finding_count=4,
        finding_label_counts={"railing": 3, "wall": 1},
        top_windows=[
            {"window": "0000_0100", "cam_id": 1, "finding_points": 40, "score_sum": 120},
            {"window": "0000_0100", "cam_id": 2, "finding_points": 10, "score_sum": 20},
        ],
    )

    comparison = module.build_comparison([("baseline", baseline), ("refined", refined)])

    delta = comparison["deltas_from_baseline"]["refined"]
    assert delta["target_count"] == 2
    assert delta["finding_count"] == 2
    assert delta["finding_points"] == 20
    assert delta["top_window_score_sum"] == 70
    assert delta["finding_label_counts"]["railing"] == 2


def test_markdown_contains_delta_section(tmp_path: Path):
    baseline = write_report(tmp_path / "baseline.json")
    comparison = module.build_comparison([("baseline", baseline)])

    text = module.markdown(comparison)

    assert "Frame-Target Geometry Conflict Comparison" in text
    assert "Delta From Baseline" in text
