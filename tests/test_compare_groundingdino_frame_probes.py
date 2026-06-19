from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "compare_groundingdino_frame_probes_for_test",
    SCRIPTS / "compare_groundingdino_frame_probes.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def write_report(path: Path, **overrides):
    data = {
        "model": "tiny",
        "image_count": 10,
        "box_threshold": 0.35,
        "text_threshold": 0.25,
        "large_box_ratio": 0.12,
        "detection_counts": {"railing": 4, "car": 2},
        "large_box_counts": {"railing": 2},
        "mean_box_area_ratio_by_label": {"railing": 0.2, "car": 0.05},
        "max_box_area_ratio_by_label": {"railing": 0.6, "car": 0.1},
    }
    data.update(overrides)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_parse_named_report():
    name, path = module.parse_named_report("base=/tmp/report.json")

    assert name == "base"
    assert path == Path("/tmp/report.json")


def test_build_comparison_reports_counts_and_rates(tmp_path: Path):
    tiny = write_report(tmp_path / "tiny.json")
    base = write_report(
        tmp_path / "base.json",
        model="base",
        detection_counts={"railing": 5, "car": 1},
        large_box_counts={"railing": 1, "car": 1},
        mean_box_area_ratio_by_label={"railing": 0.12, "car": 0.4},
    )

    comparison = module.build_comparison([("tiny", tiny), ("base", base)])

    assert comparison["labels"] == ["railing", "car"]
    assert comparison["versions"]["tiny"]["large_box_rates"]["railing"] == 0.5
    delta = comparison["deltas_from_baseline"]["base"]
    assert delta["detection_counts"]["railing"] == 1
    assert delta["detection_counts"]["car"] == -1
    assert delta["large_box_counts"]["railing"] == -1
    assert delta["large_box_rates"]["car"] == 1.0


def test_markdown_contains_main_sections(tmp_path: Path):
    tiny = write_report(tmp_path / "tiny.json")
    comparison = module.build_comparison([("tiny", tiny)])

    text = module.markdown(comparison)

    assert "GroundingDINO Frame Probe Comparison" in text
    assert "Large Box Rates" in text
    assert "Delta From Baseline" in text
