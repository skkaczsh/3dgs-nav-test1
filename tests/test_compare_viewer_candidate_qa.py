from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "compare_viewer_candidate_qa_for_test",
    SCRIPTS / "compare_viewer_candidate_qa.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def write_report(path: Path, **overrides):
    data = {
        "status": "ok",
        "warnings": ["large car/railing objects exist; inspect for surface swallowing"],
        "errors": [],
        "ply": {"data_rows": 100},
        "objects": {
            "object_count": 2,
            "label_point_counts": {"ground": 40, "wall": 30, "railing": 30},
            "label_object_counts": {"ground": 1, "railing": 1},
            "status_counts": {"stable": 1, "single_target": 1},
            "large_fine_objects": [
                {"object_id": 7, "semantic_label": "railing", "point_count": 30},
            ],
        },
    }
    data.update(overrides)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_parse_named_report():
    name, path = module.parse_named_report("candidate=/tmp/viewer_candidate_qa.json")

    assert name == "candidate"
    assert path == Path("/tmp/viewer_candidate_qa.json")


def test_build_comparison_deltas_label_points_and_large_fine_objects(tmp_path: Path):
    baseline = write_report(tmp_path / "baseline.json")
    candidate = write_report(
        tmp_path / "candidate.json",
        warnings=[],
        objects={
            "object_count": 3,
            "label_point_counts": {"ground": 42, "wall": 33, "unknown": 25},
            "label_object_counts": {"ground": 2, "wall": 1},
            "status_counts": {"stable": 2, "priority_unknown_local_geometry_child": 1},
            "large_fine_objects": [],
        },
    )

    comparison = module.build_comparison([("baseline", baseline), ("candidate", candidate)])

    delta = comparison["deltas_from_baseline"]["candidate"]
    assert delta["object_count"] == 1
    assert delta["large_fine_object_count"] == -1
    assert delta["large_fine_object_points"] == -30
    assert delta["label_point_counts"]["railing"] == -30
    assert delta["label_point_counts"]["unknown"] == 25


def test_markdown_contains_warning_and_label_tables(tmp_path: Path):
    baseline = write_report(tmp_path / "baseline.json")
    comparison = module.build_comparison([("baseline", baseline)])

    markdown = module.markdown_report(comparison)

    assert "Label Point Counts" in markdown
    assert "large car/railing objects exist" in markdown
    assert "large fine object ids" in markdown
