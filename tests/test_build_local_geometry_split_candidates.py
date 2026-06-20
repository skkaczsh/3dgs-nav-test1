import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "build_local_geometry_split_candidates_for_test",
        SCRIPTS / "build_local_geometry_split_candidates.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def args(**kwargs):
    defaults = {
        "labels": "railing",
        "min_points": 2000,
        "require_reasons": "large_single_target_object,railing_not_linear,railing_extent_too_large",
        "limit": 0,
    }
    defaults.update(kwargs)
    return type("Args", (), defaults)()


def test_selects_large_risky_railing_candidate():
    module = load_module()
    rows = [
        {
            "object_id": 10,
            "semantic_label": "railing",
            "status": "single_target",
            "point_count": 3000,
            "target_count": 1,
            "bbox_3d": {"min": [0, 0, 0], "max": [4, 4, 2.5]},
            "geometry_stats": {"linearity_mean": 0.1, "planarity_mean": 0.6},
            "label_votes": {"railing": 3000},
        },
        {
            "object_id": 11,
            "semantic_label": "railing",
            "status": "single_target",
            "point_count": 100,
            "target_count": 1,
            "bbox_3d": {"min": [0, 0, 0], "max": [1, 0.1, 0.1]},
            "geometry_stats": {"linearity_mean": 0.9, "planarity_mean": 0.1},
        },
        {
            "object_id": 12,
            "semantic_label": "wall",
            "status": "stable",
            "point_count": 5000,
            "target_count": 2,
            "bbox_3d": {"min": [0, 0, 0], "max": [4, 4, 4]},
        },
    ]

    selected, summary = module.select_candidates(rows, args())

    assert [row["object_id"] for row in selected] == [10]
    assert selected[0]["suggested_action"] == "split_railing_candidate"
    assert "railing_not_linear" in selected[0]["matched_reasons"]
    assert summary["selected_candidates"] == 1
    assert summary["skipped"]["min_points"] == 1
    assert summary["skipped"]["label"] == 1


def test_limit_keeps_highest_score_first():
    module = load_module()
    rows = [
        {
            "object_id": 1,
            "semantic_label": "railing",
            "status": "stable",
            "point_count": 2500,
            "target_count": 2,
            "bbox_3d": {"min": [0, 0, 0], "max": [1, 1, 0.2]},
            "geometry_stats": {"linearity_mean": 0.1},
        },
        {
            "object_id": 2,
            "semantic_label": "railing",
            "status": "single_target",
            "point_count": 9000,
            "target_count": 1,
            "bbox_3d": {"min": [0, 0, 0], "max": [10, 1, 2]},
            "geometry_stats": {"linearity_mean": 0.1},
        },
    ]

    selected, _summary = module.select_candidates(rows, args(limit=1))

    assert [row["object_id"] for row in selected] == [2]


def test_select_candidates_uses_viewer_object_id_for_string_object_ids():
    module = load_module()
    rows = [
        {
            "object_id": "obj_000028",
            "viewer_object_id": 28,
            "semantic_label": "railing",
            "status": "stable",
            "point_count": 11537,
            "target_count": 2,
            "bbox_3d": {"min": [0, 0, 0], "max": [9, 0.5, 1.9]},
            "geometry_stats": {"linearity_mean": 0.9, "planarity_mean": 0.05},
            "label_votes": {"railing": 11537},
        }
    ]

    selected, _summary = module.select_candidates(rows, args(require_reasons="railing_extent_too_large,large_fine_object"))

    assert selected[0]["object_id"] == 28
    assert selected[0]["source_object_id"] == "obj_000028"
    assert "large_fine_object" in selected[0]["matched_reasons"]
