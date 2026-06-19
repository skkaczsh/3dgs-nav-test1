from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "diagnose_frame_target_geometry_conflicts_for_test",
        SCRIPTS / "diagnose_frame_target_geometry_conflicts.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def args(**overrides):
    values = {
        "window_size": 100,
        "min_score": 25,
        "top_windows": 30,
        "top_findings": 50,
        "min_fine_points": 80,
        "fine_horizontal_normal_z": 0.92,
        "fine_surface_like_normal_z": 0.85,
        "fine_surface_like_planarity": 0.55,
        "car_flat_max_z_span": 0.25,
        "car_max_z_span": 3.0,
        "car_max_horizontal_extent": 10.0,
        "railing_flat_max_z_span": 0.25,
        "railing_flat_min_planarity": 0.35,
        "railing_keep_linearity": 0.82,
        "railing_max_z_span": 2.2,
        "railing_max_horizontal_extent": 8.0,
        "surface_min_planarity": 0.35,
        "ground_min_normal_z": 0.55,
        "ground_max_z_span": 0.9,
        "ground_height_span_max_points": 5000,
        "wall_horizontal_normal_z": 0.75,
        "wall_min_z_span": 0.4,
        "wall_flat_min_horizontal_extent": 1.5,
        "ceiling_min_normal_z": 0.75,
    }
    values.update(overrides)
    return type("Args", (), values)()


def target(label: str, **overrides):
    row = {
        "target_id": f"t_{label}",
        "target_index": 1,
        "frame_id": 1234,
        "cam_id": 2,
        "mask_id": 5,
        "label": label,
        "raw_label": label,
        "cluster_size": 100,
        "bbox_3d": {"min": [0, 0, 0], "max": [2, 1, 0.1]},
        "centroid": [1, 0.5, 0.05],
        "pca": {"normal": [0, 0, 1], "linearity": 0.2, "planarity": 0.7},
        "image_path": "image.jpg",
        "mask_path": "mask.png",
    }
    row.update(overrides)
    return row


def test_detects_flat_horizontal_railing_target():
    module = load_module()

    score, reasons, action = module.assess_target(target("railing"), args())

    assert score >= 75
    assert "railing_flat_horizontal_surface" in reasons
    assert action == "mask_review_or_demote_surface"


def test_keeps_linear_railing_target():
    module = load_module()
    row = target("railing", pca={"normal": [0, 0, 1], "linearity": 0.9, "planarity": 0.1})

    score, reasons, _action = module.assess_target(row, args())

    assert "railing_flat_horizontal_surface" not in reasons
    assert score == 0


def test_detects_flat_car_target():
    module = load_module()

    score, reasons, action = module.assess_target(target("car"), args())

    assert score >= 70
    assert "car_flat_horizontal_surface" in reasons
    assert action == "mask_review_or_demote_surface"


def test_diagnose_aggregates_top_windows():
    module = load_module()
    rows = [
        target("car", target_id="car1", frame_id=1234, cam_id=0),
        target("wall", target_id="wall1", frame_id=1240, cam_id=0),
        target(
            "ground",
            target_id="ground_ok",
            frame_id=1300,
            cam_id=1,
            bbox_3d={"min": [0, 0, 0], "max": [2, 2, 0.05]},
            pca={"normal": [0, 0, 1], "linearity": 0.2, "planarity": 0.7},
        ),
    ]

    findings, report = module.diagnose_targets(rows, args())

    assert len(findings) == 2
    assert report["finding_label_counts"]["car"] == 1
    assert report["finding_label_counts"]["wall"] == 1
    assert report["top_windows"][0]["window"] == "001200_001300"
    assert report["top_windows"][0]["cam_id"] == 0
