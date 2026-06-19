from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "trace_risky_objects_to_targets_for_test",
    SCRIPTS / "trace_risky_objects_to_targets.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_trace_objects_compacts_top_target_evidence():
    objects = [
        {
            "object_id": "obj_ground_bad",
            "semantic_label": "ground",
            "status": "single_target",
            "point_count": 1000,
            "target_count": 1,
            "targets": ["t2", "missing"],
            "bbox_3d": {"min": [0, 0, 0], "max": [1, 1, 2]},
            "label_votes": {"ground": 1000},
        },
        {
            "object_id": "obj_clean",
            "semantic_label": "wall",
            "status": "stable",
            "point_count": 100,
            "target_count": 2,
            "targets": ["t1"],
            "bbox_3d": {"min": [0, 0, 0], "max": [0.1, 0.1, 2]},
            "normal": [1, 0, 0],
            "geometry_stats": {"planarity_mean": 0.8},
            "label_votes": {"wall": 100},
        },
    ]
    targets = [
        {
            "target_id": "t1",
            "frame_id": 10,
            "cam_id": 0,
            "mask_id": 2,
            "label": "wall",
            "cluster_size": 20,
            "point_indices": [1, 2, 3],
        },
        {
            "target_id": "t2",
            "frame_id": 20,
            "cam_id": 1,
            "mask_id": 1,
            "label": "ground",
            "raw_label": "ground",
            "cluster_size": 500,
            "bbox_3d": {"min": [0, 0, 0], "max": [1, 2, 3]},
            "bbox_2d": {"xyxy": [1, 2, 3, 4]},
            "pca": {"linearity": 0.1, "planarity": 0.9, "normal": [0, 0, 1]},
            "image_path": "/tmp/frame.jpg",
            "mask_path": "/tmp/mask.png",
            "point_indices": list(range(100)),
        },
    ]

    rows = module.trace_objects(objects, targets, limit=10, evidence_per_object=1)

    assert len(rows) == 1
    row = rows[0]
    assert row["object_id"] == "obj_ground_bad"
    assert "ground_has_large_height_span" in row["risk_reasons"]
    assert row["missing_target_count"] == 1
    assert row["top_targets"][0]["target_id"] == "t2"
    assert "point_indices" not in row["top_targets"][0]


def test_write_outputs_jsonl_and_csv(tmp_path: Path):
    rows = [
        {
            "object_id": "obj_1",
            "semantic_label": "car",
            "status": "single_target",
            "risk_score": 90,
            "risk_reasons": ["car_extent_suspicious"],
            "point_count": 800,
            "target_count": 1,
            "top_targets": [
                {
                    "target_id": "t1",
                    "frame_id": 30,
                    "cam_id": 2,
                    "mask_id": 8,
                    "cluster_size": 123,
                    "bbox_3d": {"min": [0, 0, 0], "max": [1, 2, 0.2]},
                    "normal": [0.0, 0.0, 1.0],
                    "linearity": 0.8,
                    "planarity": 0.2,
                    "refinement_reasons": ["example_reason"],
                    "image_path": "/tmp/image.jpg",
                    "mask_path": "/tmp/mask.png",
                }
            ],
        }
    ]
    jsonl = tmp_path / "trace.jsonl"
    csv_path = tmp_path / "trace.csv"

    module.write_jsonl(rows, jsonl)
    module.write_csv(rows, csv_path)

    assert json.loads(jsonl.read_text().strip())["object_id"] == "obj_1"
    with csv_path.open(newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert csv_rows[0]["first_frame"] == "30"
    assert csv_rows[0]["risk_reasons"] == "car_extent_suspicious"
    assert csv_rows[0]["first_cluster_size"] == "123"
    assert csv_rows[0]["first_bbox_dz"] == "0.2"
    assert csv_rows[0]["first_normal_z"] == "1.0"
    assert csv_rows[0]["first_refinement_reasons"] == "example_reason"
