from __future__ import annotations

import json
from pathlib import Path

from scripts import qa_object_energy as module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_scores_mixed_surface_and_teacher_conflict(tmp_path: Path) -> None:
    objects = tmp_path / "objects.jsonl"
    write_jsonl(
        objects,
        [
            {
                "object_id": 1,
                "semantic_label": "wall",
                "geometry_type": "mixed",
                "voxel_count": 1000,
                "patch_count": 3,
                "bucket_counts": {"horizontal": 450, "vertical": 450, "unknown": 100},
                "bbox_3d": {"min": [0, 0, 0], "max": [10, 1, 2]},
                "teacher_semantic_confidence": 0.1,
                "teacher_semantic_votes": {"floor": 300, "wall": 10},
                "teacher_allowed_votes": {"wall": 10},
                "teacher_vetoed_votes": {"floor": 300},
            },
            {
                "object_id": 2,
                "semantic_label": "floor",
                "geometry_type": "horizontal",
                "voxel_count": 900,
                "patch_count": 1,
                "bucket_counts": {"horizontal": 900},
                "bbox_3d": {"min": [0, 0, 0], "max": [3, 3, 0.1]},
            },
        ],
    )
    report = module.analyze(
        objects_jsonl=objects,
        ply=None,
        output_dir=tmp_path / "qa",
        voxel_size=0.2,
        max_overlap_pairs=10,
        top_n=5,
        min_mixed_bucket_ratio=0.2,
    )

    top = report["top_problem_objects"][0]
    assert top["object_id"] == 1
    assert "horizontal_vertical_same_object" in top["flags"]
    assert "surface_label_on_mixed_geometry" in top["flags"]
    assert "teacher_vetoed_votes_dominate_allowed_votes" in top["flags"]
    assert report["flag_counts"]["horizontal_vertical_same_object"] == 1
    assert report["risk_flag_counts"]["horizontal_vertical_same_object"] == 1
    assert report["high_risk_object_count"] == 1
    assert (tmp_path / "qa" / "object_energy_qa_report.json").exists()
    assert (tmp_path / "qa" / "object_energy_qa.md").exists()


def test_normalizes_numeric_bucket_ids() -> None:
    row = {"bucket_counts": {"1": 4, "2": 1, "4": 2}}
    counts = module.normalized_bucket_counts(row)
    assert counts["horizontal"] == 4
    assert counts["vertical"] == 1
    assert counts["rough_mixed"] == 2


def test_geometry_supported_surface_downgrades_teacher_conflict() -> None:
    row = {
        "object_id": 1,
        "semantic_label": "floor",
        "geometry_type": "horizontal",
        "voxel_count": 1000,
        "bucket_counts": {"horizontal": 1000},
        "teacher_semantic_confidence": 0.01,
        "teacher_semantic_votes": {"wall": 500, "floor": 1},
        "teacher_allowed_votes": {"floor": 1},
        "teacher_vetoed_votes": {"wall": 500},
    }

    scored = module.score_object(row, overlap_stats={}, min_mixed_bucket_ratio=0.2)

    assert "low_visual_support_for_geometry_supported_label" in scored["flags"]
    assert "teacher_vetoed_votes_resolved_by_geometry" in scored["flags"]
    assert "teacher_vetoed_votes_dominate_allowed_votes" not in scored["flags"]
    assert scored["energy_score"] < 1.0
    assert module.is_evidence_only_flag("low_visual_support_for_geometry_supported_label")
    assert not module.is_evidence_only_flag("surface_label_on_mixed_geometry")
