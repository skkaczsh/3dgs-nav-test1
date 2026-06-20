import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import apply_geometry_conflict_demotions as mod


def test_apply_demotions_only_changes_matching_label_and_action(tmp_path: Path) -> None:
    objects = tmp_path / "objects.jsonl"
    objects.write_text(
        '{"object_id":"obj_000001","viewer_object_id":1,"semantic_label":"car","point_count":10}\n'
        '{"object_id":"obj_000002","viewer_object_id":2,"semantic_label":"car","point_count":20}\n'
        '{"object_id":"obj_000003","viewer_object_id":3,"semantic_label":"wall","point_count":30}\n',
        encoding="utf-8",
    )
    conflicts = tmp_path / "conflicts.jsonl"
    conflicts.write_text(
        '{"object_id":1,"semantic_label":"car","suggested_action":"demote_or_visual_review","reasons":["car_too_flat"],"metrics":{"z_extent":0.1}}\n'
        '{"object_id":2,"semantic_label":"car","suggested_action":"relabel_car_to_wall","reasons":["car_on_vertical_surface_region"],"metrics":{}}\n'
        '{"object_id":3,"semantic_label":"wall","suggested_action":"demote_or_visual_review","reasons":["wall_low_planarity"],"metrics":{}}\n'
        '{"object_id":9,"semantic_label":"car","suggested_action":"demote_or_visual_review","reasons":["car_high_centroid_z"],"metrics":{}}\n',
        encoding="utf-8",
    )

    rows, report = mod.apply_demotions(
        objects,
        conflicts,
        source_label="car",
        action="demote_or_visual_review",
        target_label="unknown",
        status="geometry_demoted_visual_review",
    )

    assert rows[0]["semantic_label"] == "unknown"
    assert rows[0]["semantic_label_original"] == "car"
    assert rows[0]["semantic_id"] == 0
    assert rows[0]["status"] == "geometry_demoted_visual_review"
    assert rows[0]["geometry_demotion_reasons"] == ["car_too_flat"]
    assert rows[1]["semantic_label"] == "car"
    assert rows[2]["semantic_label"] == "wall"
    assert report["candidate_count"] == 2
    assert report["applied_count"] == 1
    assert report["missing_object_ids"] == [9]
    assert report["label_counts_after"] == {"unknown": 1, "car": 1, "wall": 1}


def test_apply_demotions_rejects_unknown_target_label(tmp_path: Path) -> None:
    objects = tmp_path / "objects.jsonl"
    conflicts = tmp_path / "conflicts.jsonl"
    objects.write_text("", encoding="utf-8")
    conflicts.write_text("", encoding="utf-8")

    try:
        mod.apply_demotions(objects, conflicts, "car", "demote_or_visual_review", "bad_label", "status")
    except ValueError as exc:
        assert "Unknown target label" in str(exc)
    else:
        raise AssertionError("expected ValueError")
