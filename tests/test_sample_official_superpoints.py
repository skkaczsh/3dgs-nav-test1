from pathlib import Path

import numpy as np

from scripts.sample_official_superpoints import read_jsonl, reservoir_update, select_rows, source_frame_ids


def test_select_rows_stratifies_only_supported_objects() -> None:
    rows = [
        {"object_id": 1, "geometry_type": "horizontal"},
        {"object_id": 2, "geometry_type": "horizontal"},
        {"object_id": 3, "geometry_type": "vertical"},
    ]
    selected = select_rows(rows, {1, 3}, per_geometry=2, min_object_points=0, seed=17)
    assert [row["object_id"] for row in selected] == [1, 3]


def test_select_rows_can_require_a_reviewable_point_count() -> None:
    rows = [
        {"object_id": 1, "geometry_type": "horizontal", "count": 40},
        {"object_id": 2, "geometry_type": "horizontal", "count": 500},
    ]
    selected = select_rows(rows, {1, 2}, per_geometry=2, min_object_points=100, seed=17)
    assert [row["object_id"] for row in selected] == [2]


def test_preselected_rows_keep_candidate_metadata(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    path.write_text('{"object_id": 3, "seed_candidate_score": 2.5}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"object_id": 3, "seed_candidate_score": 2.5}]


def test_source_frame_ids_filters_to_selected_superpoints(tmp_path: Path) -> None:
    path = tmp_path / "support.jsonl"
    path.write_text(
        '{"object_id": 3, "top_source_frames": [{"frame_id": 10}]}\n'
        '{"object_id": 4, "top_source_frames": [{"frame_id": 20}]}\n',
        encoding="utf-8",
    )
    assert source_frame_ids(path, {4}) == {4: [20]}


def test_reservoir_update_keeps_bounded_sample() -> None:
    points, keys = {}, {}
    reservoir_update(points, keys, 7, np.arange(30, dtype=np.float32).reshape(10, 3), 3, np.random.default_rng(17))
    assert points[7].shape == (3, 3)
    assert keys[7].shape == (3,)
