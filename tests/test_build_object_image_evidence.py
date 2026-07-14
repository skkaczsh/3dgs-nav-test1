import numpy as np

from scripts import build_object_image_evidence as module


def test_projected_view_selection_beats_nearest_backfacing_pose(monkeypatch) -> None:
    poses = [
        {"pos": [0.0, 0.0, 0.0], "name": "near_backfacing"},
        {"pos": [10.0, 0.0, 0.0], "name": "far_visible"},
    ]

    def project(points, pose, cam_id, min_depth):
        count = 0 if pose["name"] == "near_backfacing" else 4
        return np.zeros((count, 2), dtype=np.float32), np.ones(count, dtype=np.float32)

    monkeypatch.setattr(module, "project_points", project)
    selected = module.choose_frame_pool(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32), poses, 1, 0.0, "projected", 0.1
    )
    assert selected[0]["name"] == "far_visible"


def test_source_frame_selection_uses_only_raw_section_support() -> None:
    poses = {
        10: {"frame_id": 10, "name": "raw_source"},
        20: {"frame_id": 20, "name": "second_source"},
        30: {"frame_id": 30, "name": "unrelated_visible"},
    }
    selected = module.choose_source_frame_pool(7, {7: [20, 10]}, poses, 8)
    assert [pose["name"] for pose in selected] == ["second_source", "raw_source"]
