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
