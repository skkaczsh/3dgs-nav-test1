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


def test_evidence_pose_uses_the_crop_frame_not_the_last_candidate() -> None:
    poses = {10: {"frame_id": 10, "name": "crop"}, 20: {"frame_id": 20, "name": "last_candidate"}}
    assert module.pose_for_evidence({"frame_id": 10}, poses)["name"] == "crop"


def test_depth_cache_evicts_oldest_frame_camera_pair() -> None:
    cache = module.OrderedDict()
    module.remember_depth_buffer(cache, (1, 0), np.ones((1, 1), dtype=np.float32), 1)
    module.remember_depth_buffer(cache, (2, 0), np.ones((1, 1), dtype=np.float32), 1)
    assert list(cache) == [(2, 0)]


def test_draw_world_up_arrow_changes_image() -> None:
    image = np.zeros((160, 160, 3), dtype=np.uint8)
    module.draw_world_up_arrow(image, {"world_up_image_unit_xy": [0.0, -1.0]})
    assert int(image.sum()) > 0


def test_camera_pose_context_uses_projection_chain(monkeypatch) -> None:
    identity = np.eye(4, dtype=np.float64)
    monkeypatch.setattr(module.config, "Til", identity)
    monkeypatch.setattr(module.config, "Tcl", [identity])
    hint = module.camera_pose_context(
        np.array([[0.0, 0.0, 2.0]], dtype=np.float32),
        {"T_world_robot": identity},
        0,
    )
    assert hint["camera_pose_hint"] == "calibrated"
    assert hint["camera_center_world"] == [0.0, 0.0, 0.0]
    assert hint["camera_forward_world_unit"] == [0.0, 0.0, 1.0]
    assert hint["camera_image_up_world_unit"] == [0.0, -1.0, 0.0]
    assert hint["object_relative_height_m"] == 2.0
    assert hint["object_view_elevation_deg"] == 90.0
