from pathlib import Path
import sys

import numpy as np

from scripts import build_object_image_evidence as module


def test_dataset_cli_configures_calibration_paths(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sys, "argv", ["evidence.py", "--data-dir", str(tmp_path)])
    monkeypatch.delenv("SCAN_DATA_DIR", raising=False)
    monkeypatch.delenv("SCAN_IMAGE_DIR", raising=False)
    module.configure_dataset_from_cli()
    assert module.os.environ["SCAN_DATA_DIR"] == str(tmp_path.resolve())
    assert module.os.environ["SCAN_IMAGE_DIR"] == str(tmp_path.resolve() / "image")


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


def test_batched_projected_selection_matches_calibrated_scalar_chain(monkeypatch) -> None:
    monkeypatch.setattr(module.config, "IMAGE_WIDTH", 100)
    monkeypatch.setattr(module.config, "IMAGE_HEIGHT", 100)
    monkeypatch.setattr(module.config, "Til", np.eye(4, dtype=np.float64))
    monkeypatch.setattr(module.config, "Tcl", [np.eye(4, dtype=np.float64) for _ in range(3)])
    monkeypatch.setattr(
        module.config,
        "CAMERA_PARAMS",
        [{"K": np.array([[40.0, 0.0, 50.0], [0.0, 40.0, 50.0], [0.0, 0.0, 1.0]])} for _ in range(3)],
    )
    identity = np.eye(4, dtype=np.float64)
    shifted = np.eye(4, dtype=np.float64)
    shifted[0, 3] = 10.0
    poses = [{"frame_id": 10, "T_world_robot": identity}, {"frame_id": 20, "T_world_robot": shifted}]
    points = np.array([[0.0, 0.0, 2.0], [0.2, 0.1, 2.0]], dtype=np.float32)

    scalar = module.choose_frame_pool(points, poses, 2, 0.0, "projected", 0.1)
    batched = module.choose_projected_frame_pool_batched(points, poses, 2, 0.1, pose_batch=1)

    assert [row["frame_id"] for row in batched] == [row["frame_id"] for row in scalar]


def test_first_touch_visibility_rejects_occluded_and_sky_pixels(monkeypatch) -> None:
    monkeypatch.setattr(module.config, "IMAGE_WIDTH", 8)
    monkeypatch.setattr(module.config, "IMAGE_HEIGHT", 8)

    def project(_points, _pose, _cam_id, _min_depth):
        return np.array([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=np.float32), np.array([2.0, 5.0, 2.0], dtype=np.float32)

    monkeypatch.setattr(module, "project_points", project)
    depth = np.full((8, 8), np.inf, dtype=np.float32)
    depth[2, 2] = 2.0
    depth[3, 3] = 2.0  # second point is behind the first touch.
    depth[4, 4] = 2.0
    sky = np.zeros((8, 8), dtype=np.uint8)
    sky[4, 4] = 255

    count = module.first_touch_visible_count(
        np.zeros((3, 3), dtype=np.float32), {}, 0, depth, sky, 0.1, 0.2, 0, 128,
    )

    assert count == 1


def test_source_frame_selection_uses_only_raw_section_support() -> None:
    poses = {
        10: {"frame_id": 10, "name": "raw_source"},
        20: {"frame_id": 20, "name": "second_source"},
        30: {"frame_id": 30, "name": "unrelated_visible"},
    }
    selected = module.choose_source_frame_pool(7, {7: [20, 10]}, poses, 8)
    assert [pose["name"] for pose in selected] == ["second_source", "raw_source"]


def test_source_frame_selection_skips_frames_without_materialized_points() -> None:
    poses = {10: {"frame_id": 10}, 20: {"frame_id": 20}, 30: {"frame_id": 30}}
    selected = module.choose_source_frame_pool(7, {7: [10, 20, 30]}, poses, 2, {20, 30})
    assert [pose["frame_id"] for pose in selected] == [20, 30]


def test_global_view_plan_reuses_only_declared_calibrated_poses(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(
        '{"schema":"global-evidence-view-plan/v1","objects":[{"object_id":7,"frame_ids":[20,10]}]}',
        encoding="utf-8",
    )
    poses = {10: {"frame_id": 10}, 20: {"frame_id": 20}}

    loaded = module.load_global_view_plan(plan, {7}, poses)

    assert [pose["frame_id"] for pose in loaded[7]] == [20, 10]


def test_global_view_plan_rejects_missing_requested_object(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text('{"schema":"global-evidence-view-plan/v1","objects":[]}', encoding="utf-8")

    try:
        module.load_global_view_plan(plan, {7}, {10: {"frame_id": 10}})
    except ValueError as exc:
        assert "missing requested object ids" in str(exc)
    else:
        raise AssertionError("missing object must not trigger an implicit global re-search")


def test_evidence_pose_uses_the_crop_frame_not_the_last_candidate() -> None:
    poses = {10: {"frame_id": 10, "name": "crop"}, 20: {"frame_id": 20, "name": "last_candidate"}}
    assert module.pose_for_evidence({"frame_id": 10}, poses)["name"] == "crop"


def test_source_frame_points_do_not_leak_between_observations() -> None:
    points = np.array([[1.0, 2.0, 3.0, 10.0], [4.0, 5.0, 6.0, 20.0], [7.0, 8.0, 9.0, 20.0]], dtype=np.float32)
    assert module.points_for_source_frame(points, 20).tolist() == [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    assert module.available_source_frames(points) == {10, 20}
    assert module.available_source_frames(points, 2) == {20}


def test_depth_cache_evicts_oldest_frame_camera_pair() -> None:
    cache = module.OrderedDict()
    module.remember_depth_buffer(cache, (1, 0), np.ones((1, 1), dtype=np.float32), 1)
    module.remember_depth_buffer(cache, (2, 0), np.ones((1, 1), dtype=np.float32), 1)
    assert list(cache) == [(2, 0)]


def test_sky_mask_path_accepts_current_and_legacy_names(tmp_path: Path) -> None:
    current = tmp_path / "cam1_0000249_sky.png"
    current.write_bytes(b"")
    assert module.sky_mask_path(tmp_path, 1, 249) == current
    current.unlink()
    legacy = tmp_path / "cam1_00249_sky.png"
    legacy.write_bytes(b"")
    assert module.sky_mask_path(tmp_path, 1, 249) == legacy


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
