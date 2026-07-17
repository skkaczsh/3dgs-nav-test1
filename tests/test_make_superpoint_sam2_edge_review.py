from scripts.make_superpoint_sam2_edge_review import choose_shared_views


def test_choose_shared_views_requires_both_edge_endpoints_and_prefers_weaker_support() -> None:
    rows = [
        {"object_id": 1, "frame_id": 10, "cam_id": 0, "projected_points": 50},
        {"object_id": 2, "frame_id": 10, "cam_id": 0, "projected_points": 30},
        {"object_id": 1, "frame_id": 20, "cam_id": 1, "projected_points": 90},
        {"object_id": 2, "frame_id": 20, "cam_id": 1, "projected_points": 10},
        {"object_id": 1, "frame_id": 30, "cam_id": 0, "projected_points": 99},
    ]
    shared = choose_shared_views((1, 2), rows, max_views=3)
    assert [(a["frame_id"], b["frame_id"]) for a, b in shared] == [(10, 10), (20, 20)]
