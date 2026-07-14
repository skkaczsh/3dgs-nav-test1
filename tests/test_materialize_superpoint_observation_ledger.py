from scripts.materialize_superpoint_observation_ledger import materialize


def test_ledger_preserves_one_view_and_only_marks_true_source_frame() -> None:
    rows = materialize(
        [{"object_id": 7, "frame_id": 11, "cam_id": 1, "rank": 1, "projected_points": 12, "score": 4.0}],
        [{"object_id": 7, "count": 44, "geometry_type": "vertical_surface"}],
        [{"object_id": 7, "parsed": {"controlled_label": "wall", "description_zh": "white wall", "confidence": 0.9}}],
        [{"object_id": 7, "top_source_frames": [{"frame_id": 11}]}],
    )
    assert rows == [{
        "observation_id": "sp7:f11:c1:r1", "superpoint_id": 7, "frame_id": 11, "cam_id": 1,
        "rank": 1, "source_frame_confirmed": True, "geometry_type": "vertical_surface", "point_count": 44,
        "projected_points": 12, "depth_visible_ratio": 0.0, "sky_filtered_ratio": 0.0, "median_depth": 0.0,
        "evidence_score": 4.0, "crop_path": "", "overlay_path": "", "vlm_description_zh": "white wall",
        "vlm_candidate_label": "wall", "vlm_confidence": 0.9, "vlm_surface_fragment": False,
    }]
