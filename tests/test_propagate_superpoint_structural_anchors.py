from scripts.propagate_superpoint_structural_anchors import propagate


def test_propagation_stops_after_short_color_compatible_path() -> None:
    edges = [
        {"object_a": 1, "object_b": 2, "shared_voxel_faces": 100, "contact_rgb_distance": 0.0},
        {"object_a": 2, "object_b": 3, "shared_voxel_faces": 100, "contact_rgb_distance": 0.0},
        {"object_a": 3, "object_b": 4, "shared_voxel_faces": 100, "contact_rgb_distance": 0.0},
        {"object_a": 1, "object_b": 5, "shared_voxel_faces": 100, "contact_rgb_distance": 100.0},
    ]
    anchors = [{"object_id": 1, "anchor_label": "floor", "propagation_eligible": True}]
    rows, report = propagate(edges, anchors, 10, 100, 40.0, 2, 0.35, 0.15)
    by_id = {row["object_id"]: row for row in rows}
    assert by_id[2]["propagation_status"] == "promoted"
    assert by_id[3]["propagation_status"] == "promoted"
    assert 4 not in by_id
    assert by_id[5]["propagation_status"] == "ambiguous_or_weak"
    assert report["retained_edges"] == 4
