from scripts.propagate_superpoint_structural_anchors import graph_node_ids, propagate


def test_propagation_stops_after_short_color_compatible_path() -> None:
    edges = [
        {"object_a": 1, "object_b": 2, "shared_voxel_faces": 100, "contact_rgb_distance": 0.0},
        {"object_a": 2, "object_b": 3, "shared_voxel_faces": 100, "contact_rgb_distance": 0.0},
        {"object_a": 3, "object_b": 4, "shared_voxel_faces": 100, "contact_rgb_distance": 0.0},
        {"object_a": 1, "object_b": 5, "shared_voxel_faces": 100, "contact_rgb_distance": 100.0},
    ]
    anchors = [{"object_id": 1, "anchor_label": "floor", "propagation_eligible": True}]
    rows, report = propagate(
        edges, anchors, 10, 100, 40.0, 2, 0.35, 0.15,
        {object_id: "horizontal" for object_id in range(1, 6)},
    )
    by_id = {row["object_id"]: row for row in rows}
    assert by_id[2]["propagation_status"] == "promoted"
    assert by_id[3]["propagation_status"] == "promoted"
    assert by_id[3]["structural_source_anchor"] == 1
    assert by_id[3]["structural_hops"] == 2
    assert 4 not in by_id
    assert by_id[5]["propagation_status"] == "ambiguous_or_weak"
    assert report["retained_edges"] == 4


def test_propagation_cannot_cross_into_geometry_that_conflicts_with_label() -> None:
    edges = [{"object_a": 1, "object_b": 2, "shared_voxel_faces": 100, "contact_rgb_distance": 0.0}]
    anchors = [
        {"object_id": 1, "geometry_type": "horizontal", "anchor_label": "floor", "propagation_eligible": True},
        {"object_id": 2, "geometry_type": "vertical_surface", "anchor_label": "unknown", "propagation_eligible": False},
    ]
    rows, _report = propagate(edges, anchors, 10, 100, 40.0, 2, 0.35, 0.15)
    assert {row["object_id"] for row in rows} == {1}


def test_propagation_does_not_invent_geometry_for_unlisted_graph_nodes() -> None:
    edges = [{"object_a": 1, "object_b": 2, "shared_voxel_faces": 100, "contact_rgb_distance": 0.0}]
    anchors = [{"object_id": 1, "geometry_type": "horizontal", "anchor_label": "floor", "propagation_eligible": True}]
    rows, report = propagate(edges, anchors, 10, 100, 40.0, 2, 0.35, 0.15)
    assert {row["object_id"] for row in rows} == {1}
    assert report["missing_geometry_skips"] == 1


def test_graph_node_ids_covers_both_edge_endpoints() -> None:
    edges = [{"object_a": 2, "object_b": 9, "shared_voxel_faces": 1, "contact_rgb_distance": 0.0}]
    assert graph_node_ids(edges) == {2, 9}
