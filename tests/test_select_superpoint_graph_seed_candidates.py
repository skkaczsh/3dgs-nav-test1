from scripts.select_superpoint_graph_seed_candidates import graph_candidates


def test_candidates_prefer_uncovered_supported_contact_neighborhoods() -> None:
    objects = [
        {"object_id": 1, "count": 1000, "geometry_type": "horizontal", "structural_region_dominant": "ground_like_region"},
        {"object_id": 2, "count": 1000, "geometry_type": "horizontal", "structural_region_dominant": "ground_like_region"},
        {"object_id": 3, "count": 1000, "geometry_type": "vertical", "structural_region_dominant": "vertical_surface_region"},
    ]
    edges = [
        {"object_a": 1, "object_b": 2, "shared_voxel_faces": 100, "contact_rgb_distance": 0.0},
        {"object_a": 2, "object_b": 3, "shared_voxel_faces": 100, "contact_rgb_distance": 0.0},
    ]
    rows = graph_candidates(objects, edges, {1, 2, 3}, {1}, {1}, 500, 2, 10)
    assert [row["object_id"] for row in rows] == [2, 3]
    assert rows[0]["new_one_hop_neighbors"] == 1
