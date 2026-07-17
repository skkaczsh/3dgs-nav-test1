from scripts.infer_superpoint_semantic_graph import attenuation_factor, edge_affinity, infer


def test_unobserved_nodes_are_neither_sources_nor_promoted() -> None:
    rows = [
        {"object_id": 1, "state": "reviewed", "geometry_type": "horizontal", "alpha": {"floor": 0.9, "unknown": 0.1}},
        {"object_id": 2, "state": "unobserved", "geometry_type": "horizontal", "alpha": {}},
        {"object_id": 3, "state": "unobserved", "geometry_type": "vertical", "alpha": {}},
    ]
    edges = [
        {"object_a": 1, "object_b": 2, "shared_voxel_faces": 100, "contact_ratio_min": 0.2, "contact_rgb_distance": 0.0},
        {"object_a": 2, "object_b": 3, "shared_voxel_faces": 100, "contact_ratio_min": 0.2, "contact_rgb_distance": 0.0},
    ]
    result, report = infer(rows, edges)
    by_id = {row["object_id"]: row for row in result}
    assert by_id[2]["semantic_status"] == "local_proposal_not_promoted"
    assert by_id[3]["semantic_status"] == "unobserved_or_unlabeled"
    assert report["reviewed_nodes"] == 1


def test_stable_geometry_veto_blocks_wall_on_horizontal_node() -> None:
    rows = [
        {"object_id": 1, "state": "reviewed", "geometry_type": "vertical", "alpha": {"wall": 1.0}},
        {"object_id": 2, "state": "observed_unlabeled", "geometry_type": "horizontal", "alpha": {}},
    ]
    edges = [{"object_a": 1, "object_b": 2, "shared_voxel_faces": 100, "contact_ratio_min": 0.2, "contact_rgb_distance": 0.0}]
    result, _report = infer(rows, edges)
    assert {row["object_id"]: row for row in result}[2]["semantic_posterior"] == {}


def test_object_like_label_is_not_vetoed_by_horizontal_local_geometry() -> None:
    rows = [
        {"object_id": 1, "state": "reviewed", "geometry_type": "rough_mixed", "alpha": {"car": 1.0}},
        {"object_id": 2, "state": "observed_unlabeled", "geometry_type": "horizontal", "alpha": {}},
    ]
    edges = [{"object_a": 1, "object_b": 2, "shared_voxel_faces": 100, "contact_ratio_min": 0.2, "contact_rgb_distance": 0.0}]
    result, _report = infer(rows, edges)
    assert {row["object_id"]: row for row in result}[2]["semantic_posterior"]["car"] > 0


def test_edge_affinity_requires_contact_support_and_color_continuity() -> None:
    good = {"shared_voxel_faces": 100, "contact_ratio_min": 0.2, "contact_rgb_distance": 0.0}
    weak = {"shared_voxel_faces": 2, "contact_ratio_min": 0.2, "contact_rgb_distance": 0.0}
    distant = {"shared_voxel_faces": 100, "contact_ratio_min": 0.2, "contact_rgb_distance": 120.0}
    assert edge_affinity(good, 10, 0.01, 35.0) > 0.8
    assert edge_affinity(weak, 10, 0.01, 35.0) == 0.0
    assert edge_affinity(distant, 10, 0.01, 35.0) < 0.01


def test_repeated_photometric_boundary_can_disable_an_otherwise_valid_edge() -> None:
    rows = [
        {"object_id": 1, "state": "reviewed", "geometry_type": "rough_mixed", "alpha": {"car": 1.0}},
        {"object_id": 2, "state": "observed_unlabeled", "geometry_type": "rough_mixed", "alpha": {}},
    ]
    edges = [{"object_a": 1, "object_b": 2, "shared_voxel_faces": 100, "contact_ratio_min": 0.2, "contact_rgb_distance": 0.0}]
    photo = [{"object_a": 1, "object_b": 2, "photometric_affinity": 0.0, "view_count": 3}]
    result, report = infer(rows, edges, photometric_rows=photo)
    by_id = {row["object_id"]: row for row in result}
    assert by_id[2]["semantic_posterior"] == {}
    assert report["photometric_edges"] == 1


def test_repeated_sam2_separation_can_disable_an_otherwise_valid_edge() -> None:
    rows = [
        {"object_id": 1, "state": "reviewed", "geometry_type": "rough_mixed", "alpha": {"car": 1.0}},
        {"object_id": 2, "state": "observed_unlabeled", "geometry_type": "rough_mixed", "alpha": {}},
    ]
    edges = [{"object_a": 1, "object_b": 2, "shared_voxel_faces": 100, "contact_ratio_min": 0.2, "contact_rgb_distance": 0.0}]
    sam2 = [{"object_a": 1, "object_b": 2, "sam2_affinity": 0.0, "view_count": 3}]
    result, report = infer(rows, edges, sam2_rows=sam2)
    assert {row["object_id"]: row for row in result}[2]["semantic_posterior"] == {}
    assert report["sam2_comask_edges"] == 1
    assert report["sam2_observed_viable_edges"] == 1
    assert report["sam2_strong_separation_viable_edges"] == 1


def test_unary_support_strength_is_preserved_before_final_normalization() -> None:
    target = {"object_id": 1, "state": "reviewed", "geometry_type": "horizontal", "alpha": {"unknown": 1.0}}
    edge = {"object_a": 1, "object_b": 2, "shared_voxel_faces": 100, "contact_ratio_min": 0.2, "contact_rgb_distance": 0.0}
    weak_source = {"object_id": 2, "state": "reviewed", "geometry_type": "horizontal", "alpha": {"floor": 0.1}}
    strong_source = {"object_id": 2, "state": "reviewed", "geometry_type": "horizontal", "alpha": {"floor": 0.9}}

    weak, _ = infer([target, weak_source], [edge])
    strong, _ = infer([target, strong_source], [edge])

    weak_floor = {row["object_id"]: row for row in weak}[1]["semantic_posterior"]["floor"]
    strong_floor = {row["object_id"]: row for row in strong}[1]["semantic_posterior"]["floor"]
    assert strong_floor > weak_floor


def test_image_evidence_weight_is_clamped_to_a_non_boosting_attenuation() -> None:
    assert attenuation_factor(0.0, -1.0) == 1.0
    assert attenuation_factor(0.0, 2.0) == 0.0
    assert attenuation_factor(0.4, 0.5) == 0.7
