from argparse import Namespace

import numpy as np

from scripts.cluster_superpoint_graph import cluster, edge_score


def args(**overrides):
    base = dict(
        min_edge_score=0.5,
        max_color_distance=100.0,
        max_merged_entropy=1.1,
        fh_k=0.0,
        min_patch_voxels=1,
        disable_contact_bridge=False,
        contact_bridge_min_support=0.25,
        contact_bridge_max_color_distance=65.0,
        contact_bridge_max_color_p90=80.0,
        disable_near_bbox_candidates=False,
        near_candidate_min_voxels=1,
        near_candidate_max_gap=0.2,
        near_candidate_max_color_distance=70.0,
        near_candidate_min_normal_score=0.65,
        near_candidate_max_per_patch=8,
        enable_uncertain_fragment_candidates=False,
        uncertain_cell_size=0.05,
        uncertain_radius=1,
        uncertain_min_stable_voxels=10000,
        uncertain_max_fragment_voxels=5000,
        uncertain_min_contact_points=16,
        uncertain_max_color_distance=75.0,
        uncertain_max_bbox_gap=0.06,
        uncertain_max_cells_per_patch=30000,
        uncertain_max_stable_patches=200,
        uncertain_max_candidates_per_stable=8,
        enable_structural_merge_veto=False,
        structural_veto_min_bucket_ratio=0.2,
        structural_veto_min_voxels=1,
    )
    base.update(overrides)
    return Namespace(**base)


def test_edge_score_rewards_contact_support():
    feature = {
        "contact_color_distance": 40.0,
        "contact_color_p90": 50.0,
        "contact_normal_score": 0.4,
        "contact_roughness_delta": 0.2,
        "contact_planarity_delta": 0.2,
        "contact_linearity_delta": 0.2,
    }

    weak = edge_score({**feature, "contact_support": 0.05}, 100.0)
    strong = edge_score({**feature, "contact_support": 0.90}, 100.0)

    assert strong > weak + 0.10


def test_external_edge_evidence_can_lower_score():
    feature = {
        "contact_color_distance": 5.0,
        "contact_color_p90": 5.0,
        "contact_support": 1.0,
        "contact_normal_score": 1.0,
        "contact_roughness_delta": 0.0,
        "contact_planarity_delta": 0.0,
        "contact_linearity_delta": 0.0,
    }

    base = edge_score(feature, 100.0)
    lowered = edge_score({**feature, "external_similarity": 0.0, "external_edge_weight": 0.5}, 100.0)

    assert base > 0.9
    assert lowered < base - 0.4


def test_superpoint_graph_merges_similar_neighbors():
    arrays = {
        "xyz": np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        "rgb": np.array([[10, 10, 10], [12, 10, 10]], dtype=np.float32),
        "normal": np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32),
        "roughness": np.array([0.1, 0.1], dtype=np.float32),
        "planarity": np.array([0.9, 0.9], dtype=np.float32),
        "linearity": np.array([0.1, 0.1], dtype=np.float32),
        "local_color_std": np.array([1, 1], dtype=np.float32),
        "height_range": np.array([0, 0], dtype=np.float32),
        "buckets": np.array([1, 1], dtype=np.int16),
    }
    labels = np.array([1, 2], dtype=np.int32)
    out, report = cluster(arrays, labels, np.array([0], dtype=np.int32), np.array([1], dtype=np.int32), args())
    assert len(set(out.tolist())) == 1
    assert report["accepted_edges"] == 1


def test_external_edge_evidence_participates_in_cluster_decision():
    arrays = {
        "xyz": np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        "rgb": np.array([[10, 10, 10], [12, 10, 10]], dtype=np.float32),
        "normal": np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32),
        "roughness": np.array([0.1, 0.1], dtype=np.float32),
        "planarity": np.array([0.9, 0.9], dtype=np.float32),
        "linearity": np.array([0.1, 0.1], dtype=np.float32),
        "local_color_std": np.array([1, 1], dtype=np.float32),
        "height_range": np.array([0, 0], dtype=np.float32),
        "buckets": np.array([1, 1], dtype=np.int16),
    }
    labels = np.array([1, 2], dtype=np.int32)
    out, report = cluster(
        arrays,
        labels,
        np.array([0], dtype=np.int32),
        np.array([1], dtype=np.int32),
        args(
            min_edge_score=0.8,
            disable_contact_bridge=True,
            external_edge_evidence={(1, 2): 0.0},
            external_edge_weight=0.5,
        ),
    )

    assert len(set(out.tolist())) == 2
    assert report["external_edge_evidence_count"] == 1
    assert report["reject_counts"]["score"] == 1


def test_contact_bridge_accepts_same_geometry_high_support_edge_below_score_threshold():
    arrays = {
        "xyz": np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
        "rgb": np.array([[10, 10, 10], [10, 10, 10], [14, 10, 10], [14, 10, 10]], dtype=np.float32),
        "normal": np.array([[0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=np.float32),
        "roughness": np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        "planarity": np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float32),
        "linearity": np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        "local_color_std": np.array([1, 1, 1, 1], dtype=np.float32),
        "height_range": np.array([0, 0, 0, 0], dtype=np.float32),
        "buckets": np.array([1, 1, 1, 1], dtype=np.int16),
    }
    labels = np.array([1, 1, 2, 2], dtype=np.int32)
    out, report = cluster(
        arrays,
        labels,
        np.array([0, 1], dtype=np.int32),
        np.array([2, 3], dtype=np.int32),
        args(min_edge_score=0.95),
    )

    assert len(set(out.tolist())) == 1
    assert report["accepted_reasons"]["contact_bridge"] == 1


def test_near_bbox_candidate_merges_same_geometry_without_touch_edge():
    arrays = {
        "xyz": np.array([[0, 0, 0], [1, 0, 0], [1.1, 0, 0], [2.1, 0, 0]], dtype=np.float32),
        "rgb": np.array([[10, 10, 10], [10, 10, 10], [12, 10, 10], [12, 10, 10]], dtype=np.float32),
        "normal": np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32),
        "roughness": np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        "planarity": np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float32),
        "linearity": np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        "local_color_std": np.array([1, 1, 1, 1], dtype=np.float32),
        "height_range": np.array([0, 0, 0, 0], dtype=np.float32),
        "buckets": np.array([1, 1, 1, 1], dtype=np.int16),
    }
    labels = np.array([1, 1, 2, 2], dtype=np.int32)
    out, report = cluster(
        arrays,
        labels,
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        args(min_edge_score=0.95),
    )

    assert len(set(out.tolist())) == 1
    assert report["near_bbox_candidate_count"] == 1
    assert report["accepted_reasons"]["near_bbox_bridge"] == 1


def test_near_bbox_candidate_does_not_cross_geometry_type():
    arrays = {
        "xyz": np.array([[0, 0, 0], [1, 0, 0], [1.1, 0, 0], [2.1, 0, 0]], dtype=np.float32),
        "rgb": np.array([[10, 10, 10], [10, 10, 10], [12, 10, 10], [12, 10, 10]], dtype=np.float32),
        "normal": np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0]], dtype=np.float32),
        "roughness": np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        "planarity": np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float32),
        "linearity": np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        "local_color_std": np.array([1, 1, 1, 1], dtype=np.float32),
        "height_range": np.array([0, 0, 0, 0], dtype=np.float32),
        "buckets": np.array([1, 1, 2, 2], dtype=np.int16),
    }
    labels = np.array([1, 1, 2, 2], dtype=np.int32)
    out, report = cluster(
        arrays,
        labels,
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        args(min_edge_score=0.95),
    )

    assert len(set(out.tolist())) == 2
    assert report["near_bbox_candidate_count"] == 0


def test_superpoint_graph_vetoes_stable_surface_crossing():
    arrays = {
        "xyz": np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        "rgb": np.array([[10, 10, 10], [10, 10, 10]], dtype=np.float32),
        "normal": np.array([[0, 0, 1], [1, 0, 0]], dtype=np.float32),
        "roughness": np.array([0.1, 0.1], dtype=np.float32),
        "planarity": np.array([0.9, 0.9], dtype=np.float32),
        "linearity": np.array([0.1, 0.1], dtype=np.float32),
        "local_color_std": np.array([1, 1], dtype=np.float32),
        "height_range": np.array([0, 0], dtype=np.float32),
        "buckets": np.array([1, 2], dtype=np.int16),
    }
    labels = np.array([1, 2], dtype=np.int32)
    out, report = cluster(
        arrays,
        labels,
        np.array([0], dtype=np.int32),
        np.array([1], dtype=np.int32),
        args(enable_structural_merge_veto=True),
    )
    assert len(set(out.tolist())) == 2
    assert report["reject_counts"]["structural_horizontal_vertical_veto"] == 1


def test_fh_threshold_rejects_edge_that_is_too_weak_for_component_size():
    arrays = {
        "xyz": np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
        "rgb": np.array([[10, 10, 10], [10, 10, 10], [40, 10, 10], [40, 10, 10]], dtype=np.float32),
        "normal": np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32),
        "roughness": np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        "planarity": np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float32),
        "linearity": np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        "local_color_std": np.array([1, 1, 1, 1], dtype=np.float32),
        "height_range": np.array([0, 0, 0, 0], dtype=np.float32),
        "buckets": np.array([1, 1, 1, 1], dtype=np.int16),
    }
    labels = np.array([1, 1, 2, 2], dtype=np.int32)
    out, report = cluster(
        arrays,
        labels,
        np.array([0, 1], dtype=np.int32),
        np.array([2, 3], dtype=np.int32),
        args(min_edge_score=0.1, fh_k=0.1),
    )

    assert len(set(out.tolist())) == 2
    assert report["reject_counts"]["fh_threshold"] == 1


def test_uncertain_fragment_candidate_attaches_small_unknown_to_stable_surface():
    arrays = {
        "xyz": np.array([[0, 0, 0], [0.03, 0, 0], [0.06, 0, 0]], dtype=np.float32),
        "rgb": np.array([[10, 10, 10], [10, 10, 10], [12, 10, 10]], dtype=np.float32),
        "normal": np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32),
        "roughness": np.array([0.1, 0.1, 0.12], dtype=np.float32),
        "planarity": np.array([0.9, 0.9, 0.4], dtype=np.float32),
        "linearity": np.array([0.1, 0.1, 0.2], dtype=np.float32),
        "local_color_std": np.array([1, 1, 2], dtype=np.float32),
        "height_range": np.array([0, 0, 0], dtype=np.float32),
        "buckets": np.array([1, 1, 0], dtype=np.int16),
    }
    labels = np.array([1, 1, 2], dtype=np.int32)
    out, report = cluster(
        arrays,
        labels,
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        args(
            min_edge_score=0.95,
            enable_uncertain_fragment_candidates=True,
            uncertain_cell_size=0.03,
            uncertain_min_stable_voxels=2,
            uncertain_min_contact_points=1,
            uncertain_max_fragment_voxels=4,
        ),
    )

    assert len(set(out.tolist())) == 1
    assert report["uncertain_fragment_candidate_count"] == 1
    assert report["accepted_reasons"]["uncertain_fragment_bridge"] == 1
