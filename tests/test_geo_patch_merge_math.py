from argparse import Namespace
from collections import Counter

import numpy as np

from scripts.optimize_geo_patch_merges import build_patch_edges
from scripts.optimize_patch_graph_energy import PatchStats, attachment_merge_decision, build_edge_features, edge_weight_from_candidate, fh_threshold, merge_patch_stats, structural_merge_veto


def test_build_patch_edges_counts_unordered_pairs():
    labels = np.array([5, 2, 5, 9, 2, 9], dtype=np.int32)
    src = np.array([0, 1, 2, 3, 4, 0, 1, 5], dtype=np.int32)
    dst = np.array([1, 2, 3, 4, 5, 2, 4, 3], dtype=np.int32)
    assert build_patch_edges(labels, src, dst) == Counter({(2, 5): 2, (2, 9): 2, (5, 9): 1})


def test_fh_edge_weight_updates_internal_diff():
    args = Namespace(max_color_distance=100.0, fh_color_weight=0.75, fh_color_p90_weight=0.0, fh_normal_weight=0.25, fh_shape_weight=0.0, fh_roughness_scale=0.35, fh_planarity_scale=0.35, fh_linearity_scale=0.35, fh_k=10.0)
    a = PatchStats(1, 10, np.zeros(3), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.zeros(3), np.ones(3), Counter({1: 10}), "horizontal", {1})
    b = PatchStats(2, 5, np.zeros(3), np.array([50.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.zeros(3), np.ones(3), Counter({1: 5}), "horizontal", {2})
    edge_weight = edge_weight_from_candidate({"contact_color_distance": 50.0, "contact_normal_score": 1.0}, a, b, args)
    merged = merge_patch_stats(a, b)
    merged.internal_diff = max(merged.internal_diff, edge_weight)
    assert edge_weight == 0.375
    assert fh_threshold(merged, args.fh_k) > edge_weight


def test_build_edge_features_tracks_contact_shape_and_p90():
    labels = np.array([1, 1, 2, 2], dtype=np.int32)
    src = np.array([0, 1], dtype=np.int32)
    dst = np.array([2, 3], dtype=np.int32)
    arrays = {
        "rgb": np.array([[0, 0, 0], [0, 0, 0], [30, 40, 0], [300, 400, 0]], dtype=np.float64),
        "normal": np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0]], dtype=np.float64),
        "roughness": np.array([0.1, 0.2, 0.4, 0.8], dtype=np.float64),
        "planarity": np.array([0.9, 0.8, 0.7, 0.4], dtype=np.float64),
        "linearity": np.array([0.2, 0.3, 0.5, 0.9], dtype=np.float64),
    }
    f = build_edge_features(labels, src, dst, arrays)[(1, 2)]
    assert f["shared_edges"] == 2
    assert round(f["contact_color_distance"], 6) == 275.0
    assert round(f["contact_color_p90"], 6) == 500.0
    assert round(f["contact_roughness_delta"], 6) == 0.45
    assert round(f["contact_planarity_delta"], 6) == 0.3
    assert round(f["contact_linearity_delta"], 6) == 0.45


def test_attachment_score_uses_contact_shape_when_weighted():
    args = Namespace(
        enable_attachment_merge=True,
        attachment_min_anchor_voxels=10,
        attachment_max_fragment_voxels=10,
        attachment_min_size_ratio=2.0,
        attachment_min_score=0.75,
        attachment_min_contact_ratio=0.1,
        attachment_min_shared_edges=1,
        attachment_max_color_distance=100.0,
        attachment_min_normal_score=0.4,
        attachment_max_bbox_gap=0.1,
        attachment_contact_norm=0.5,
        attachment_color_weight=0.25,
        attachment_normal_weight=0.25,
        attachment_bucket_weight=0.0,
        attachment_contact_weight=0.25,
        attachment_gap_weight=0.0,
        attachment_shape_weight=0.25,
        attachment_roughness_scale=0.35,
        attachment_planarity_scale=0.35,
        attachment_linearity_scale=0.35,
        attachment_use_contact_evidence=True,
    )
    anchor = PatchStats(1, 20, np.zeros(3), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.zeros(3), np.ones(3), Counter({1: 20}), "rough_mixed", {1})
    fragment = PatchStats(2, 5, np.zeros(3), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.zeros(3), np.ones(3), Counter({1: 5}), "mixed", {2})
    good, _, good_detail = attachment_merge_decision(anchor, fragment, 2, 0.4, {"contact_color_distance": 0.0, "contact_normal_score": 1.0}, args)
    bad, reason, bad_detail = attachment_merge_decision(
        anchor,
        fragment,
        2,
        0.4,
        {"contact_color_distance": 0.0, "contact_normal_score": 1.0, "contact_roughness_delta": 1.0, "contact_planarity_delta": 1.0, "contact_linearity_delta": 1.0},
        args,
    )
    assert good
    assert good_detail["attachment_shape"] == 1.0
    assert not bad
    assert reason == "attachment_score"
    assert bad_detail["attachment_shape"] == 0.0


def test_structural_veto_blocks_stable_horizontal_vertical_merge():
    args = Namespace(enable_structural_merge_veto=True, structural_veto_min_bucket_ratio=0.2, structural_veto_min_voxels=100)
    horizontal = PatchStats(1, 200, np.zeros(3), np.zeros(3), np.array([0.0, 0.0, 1.0]), np.zeros(3), np.ones(3), Counter({1: 200}), "horizontal", {1})
    vertical = PatchStats(2, 180, np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0]), np.zeros(3), np.ones(3), Counter({2: 180}), "vertical", {2})

    vetoed, reason, detail = structural_merge_veto(horizontal, vertical, args)

    assert vetoed
    assert reason == "structural_horizontal_vertical_veto"
    assert detail["structural_veto_a_horizontal"] == 1.0
    assert detail["structural_veto_b_vertical"] == 1.0


def test_structural_veto_allows_same_surface_family():
    args = Namespace(enable_structural_merge_veto=True, structural_veto_min_bucket_ratio=0.2, structural_veto_min_voxels=100)
    a = PatchStats(1, 200, np.zeros(3), np.zeros(3), np.array([0.0, 0.0, 1.0]), np.zeros(3), np.ones(3), Counter({1: 180, 0: 20}), "horizontal", {1})
    b = PatchStats(2, 180, np.zeros(3), np.zeros(3), np.array([0.0, 0.0, 1.0]), np.zeros(3), np.ones(3), Counter({1: 160, 0: 20}), "horizontal", {2})

    vetoed, reason, _ = structural_merge_veto(a, b, args)

    assert not vetoed
    assert reason == ""
