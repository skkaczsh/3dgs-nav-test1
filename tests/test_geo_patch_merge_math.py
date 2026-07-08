from argparse import Namespace
from collections import Counter

import numpy as np

from scripts.optimize_geo_patch_merges import build_patch_edges
from scripts.optimize_patch_graph_energy import PatchStats, build_edge_features, edge_weight_from_candidate, fh_threshold, merge_patch_stats


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
