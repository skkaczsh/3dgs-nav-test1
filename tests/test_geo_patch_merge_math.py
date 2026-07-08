from argparse import Namespace
from collections import Counter

import numpy as np

from scripts.optimize_geo_patch_merges import build_patch_edges
from scripts.optimize_patch_graph_energy import PatchStats, edge_weight_from_candidate, fh_threshold, merge_patch_stats


def test_build_patch_edges_counts_unordered_pairs():
    labels = np.array([5, 2, 5, 9, 2, 9], dtype=np.int32)
    src = np.array([0, 1, 2, 3, 4, 0, 1, 5], dtype=np.int32)
    dst = np.array([1, 2, 3, 4, 5, 2, 4, 3], dtype=np.int32)
    assert build_patch_edges(labels, src, dst) == Counter({(2, 5): 2, (2, 9): 2, (5, 9): 1})


def test_fh_edge_weight_updates_internal_diff():
    args = Namespace(max_color_distance=100.0, fh_color_weight=0.75, fh_normal_weight=0.25, fh_k=10.0)
    a = PatchStats(1, 10, np.zeros(3), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.zeros(3), np.ones(3), Counter({1: 10}), "horizontal", {1})
    b = PatchStats(2, 5, np.zeros(3), np.array([50.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.zeros(3), np.ones(3), Counter({1: 5}), "horizontal", {2})
    edge_weight = edge_weight_from_candidate({"contact_color_distance": 50.0, "contact_normal_score": 1.0}, a, b, args)
    merged = merge_patch_stats(a, b)
    merged.internal_diff = max(merged.internal_diff, edge_weight)
    assert edge_weight == 0.375
    assert fh_threshold(merged, args.fh_k) > edge_weight
