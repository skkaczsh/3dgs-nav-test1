from __future__ import annotations

import numpy as np

from scripts.analyze_superpoint_graph_edges import summarize


def test_summarize_reports_isolated_patches():
    arrays = {
        "xyz": np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0]], dtype=np.float32),
        "rgb": np.zeros((3, 3), dtype=np.float32),
        "normal": np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0]], dtype=np.float32),
        "roughness": np.zeros(3, dtype=np.float32),
        "planarity": np.ones(3, dtype=np.float32),
        "linearity": np.zeros(3, dtype=np.float32),
        "local_color_std": np.zeros(3, dtype=np.float32),
        "height_range": np.zeros(3, dtype=np.float32),
        "buckets": np.array([1, 1, 2], dtype=np.int16),
    }
    labels = np.array([1, 2, 3], dtype=np.int32)

    report = summarize(arrays, labels, np.array([0], dtype=np.int32), np.array([1], dtype=np.int32), 1)

    assert report["patch_count"] == 3
    assert report["edge_pair_count"] == 1
    assert report["isolated_patch_count"] == 1
    assert report["isolated_size_bins"]["1"] == 1
    assert report["patch_size_bins"]["1"] == 3
    assert report["isolated_geometry_counts"] == {"vertical": 1}
    assert report["large_isolated_top20"][0]["patch_id"] == 3
