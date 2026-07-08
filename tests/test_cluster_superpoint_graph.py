from argparse import Namespace

import numpy as np

from scripts.cluster_superpoint_graph import cluster


def args(**overrides):
    base = dict(
        min_edge_score=0.5,
        max_color_distance=100.0,
        max_merged_entropy=1.1,
        min_patch_voxels=1,
        enable_structural_merge_veto=False,
        structural_veto_min_bucket_ratio=0.2,
        structural_veto_min_voxels=1,
    )
    base.update(overrides)
    return Namespace(**base)


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
