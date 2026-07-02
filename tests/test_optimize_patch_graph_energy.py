from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from scripts import optimize_patch_graph_energy as module


def make_split_args(enable_bucket_split: bool) -> argparse.Namespace:
    return argparse.Namespace(
        split_min_component_voxels=1,
        residual_component_voxels=1,
        internal_color_distance=55.0,
        internal_normal_dot=0.52,
        enable_bucket_connectivity_split=enable_bucket_split,
        bucket_split_min_bucket_ratio=0.20,
        bucket_split_target_buckets="unknown,thin_linear,rough_mixed",
    )


def make_chain_fixture() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    arrays = {
        "xyz": np.array([[float(i), 0.0, 0.0] for i in range(6)], dtype=np.float32),
        "rgb": np.zeros((6, 3), dtype=np.float32),
        "normal": np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (6, 1)),
        "buckets": np.array([0, 0, 0, 4, 4, 4], dtype=np.int16),
    }
    labels = np.ones(6, dtype=np.int32)
    src = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    dst = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    return arrays, labels, src, dst


def test_bucket_connectivity_split_breaks_mixed_spectrum_bridge() -> None:
    arrays, labels, src, dst = make_chain_fixture()

    out_labels, next_id, logs = module.split_component(
        patch_id=1,
        point_ids=np.arange(6, dtype=np.int64),
        arrays=arrays,
        labels=labels.copy(),
        src=src,
        dst=dst,
        next_id=2,
        args=make_split_args(enable_bucket_split=True),
    )

    assert next_id == 3
    assert len(set(out_labels.tolist())) == 2
    assert logs[0]["reason"] == "bucket_connectivity_split"
    assert out_labels[0] == out_labels[1] == out_labels[2]
    assert out_labels[3] == out_labels[4] == out_labels[5]
    assert out_labels[0] != out_labels[3]


def test_bucket_connectivity_split_disabled_keeps_feature_continuous_chain() -> None:
    arrays, labels, src, dst = make_chain_fixture()

    out_labels, next_id, logs = module.split_component(
        patch_id=1,
        point_ids=np.arange(6, dtype=np.int64),
        arrays=arrays,
        labels=labels.copy(),
        src=src,
        dst=dst,
        next_id=2,
        args=make_split_args(enable_bucket_split=False),
    )

    assert next_id == 2
    assert out_labels.tolist() == [1, 1, 1, 1, 1, 1]
    assert logs == []


def test_significant_bucket_selection_uses_ratio_and_targets() -> None:
    counts = Counter({0: 60, 4: 30, 1: 10})

    selected = module.significant_buckets(
        counts,
        min_ratio=0.20,
        target_buckets=module.parse_bucket_id_set("unknown,rough_mixed"),
    )

    assert selected == {0, 4}
