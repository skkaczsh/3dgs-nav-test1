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


def make_attachment_args() -> argparse.Namespace:
    return argparse.Namespace(
        enable_attachment_merge=True,
        attachment_min_anchor_voxels=20000,
        attachment_max_fragment_voxels=5000,
        attachment_min_size_ratio=4.0,
        attachment_min_score=0.78,
        attachment_min_contact_ratio=0.08,
        attachment_min_shared_edges=8,
        attachment_max_color_distance=70.0,
        attachment_min_normal_score=0.42,
        attachment_max_bbox_gap=0.08,
        attachment_contact_norm=0.18,
        attachment_color_weight=0.32,
        attachment_normal_weight=0.16,
        attachment_bucket_weight=0.14,
        attachment_contact_weight=0.30,
        attachment_gap_weight=0.08,
        attachment_shape_weight=0.0,
        attachment_roughness_scale=0.35,
        attachment_planarity_scale=0.35,
        attachment_linearity_scale=0.35,
        attachment_use_contact_evidence=True,
        enable_structural_merge_veto=False,
        split_attachment_min_anchor_voxels=5000,
        split_attachment_max_fragment_voxels=5000,
        split_attachment_min_size_ratio=2.5,
        split_attachment_min_score=0.78,
        split_attachment_min_contact_ratio=0.08,
        split_attachment_min_shared_edges=8,
        split_attachment_max_color_distance=70.0,
        split_attachment_min_normal_score=0.42,
        split_attachment_max_bbox_gap=0.08,
        fragment_attachment_min_anchor_voxels=5000,
        fragment_attachment_max_fragment_voxels=4000,
        fragment_attachment_min_size_ratio=2.5,
        fragment_attachment_min_score=0.80,
        fragment_attachment_min_contact_ratio=0.10,
        fragment_attachment_min_shared_edges=12,
        fragment_attachment_max_color_distance=62.0,
        fragment_attachment_min_normal_score=0.44,
        fragment_attachment_max_bbox_gap=0.07,
    )


def make_patch(
    patch_id: int,
    count: int,
    mean_rgb: list[float],
    geometry_type: str = "rough_mixed",
) -> module.PatchStats:
    return module.PatchStats(
        patch_id=patch_id,
        count=count,
        centroid=np.zeros(3, dtype=np.float64),
        mean_rgb=np.array(mean_rgb, dtype=np.float64),
        mean_normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        bbox_min=np.zeros(3, dtype=np.float64),
        bbox_max=np.array([1.0, 1.0, 0.02], dtype=np.float64),
        bucket_counts=Counter({4: count}),
        geometry_type=geometry_type,
        source_patch_ids={patch_id},
    )


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


def test_split_provenance_attachment_uses_dedicated_anchor_threshold() -> None:
    args = make_attachment_args()
    anchor = make_patch(1, 8000, [10.0, 10.0, 10.0])
    fragment = make_patch(2, 1000, [12.0, 10.0, 10.0])
    candidate = {"contact_color_distance": 2.0, "contact_normal_score": 0.95}

    ok, reason, _detail = module.attachment_merge_decision(
        anchor,
        fragment,
        shared_edges=200,
        candidate_support=0.2,
        candidate=candidate,
        args=args,
        provenance_relaxed=False,
    )
    assert not ok
    assert reason == "attachment_anchor_too_small"

    ok, reason, detail = module.attachment_merge_decision(
        anchor,
        fragment,
        shared_edges=200,
        candidate_support=0.2,
        candidate=candidate,
        args=args,
        provenance_relaxed=True,
    )
    assert ok
    assert reason == "accepted_split_attachment"
    assert detail["attachment_provenance_relaxed"] == 1.0


def test_fragment_evidence_attachment_uses_dedicated_profile() -> None:
    args = make_attachment_args()
    anchor = make_patch(1, 8000, [10.0, 10.0, 10.0])
    fragment = make_patch(2, 1000, [12.0, 10.0, 10.0])
    candidate = {"contact_color_distance": 2.0, "contact_normal_score": 0.95}

    ok, reason, detail = module.attachment_merge_decision(
        anchor,
        fragment,
        shared_edges=200,
        candidate_support=0.2,
        candidate=candidate,
        args=args,
        fragment_relaxed=True,
    )

    assert ok
    assert reason == "accepted_fragment_attachment"
    assert detail["attachment_fragment_relaxed"] == 1.0
    assert detail["attachment_provenance_relaxed"] == 0.0
