from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "build_frame_targets_from_priority_for_test",
        SCRIPTS / "build_frame_targets_from_priority.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_split_local_indices_by_image_components_separates_disconnected_regions():
    module = load_module()
    label_mask = np.zeros((8, 8), dtype=np.uint8)
    label_mask[1:3, 1:3] = 1
    label_mask[5:7, 5:7] = 1
    uu = np.array([1, 2, 5, 6], dtype=np.int32)
    vv = np.array([1, 2, 5, 6], dtype=np.int32)
    local_indices = np.arange(4, dtype=np.int64)

    groups = module.split_local_indices_by_image_components(label_mask, uu, vv, local_indices, min_pixels=1)

    assert [group.tolist() for group in groups] == [[0, 1], [2, 3]]


def test_split_local_indices_by_image_components_keeps_small_components_as_residual():
    module = load_module()
    label_mask = np.zeros((8, 8), dtype=np.uint8)
    label_mask[1:4, 1:4] = 1
    label_mask[6, 6] = 1
    uu = np.array([1, 2, 6], dtype=np.int32)
    vv = np.array([1, 2, 6], dtype=np.int32)
    local_indices = np.arange(3, dtype=np.int64)

    groups = module.split_local_indices_by_image_components(label_mask, uu, vv, local_indices, min_pixels=4)

    assert [group.tolist() for group in groups] == [[0, 1], [2]]


def test_split_local_indices_by_depth_support_splits_depth_jump():
    module = load_module()
    uu = np.array([10, 12, 14, 16], dtype=np.int32)
    vv = np.array([10, 11, 10, 11], dtype=np.int32)
    depths = np.array([1.0, 1.1, 4.0, 4.1], dtype=np.float32)
    local_indices = np.arange(4, dtype=np.int64)

    groups = module.split_local_indices_by_depth_support(
        uu,
        vv,
        depths,
        local_indices,
        pixel_radius=4,
        max_depth_gap=0.5,
        min_points=1,
    )

    assert [group.tolist() for group in groups] == [[0, 1], [2, 3]]


def test_split_local_indices_by_depth_support_keeps_continuous_depth():
    module = load_module()
    uu = np.array([10, 12, 14, 16], dtype=np.int32)
    vv = np.array([10, 11, 10, 11], dtype=np.int32)
    depths = np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float32)
    local_indices = np.arange(4, dtype=np.int64)

    groups = module.split_local_indices_by_depth_support(
        uu,
        vv,
        depths,
        local_indices,
        pixel_radius=4,
        max_depth_gap=0.5,
        min_points=1,
    )

    assert [group.tolist() for group in groups] == [[0, 1, 2, 3]]
