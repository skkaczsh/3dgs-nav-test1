import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_single_frame_mask_matrix import (  # noqa: E402
    connected_components_gated,
    labels_from_sam_split,
    spectral_scalar,
)


def test_connected_components_respects_blocked_boundary():
    valid = np.ones((4, 6), dtype=bool)
    blocked = np.zeros_like(valid)
    blocked[:, 3] = True
    labels = connected_components_gated(valid, None, blocked, feature_threshold=0.0, min_area=1)

    left_ids = set(np.unique(labels[:, :3])) - {0}
    right_ids = set(np.unique(labels[:, 4:])) - {0}

    assert left_ids
    assert right_ids
    assert left_ids.isdisjoint(right_ids)


def test_connected_components_respects_feature_threshold():
    valid = np.ones((3, 6), dtype=bool)
    features = np.zeros((3, 6, 1), dtype=np.float32)
    features[:, 3:, 0] = 1.0
    labels = connected_components_gated(valid, features, None, feature_threshold=0.25, min_area=1)

    assert labels[1, 1] != labels[1, 4]


def test_spectral_scalar_tracks_main_image_contrast():
    valid = np.ones((8, 8), dtype=bool)
    depth = np.zeros((8, 8), dtype=np.float32)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, 4:, 0] = 255

    spectral = spectral_scalar(depth, rgb, valid)

    assert abs(float(spectral[:, :4].mean()) - float(spectral[:, 4:].mean())) > 0.5


def test_sam_masks_are_split_by_geometry_components():
    valid = np.ones((4, 6), dtype=bool)
    base_labels = np.zeros((4, 6), dtype=np.int32)
    base_labels[:, :3] = 1
    base_labels[:, 3:] = 2
    sam_mask = np.ones((4, 6), dtype=bool)

    split = labels_from_sam_split([sam_mask], base_labels, valid, min_area=1, target_size=(6, 4))

    ids = set(np.unique(split)) - {0}
    assert len(ids) == 2
    assert split[1, 1] != split[1, 4]
