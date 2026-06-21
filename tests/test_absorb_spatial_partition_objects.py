import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from absorb_spatial_partition_objects import absorb_components  # noqa: E402


def _row(key, oid, label, rgb=(100, 100, 100)):
    return {
        "key": key,
        "object_id": oid,
        "label": label,
        "semantic": 2,
        "xyz": np.array(key, dtype=float),
        "rgb": np.array(rgb, dtype=float),
    }


def _object(oid, label, status, count):
    return {
        "object_id": oid,
        "semantic_label": label,
        "status": status,
        "voxel_count": count,
        "pca_normal": [0.0, 0.0, 1.0],
    }


def test_same_label_small_component_absorbs_to_neighbor_anchor():
    rows = [_row((0, 0, 0), 1, "wall"), _row((1, 0, 0), 1, "wall"), _row((2, 0, 0), 2, "wall")]
    objects = {1: _object(1, "wall", "spatial_connected_component", 2), 2: _object(2, "wall", "small_component", 1)}
    owner = {r["key"]: r["object_id"] for r in rows}

    remap, report = absorb_components(rows, objects, owner, 1, 50.0, 30.0, False, False)

    assert remap[2] == 1
    assert report["absorbed_component_count"] == 1


def test_different_label_small_component_does_not_absorb_by_default():
    rows = [_row((0, 0, 0), 1, "wall"), _row((1, 0, 0), 2, "car")]
    objects = {1: _object(1, "wall", "spatial_connected_component", 1), 2: _object(2, "car", "small_component", 1)}
    owner = {r["key"]: r["object_id"] for r in rows}

    remap, report = absorb_components(rows, objects, owner, 1, 50.0, 30.0, False, False)

    assert remap[2] == 2
    assert report["residual_small_component_count"] == 1


def test_unknown_can_absorb_when_enabled():
    rows = [_row((0, 0, 0), 1, "floor"), _row((1, 0, 0), 2, "unknown")]
    objects = {1: _object(1, "floor", "spatial_connected_component", 1), 2: _object(2, "unknown", "small_component", 1)}
    owner = {r["key"]: r["object_id"] for r in rows}

    remap, _report = absorb_components(rows, objects, owner, 1, 50.0, 30.0, True, False)

    assert remap[2] == 1
