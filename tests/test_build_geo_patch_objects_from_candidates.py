from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.build_geo_patch_objects_from_candidates import write_objects_jsonl


def arrays() -> dict[str, np.ndarray]:
    return {
        "xyz": np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
        "rgb": np.array([[10, 20, 30], [20, 30, 40], [30, 40, 50]], dtype=np.float32),
        "normal": np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0]], dtype=np.float32),
        "buckets": np.array([1, 1, 2], dtype=np.int16),
    }


def test_objects_jsonl_keeps_geometry_separate_from_semantic_label(tmp_path: Path) -> None:
    patch_labels = np.array([10, 10, 20], dtype=np.int32)
    object_labels = np.array([1, 1, 2], dtype=np.int32)
    patch_to_object = {10: 1, 20: 2}
    path = tmp_path / "objects.jsonl"

    count = write_objects_jsonl(path, arrays(), object_labels, patch_labels, patch_to_object)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert count == 2
    assert rows[0]["geometry_type"] == "horizontal"
    assert rows[0]["geometry_label"] == "horizontal"
    assert rows[0]["semantic_label"] == "unknown"
    assert rows[0]["semantic_status"] == "geometry_only_unlabeled"
    assert rows[0]["label_policy"] == "geometry_is_not_semantic"
    assert rows[0]["description"].startswith("geometry-only object")
