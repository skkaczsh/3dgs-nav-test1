from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "refine_priority_masks_with_geometry_for_test",
    SCRIPTS / "refine_priority_masks_with_geometry.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def args(**overrides):
    base = {
        "surface_override_from": [0],
        "guarded_fine_surface_override": False,
        "fine_surface_min_pixels": 4,
        "fine_surface_min_ratio": 0.35,
        "fine_surface_dominant_ratio": 0.70,
        "fine_surface_neighbor_radius": 0,
        "fine_surface_neighbor_min_support": 1,
        "cut_fine_at_depth_edge": False,
        "min_fine_component_area": 0,
        "component_min_area": 1,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_guarded_fine_surface_override_recovers_surface_supported_railing():
    priority = np.zeros((8, 8), dtype=np.uint8)
    priority[1:7, 1:7] = 5
    semantic = np.zeros((8, 8), dtype=np.uint8)
    semantic[1:7, 1:7] = 2
    valid = np.zeros((8, 8), dtype=np.uint8)
    valid[1:7, 1:7] = 255
    edge = np.zeros((8, 8), dtype=np.uint8)

    refined, stats = module.refine(
        priority,
        semantic,
        valid,
        edge,
        args(guarded_fine_surface_override=True),
    )

    assert np.all(refined[1:7, 1:7] == 2)
    assert stats["guarded_fine_surface_override_pixels"] == 36
    assert stats["override_pairs"]["railing->wall"] == 36


def test_guarded_fine_surface_override_keeps_sparse_surface_overlap():
    priority = np.zeros((8, 8), dtype=np.uint8)
    priority[1:7, 1:7] = 5
    semantic = np.zeros((8, 8), dtype=np.uint8)
    semantic[1:3, 1:3] = 2
    valid = np.zeros((8, 8), dtype=np.uint8)
    valid[1:7, 1:7] = 255
    edge = np.zeros((8, 8), dtype=np.uint8)

    refined, stats = module.refine(
        priority,
        semantic,
        valid,
        edge,
        args(guarded_fine_surface_override=True, fine_surface_min_ratio=0.50),
    )

    assert np.all(refined[1:7, 1:7] == 5)
    assert stats["guarded_fine_surface_override_pixels"] == 0


def test_priority_path_accepts_suffix():
    path = module.priority_path(Path("/tmp/priority_base"), cam_id=2, frame_id=123, suffix="_priority_refined")

    assert path == Path("/tmp/priority_base/priority/cam2_000123_priority_refined.png")
