import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module(path: Path, name: str):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_target_connected_components_splits_voxel_groups():
    module = load_module(SCRIPTS / "build_targets_from_masks.py", "build_targets_for_repo_test")
    points = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.03, 0.00, 0.00],
            [1.00, 1.00, 1.00],
            [1.03, 1.00, 1.00],
            [3.00, 3.00, 3.00],
        ],
        dtype=np.float32,
    )

    components, residual = module.connected_components(points, voxel_size=0.08, min_points=2)

    assert [len(c) for c in components] == [2, 2]
    assert residual.tolist().count(True) == 1


def _target(target_id, frame_id, label, centroid, parent="surface", point_start=0):
    c = np.array(centroid, dtype=float)
    return {
        "target_id": target_id,
        "frame_id": frame_id,
        "label": label,
        "parent_class": parent,
        "cluster_size": 10,
        "point_indices": list(range(point_start, point_start + 10)),
        "bbox_3d": {"min": (c - 0.05).tolist(), "max": (c + 0.05).tolist()},
        "centroid": c.tolist(),
        "mean_color": [100, 100, 100],
        "pca": {"normal": [0, 0, 1], "planarity": 0.8, "linearity": 0.1},
    }


def test_fuse_targets_merges_near_same_label_and_splits_far_target():
    module = load_module(SCRIPTS / "fuse_targets_to_objects.py", "fuse_targets_for_repo_test")
    args = type("Args", (), {
        "centroid_distance": 0.35,
        "bbox_distance": 0.35,
        "color_distance": 70.0,
        "normal_angle": 25.0,
        "zone_size": 100,
        "active_zone_window": 1,
    })()

    objects, decisions = module.fuse_targets(
        [
            _target("t1", 0, "floor", [0, 0, 0], point_start=0),
            _target("t2", 1, "floor", [0.12, 0, 0], point_start=100),
            _target("t3", 1, "floor", [2.0, 0, 0], point_start=200),
        ],
        args,
    )
    finalized = [module.finalize_object(o) for o in objects]

    assert len(finalized) == 2
    assert finalized[0]["target_count"] == 2
    assert decisions[1]["action"] == "merge"
    assert decisions[2]["action"] == "new_object"


def test_fuse_targets_marks_same_parent_label_conflict_ambiguous():
    module = load_module(SCRIPTS / "fuse_targets_to_objects.py", "fuse_targets_conflict_for_repo_test")
    args = type("Args", (), {
        "centroid_distance": 0.35,
        "bbox_distance": 0.35,
        "color_distance": 70.0,
        "normal_angle": 25.0,
        "zone_size": 100,
        "active_zone_window": 1,
    })()

    objects, _ = module.fuse_targets(
        [
            _target("t1", 0, "floor", [0, 0, 0], parent="surface", point_start=0),
            _target("t2", 1, "wall", [0.10, 0, 0], parent="surface", point_start=100),
        ],
        args,
    )
    finalized = [module.finalize_object(o) for o in objects]

    assert len(finalized) == 1
    assert finalized[0]["semantic_label"] == "ambiguous"
    assert finalized[0]["status"] == "ambiguous_object"


def test_finalize_keeps_high_vote_conflict_stable():
    module = load_module(SCRIPTS / "fuse_targets_to_objects.py", "fuse_targets_finalize_for_repo_test")
    obj = module.create_object("obj_000001", _target("t1", 0, "floor", [0, 0, 0], point_start=0))
    for idx in range(1, 9):
        module.update_object(obj, _target(f"t_floor_{idx}", idx, "floor", [0.01 * idx, 0, 0], point_start=idx * 100))
    module.update_object(obj, _target("t_wall", 9, "wall", [0.09, 0, 0], parent="surface", point_start=900))

    finalized = module.finalize_object(obj)
    assert finalized["semantic_label"] == "floor"
    assert finalized["dominant_label_ratio"] >= 0.8
    assert finalized["status"] == "stable"
