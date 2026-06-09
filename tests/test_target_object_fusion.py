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


def _fine_candidate(candidate_id, frame_min, semantic, centroid, source_cluster=1, color=(100, 100, 100)):
    c = np.array(centroid, dtype=float)
    return {
        "candidate_id": candidate_id,
        "semantic": semantic,
        "source_type": 2,
        "source_cluster": source_cluster,
        "subcluster": 0,
        "points": 10,
        "bbox_3d": {"min": (c - 0.03).tolist(), "max": (c + 0.03).tolist()},
        "centroid": c.tolist(),
        "mean_visual_color": list(color),
        "frame_min": frame_min,
        "frame_max": frame_min + 2,
        "frame_count": 3,
        "camera_counts": {1: 10},
        "mask_count": 1,
        "linearity": 0.3,
        "planarity": 0.3,
        "scattering": 0.1,
        "normal": [0, 0, 1],
    }


def test_incremental_fine_fusion_uses_frame_window_and_semantic_gate():
    module = load_module(SCRIPTS / "fuse_enriched_fine_objects_incremental.py", "incremental_fine_fusion_for_repo_test")
    args = type("Args", (), {
        "centroid_distance": 0.45,
        "cross_source_centroid_distance": 0.25,
        "bbox_distance": 0.05,
        "color_distance": 30.0,
        "active_frame_window": 20,
    })()

    objects, decisions = module.fuse(
        [
            _fine_candidate(1, 10, 16, [0, 0, 0]),
            _fine_candidate(2, 15, 16, [0.05, 0, 0]),
            _fine_candidate(3, 18, 17, [0.06, 0, 0]),
            _fine_candidate(4, 100, 16, [0.07, 0, 0]),
        ],
        args,
    )
    finalized = [module.finalize_object(o) for o in objects]

    assert len(finalized) == 3
    assert finalized[0]["candidate_count"] == 2
    assert decisions[1]["action"] == "merge"
    assert decisions[2]["action"] == "new_object"
    assert decisions[3]["action"] == "new_object"


def test_frame_fine_target_builder_splits_components_and_keeps_point_indices():
    module = load_module(SCRIPTS / "build_frame_fine_targets_from_enriched.py", "frame_fine_targets_for_repo_test")
    props = [
        "x",
        "y",
        "z",
        "red",
        "green",
        "blue",
        "semantic",
        "accepted_candidate",
        "fine_object",
        "source_type",
        "source_cluster",
        "subcluster",
        "visual_red",
        "visual_green",
        "visual_blue",
        "frame",
        "camera",
        "mask",
        "point_index",
        "trace_status",
    ]
    rows = []
    for point_index, x in enumerate([0.00, 0.03, 1.00, 1.03, 4.00]):
        rows.append(
            [
                x,
                0,
                0,
                255,
                0,
                255,
                16,
                200001,
                1,
                2,
                9,
                1,
                90,
                100,
                110,
                7,
                1,
                3,
                point_index,
                1,
            ]
        )
    args = type("Args", (), {"voxel_size": 0.08, "min_target_points": 2})()

    targets, report, _ = module.build_targets(props, np.array(rows, dtype=float), args)

    assert report["targets"] == 2
    assert report["small_residual_points"] == 1
    assert [t["cluster_size"] for t in targets] == [2, 2]
    assert targets[0]["frame_id"] == 7
    assert targets[0]["label"] == "equipment"
    assert targets[0]["parent_class"] == "fine_object"
    assert targets[0]["point_indices"] == [0, 1]
