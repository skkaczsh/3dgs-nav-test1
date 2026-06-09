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


def _track_target(target_id, frame_id, centroid, color=(100, 100, 100)):
    c = np.array(centroid, dtype=float)
    return {
        "target_id": target_id,
        "frame_id": frame_id,
        "label": "equipment",
        "label_id": 16,
        "parent_class": "fine_object",
        "cluster_size": 5,
        "point_indices": list(range(frame_id, frame_id + 5)),
        "bbox_3d": {"min": (c - 0.02).tolist(), "max": (c + 0.02).tolist()},
        "centroid": c.tolist(),
        "mean_color": list(color),
        "pca": {"normal": [0, 0, 1], "linearity": 0.1, "planarity": 0.1},
    }


def test_tracklet_builder_merges_short_gap_and_splits_long_gap():
    module = load_module(SCRIPTS / "build_tracklets_from_frame_targets.py", "tracklets_for_repo_test")
    args = type("Args", (), {
        "max_frame_gap": 10,
        "centroid_distance": 0.2,
        "bbox_distance": 0.05,
        "color_distance": 30.0,
        "normal_angle": 180.0,
    })()

    tracklets, decisions = module.build_tracklets(
        [
            _track_target("t1", 0, [0, 0, 0]),
            _track_target("t2", 5, [0.04, 0, 0]),
            _track_target("t3", 30, [0.05, 0, 0]),
            _track_target("t4", 35, [0.06, 0, 0], color=(220, 220, 220)),
        ],
        args,
    )
    finalized = [module.finalize_tracklet(t) for t in tracklets]

    assert len(finalized) == 3
    assert finalized[0]["target_count"] == 2
    assert decisions[1]["action"] == "merge"
    assert decisions[2]["action"] == "new_tracklet"
    assert decisions[3]["action"] == "new_tracklet"


def _long_tracklet(tracklet_id, frame_min, centroid, candidate="200001", source="9", color=(100, 100, 100)):
    c = np.array(centroid, dtype=float)
    return {
        "tracklet_id": tracklet_id,
        "target_id": tracklet_id,
        "label": "equipment",
        "label_id": 16,
        "parent_class": "fine_object",
        "frames": [frame_min],
        "frame_id": frame_min,
        "frame_min": frame_min,
        "frame_max": frame_min,
        "target_count": 2,
        "cluster_size": 10,
        "point_count": 10,
        "bbox_3d": {"min": (c - 0.03).tolist(), "max": (c + 0.03).tolist()},
        "centroid": c.tolist(),
        "mean_color": list(color),
        "accepted_candidate_votes": {candidate: 10},
        "source_cluster_votes": {source: 10},
    }


def test_long_range_association_uses_accepted_candidate_evidence():
    module = load_module(SCRIPTS / "associate_tracklets_long_range.py", "long_range_assoc_for_repo_test")
    args = type("Args", (), {
        "same_candidate_centroid_distance": 1.5,
        "same_candidate_bbox_distance": 0.5,
        "same_candidate_color_distance": 90.0,
        "source_frame_gap": 30,
        "source_centroid_distance": 0.4,
        "source_bbox_distance": 0.1,
        "source_color_distance": 40.0,
        "cross_frame_gap": 10,
        "cross_centroid_distance": 0.2,
        "cross_bbox_distance": 0.05,
        "cross_color_distance": 25.0,
    })()

    objects, decisions = module.associate(
        [
            _long_tracklet("trk1", 0, [0, 0, 0], candidate="200001"),
            _long_tracklet("trk2", 300, [0.4, 0, 0], candidate="200001"),
            _long_tracklet("trk3", 310, [4.0, 0, 0], candidate="200002", source="99"),
        ],
        args,
    )
    finalized = [module.finalize_object(o) for o in objects]

    assert len(finalized) == 2
    assert finalized[0]["tracklet_count"] == 2
    assert decisions[1]["action"] == "merge"
    assert decisions[1]["reason"] == "same_accepted_candidate"
    assert decisions[2]["action"] == "new_object"


def _long_object(object_id, centroid, candidate, source="99", color=(100, 100, 100), label="equipment"):
    c = np.array(centroid, dtype=float)
    return {
        "long_object_id": object_id,
        "label": label,
        "point_count": 100,
        "tracklet_count": 3,
        "frame_min": 0,
        "frame_max": 20,
        "bbox_3d": {"min": (c - 0.1).tolist(), "max": (c + 0.1).tolist()},
        "centroid": c.tolist(),
        "mean_color": list(color),
        "dominant_accepted_candidate": candidate,
        "dominant_accepted_candidate_ratio": 0.95,
        "dominant_source_cluster": source,
    }


def test_cross_candidate_merge_proposals_filter_and_rank_candidates():
    module = load_module(SCRIPTS / "propose_cross_candidate_object_merges.py", "cross_candidate_proposals_for_repo_test")
    args = type("Args", (), {
        "centroid_distance": 1.2,
        "bbox_distance": 0.35,
        "min_bbox_overlap": 0.05,
        "color_distance": 80.0,
        "frame_gap": 360,
        "auto_review_score": 1.2,
        "max_proposals": 20,
    })()

    proposals = module.propose(
        [
            _long_object("o1", [0, 0, 0], "200001"),
            _long_object("o2", [0.4, 0, 0], "200002"),
            _long_object("o3", [5, 0, 0], "200003"),
            _long_object("o4", [10, 0, 0], "200001"),
        ],
        args,
    )

    assert len(proposals) == 1
    assert proposals[0]["object_a"] == "o1"
    assert proposals[0]["object_b"] == "o2"
    assert proposals[0]["same_source_cluster"] is True
