import importlib.util
import json
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


def test_target_label_loader_preserves_vlm_quality_fields(tmp_path: Path):
    module = load_module(SCRIPTS / "build_targets_from_masks.py", "build_targets_label_loader_for_repo_test")
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            {
                "labels": {
                    "1": "floor",
                    "2": {
                        "label": "railing",
                        "confidence": 0.73,
                        "mixed": True,
                        "is_large_surface": False,
                        "can_merge_to_surface": False,
                        "ambiguous_with": ["pipe", "equipment"],
                        "reason": "thin foreground structure",
                    },
                    "bad": {"label": "ignore"},
                }
            }
        ),
        encoding="utf-8",
    )

    records = module.load_label_records(path)
    labels = module.load_labels(path)

    assert labels == {1: "floor", 2: "railing"}
    assert records[1]["confidence"] == 1.0
    assert records[1]["mixed"] is False
    assert records[2]["confidence"] == 0.73
    assert records[2]["mixed"] is True
    assert records[2]["ambiguous_with"] == ["pipe", "equipment"]
    assert records[2]["reason"] == "thin foreground structure"


def test_target_label_loader_accepts_item_list_schema(tmp_path: Path):
    module = load_module(SCRIPTS / "build_targets_from_masks.py", "build_targets_label_list_for_repo_test")
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            [
                {"mask_id": "3", "label": "equipment", "confidence": "0.5", "ambiguous_with": "railing"},
                {"id": 4, "name": "wall", "confidence": 2.0},
            ]
        ),
        encoding="utf-8",
    )

    records = module.load_label_records(path)

    assert records[3]["label"] == "equipment"
    assert records[3]["confidence"] == 0.5
    assert records[3]["ambiguous_with"] == ["railing"]
    assert records[4]["label"] == "wall"
    assert records[4]["confidence"] == 1.0


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


def test_cross_candidate_review_pack_parses_target_and_selects_representatives():
    module = load_module(SCRIPTS / "build_cross_candidate_review_pack.py", "review_pack_for_repo_test")
    meta = module.parse_target_id("fine_t_000579_cam1_mask0008_sem16_cc02")
    assert meta == {"frame": 579, "cam": 1, "mask": 8, "semantic": 16, "cc": 2}

    tracklets = {
        "trk_a": {
            "tracklet_id": "trk_a",
            "point_count": 20,
            "target_count": 2,
            "target_ids": ["fine_t_000579_cam1_mask0008_sem16_cc02"],
            "accepted_candidate_votes": {"200001": 20},
        },
        "trk_b": {
            "tracklet_id": "trk_b",
            "point_count": 100,
            "target_count": 4,
            "target_ids": ["fine_t_000580_cam1_mask0009_sem16_cc00"],
            "accepted_candidate_votes": {"200002": 100},
        },
    }
    obj = {"tracklet_ids": ["trk_a", "trk_b"]}
    reps = module.choose_representative_tracklets(obj, tracklets, "200002", 1)

    assert reps[0]["tracklet_id"] == "trk_b"


def test_cross_candidate_review_pack_prefers_existing_artifact_and_scaled_raw(tmp_path):
    module = load_module(SCRIPTS / "build_cross_candidate_review_pack.py", "review_pack_paths_for_repo_test")
    artifact_a = tmp_path / "semantic_a"
    artifact_b = tmp_path / "semantic_b"
    raw_dir = tmp_path / "raw"
    overlay_dir = artifact_a / "images" / "cam1_000189" / "combo"
    overlay_dir.mkdir(parents=True)
    (overlay_dir / "overlay.png").write_bytes(b"png")
    raw_dir.mkdir()
    (raw_dir / "cam1_001890.png").write_bytes(b"raw")

    meta = module.parse_target_id("fine_t_000189_cam1_mask0023_sem16_cc00")
    paths = module.resolve_artifact_paths([artifact_b, artifact_a], "combo", meta)
    raw = module.raw_image_path(raw_dir, meta, frame_scale=10)

    assert paths["overlay"].endswith("semantic_a/images/cam1_000189/combo/overlay.png")
    assert raw.endswith("raw/cam1_001890.png")


def test_review_contact_sheet_builder_writes_sheet(tmp_path):
    from PIL import Image

    module = load_module(SCRIPTS / "make_review_contact_sheets.py", "review_contact_sheet_for_repo_test")
    asset = tmp_path / "overlay.png"
    Image.new("RGB", (32, 24), (255, 0, 0)).save(asset)
    item = {
        "review_id": "review_001",
        "proposal": {
            "object_a": "o1",
            "object_b": "o2",
            "score": 0.12,
            "candidate_a": "200001",
            "candidate_b": "200002",
        },
        "representatives": [
            {
                "side": "a",
                "rep_index": 0,
                "tracklet_id": "trk_1",
                "target_meta": {"frame": 1, "cam": 2},
                "copied_overlay": str(asset),
            }
        ],
    }

    out = module.make_sheet(item, tmp_path / "sheets", 160, 90)

    assert out.exists()
    assert out.name == "review_001_contact_sheet.jpg"


def test_review_contact_sheet_remaps_server_asset_paths(tmp_path):
    from PIL import Image

    module = load_module(SCRIPTS / "make_review_contact_sheets.py", "review_contact_sheet_remap_for_repo_test")
    pack_dir = tmp_path / "pack"
    asset = pack_dir / "assets" / "proposal_001" / "a0_overlay.png"
    asset.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24), (0, 255, 0)).save(asset)
    rep = {"copied_overlay": "/root/epfs/some_pack/assets/proposal_001/a0_overlay.png"}

    resolved = module.image_path(rep, pack_dir)

    assert resolved == asset


def test_vlm_merge_review_extracts_and_normalizes_json():
    module = load_module(SCRIPTS / "review_cross_candidate_merges_vlm.py", "vlm_merge_review_for_repo_test")
    parsed = module.extract_json(
        """```json
        {"decision": "MERGE", "confidence": 1.3, "physical_relation": "same_object", "reason": "continuous railing", "evidence": "same line", "risk": "overmerge"}
        ```"""
    )
    normalized = module.normalize_decision(parsed)

    assert normalized["decision"] == "merge"
    assert normalized["confidence"] == 1.0
    assert normalized["evidence"] == ["same line"]


def test_vlm_merge_review_resizes_image_data_url(tmp_path):
    import base64
    from io import BytesIO

    from PIL import Image

    module = load_module(SCRIPTS / "review_cross_candidate_merges_vlm.py", "vlm_merge_review_image_for_repo_test")
    path = tmp_path / "sheet.jpg"
    Image.new("RGB", (400, 100), (10, 20, 30)).save(path)

    data_url = module.encode_image_data_url(path, long_edge=100, jpeg_quality=80)
    payload = base64.b64decode(data_url.split(",", 1)[1])
    resized = Image.open(BytesIO(payload))

    assert data_url.startswith("data:image/jpeg;base64,")
    assert max(resized.size) == 100


def _review_object(object_id, centroid, points=10):
    c = np.array(centroid, dtype=float)
    return {
        "long_object_id": object_id,
        "label": "equipment",
        "point_count": points,
        "tracklet_ids": [f"trk_{object_id}"],
        "tracklet_count": 1,
        "target_count": 2,
        "frame_min": 1,
        "frame_max": 3,
        "frame_count": 3,
        "bbox_3d": {"min": (c - 0.1).tolist(), "max": (c + 0.1).tolist()},
        "centroid": c.tolist(),
        "mean_color": [10, 20, 30],
        "accepted_candidate_votes": {object_id: points},
        "source_cluster_votes": {"1": points},
        "status": "stable_long_object",
    }


def test_apply_cross_candidate_reviews_merges_only_accepted_pairs():
    module = load_module(SCRIPTS / "apply_cross_candidate_merge_reviews.py", "apply_reviews_for_repo_test")
    objects = [
        _review_object("long_obj_0001", [0, 0, 0], points=10),
        _review_object("long_obj_0002", [1, 0, 0], points=20),
        _review_object("long_obj_0003", [5, 0, 0], points=30),
    ]
    reviews = [
        {
            "review_id": "r1",
            "object_a": "long_obj_0001",
            "object_b": "long_obj_0002",
            "vlm": {"decision": "merge", "confidence": 0.9},
        },
        {
            "review_id": "r2",
            "object_a": "long_obj_0002",
            "object_b": "long_obj_0003",
            "vlm": {"decision": "merge", "confidence": 0.2},
        },
    ]

    merged, decisions = module.apply_reviews(objects, reviews, min_confidence=0.75)
    groups = sorted(row["source_long_object_ids"] for row in merged)

    assert len(merged) == 2
    assert ["long_obj_0001", "long_obj_0002"] in groups
    assert ["long_obj_0003"] in groups
    assert [d["accepted"] for d in decisions] == [True, False]
    assert max(row["point_count"] for row in merged) == 30


def test_cross_candidate_review_html_writes_template_and_context(tmp_path):
    module = load_module(SCRIPTS / "make_cross_candidate_review_html.py", "review_html_for_repo_test")
    items = [
        {
            "review_id": "review_001",
            "proposal": {
                "object_a": "long_obj_0001",
                "object_b": "long_obj_0002",
                "candidate_a": "200001",
                "candidate_b": "200002",
                "same_source_cluster": True,
                "score": 0.12,
                "centroid_distance": 0.5,
                "bbox_distance": 0.0,
                "bbox_overlap_ratio": 0.9,
                "color_distance": 4.0,
            },
        }
    ]
    output = tmp_path / "out"
    output.mkdir()
    csv_path = output / "manual_merge_decisions.csv"
    module.write_decision_template(items, csv_path)
    html_text = module.render_html(items, tmp_path / "sheets", output, csv_path)

    assert "review_001" in csv_path.read_text()
    assert "rooftop scan" in html_text
    assert "../sheets/review_001_contact_sheet.jpg" in html_text
    assert "merge</code>" in html_text


def test_normalize_manual_merge_decisions_validates_rows(tmp_path):
    module = load_module(SCRIPTS / "normalize_manual_merge_decisions.py", "normalize_manual_review_for_repo_test")
    review_jsonl = tmp_path / "items.jsonl"
    review_jsonl.write_text(
        '{"review_id":"review_001","proposal":{"object_a":"long_obj_0001","object_b":"long_obj_0002","candidate_a":"200001","candidate_b":"200002"}}\n'
        '{"review_id":"review_002","proposal":{"object_a":"long_obj_0003","object_b":"long_obj_0004","candidate_a":"200003","candidate_b":"200004"}}\n',
        encoding="utf-8",
    )
    csv_path = tmp_path / "manual.csv"
    csv_path.write_text(
        "review_id,object_a,object_b,decision,confidence,reviewer,notes\n"
        "review_001,long_obj_0001,long_obj_0002,merge,0.8,skk,same railing\n"
        "review_002,long_obj_0003,long_obj_0004,pending,,,\n"
        "review_003,long_obj_0005,long_obj_0006,bad,0.5,,\n",
        encoding="utf-8",
    )

    rows, errors = module.normalize(csv_path, review_jsonl)

    assert len(rows) == 1
    assert rows[0]["vlm"]["decision"] == "merge"
    assert rows[0]["vlm"]["confidence"] == 0.8
    assert [e["error"] for e in errors] == ["pending", "unknown_review_id"]


def test_manual_merge_review_workflow_end_to_end(tmp_path):
    module = load_module(SCRIPTS / "run_manual_merge_review_workflow.py", "manual_workflow_for_repo_test")
    review_jsonl = tmp_path / "items.jsonl"
    review_jsonl.write_text(
        '{"review_id":"review_001","proposal":{"object_a":"long_obj_0001","object_b":"long_obj_0002","candidate_a":"200001","candidate_b":"200002"}}\n',
        encoding="utf-8",
    )
    manual_csv = tmp_path / "manual.csv"
    manual_csv.write_text(
        "review_id,object_a,object_b,decision,confidence,reviewer,notes\n"
        "review_001,long_obj_0001,long_obj_0002,merge,0.9,skk,same object\n",
        encoding="utf-8",
    )
    objects = tmp_path / "objects.jsonl"
    objects.write_text(
        json.dumps(_review_object("long_obj_0001", [0, 0, 0], points=10)) + "\n"
        + json.dumps(_review_object("long_obj_0002", [0.1, 0, 0], points=20)) + "\n",
        encoding="utf-8",
    )

    rows, errors = module.normalize(manual_csv, review_jsonl)
    merged, decisions = module.apply_reviews(module.load_objects(objects), rows, min_confidence=0.75)
    qa_report = module.qa_reviewed_merge(module.load_objects(objects), merged, decisions)

    assert errors == []
    assert len(merged) == 1
    assert decisions[0]["accepted"] is True
    assert qa_report["passed"] is True


def test_qa_reviewed_merge_results_checks_invariants():
    module = load_module(SCRIPTS / "qa_reviewed_merge_results.py", "qa_review_merge_for_repo_test")
    inputs = [
        _review_object("long_obj_0001", [0, 0, 0], points=10),
        _review_object("long_obj_0002", [0.1, 0, 0], points=20),
    ]
    outputs = [
        {
            **_review_object("review_obj_0001", [0.05, 0, 0], points=30),
            "source_long_object_ids": ["long_obj_0001", "long_obj_0002"],
            "source_object_count": 2,
        }
    ]
    decisions = [{"object_a": "long_obj_0001", "object_b": "long_obj_0002", "accepted": True}]

    report = module.qa(inputs, outputs, decisions)
    bad_report = module.qa(inputs, [{**outputs[0], "point_count": 29}], decisions)

    assert report["passed"] is True
    assert bad_report["passed"] is False
    assert bad_report["checks"]["point_count_preserved"] is False


def test_summarize_cross_candidate_review_stage_marks_missing_qwen(tmp_path):
    module = load_module(SCRIPTS / "summarize_cross_candidate_review_stage.py", "summary_review_stage_for_repo_test")
    pack = tmp_path / "pack"
    (pack / "contact_sheets").mkdir(parents=True)
    (pack / "review_html").mkdir()
    (pack / "manual_review_normalized").mkdir()
    (pack / "manual_workflow_pending" / "applied").mkdir(parents=True)
    (pack / "cross_candidate_review_items.jsonl").write_text("{}\n", encoding="utf-8")
    (pack / "cross_candidate_review_pack_report.json").write_text(
        json.dumps({"item_count": 1, "items_with_any_image": 1, "copied_overlay_count": 1}),
        encoding="utf-8",
    )
    (pack / "contact_sheets" / "contact_sheet_report.json").write_text(
        json.dumps({"sheet_count": 1}),
        encoding="utf-8",
    )
    (pack / "review_html" / "review_html_report.json").write_text(
        json.dumps({"html": "index.html", "decision_template": "manual.csv"}),
        encoding="utf-8",
    )
    (pack / "manual_review_normalized" / "manual_merge_review_report.json").write_text("{}", encoding="utf-8")
    (pack / "manual_workflow_pending" / "manual_merge_workflow_report.json").write_text(
        json.dumps({"manual_review_count": 0, "manual_error_count": 1, "input_object_count": 2, "output_object_count": 2, "accepted_merge_count": 0}),
        encoding="utf-8",
    )
    (pack / "manual_workflow_pending" / "applied" / "review_merge_report.json").write_text(
        json.dumps({"accepted_merge_count": 0, "objects_path": "objects.jsonl"}),
        encoding="utf-8",
    )
    (pack / "manual_workflow_pending" / "qa_reviewed_merge_report.json").write_text(
        json.dumps({"passed": True}),
        encoding="utf-8",
    )

    summary = module.build_summary(pack)
    markdown = module.render_markdown(summary)

    assert summary["stage_status"]["review_pack_ready"] is True
    assert summary["stage_status"]["qwen_review_ready"] is False
    assert "run_manual_merge_review_workflow.py" in markdown


def test_package_cross_candidate_review_copies_manifest_and_zip(tmp_path):
    module = load_module(SCRIPTS / "package_cross_candidate_review.py", "package_review_for_repo_test")
    pack = tmp_path / "pack"
    for rel in (
        "cross_candidate_review_items.jsonl",
        "review_html/index.html",
        "review_html/manual_merge_decisions.csv",
        "contact_sheets/review_001_contact_sheet.jpg",
    ):
        path = pack / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    out = tmp_path / "out"
    copied, missing = module.collect_files(pack, out)
    manifest = module.build_manifest(out, copied, missing, pack)
    manifest_path = out / "manifest_sha256.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    zip_path = tmp_path / "review.zip"
    module.write_zip(out, zip_path)

    copied_paths = {row["path"] for row in manifest["files"]}
    assert "cross_candidate_review_items.jsonl" in copied_paths
    assert "contact_sheets/review_001_contact_sheet.jpg" in copied_paths
    assert manifest["file_count"] == len(copied)
    assert zip_path.exists()


def test_verify_review_delivery_manifest_detects_corruption(tmp_path):
    package = load_module(SCRIPTS / "package_cross_candidate_review.py", "package_review_for_verify_test")
    verify = load_module(SCRIPTS / "verify_review_delivery_manifest.py", "verify_review_delivery_for_repo_test")
    delivery = tmp_path / "delivery"
    (delivery / "contact_sheets").mkdir(parents=True)
    file_path = delivery / "contact_sheets" / "review_001_contact_sheet.jpg"
    file_path.write_text("ok", encoding="utf-8")
    manifest = package.build_manifest(delivery, [file_path], [], tmp_path / "source")
    (delivery / "manifest_sha256.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = verify.verify_dir(delivery)
    file_path.write_text("corrupt", encoding="utf-8")
    bad_report = verify.verify_dir(delivery)

    assert report["passed"] is True
    assert bad_report["passed"] is False
    assert bad_report["errors"][0]["error"] in {"size_mismatch", "sha256_mismatch"}


def test_diagnose_server_connectivity_marks_missing_bind(monkeypatch):
    module = load_module(SCRIPTS / "diagnose_server_connectivity.py", "diagnose_connectivity_for_repo_test")

    def fake_run(cmd, timeout=10.0):
        if cmd == ["ifconfig"]:
            return 0, "en0: flags=x\n\tinet 192.168.0.3 netmask x\n", ""
        if cmd == ["ssh", "-G", "scan-train"]:
            return 0, "hostname 10.0.8.114\nport 31909\nbindaddress 192.168.100.115\nuser root\n", ""
        return 1, "", "unexpected"

    monkeypatch.setattr(module, "run", fake_run)
    monkeypatch.setattr(module, "tcp_check", lambda hostname, port, timeout: {"hostname": hostname, "port": port, "reachable": False, "error": "timeout"})

    report = module.diagnose(["scan-train"], timeout=1.0)

    assert report["all_reachable"] is False
    assert report["hosts"][0]["bind_address_present_locally"] is False
    assert report["hosts"][0]["tcp"]["error"] == "timeout"


def test_summarize_route_status_renders_blocker_and_side_tracks(tmp_path):
    module = load_module(SCRIPTS / "summarize_route_status.py", "route_status_for_repo_test")
    connectivity = tmp_path / "connectivity.json"
    connectivity.write_text(
        json.dumps(
            {
                "all_reachable": False,
                "hosts": [
                    {
                        "host": "scan-train",
                        "ssh_config": {"hostname": "10.0.8.114", "port": "31909"},
                        "tcp": {"reachable": False, "error": "timed out"},
                        "bind_address_present_locally": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    stage = tmp_path / "stage.json"
    stage.write_text(
        json.dumps(
            {
                "stage_status": {"review_pack_ready": True, "contact_sheets_ready": True, "manual_html_ready": True, "pending_apply_safe": True, "qwen_review_ready": False},
                "manual_merge_qa": {"passed": True, "input_point_count": 1, "output_point_count": 1},
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"file_count": 1, "missing": []}), encoding="utf-8")
    conceptseg = tmp_path / "conceptseg.md"
    conceptseg.write_text("side track", encoding="utf-8")
    old = tmp_path / "old.json"
    old.write_text("{}", encoding="utf-8")
    args = type("Args", (), {
        "connectivity": connectivity,
        "stage_summary": stage,
        "delivery_manifest": manifest,
        "delivery_zip": tmp_path / "delivery.zip",
        "conceptseg_report": conceptseg,
        "old_route_summary": old,
    })()

    status = module.build_status(args)
    markdown = module.render_markdown(status)

    assert status["connectivity"]["all_reachable"] is False
    assert "ConceptSeg-R1" in markdown
    assert "Qwen review ready: `False`" in markdown


def test_append_route_status_snapshot_writes_history(tmp_path):
    module = load_module(SCRIPTS / "append_route_status_snapshot.py", "route_snapshot_for_repo_test")
    status = {
        "connectivity": {"all_reachable": False},
        "main_route": {
            "stage_status": {"qwen_review_ready": False, "review_pack_ready": True, "contact_sheets_ready": True, "manual_html_ready": True, "pending_apply_safe": True},
            "manual_workflow_pending": {"manual_review_count": 0, "accepted_merge_count": 0, "input_object_count": 66, "output_object_count": 66},
            "manual_merge_qa": {"passed": True, "input_point_count": 1, "output_point_count": 1},
        },
        "delivery": {"file_count": 21, "missing": []},
        "new_model_side_track": {"status": "side_track_only"},
        "old_route_side_track": {"status": "visual_reference_only"},
    }
    status_path = tmp_path / "status.json"
    history = tmp_path / "history.jsonl"
    status_path.write_text(json.dumps(status), encoding="utf-8")

    snapshot = module.append_snapshot(status_path, history, "2026-06-10T00:00:00+00:00")

    assert snapshot["all_servers_reachable"] is False
    assert snapshot["delivery_missing_count"] == 0
    assert len(history.read_text().splitlines()) == 1
