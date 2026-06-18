import importlib.util
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "build_frame_local_object_qa_pack_for_test",
        SCRIPTS / "build_frame_local_object_qa_pack.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_object_risk_flags_ambiguous_and_large_single_target():
    module = load_module()
    obj = {
        "object_id": "obj_1",
        "semantic_label": "ambiguous",
        "status": "ambiguous_object",
        "target_count": 1,
        "point_count": 1000,
        "label_votes": {"wall": 60, "ground": 40},
        "bbox_3d": {"min": [0, 0, 0], "max": [3, 3, 2]},
        "normal": [0, 0, 1],
        "geometry_stats": {"planarity_mean": 0.5, "linearity_mean": 0.1},
    }

    score, reasons = module.object_risk(obj)

    assert score > 100
    assert "label_vote_conflict" in reasons
    assert "large_single_target_object" in reasons


def test_select_candidates_orders_by_risk_score():
    module = load_module()
    objects = [
        {"object_id": "obj_low", "semantic_label": "wall", "status": "stable", "target_count": 2, "point_count": 10, "label_votes": {"wall": 10}},
        {"object_id": "obj_high", "semantic_label": "railing", "status": "stable", "target_count": 1, "point_count": 100, "label_votes": {"railing": 100}, "bbox_3d": {"min": [0, 0, 0], "max": [20, 1, 3]}, "geometry_stats": {"linearity_mean": 0.1}},
    ]

    candidates = module.select_candidates(objects, limit=10)

    assert candidates[0]["object_id"] == "obj_high"
    assert "railing_extent_too_large" in candidates[0]["risk_reasons"]


def test_pick_evidence_targets_uses_largest_targets_first():
    module = load_module()
    candidate = {"targets": ["t1", "t2", "missing"]}
    targets = {
        "t1": {"target_id": "t1", "cluster_size": 5, "frame_id": 1, "cam_id": 0},
        "t2": {"target_id": "t2", "cluster_size": 10, "frame_id": 0, "cam_id": 0},
    }

    picked = module.pick_evidence_targets(candidate, targets, per_object=1)

    assert [x["target_id"] for x in picked] == ["t2"]


def test_crop_target_image_overlays_source_priority_mask(tmp_path: Path):
    module = load_module()
    image = tmp_path / "image.jpg"
    mask = tmp_path / "mask.png"
    output = tmp_path / "crop.jpg"
    Image.new("RGB", (20, 20), (100, 100, 100)).save(image)
    mask_im = Image.new("L", (20, 20), 0)
    for x in range(5, 15):
        for y in range(5, 15):
            mask_im.putpixel((x, y), 5)
    mask_im.save(mask)
    candidate = {"object_id": "obj_1", "semantic_label": "railing", "risk_score": 1}
    target = {
        "target_id": "t1",
        "frame_id": 1,
        "cam_id": 0,
        "label": "railing",
        "raw_label": "railing",
        "cluster_size": 10,
        "image_path": str(image),
        "mask_path": str(mask),
        "bbox_2d": {"xyxy": [5, 5, 14, 14]},
    }

    made, overlay_status = module.crop_target_image(tmp_path, candidate, target, output, margin=0, mask_overlay_alpha=0.5)

    assert made == output
    assert output.exists()
    assert overlay_status == "source_priority_5"
