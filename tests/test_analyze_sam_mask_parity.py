import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_sam_mask_parity.py"

spec = importlib.util.spec_from_file_location("analyze_sam_mask_parity", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules["analyze_sam_mask_parity"] = module
spec.loader.exec_module(module)


def write_masks(path: Path, image_id: str, masks):
    payload = {
        "image_name": image_id,
        "masks": [
            {
                "segmentation": mask,
                "area": int(sum(sum(1 for x in row if x) for row in mask)),
                "bbox": [0, 0, 1, 1],
                "predicted_iou": 1.0,
                "stability_score": 1.0,
            }
            for mask in masks
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_decode_rle_matches_coco_column_major_order():
    mask = module.decode_segmentation({"size": [2, 3], "counts": [1, 2, 3]})
    assert mask.tolist() == [[False, True, False], [True, False, False]]


def test_compare_image_reports_extra_and_missing_pixels(tmp_path):
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    image_id = "cam0_000001"
    write_masks(
        baseline_dir / f"{image_id}_sam_masks.json",
        image_id,
        [[[True, False], [False, False]]],
    )
    write_masks(
        candidate_dir / f"{image_id}_sam_masks.json",
        image_id,
        [[[False, True], [False, False]]],
    )

    row, details = module.compare_image(image_id, baseline_dir, candidate_dir, min_area=0, match_iou=0.5)

    assert row.baseline_coverage == 0.25
    assert row.candidate_coverage == 0.25
    assert row.extra_pixel_ratio == 0.25
    assert row.missing_pixel_ratio == 0.25
    assert row.union_iou == 0.0
    assert row.unmatched_baseline_masks == 1
    assert row.unmatched_candidate_masks == 1
    assert {detail.side for detail in details} == {"baseline", "candidate"}
