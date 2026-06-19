from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "probe_fine_mask_color_geometry_for_test",
    SCRIPTS / "probe_fine_mask_color_geometry.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def args(tmp_path: Path):
    return SimpleNamespace(
        output_dir=tmp_path,
        bbox_pad=0,
        large_bbox_ratio=0.10,
        high_fill_ratio=0.18,
        max_components=2,
        min_thin_aspect=3.0,
        min_boundary_lab_contrast=8.0,
        max_lab_std_mean=38.0,
    )


def test_clip_bbox_clamps_to_image():
    assert module.clip_bbox([-10, 5, 20, 99], width=16, height=16, pad=2) == (0, 3, 15, 15)


def test_largest_component_stats_reports_thin_aspect():
    mask = np.zeros((80, 80), dtype=bool)
    mask[40:43, 10:70] = True

    stats = module.largest_component_stats(mask)

    assert stats["components"] == 1
    assert stats["largest_area"] == int(mask.sum())
    assert stats["min_rect_aspect"] > 10


def test_lab_stats_detects_boundary_contrast():
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[:, :20] = (0, 0, 0)
    image[:, 20:] = (255, 255, 255)
    mask = np.zeros((40, 40), dtype=bool)
    mask[:, :20] = True

    stats = module.lab_stats(image, mask)

    assert stats["boundary_lab_contrast"] > 20
    assert stats["boundary_sobel_mean"] > 0


def test_process_item_writes_preview_and_flags_broad_mask(tmp_path: Path):
    image = np.full((100, 100, 3), 120, dtype=np.uint8)
    priority = np.zeros((100, 100), dtype=np.uint8)
    priority[10:80, 10:80] = 5
    image_path = tmp_path / "image.jpg"
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), priority)
    item = {
        "sample_id": "s1",
        "object_id": 1,
        "target_id": "t1",
        "semantic_label": "railing",
        "frame_id": 1,
        "cam_id": 0,
        "bbox_2d": {"xyxy": [10, 10, 79, 79]},
        "prepared_image": str(image_path),
        "prepared_current_mask": str(mask_path),
    }

    row = module.process_item(item, args(tmp_path))

    assert row["status"] == "ok"
    assert "high_fill_ratio" in row["risk_flags"]
    assert "not_thin" in row["risk_flags"]
    assert Path(row["preview_path"]).exists()
