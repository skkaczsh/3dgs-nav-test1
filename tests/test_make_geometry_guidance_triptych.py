from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_geometry_guidance_triptych.py"
    spec = importlib.util.spec_from_file_location("make_geometry_guidance_triptych", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_overlay_points_only_changes_rendered_pixels():
    module = load_module()
    original = np.zeros((2, 2, 3), dtype=np.uint8)
    rendered = np.zeros_like(original)
    rendered[0, 0] = [100, 100, 100]

    overlay = module.overlay_points(original, rendered, 1.0)

    assert overlay[0, 0].tolist() == [100, 100, 100]
    assert overlay[1, 1].tolist() == [0, 0, 0]


def test_make_triptych_writes_image(tmp_path: Path):
    module = load_module()
    paths = []
    for name, value in [("orig.jpg", 40), ("depth.jpg", 80), ("render.jpg", 120)]:
        path = tmp_path / name
        cv2.imwrite(str(path), np.full((4, 5, 3), value, dtype=np.uint8))
        paths.append(path)

    out = tmp_path / "triptych.jpg"
    summary = module.make_triptych(paths[0], paths[1], paths[2], out, 0.5)

    assert out.exists()
    img = cv2.imread(str(out))
    assert img is not None
    assert img.shape[1] == 15
    assert summary["valid_render_pixels"] == 20
    assert summary["overlay_source"] == "rendered_rgb"


def test_make_triptych_falls_back_to_depth_valid_overlay(tmp_path: Path):
    module = load_module()
    original = tmp_path / "orig.jpg"
    depth = tmp_path / "depth.jpg"
    rendered = tmp_path / "render.jpg"
    npz = tmp_path / "geom.npz"
    cv2.imwrite(str(original), np.zeros((4, 5, 3), dtype=np.uint8))
    cv2.imwrite(str(depth), np.full((4, 5, 3), 80, dtype=np.uint8))
    cv2.imwrite(str(rendered), np.zeros((4, 5, 3), dtype=np.uint8))
    valid = np.zeros((4, 5), dtype=np.uint8)
    valid[1:3, 1:4] = 255
    depth_values = np.ones((4, 5), dtype=np.float32)
    np.savez_compressed(npz, valid=valid, depth=depth_values)

    summary = module.make_triptych(original, depth, rendered, tmp_path / "triptych_depth.jpg", 0.5, npz)

    assert summary["valid_render_pixels"] == 0
    assert summary["valid_overlay_pixels"] == 6
    assert summary["overlay_source"] == "depth_valid"


def test_write_review_html_uses_relative_image_paths(tmp_path: Path):
    module = load_module()
    image = tmp_path / "triptych" / "cam0_000001_triptych.jpg"
    image.parent.mkdir()
    image.write_bytes(b"fake")
    html_path = tmp_path / "triptych" / "review.html"
    module.write_review_html(
        {
            "items": [
                {
                    "output_path": str(image),
                    "overlay_source": "depth_valid",
                    "valid_overlay_pixels": 5,
                    "image_pixels": 10,
                }
            ]
        },
        html_path,
        "Review",
    )

    text = html_path.read_text(encoding="utf-8")
    assert "cam0_000001_triptych.jpg" in text
    assert "overlay coverage" in text
