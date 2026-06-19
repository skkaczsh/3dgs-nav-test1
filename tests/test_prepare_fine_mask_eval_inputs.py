from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "prepare_fine_mask_eval_inputs_for_test",
    SCRIPTS / "prepare_fine_mask_eval_inputs.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_rewrite_path_replaces_prefix():
    path = module.rewrite_path("/remote/a/image.jpg", "/remote", "/local")

    assert path == Path("/local/a/image.jpg")


def test_prepare_creates_symlink_package(tmp_path: Path):
    source = tmp_path / "source"
    image = source / "frames/cam1/frame_000100.jpg"
    mask = source / "masks/cam1_000100.png"
    crop = source / "crops/a.jpg"
    for path in [image, mask, crop]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    manifest = {
        "items": [
            {
                "sample_id": "0001_obj7_cam1_frame000100",
                "image_path": "/remote/frames/cam1/frame_000100.jpg",
                "current_mask_path": "/remote/masks/cam1_000100.png",
                "crop_path": "/remote/crops/a.jpg",
            }
        ]
    }
    args = SimpleNamespace(
        manifest=tmp_path / "manifest.json",
        output_dir=tmp_path / "out",
        mode="symlink",
        source_prefix="/remote",
        target_prefix=str(source),
        overwrite=False,
    )

    report = module.prepare(manifest, args)

    assert report["sample_count"] == 1
    assert report["ready_images"] == 1
    assert report["ready_current_masks"] == 1
    assert report["ready_crops"] == 1
    assert report["missing_count"] == 0
    assert (tmp_path / "out/images/0001_obj7_cam1_frame000100.jpg").is_symlink()
    assert (tmp_path / "out/images.txt").read_text(encoding="utf-8").strip().endswith(".jpg")


def test_prepare_reports_missing_files(tmp_path: Path):
    manifest = {
        "items": [
            {
                "sample_id": "0001_obj7_cam1_frame000100",
                "image_path": "/missing/frame.jpg",
                "current_mask_path": "/missing/mask.png",
            }
        ]
    }
    args = SimpleNamespace(
        manifest=tmp_path / "manifest.json",
        output_dir=tmp_path / "out",
        mode="symlink",
        source_prefix="",
        target_prefix="",
        overwrite=False,
    )

    report = module.prepare(manifest, args)

    assert report["ready_images"] == 0
    assert report["ready_current_masks"] == 0
    assert report["missing_count"] == 2
