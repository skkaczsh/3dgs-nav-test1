from pathlib import Path

from scripts.make_sam2_input_links import evidence_views, fallback_frame_path


def test_evidence_views_deduplicates_irregular_camera_frames() -> None:
    rows = [
        {"frame_id": 40, "cam_id": 2},
        {"frame_id": 10, "cam_id": 1, "image_path": "/data/cam1_000010.jpg"},
        {"frame_id": 40, "cam_id": 2, "image_path": "/data/cam2_000040.png"},
        {"frame_id": 10, "cam_id": 1, "image_path": "/data/duplicate.jpg"},
    ]

    assert evidence_views(rows) == [
        (10, 1, Path("/data/cam1_000010.jpg")),
        (40, 2, Path("/data/cam2_000040.png")),
    ]


def test_fallback_frame_path_supports_current_jpeg_layout(tmp_path: Path) -> None:
    source = tmp_path / "cam0" / "frame_000120.jpg"
    source.parent.mkdir()
    source.touch()

    assert fallback_frame_path(tmp_path, 0, 120) == source
