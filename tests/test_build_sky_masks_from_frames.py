from pathlib import Path


def test_frame_path_prefers_current_six_digit_jpeg(tmp_path: Path) -> None:
    from scripts import build_sky_masks_from_frames as module

    frame = tmp_path / "cam1" / "frame_003000.jpg"
    frame.parent.mkdir()
    frame.write_bytes(b"")
    assert module.frame_path(tmp_path, 1, 3000) == frame


def test_frame_path_supports_legacy_four_digit_png(tmp_path: Path) -> None:
    from scripts import build_sky_masks_from_frames as module

    frame = tmp_path / "cam0" / "frame_0042.png"
    frame.parent.mkdir()
    frame.write_bytes(b"")
    assert module.frame_path(tmp_path, 0, 42) == frame


def test_read_frame_ids_accepts_json_and_newline_lists(tmp_path: Path) -> None:
    from scripts import build_sky_masks_from_frames as module

    json_list = tmp_path / "frames.json"
    json_list.write_text("[30, 10, 30]", encoding="utf-8")
    assert module.read_frame_ids(json_list) == [10, 30]
    lines = tmp_path / "frames.txt"
    lines.write_text("30\n10\n30\n", encoding="utf-8")
    assert module.read_frame_ids(lines) == [10, 30]
