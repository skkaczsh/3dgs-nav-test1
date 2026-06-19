import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "extract_undistorted_frames_jpeg.py"
    spec = importlib.util.spec_from_file_location("extract_undistorted_frames_jpeg", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeCap:
    opened = []

    def __init__(self, path):
        self.path = path
        self.pos = 0
        self.reads = []
        FakeCap.opened.append(self)

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.pos = int(value)

    def read(self):
        self.reads.append(self.pos)
        image = np.full((4, 4, 3), self.pos % 255, dtype=np.uint8)
        return True, image

    def release(self):
        pass


def test_extract_cam_uses_explicit_frame_map_without_direct_fallback(tmp_path: Path, monkeypatch):
    module = load_module()
    FakeCap.opened.clear()
    monkeypatch.setitem(module.config.VIDEO_FILES, 0, "cam0.mkv")
    monkeypatch.setattr(module.cv2, "VideoCapture", FakeCap)
    monkeypatch.setattr(module, "load_video_timestamps", lambda _path: [])
    monkeypatch.setattr(module, "make_maps", lambda: {0: (None, None)})
    monkeypatch.setattr(module.cv2, "remap", lambda frame, _map1, _map2, _interp: frame)
    writes = []
    monkeypatch.setattr(module.cv2, "imwrite", lambda path, image, params: writes.append((path, int(image[0, 0, 0]))) or True)

    result = module.extract_cam(
        cam_id=0,
        ids=[10, 20],
        output_dir=tmp_path,
        quality=90,
        skip_existing=False,
        sync_mode="frame-map",
        time_scale=0.1,
        max_delta=0.15,
        frame_map={(10, 0): 12},
        require_frame_map=True,
    )

    assert result["ok"] == 1
    assert result["failed"] == 1
    assert result["mapped_reads"] == 1
    assert result["direct_reads"] == 0
    assert result["missing_frame_map_reads"] == 1
    assert FakeCap.opened[0].reads == [12]
    assert writes == [(str(tmp_path / "cam0" / "frame_000010.jpg"), 12)]
    assert result["frame_map"][0]["video_idx"] == 12
    assert result["frame_map"][1]["status"] == "missing_frame_map"


def test_cli_requires_frame_map_jsonl_when_frame_map_is_required(tmp_path: Path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(sys, "argv", [
        "extract_undistorted_frames_jpeg.py",
        "--output-dir", str(tmp_path),
        "--sync-mode", "frame-map",
        "--require-frame-map",
    ])

    try:
        module.main()
    except SystemExit as exc:
        assert str(exc) == "--frame-map-jsonl is required when --sync-mode frame-map --require-frame-map is used"
    else:
        raise AssertionError("main() should reject missing --frame-map-jsonl")
