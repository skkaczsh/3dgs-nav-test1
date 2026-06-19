from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_video_pts_timing.py"
    spec = importlib.util.spec_from_file_location("audit_video_pts_timing", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_ffprobe_frame_rows_uses_first_numeric_timestamp_after_keyframe():
    module = load_module()
    rows = module.parse_ffprobe_frame_rows(
        "frame,1,0.000000,0.000000,I\n"
        "frame,0,N/A,0.100000,P\n"
        "frame,0,0.200000,N/A,B\n"
    )

    assert [row["pts"] for row in rows] == [0.0, 0.1, 0.2]
    assert rows[0]["key_frame"] == 1
    assert rows[0]["pict_type"] == "I"


def test_percentile_interpolates():
    module = load_module()
    assert module.percentile([0.0, 1.0, 2.0], 50) == 1.0
