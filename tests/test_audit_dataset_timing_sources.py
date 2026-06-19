from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_dataset_timing_sources.py"
    spec = importlib.util.spec_from_file_location("audit_dataset_timing_sources", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_read_lx_headers_parses_section_count_and_header_fields(tmp_path):
    module = load_module()
    path = tmp_path / "sample.lx"
    header0 = struct.pack("<12f", 0.0, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9, 0.0, 0.0, 27.0, 0.0)
    points0 = np.zeros(2, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("marker", "<u4")])
    points0["marker"] = [1, 2]
    header1_values = [0.0] * 12
    header1_values[8] = 1.0
    header1 = struct.pack("<12f", *header1_values)
    points1 = np.zeros(1, dtype=points0.dtype)
    path.write_bytes(
        header0
        + struct.pack("<I", len(points0))
        + points0.tobytes()
        + header1
        + struct.pack("<I", len(points1))
        + points1.tobytes()
    )

    rows = module.read_lx_headers(path)
    assert len(rows) == 2
    assert rows[0]["count"] == 2
    assert rows[0]["data_offset"] == 52
    assert rows[0]["floats"][1:4] == [1.0, 2.0, 3.0]
    assert rows[1]["section"] == 1


def test_pose_match_summary_checks_lx_pose_convention():
    module = load_module()
    headers = [
        {
            "section": 0,
            "floats": [0.0, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9, 0.0, 0.0, 27.0, 0.0],
            "uints": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    ]
    poses = [
        {
            "frame_id": 0,
            "pos": np.asarray([1.0, 2.0, 3.0]),
            "quat": np.asarray([0.9, 0.1, 0.2, 0.3]),
        }
    ]
    summary = module.pose_match_summary(headers, poses)
    assert summary["uint8_matches_frame_id_ratio"] == 1.0
    assert summary["header_pos_float1_3_vs_img_pos_error"]["max"] == 0.0
    assert summary["header_quat_float4_7_xyzw_vs_img_pos_error"]["max"] == 0.0
