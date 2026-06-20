import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_mimo_scene_prior.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_mimo_scene_prior_for_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stride_indices_include_last_frame():
    module = load_module()

    assert module.stride_indices(100, 30) == [0, 30, 60, 90, 99]
    assert module.stride_indices(1, 30) == [0]


def test_parse_json_object_accepts_fenced_json():
    module = load_module()

    parsed, error = module.parse_json_object('```json\n{"ok": true, "items": [1]}\n```')

    assert error == ""
    assert parsed == {"ok": True, "items": [1]}


def test_add_frame_bounds_maps_segment_ranks_to_frames():
    module = load_module()
    prior = {
        "segments": [
            {"start_rank": 1, "end_rank": 2, "area_type": "outdoor_parking"},
            {"start_rank": 99, "end_rank": 100, "area_type": "unknown"},
        ]
    }
    records = [
        {"rank": 0, "frame_index": 0, "time_sec": 0.0},
        {"rank": 1, "frame_index": 30, "time_sec": 3.0},
        {"rank": 2, "frame_index": 60, "time_sec": 6.0},
    ]

    out = module.add_frame_bounds(prior, records)

    assert out["segments"][0]["segment_id"] == "scene_000"
    assert out["segments"][0]["start_frame"] == 30
    assert out["segments"][0]["end_frame"] == 60
    assert out["segments"][0]["start_time_sec"] == 3.0
    assert out["segments"][0]["end_time_sec"] == 6.0
    assert out["segments"][1]["start_frame"] == 0
    assert out["segments"][1]["end_frame"] == 60
