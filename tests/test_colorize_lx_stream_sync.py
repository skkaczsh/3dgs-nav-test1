import importlib.util
import sys
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "colorize_lx_stream.py"
    spec = importlib.util.spec_from_file_location("colorize_lx_stream", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_requires_frame_map_jsonl_when_frame_map_is_required(tmp_path: Path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(sys, "argv", [
        "colorize_lx_stream.py",
        "--lx-file", str(tmp_path / "missing.lx"),
        "--output", str(tmp_path / "out.ply"),
        "--require-frame-map",
    ])

    try:
        module.main()
    except SystemExit as exc:
        assert str(exc) == "--frame-map-jsonl is required when --require-frame-map is used"
    else:
        raise AssertionError("main() should reject missing --frame-map-jsonl")
