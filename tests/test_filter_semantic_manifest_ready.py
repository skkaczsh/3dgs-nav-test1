import importlib.util
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "filter_semantic_manifest_ready_for_test",
        SCRIPTS / "filter_semantic_manifest_ready.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sam_ready_validates_json_and_age(tmp_path: Path):
    module = load_module()
    valid = tmp_path / "valid_sam_masks.json"
    valid.write_text(json.dumps({"masks": []}), encoding="utf-8")
    old = time.time() - 120
    os.utime(valid, (old, old))

    assert module.sam_ready(valid, validate_json=True, min_age_seconds=30) == (True, None)


def test_sam_ready_rejects_invalid_json(tmp_path: Path):
    module = load_module()
    invalid = tmp_path / "bad_sam_masks.json"
    invalid.write_text('{"masks": [', encoding="utf-8")
    old = time.time() - 120
    os.utime(invalid, (old, old))

    assert module.sam_ready(invalid, validate_json=True, min_age_seconds=30) == (False, "sam_invalid_json")


def test_sam_ready_rejects_unstable_recent_file(tmp_path: Path):
    module = load_module()
    recent = tmp_path / "recent_sam_masks.json"
    recent.write_text(json.dumps({"masks": []}), encoding="utf-8")

    assert module.sam_ready(recent, validate_json=True, min_age_seconds=30) == (False, "sam_unstable")
