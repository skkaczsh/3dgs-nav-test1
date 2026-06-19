from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "prepare_consolidated_viewer_objects_for_test",
        SCRIPTS / "prepare_consolidated_viewer_objects.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_convert_objects_uses_sidecar_ids_and_preserves_original_id():
    module = load_module()
    rows = [
        {"object_id": "surf_wall_00001", "semantic_label": "wall"},
        {"object_id": "pass_obj_000002", "semantic_label": "ground"},
    ]
    mapping = {"surf_wall_00001": 7, "pass_obj_000002": 8}

    converted, missing = module.convert_objects(rows, mapping)

    assert missing == []
    assert converted[0]["object_id"] == 7
    assert converted[0]["viewer_object_id"] == 7
    assert converted[0]["original_object_id"] == "surf_wall_00001"
    assert converted[1]["object_id"] == 8


def test_convert_objects_reports_missing_mapping():
    module = load_module()
    converted, missing = module.convert_objects(
        [{"object_id": "surf_wall_00001"}, {"object_id": "missing"}],
        {"surf_wall_00001": 7},
    )

    assert len(converted) == 1
    assert missing == ["missing"]
