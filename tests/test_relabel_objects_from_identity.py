import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "relabel_objects_from_identity_for_test",
        SCRIPTS / "relabel_objects_from_identity.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_identity_relabels_guardrail_mislabeled_as_equipment():
    module = load_module()
    obj = {
        "object_id": "obj_000001",
        "semantic_label": "equipment",
        "description": "yellow metal guardrail",
        "label_votes": {"equipment": 120},
    }

    out, decision = module.relabel_object(obj)

    assert out["semantic_label"] == "railing"
    assert out["original_semantic_label"] == "equipment"
    assert decision["changed"] is True
    assert decision["reason"] == "identity_railing"


def test_identity_keeps_hvac_even_when_fence_is_visible():
    module = load_module()
    obj = {
        "object_id": "obj_000002",
        "semantic_label": "equipment",
        "description": "multiple white HVAC outdoor units behind a fence",
        "label_votes": {"equipment": 120},
    }

    out, decision = module.relabel_object(obj)

    assert out["semantic_label"] == "equipment"
    assert decision["changed"] is False


def test_identity_relabels_ambiguous_rooftop_surface_to_floor():
    module = load_module()
    obj = {
        "object_id": "obj_000003",
        "semantic_label": "ambiguous",
        "description": "large rooftop surface",
        "label_votes": {"floor": 60, "wall": 50},
    }

    out, decision = module.relabel_object(obj)

    assert out["semantic_label"] == "floor"
    assert decision["reason"] == "identity_floor"


def test_identity_ignores_minority_description_votes_when_primary_is_clear():
    module = load_module()
    obj = {
        "object_id": "obj_000004",
        "semantic_label": "building",
        "description": "distant high-rise building facade with grid windows",
        "description_votes": {
            "distant high-rise building facade": 240,
            "yellow metal guardrail": 12,
        },
        "label_votes": {"building": 238, "railing": 55},
    }

    out, decision = module.relabel_object(obj)

    assert out["semantic_label"] == "building"
    assert decision["changed"] is False


def test_identity_primary_description_beats_secondary_attributes():
    module = load_module()
    obj = {
        "object_id": "obj_000005",
        "semantic_label": "ambiguous",
        "description": "distant high-rise building facade",
        "dominant_attributes": {"function": {"value": "railing", "vote_ratio": 0.3}},
        "label_votes": {"building": 68, "railing": 80},
    }

    out, decision = module.relabel_object(obj)

    assert out["semantic_label"] == "building"
    assert decision["reason"] == "identity_building"


def test_remap_ply_updates_semantic_and_palette(tmp_path: Path):
    module = load_module()
    src = tmp_path / "in.ply"
    dst = tmp_path / "out.ply"
    src.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "property int object",
                "property uchar semantic",
                "end_header",
                "0 0 0 30 210 190 1 16",
                "1 0 0 30 210 190 2 16",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    objects = [
        {"object_id": "obj_000001", "semantic_label": "railing"},
        {"object_id": "obj_000002", "semantic_label": "equipment"},
    ]

    report = module.remap_ply(src, dst, objects)

    assert report["changed_vertices"] == 1
    rows = dst.read_text(encoding="utf-8").splitlines()[-2:]
    assert json.dumps(rows)
    assert rows[0].endswith("1 9")
    assert "255 210 40" in rows[0]
