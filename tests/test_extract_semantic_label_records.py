import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "extract_semantic_label_records_for_test",
        SCRIPTS / "extract_semantic_label_records.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_records_preserves_identity_fields_from_vlm_raw_json():
    module = load_module()
    summary = {
        "combo": "sam2_prompt_v3_sky_label_merge_completion",
        "vlm": {
            "chunks": [
                {
                    "raw": json.dumps(
                        {
                            "items": [
                                {
                                    "mask_id": "1",
                                    "label": "equipment",
                                    "confidence": 0.91,
                                    "description": "white HVAC outdoor unit",
                                    "identity_hint": "white rectangular HVAC unit near parapet",
                                    "attributes": {
                                        "color": "white",
                                        "material": "metal",
                                        "shape": "rectangular box",
                                        "function": "HVAC outdoor unit",
                                    },
                                }
                            ]
                        }
                    )
                }
            ]
        },
    }
    labels = {"1": "equipment"}

    records = module.extract_from_summary(summary, labels)

    assert records[1]["label"] == "equipment"
    assert records[1]["confidence"] == 0.91
    assert records[1]["description"] == "white HVAC outdoor unit"
    assert records[1]["identity_hint"] == "white rectangular HVAC unit near parapet"
    assert records[1]["attributes"]["function"] == "HVAC outdoor unit"


def test_extract_records_removes_overlay_color_leak_from_identity_fields():
    module = load_module()
    records = module.parse_raw_items(
        json.dumps(
            {
                "items": [
                    {
                        "mask_id": "2",
                        "label": "floor",
                        "description": "magenta highlighted concrete roof floor",
                        "identity_hint": "yellow roof surface",
                        "attributes": {"color": "purple", "material": "concrete"},
                    }
                ]
            }
        )
    )

    assert records[2]["label"] == "floor"
    assert records[2]["attributes"] == {"material": "concrete"}
    assert "magenta" not in records[2]["description"].lower()
    assert "yellow" not in records[2]["identity_hint"].lower()
    assert records[2]["identity_sanitized"] == "overlay_color"
