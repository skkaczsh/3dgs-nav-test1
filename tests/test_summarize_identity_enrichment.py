import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "summarize_identity_enrichment_for_test",
        SCRIPTS / "summarize_identity_enrichment.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_summarize_records_counts_identity_fields():
    module = load_module()
    summary = module.summarize_records(
        [
            {"label": "equipment", "description": "white HVAC outdoor unit", "attributes": {"color": "white"}},
            {"label": "equipment", "description": "", "attributes": {}},
            {"label": "floor", "identity_hint": "broad roof surface"},
        ]
    )

    assert summary["record_count"] == 3
    assert summary["enriched_count"] == 2
    assert summary["enriched_by_label"] == {"equipment": 1, "floor": 1}
    assert summary["top_descriptions_by_label"]["equipment"][0]["description"] == "white HVAC outdoor unit"


def test_iter_objects_and_summary(tmp_path: Path):
    module = load_module()
    objects = tmp_path / "objects.jsonl"
    objects.write_text(
        "\n".join(
            [
                json.dumps({"semantic_label": "equipment", "description": "gray electrical cabinet", "description_vote_ratio": 0.8}),
                json.dumps({"semantic_label": "floor", "description_votes": {"large roof surface": 10}}),
            ]
        ),
        encoding="utf-8",
    )

    summary = module.summarize_objects(list(module.iter_objects(objects)))

    assert summary["object_count"] == 2
    assert summary["enriched_count"] == 2
    assert summary["label_counts"] == {"equipment": 1, "floor": 1}
    assert summary["description_vote_ratio_median"] == 0.8
