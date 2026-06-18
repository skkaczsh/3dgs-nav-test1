from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "compare_frame_local_object_qa_for_test",
    SCRIPTS / "compare_frame_local_object_qa.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def write_report(path: Path, **overrides):
    data = {
        "objects": 10,
        "semantic_label_counts": {"wall": 5, "ground": 5},
        "all_candidate_count": 3,
        "candidate_count": 2,
        "all_risk_reason_counts": {"wall_normal_too_up": 2, "ground_has_large_height_span": 1},
        "candidate_risk_reason_counts": {"wall_normal_too_up": 1},
    }
    data.update(overrides)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_parse_named_report():
    name, path = module.parse_named_report("baseline=/tmp/report.json")

    assert name == "baseline"
    assert path == Path("/tmp/report.json")


def test_build_comparison_uses_full_risk_counts(tmp_path: Path):
    base = write_report(tmp_path / "base.json")
    candidate = write_report(
        tmp_path / "candidate.json",
        all_candidate_count=2,
        all_risk_reason_counts={"wall_normal_too_up": 0, "ground_has_large_height_span": 1},
        candidate_risk_reason_counts={"wall_normal_too_up": 5},
    )

    comparison = module.build_comparison([("base", base), ("candidate", candidate)])

    assert comparison["baseline"] == "base"
    assert comparison["versions"]["candidate"]["all_candidate_count"] == 2
    assert comparison["all_risk_deltas_from_baseline"]["candidate"]["wall_normal_too_up"] == -2
    assert comparison["all_risk_deltas_from_baseline"]["candidate"]["ground_has_large_height_span"] == 0


def test_candidate_counts_fall_back_to_legacy_risk_counts(tmp_path: Path):
    legacy = write_report(
        tmp_path / "legacy.json",
        all_risk_reason_counts={},
        candidate_risk_reason_counts={},
        risk_reason_counts={"legacy_reason": 4},
    )

    comparison = module.build_comparison([("legacy", legacy)])

    assert comparison["versions"]["legacy"]["candidate_risk_reason_counts"] == {"legacy_reason": 4}


def test_markdown_table_mentions_full_risk_fields(tmp_path: Path):
    base = write_report(tmp_path / "base.json")

    markdown = module.markdown_table(module.build_comparison([("base", base)]))

    assert "all_risk_reason_counts" in markdown
    assert "`base`" in markdown
