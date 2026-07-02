from __future__ import annotations

from pathlib import Path

from scripts.validate_geometry_input_contract_usage import validate, validate_script


def test_current_geometry_input_contract_usage_passes() -> None:
    report = validate()

    assert report["passed"] is True
    assert report["errors"] == []


def test_validator_rejects_missing_geometry_only_guard(tmp_path: Path) -> None:
    script = tmp_path / "bad.py"
    script.write_text(
        """
def normalized_original_label(row):
    label = row.get("semantic_label") or "unknown"
    if label == "horizontal":
        return "floor"
    return label
""",
        encoding="utf-8",
    )

    report = validate_script(script)

    assert report["passed"] is False
    assert "missing_geometry_input_contract_import" in report["errors"]
    assert "normalized_original_label_missing_geometry_only_guard" in report["errors"]
