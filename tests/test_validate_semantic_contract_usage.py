import json
import subprocess
import sys
from pathlib import Path

from scripts import validate_semantic_contract_usage as module


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_semantic_contract_usage.py"


def test_current_mainline_scripts_reuse_semantic_contract() -> None:
    report = module.validate()

    assert report["passed"] is True
    assert report["errors"] == []
    assert report["protected_script_count"] >= 10


def test_detects_local_semantic_assignment_without_contract(tmp_path: Path) -> None:
    script = tmp_path / "bad.py"
    script.write_text('LABEL_TO_SEMANTIC = {"unknown": 0}\n', encoding="utf-8")

    report = module.validate_script(script)

    assert report["passed"] is False
    assert "missing_semantic_label_contract_import" in report["errors"]
    assert any(error.startswith("local_semantic_contract_assignment:LABEL_TO_SEMANTIC") for error in report["errors"])


def test_allows_assignment_derived_from_contract(tmp_path: Path) -> None:
    script = tmp_path / "ok.py"
    script.write_text(
        "from scripts.semantic_label_contract import SEMANTIC_COLORS, SEMANTIC_TO_LABEL\n"
        "LABEL_COLORS = {label: SEMANTIC_COLORS[semantic] for semantic, label in SEMANTIC_TO_LABEL.items()}\n",
        encoding="utf-8",
    )

    report = module.validate_script(script)

    assert report["passed"] is True


def test_cli_json_report_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["passed"] is True
