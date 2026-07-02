from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import validate_production_input_guard_usage as module


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_production_input_guard_usage.py"


def test_current_production_input_guard_usage_passes() -> None:
    report = module.validate()

    assert report["passed"] is True
    assert report["errors"] == []
    assert report["protected_script_count"] >= 7


def test_validator_rejects_local_legacy_guard(tmp_path: Path) -> None:
    script = tmp_path / "bad.py"
    script.write_text(
        """
def reject_forbidden_path(path):
    if "frame_object_points_stride10.ply" in str(path):
        raise ValueError(path)

def main(args):
    reject_forbidden_path(args.source_ply)
""",
        encoding="utf-8",
    )

    report = module.validate_script(script)

    assert report["passed"] is False
    assert "missing_reject_forbidden_production_input_import" in report["errors"]
    assert "missing_reject_forbidden_production_input_call" in report["errors"]
    assert "local_reject_forbidden_path_definition" in report["errors"]


def test_validator_rejects_direct_matcher_import(tmp_path: Path) -> None:
    script = tmp_path / "bad.py"
    script.write_text(
        """
from scripts.current_mainline_contract import forbidden_production_input_match

def main(args):
    if forbidden_production_input_match(args.source_ply):
        raise ValueError(args.source_ply)
""",
        encoding="utf-8",
    )

    report = module.validate_script(script)

    assert report["passed"] is False
    assert "direct_forbidden_production_input_match_import" in report["errors"]


def test_validator_accepts_shared_guard(tmp_path: Path) -> None:
    script = tmp_path / "ok.py"
    script.write_text(
        """
from scripts.current_mainline_contract import reject_forbidden_production_input

def main(args):
    reject_forbidden_production_input(args.source_ply)
""",
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
