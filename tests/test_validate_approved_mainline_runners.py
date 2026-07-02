from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import validate_approved_mainline_runners as module


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_approved_mainline_runners.py"


def test_current_approved_mainline_runners_pass() -> None:
    report = module.validate()

    assert report["passed"] is True
    assert report["errors"] == []
    assert report["approved_runner_count"] == 5


def test_semantic_runner_requires_patch_gate_and_healthcheck(tmp_path: Path) -> None:
    runner = tmp_path / "bad_semantic.py"
    runner.write_text(
        """
from scripts.current_mainline_contract import reject_forbidden_production_input

def main(args):
    reject_forbidden_production_input(args.objects_jsonl)
""",
        encoding="utf-8",
    )

    report = module.validate_runner(
        {"path": runner.name, "stage": "semantic_evidence"},
        repo_root=tmp_path,
    )

    assert report["passed"] is False
    assert "missing_patch_promotion_gate" in report["errors"]
    assert "missing_mainline_healthcheck_call" in report["errors"]
    assert "missing_explicit_unpromoted_experiment_flag" in report["errors"]


def test_object_building_runner_requires_dense_allowlist(tmp_path: Path) -> None:
    runner = tmp_path / "bad_object.py"
    runner.write_text(
        """
from scripts.current_mainline_contract import reject_forbidden_production_input

def run_mainline_healthcheck(args):
    pass

def main(args):
    reject_forbidden_production_input(args.region_input)
    run_mainline_healthcheck(args)
""",
        encoding="utf-8",
    )

    report = module.validate_runner(
        {"path": runner.name, "stage": "object_building"},
        repo_root=tmp_path,
    )

    assert report["passed"] is False
    assert "missing_current_dense_input_allowlist_call" in report["errors"]
    assert "missing_validate_production_inputs_reference" in report["errors"]


def test_shell_runner_requires_preflight_and_tmux(tmp_path: Path) -> None:
    runner = tmp_path / "bad_remote.sh"
    runner.write_text(
        """
#!/usr/bin/env bash
set -euo pipefail
echo run
""",
        encoding="utf-8",
    )

    report = module.validate_runner(
        {"path": runner.name, "stage": "object_building"},
        repo_root=tmp_path,
    )

    assert report["passed"] is False
    assert "missing_run_preflight_switch" in report["errors"]
    assert "missing_mainline_preflight" in report["errors"]
    assert "missing_production_input_preflight" in report["errors"]
    assert "missing_tmux_launch" in report["errors"]


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
