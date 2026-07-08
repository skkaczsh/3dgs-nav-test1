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
    assert report["approved_runner_count"] == 7


def test_spg_review_runner_skips_legacy_production_preflight(tmp_path: Path) -> None:
    runner = tmp_path / "cluster_superpoint_graph.py"
    runner.write_text("print('review candidate')\n", encoding="utf-8")

    report = module.validate_runner(
        {"path": runner.name, "stage": "object_building", "contract": "spg_review"},
        repo_root=tmp_path,
    )

    assert report["passed"] is True
    assert report["errors"] == []
    assert report["warnings"] == ["spg_review_runner_legacy_preflight_skipped"]


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


def test_qa_viewer_export_runner_requires_explicit_qa_preview_source_flag(tmp_path: Path) -> None:
    runner = tmp_path / "bad_qa.py"
    runner.write_text(
        """
from scripts.current_mainline_contract import reject_forbidden_production_input

def validation_status(path):
    return {"passed": True}

def main(args):
    reject_forbidden_production_input(args.source_ply)
    command = ["scripts/rewrite_viewer_ply_semantics.py", "--allow-unvalidated-export"]
    return command
""",
        encoding="utf-8",
    )

    report = module.validate_runner(
        {"path": runner.name, "stage": "qa_viewer_export"},
        repo_root=tmp_path,
    )

    assert report["passed"] is False
    assert "missing_explicit_qa_preview_source_flag" in report["errors"]


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
    assert "missing_geometry_contract_rsync" in report["errors"]
    assert "missing_tmux_launch" in report["errors"]


def test_shell_runner_must_not_disable_remote_dense_allowlist(tmp_path: Path) -> None:
    runner = tmp_path / "bad_remote.sh"
    runner.write_text(
        """
#!/usr/bin/env bash
RUN_PREFLIGHT=1
validate_current_mainline.py
validate_production_inputs.py --require-current-dense "$REGION_INPUT" "$PATCH_LABELS"
rsync -az scripts/geometry_input_contract.py "$REMOTE:scripts/"
rsync -az docs/current_dense_patch_state.json "$REMOTE:docs/"
python scripts/run_dense_patch_object_refinement_v7.py --no-require-current-dense-inputs --run
tmux new-session -d -s run
""",
        encoding="utf-8",
    )

    report = module.validate_runner(
        {"path": runner.name, "stage": "object_building"},
        repo_root=tmp_path,
    )

    assert report["passed"] is False
    assert "remote_runner_disables_current_dense_allowlist" in report["errors"]


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
