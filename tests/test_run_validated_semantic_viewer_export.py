from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_validated_semantic_viewer_export.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_validated_semantic_viewer_export", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_validation(path: Path, passed: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "object-semantic-evidence-fusion-validation/v1",
                "passed": passed,
                "errors": [] if passed else ["object=1:scene_only_promotion"],
            }
        ),
        encoding="utf-8",
    )
    return path


def base_args(tmp_path: Path) -> argparse.Namespace:
    source_ply = tmp_path / "source.ply"
    objects = tmp_path / "objects.jsonl"
    source_ply.write_text("ply\nend_header\n", encoding="utf-8")
    objects.write_text("{}\n", encoding="utf-8")
    return argparse.Namespace(
        source_ply=source_ply,
        objects_jsonl=objects,
        fusion_validation=write_validation(tmp_path / "validation.json"),
        output_ply=tmp_path / "semantic.ply",
        report_json=tmp_path / "semantic_report.json",
        plan_json=None,
        python="python",
        run=False,
        allow_unvalidated_export=False,
    )


def test_runner_builds_rewrite_command_when_validation_passes(tmp_path: Path) -> None:
    module = load_module()
    args = base_args(tmp_path)
    plan = module.build_plan(args, module.validation_status(args.fusion_validation))

    assert plan["schema"] == "validated-semantic-viewer-export-plan/v1"
    assert plan["status"] == "ready"
    argv = plan["commands"][0]["argv"]
    assert "scripts/rewrite_viewer_ply_semantics.py" in argv
    assert "--source-ply" in argv
    assert "--objects-jsonl" in argv


def test_runner_blocks_failed_validation(tmp_path: Path) -> None:
    module = load_module()
    args = base_args(tmp_path)
    args.fusion_validation = write_validation(tmp_path / "validation.json", passed=False)

    plan = module.build_plan(args, module.validation_status(args.fusion_validation))

    assert plan["status"] == "blocked"
    assert plan["validation"]["passed"] is False


def test_runner_allows_explicit_unvalidated_export(tmp_path: Path) -> None:
    module = load_module()
    args = base_args(tmp_path)
    args.fusion_validation = write_validation(tmp_path / "validation.json", passed=False)
    args.allow_unvalidated_export = True

    plan = module.build_plan(args, module.validation_status(args.fusion_validation))

    assert plan["status"] == "ready"
    assert plan["allow_unvalidated_export"] is True


def test_runner_run_mode_refuses_blocked_plan(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    args = base_args(tmp_path)
    args.fusion_validation = write_validation(tmp_path / "validation.json", passed=False)
    argv = [
        "run_validated_semantic_viewer_export.py",
        "--source-ply",
        str(args.source_ply),
        "--objects-jsonl",
        str(args.objects_jsonl),
        "--fusion-validation",
        str(args.fusion_validation),
        "--output-ply",
        str(args.output_ply),
        "--run",
    ]
    monkeypatch.setattr("sys.argv", argv)

    assert module.main() == 2


def test_runner_run_mode_executes_rewrite_after_validation_passes(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    args = base_args(tmp_path)
    calls: list[str] = []

    monkeypatch.setattr(module, "run_command", lambda argv, cwd: calls.append("command"))
    argv = [
        "run_validated_semantic_viewer_export.py",
        "--source-ply",
        str(args.source_ply),
        "--objects-jsonl",
        str(args.objects_jsonl),
        "--fusion-validation",
        str(args.fusion_validation),
        "--output-ply",
        str(args.output_ply),
        "--report-json",
        str(args.report_json),
        "--run",
    ]
    monkeypatch.setattr("sys.argv", argv)

    assert module.main() == 0
    assert calls == ["command"]
