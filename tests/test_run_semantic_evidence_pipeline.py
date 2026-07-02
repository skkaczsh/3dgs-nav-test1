from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_semantic_evidence_pipeline.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_semantic_evidence_pipeline", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_gate(path: Path, status: str = "pass") -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "patch-experiment-promotion-gate/v1",
                "status": status,
                "candidate": "v2_bucket_attach",
                "reasons": [] if status == "pass" else ["visual_status_not_accepted=pending"],
            }
        ),
        encoding="utf-8",
    )
    return path


def base_args(tmp_path: Path) -> argparse.Namespace:
    source_ply = tmp_path / "source.ply"
    objects = tmp_path / "objects.jsonl"
    output_dir = tmp_path / "out"
    source_ply.write_text("ply\nend_header\n", encoding="utf-8")
    objects.write_text("{}\n", encoding="utf-8")
    prefix = "semantic_evidence"
    return argparse.Namespace(
        source_ply=source_ply,
        objects_jsonl=objects,
        output_dir=output_dir,
        output_prefix=prefix,
        fused_objects_jsonl=output_dir / f"{prefix}_objects.jsonl",
        fusion_report=output_dir / f"{prefix}_fusion_report.json",
        fusion_validation=output_dir / f"{prefix}_fusion_validation.json",
        output_ply=output_dir / f"{prefix}_viewer.ply",
        viewer_report=output_dir / f"{prefix}_viewer_report.json",
        viewer_export_plan=output_dir / f"{prefix}_viewer_export_plan.json",
        plan_json=None,
        python="python",
        run=False,
        patch_gate=write_gate(tmp_path / "gate.json"),
        allow_unpromoted_patch_experiment=False,
        allow_unvalidated_export=False,
        mainline_healthcheck=ROOT / "scripts" / "validate_current_mainline.py",
        skip_mainline_healthcheck=False,
        sam_weight=1.0,
        teacher_weight=1.25,
        scene_weight=0.35,
        min_total_weight=3.0,
        min_winner_ratio=0.58,
        min_scene_supported_ratio=0.52,
        allow_scene_only=False,
    )


def test_pipeline_plan_has_fuse_validate_export_commands(tmp_path: Path) -> None:
    module = load_module()
    args = base_args(tmp_path)
    plan = module.build_plan(args, module.patch_gate_status(args.patch_gate))

    assert plan["schema"] == "semantic-evidence-pipeline-plan/v1"
    assert plan["status"] == "ready"
    assert [row["name"] for row in plan["commands"]] == [
        "fuse_object_semantic_evidence",
        "validate_object_semantic_evidence_fusion",
        "run_validated_semantic_viewer_export",
    ]
    assert "--run" in plan["commands"][2]["argv"]


def test_pipeline_blocks_failed_patch_gate(tmp_path: Path) -> None:
    module = load_module()
    args = base_args(tmp_path)
    args.patch_gate = write_gate(tmp_path / "gate.json", status="fail")

    plan = module.build_plan(args, module.patch_gate_status(args.patch_gate))

    assert plan["status"] == "blocked"
    assert plan["patch_gate"]["passed"] is False


def test_pipeline_allows_explicit_unpromoted_experiment(tmp_path: Path) -> None:
    module = load_module()
    args = base_args(tmp_path)
    args.patch_gate = write_gate(tmp_path / "gate.json", status="fail")
    args.allow_unpromoted_patch_experiment = True

    plan = module.build_plan(args, module.patch_gate_status(args.patch_gate))

    assert plan["status"] == "ready"


def test_pipeline_run_mode_refuses_blocked_plan(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    args = base_args(tmp_path)
    args.patch_gate = write_gate(tmp_path / "gate.json", status="fail")
    argv = [
        "run_semantic_evidence_pipeline.py",
        "--source-ply",
        str(args.source_ply),
        "--objects-jsonl",
        str(args.objects_jsonl),
        "--output-dir",
        str(args.output_dir),
        "--patch-gate",
        str(args.patch_gate),
        "--run",
    ]
    monkeypatch.setattr("sys.argv", argv)

    assert module.main() == 2


def test_pipeline_run_mode_checks_health_and_runs_three_commands(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    args = base_args(tmp_path)
    calls: list[str] = []

    monkeypatch.setattr(module, "run_mainline_healthcheck", lambda parsed: calls.append("healthcheck"))
    monkeypatch.setattr(module, "run_command", lambda argv, cwd: calls.append("command"))
    argv = [
        "run_semantic_evidence_pipeline.py",
        "--source-ply",
        str(args.source_ply),
        "--objects-jsonl",
        str(args.objects_jsonl),
        "--output-dir",
        str(args.output_dir),
        "--patch-gate",
        str(args.patch_gate),
        "--run",
    ]
    monkeypatch.setattr("sys.argv", argv)

    assert module.main() == 0
    assert calls == ["healthcheck", "command", "command", "command"]
