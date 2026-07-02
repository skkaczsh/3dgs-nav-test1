from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_object_semantic_evidence_fusion.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_object_semantic_evidence_fusion", SCRIPT)
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
    objects = tmp_path / "objects.jsonl"
    objects.write_text("{}\n", encoding="utf-8")
    return argparse.Namespace(
        objects_jsonl=objects,
        output_jsonl=tmp_path / "fused.jsonl",
        report=tmp_path / "fused_report.json",
        plan_json=None,
        python="python",
        run=False,
        patch_gate=write_gate(tmp_path / "gate.json"),
        allow_unpromoted_patch_experiment=False,
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


def test_runner_builds_fusion_command_when_gate_passes(tmp_path: Path) -> None:
    module = load_module()
    args = base_args(tmp_path)
    gate = module.patch_gate_status(args.patch_gate)
    plan = module.build_plan(args, gate)

    assert plan["schema"] == "object-semantic-evidence-fusion-plan/v1"
    assert plan["status"] == "ready"
    argv = plan["commands"][0]["argv"]
    assert "scripts/fuse_object_semantic_evidence.py" in argv
    assert "--objects-jsonl" in argv
    assert "--min-winner-ratio" in argv


def test_runner_blocks_unpromoted_patch_gate(tmp_path: Path) -> None:
    module = load_module()
    args = base_args(tmp_path)
    args.patch_gate = write_gate(tmp_path / "gate.json", status="fail")

    plan = module.build_plan(args, module.patch_gate_status(args.patch_gate))

    assert plan["status"] == "blocked"
    assert plan["patch_gate"]["passed"] is False


def test_runner_allows_explicit_experimental_plan(tmp_path: Path) -> None:
    module = load_module()
    args = base_args(tmp_path)
    args.patch_gate = write_gate(tmp_path / "gate.json", status="fail")
    args.allow_unpromoted_patch_experiment = True

    plan = module.build_plan(args, module.patch_gate_status(args.patch_gate))

    assert plan["status"] == "ready"
    assert plan["allow_unpromoted_patch_experiment"] is True


def test_runner_main_writes_blocked_plan_without_run_failure(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    args = base_args(tmp_path)
    args.patch_gate = write_gate(tmp_path / "gate.json", status="fail")
    plan_path = tmp_path / "plan.json"
    argv = [
        "run_object_semantic_evidence_fusion.py",
        "--objects-jsonl",
        str(args.objects_jsonl),
        "--output-jsonl",
        str(args.output_jsonl),
        "--report",
        str(args.report),
        "--patch-gate",
        str(args.patch_gate),
        "--plan-json",
        str(plan_path),
    ]
    monkeypatch.setattr("sys.argv", argv)

    assert module.main() == 0
    assert json.loads(plan_path.read_text(encoding="utf-8"))["status"] == "blocked"


def test_runner_run_mode_refuses_blocked_plan(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    args = base_args(tmp_path)
    args.patch_gate = write_gate(tmp_path / "gate.json", status="fail")
    argv = [
        "run_object_semantic_evidence_fusion.py",
        "--objects-jsonl",
        str(args.objects_jsonl),
        "--output-jsonl",
        str(args.output_jsonl),
        "--report",
        str(args.report),
        "--patch-gate",
        str(args.patch_gate),
        "--run",
    ]
    monkeypatch.setattr("sys.argv", argv)

    assert module.main() == 2


def test_runner_run_mode_checks_mainline_before_command(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    args = base_args(tmp_path)
    calls: list[str] = []

    monkeypatch.setattr(module, "run_mainline_healthcheck", lambda parsed: calls.append("healthcheck"))
    monkeypatch.setattr(module, "run_command", lambda argv, cwd: calls.append("command"))
    argv = [
        "run_object_semantic_evidence_fusion.py",
        "--objects-jsonl",
        str(args.objects_jsonl),
        "--output-jsonl",
        str(args.output_jsonl),
        "--report",
        str(args.report),
        "--patch-gate",
        str(args.patch_gate),
        "--run",
    ]
    monkeypatch.setattr("sys.argv", argv)

    assert module.main() == 0
    assert calls == ["healthcheck", "command"]
