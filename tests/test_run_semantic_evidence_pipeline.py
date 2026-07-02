from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
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


def write_stale_gate(path: Path, visual_path: Path) -> Path:
    visual_path.write_text(
        json.dumps(
            {
                "schema": "patch-experiment-visual-acceptance/v1",
                "status": "accepted",
                "selected_candidate": "v2_bucket_attach",
                "candidate_policy": "geometry_input_only",
                "review_index_url": "http://127.0.0.1:8765/docs/patch_experiment_review_index.html",
                "comparison_summary": {
                    "v2": {
                        "patch_count": 10,
                        "high_entropy_count": 2,
                        "large_high_entropy_count": 1,
                        "large_low_purity_count": 1,
                    }
                },
                "checks": [{"id": "reviewed", "required": True, "status": "accepted"}],
            }
        ),
        encoding="utf-8",
    )
    path.write_text(
        json.dumps(
            {
                "schema": "patch-experiment-promotion-gate/v1",
                "status": "pass",
                "candidate": "v2_bucket_attach",
                "visual_acceptance": str(visual_path),
                "metrics": {"accepted": True, "selected_run": "v2", "selected_metrics": {"patch_count": 999}},
                "reasons": [],
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
        allow_qa_preview_source=False,
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


def test_pipeline_blocks_stale_patch_gate_cache(tmp_path: Path) -> None:
    module = load_module()
    args = base_args(tmp_path)
    args.patch_gate = write_stale_gate(tmp_path / "gate.json", tmp_path / "visual.json")

    plan = module.build_plan(args, module.patch_gate_status(args.patch_gate))

    assert plan["status"] == "blocked"
    assert plan["patch_gate"]["stale"] is True


def test_pipeline_allows_explicit_unpromoted_experiment(tmp_path: Path) -> None:
    module = load_module()
    args = base_args(tmp_path)
    args.patch_gate = write_gate(tmp_path / "gate.json", status="fail")
    args.allow_unpromoted_patch_experiment = True

    plan = module.build_plan(args, module.patch_gate_status(args.patch_gate))

    assert plan["status"] == "ready"


def test_pipeline_passes_explicit_qa_preview_source_to_export(tmp_path: Path) -> None:
    module = load_module()
    args = base_args(tmp_path)
    args.allow_qa_preview_source = True

    plan = module.build_plan(args, module.patch_gate_status(args.patch_gate))

    export_argv = plan["commands"][2]["argv"]
    assert "--allow-qa-preview-source" in export_argv


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


def test_pipeline_run_mode_executes_minimal_smoke(tmp_path: Path) -> None:
    source_ply = tmp_path / "source.ply"
    objects = tmp_path / "objects.jsonl"
    gate = write_gate(tmp_path / "gate.json", status="fail")
    out_dir = tmp_path / "out"
    source_ply.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "property int object",
                "property uchar semantic",
                "end_header",
                "0 0 0 0 0 0 1 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    objects.write_text(
        json.dumps(
            {
                "object_id": 1,
                "geometry_type": "horizontal",
                "semantic_label": "unknown",
                "semantic_status": "geometry_only_unlabeled",
                "label_policy": "geometry_is_not_semantic",
                "bbox_3d": {"min": [0, 0, 0], "max": [1, 1, 0.1]},
                "voxel_count": 1,
                "semantic_votes": {"floor": 10},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-ply",
            str(source_ply),
            "--objects-jsonl",
            str(objects),
            "--output-dir",
            str(out_dir),
            "--patch-gate",
            str(gate),
            "--python",
            sys.executable,
            "--allow-unpromoted-patch-experiment",
            "--skip-mainline-healthcheck",
            "--run",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    fused = json.loads((out_dir / "semantic_evidence_objects.jsonl").read_text(encoding="utf-8").splitlines()[0])
    validation = json.loads((out_dir / "semantic_evidence_fusion_validation.json").read_text(encoding="utf-8"))
    viewer_report = json.loads((out_dir / "semantic_evidence_viewer_report.json").read_text(encoding="utf-8"))
    assert fused["semantic_label"] == "floor"
    assert validation["passed"] is True
    assert viewer_report["label_counts"] == {"floor": 1}
    assert (out_dir / "semantic_evidence_viewer.ply").exists()
