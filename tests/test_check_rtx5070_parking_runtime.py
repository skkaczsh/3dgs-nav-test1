from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "check_rtx5070_parking_runtime_for_test",
    SCRIPTS / "check_rtx5070_parking_runtime.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def make_args(**overrides):
    values = {
        "min_free_vram_mib": 8000,
        "max_git_dirty_count": 1,
        "require_tmux": True,
        "require_proxy": False,
        "tmux_session": "scan_migrate",
        "required_remote_file": [],
        "no_default_required_files": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_parse_gpu_csv():
    gpus = module.parse_gpu_csv("0, NVIDIA GeForce RTX 5070 Ti, 1288, 16303, 2\n")

    assert gpus == [
        {
            "index": 0,
            "name": "NVIDIA GeForce RTX 5070 Ti",
            "memory_used_mib": 1288,
            "memory_total_mib": 16303,
            "memory_free_mib": 15015,
            "memory_used_ratio": 1288 / 16303,
            "utilization_gpu_percent": 2,
        }
    ]


def test_parse_remote_sections():
    sections = module.parse_remote_sections(
        "\n".join(
            [
                "section=system",
                "hostname=scan-rtx5070",
                "section=gpu",
                "0, NVIDIA GeForce RTX 5070 Ti, 1288, 16303, 2",
                "section=workspace",
                "repo_exists=1",
                "git_dirty_count=1",
                "work_exists=1",
                "section=runtime",
                "venv_python_exists=1",
                "module_torch=1",
                "torch_cuda_available=1",
                "section=tmux",
                "tmux_session_exists=1",
                "section=artifacts",
                "artifact=/tmp/a.ply|123|ok",
            ]
        )
    )

    assert sections["system"]["hostname"] == "scan-rtx5070"
    assert sections["gpu"][0]["memory_free_mib"] == 15015
    assert sections["workspace"]["repo_exists"] == "1"
    assert sections["artifacts"][0]["bytes"] == 123


def test_evaluate_passes_for_healthy_sections():
    sections = module.parse_remote_sections(
        "\n".join(
            [
                "section=gpu",
                "0, NVIDIA GeForce RTX 5070 Ti, 1288, 16303, 2",
                "section=workspace",
                "repo_exists=1",
                "git_dirty_count=1",
                "work_exists=1",
                "section=runtime",
                "venv_python_exists=1",
                "module_torch=1",
                "module_cv2=1",
                "module_scipy=1",
                "module_sklearn=1",
                "module_transformers=1",
                "torch_cuda_available=1",
                "section=tmux",
                "tmux_session_exists=1",
                "section=artifacts",
                "artifact=/tmp/a.ply|123|ok",
            ]
        )
    )

    errors, warnings = module.evaluate(sections, make_args())

    assert errors == []
    assert warnings == []


def test_evaluate_fails_for_low_vram_and_missing_artifact():
    sections = module.parse_remote_sections(
        "\n".join(
            [
                "section=gpu",
                "0, NVIDIA GeForce RTX 5070 Ti, 12000, 16303, 2",
                "section=workspace",
                "repo_exists=1",
                "git_dirty_count=0",
                "work_exists=1",
                "section=runtime",
                "venv_python_exists=1",
                "module_torch=1",
                "module_cv2=1",
                "module_scipy=1",
                "module_sklearn=1",
                "module_transformers=1",
                "torch_cuda_available=1",
                "section=tmux",
                "tmux_session_exists=1",
                "section=artifacts",
                "artifact=/tmp/a.ply|0|missing",
            ]
        )
    )

    errors, _ = module.evaluate(sections, make_args())

    assert any("GPU free VRAM below threshold" in error for error in errors)
    assert any("missing required remote artifacts" in error for error in errors)


def test_build_status_script_supports_absolute_required_files():
    args = make_args(
        remote_repo="/repo",
        remote_work="/work",
        venv="/venv",
        required_remote_file=["relative/a.txt", "/abs/b.txt"],
    )

    script = module.build_status_script(args)

    assert 'case "$rel" in' in script
    assert '*) path="$WORK/$rel" ;;' in script
    assert "relative/a.txt" in script
    assert "/abs/b.txt" in script
