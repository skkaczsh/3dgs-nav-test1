import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "validate_server_resume_commands_for_test",
        SCRIPTS / "validate_server_resume_commands.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for name in [
        "diagnose_server_connectivity.py",
        "resume_server_qwen_review.sh",
        "run_remote_server_semantic_completion_sharded.sh",
        "run_server_dataset_readiness.sh",
        "run_remote_server_target_object_fusion.sh",
        "validate_server_resume_outputs.py",
    ]:
        (scripts / name).write_text("# ok\n", encoding="utf-8")
    return repo


def valid_plan() -> dict:
    return {
        "readiness": {"ready_for_server_probe": True, "blockers": []},
        "phases": [
            {
                "id": "connectivity",
                "commands": [
                    {
                        "name": "diagnose_connectivity",
                        "command": "python3 scripts/diagnose_server_connectivity.py --output out.json",
                        "required": True,
                    }
                ],
            },
            {
                "id": "main_qwen_review",
                "commands": [
                    {
                        "name": "qwen_review",
                        "command": "BIND_ADDRESS=192.168.0.3 SERVER=scan-train CONCURRENCY=4 bash scripts/resume_server_qwen_review.sh",
                        "required": True,
                    }
                ],
            },
            {
                "id": "main_semantic_refresh",
                "commands": [
                    {
                        "name": "semantic_completion_sharded",
                        "command": "BIND_ADDRESS=192.168.100.125 SERVER=scan-train PATCH_SCENE_PROMPTS=1 SHARDS=4 bash scripts/run_remote_server_semantic_completion_sharded.sh",
                        "required": True,
                    }
                ],
            },
            {
                "id": "main_object_fusion",
                "commands": [
                    {
                        "name": "dataset_readiness",
                        "command": "BIND_ADDRESS=192.168.0.3 SERVER=scan-train bash scripts/run_server_dataset_readiness.sh",
                        "required": True,
                    },
                    {
                        "name": "target_object_fusion",
                        "command": "BIND_ADDRESS=192.168.100.125 SERVER=scan-train MIN_MERGE_CONFIDENCE=0.5 bash scripts/run_remote_server_target_object_fusion.sh",
                        "required": True,
                    }
                ],
            },
            {
                "id": "main_output_validation",
                "commands": [
                    {
                        "name": "strict_output_validation",
                        "command": "python3 scripts/validate_server_resume_outputs.py --strict",
                        "required": True,
                    }
                ],
            },
            {
                "id": "new_model_side_track",
                "commands": [
                    {
                        "name": "conceptseg_status",
                        "command": "ssh scan-train 'nvidia-smi'",
                        "required": False,
                    }
                ],
            },
            {
                "id": "old_route_side_track",
                "commands": [
                    {
                        "name": "old_route_smoke_status",
                        "command": "ls -lah /Users/skkac/Work/SCAN/server_old_route_smoke",
                        "required": False,
                    }
                ],
            },
        ],
    }


def test_validate_resume_commands_accepts_valid_plan(tmp_path: Path):
    module = load_module()
    repo = make_repo(tmp_path)
    shell = tmp_path / "plan.sh"
    shell.write_text(
        "\n".join(
            [
                "python3 scripts/diagnose_server_connectivity.py",
                "bash scripts/resume_server_qwen_review.sh",
                "bash scripts/run_remote_server_semantic_completion_sharded.sh",
                "bash scripts/run_server_dataset_readiness.sh",
                "bash scripts/run_remote_server_target_object_fusion.sh",
                "python3 scripts/validate_server_resume_outputs.py --strict",
                "[optional] conceptseg_status",
                "[optional] old_route_smoke_status",
            ]
        ),
        encoding="utf-8",
    )

    report = module.validate_plan(valid_plan(), repo, shell)

    assert report["passed"] is True
    assert report["errors"] == []


def test_validate_resume_commands_rejects_required_side_track(tmp_path: Path):
    module = load_module()
    repo = make_repo(tmp_path)
    plan = valid_plan()
    plan["phases"][5]["commands"][0]["required"] = True

    report = module.validate_plan(plan, repo)

    assert report["passed"] is False
    assert "side_track_command_required=new_model_side_track:conceptseg_status" in report["errors"]


def test_validate_resume_commands_rejects_non_strict_output_validation(tmp_path: Path):
    module = load_module()
    repo = make_repo(tmp_path)
    plan = valid_plan()
    plan["phases"][4]["commands"][0]["command"] = "python3 scripts/validate_server_resume_outputs.py"

    report = module.validate_plan(plan, repo)

    assert report["passed"] is False
    assert "strict_output_validation_not_strict" in report["errors"]


def test_validate_resume_commands_rejects_missing_local_script(tmp_path: Path):
    module = load_module()
    repo = make_repo(tmp_path)
    (repo / "scripts" / "resume_server_qwen_review.sh").unlink()

    report = module.validate_plan(valid_plan(), repo)

    assert report["passed"] is False
    assert "missing_local_script=scripts/resume_server_qwen_review.sh" in report["errors"]
