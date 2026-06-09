import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "prepare_server_resume_commands_for_test",
        SCRIPTS / "prepare_server_resume_commands.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def args_for(tmp_path: Path, readiness: Path):
    return type("Args", (), {
        "readiness": readiness,
        "output_json": tmp_path / "commands.json",
        "output_shell": tmp_path / "commands.sh",
        "server": "scan-train",
        "bind_address": "192.168.0.3",
        "qwen_concurrency": 4,
        "semantic_shards": 4,
        "min_merge_confidence": 0.5,
    })()


def test_resume_command_plan_orders_main_route_before_side_tracks(tmp_path: Path):
    module = load_module()
    readiness = tmp_path / "readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "ready_for_server_probe": True,
                "blockers": [],
                "offline_qa": {"git_head": "abc1234"},
            }
        ),
        encoding="utf-8",
    )

    plan = module.build_plan(args_for(tmp_path, readiness))
    phase_ids = [phase["id"] for phase in plan["phases"]]
    shell = module.render_shell(plan)

    assert phase_ids[:4] == [
        "connectivity",
        "main_qwen_review",
        "main_semantic_refresh",
        "main_object_fusion",
    ]
    assert phase_ids[-2:] == ["new_model_side_track", "old_route_side_track"]
    assert "CONCURRENCY=4" in shell
    assert "PATCH_SCENE_PROMPTS=1 SHARDS=4" in shell
    assert "MIN_MERGE_CONFIDENCE=0.5" in shell
    assert "ConceptSeg-R1" in shell
    assert "[optional]" in shell


def test_resume_command_plan_blocks_shell_when_not_ready(tmp_path: Path):
    module = load_module()
    readiness = tmp_path / "readiness.json"
    readiness.write_text(
        json.dumps({"ready_for_server_probe": False, "blockers": ["offline_qa_not_passing"]}),
        encoding="utf-8",
    )

    plan = module.build_plan(args_for(tmp_path, readiness))
    shell = module.render_shell(plan)

    assert "offline_qa_not_passing" in plan["readiness"]["blockers"]
    assert "readiness_not_ready_for_server_probe" in plan["readiness"]["blockers"]
    assert "exit 1" in shell
    assert "resume_server_qwen_review.sh" not in shell
