import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module(name: str):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_route_status_includes_offline_qa_report(tmp_path: Path):
    module = load_module("summarize_route_status")
    offline = tmp_path / "offline.json"
    offline.write_text(
        json.dumps({"passed": True, "git_head": "abc1234", "timestamp": "t", "checks": ["python_compile"]}),
        encoding="utf-8",
    )
    validation = tmp_path / "resume_validation.json"
    validation.write_text(
        json.dumps(
            {
                "passed": True,
                "errors": [],
                "warnings": [],
                "phase_ids": ["connectivity", "main_qwen_review"],
                "required_local_scripts": ["scripts/diagnose_server_connectivity.py"],
            }
        ),
        encoding="utf-8",
    )
    output_validation = tmp_path / "resume_outputs.json"
    output_validation.write_text(
        json.dumps({"passed": False, "blockers": ["qwen_review"], "next_gate": "finish_main_route_outputs_first"}),
        encoding="utf-8",
    )
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    concept = tmp_path / "concept.md"
    concept.write_text("ok", encoding="utf-8")
    args = type("Args", (), {
        "connectivity": empty,
        "stage_summary": empty,
        "delivery_manifest": empty,
        "delivery_zip": tmp_path / "delivery.zip",
        "conceptseg_report": concept,
        "old_route_summary": empty,
        "offline_qa_report": offline,
        "resume_command_validation": validation,
        "resume_output_validation": output_validation,
    })()

    status = module.build_status(args)
    markdown = module.render_markdown(status)

    assert status["offline_qa"]["passed"] is True
    assert status["offline_qa"]["git_head"] == "abc1234"
    assert status["server_resume_command_plan"]["passed"] is True
    assert status["server_resume_command_plan"]["phase_ids"] == ["connectivity", "main_qwen_review"]
    assert status["server_resume_outputs"]["passed"] is False
    assert status["server_resume_outputs"]["blockers"] == ["qwen_review"]
    assert "## Offline QA" in markdown
    assert "## Server Resume Plan" in markdown
    assert "## Server Resume Outputs" in markdown


def test_route_status_overlays_resume_output_success_on_stale_stage_summary(tmp_path: Path):
    module = load_module("summarize_route_status")
    stage = tmp_path / "stage.json"
    stage.write_text(
        json.dumps(
            {
                "stage_status": {"qwen_review_ready": False},
                "manual_merge_qa": {"passed": False, "accepted_merge_count": 0},
            }
        ),
        encoding="utf-8",
    )
    output_validation = tmp_path / "resume_outputs.json"
    output_validation.write_text(
        json.dumps(
            {
                "passed": True,
                "blockers": [],
                "checks": [
                    {"name": "qwen_review", "passed": True, "detail": {"error_count": 0}},
                    {
                        "name": "reviewed_merge_qa",
                        "passed": True,
                        "detail": {"accepted_merge_count": 4, "checks": {"point_count_preserved": True}},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    concept = tmp_path / "concept.md"
    concept.write_text("ok", encoding="utf-8")
    args = type("Args", (), {
        "connectivity": empty,
        "stage_summary": stage,
        "delivery_manifest": empty,
        "delivery_zip": tmp_path / "delivery.zip",
        "conceptseg_report": concept,
        "old_route_summary": empty,
        "offline_qa_report": None,
        "resume_command_validation": None,
        "resume_output_validation": output_validation,
    })()

    status = module.build_status(args)

    assert status["main_route"]["stage_status"]["qwen_review_ready"] is True
    assert status["main_route"]["manual_merge_qa"]["passed"] is True
    assert status["main_route"]["manual_merge_qa"]["accepted_merge_count"] == 4


def test_compact_snapshot_keeps_offline_qa_fields():
    module = load_module("append_route_status_snapshot")

    snapshot = module.compact(
        {
            "offline_qa": {"passed": True, "git_head": "abc1234"},
            "server_resume_command_plan": {"passed": True, "errors": []},
            "server_resume_outputs": {"passed": False, "blockers": ["qwen_review"]},
        },
        "now",
    )

    assert snapshot["offline_qa_passed"] is True
    assert snapshot["offline_qa_git_head"] == "abc1234"
    assert snapshot["resume_command_plan_passed"] is True
    assert snapshot["resume_command_plan_error_count"] == 0
    assert snapshot["resume_outputs_passed"] is False
    assert snapshot["resume_outputs_blocker_count"] == 1


def test_compact_snapshot_prefers_reviewed_merge_qa_count():
    module = load_module("append_route_status_snapshot")

    snapshot = module.compact(
        {
            "main_route": {
                "manual_workflow_pending": {"accepted_merge_count": 0},
                "manual_merge_qa": {"accepted_merge_count": 4},
            }
        },
        "now",
    )

    assert snapshot["accepted_merge_count"] == 4
