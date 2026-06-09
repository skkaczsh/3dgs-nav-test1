import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "prepare_server_resume_report_for_test",
        SCRIPTS / "prepare_server_resume_report.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resume_report_marks_ready_when_local_artifacts_exist(tmp_path: Path):
    module = load_module()
    route = tmp_path / "route.json"
    latest = tmp_path / "latest.json"
    offline = tmp_path / "offline.json"
    delivery = tmp_path / "delivery.zip"
    manual = tmp_path / "manual.csv"
    review = tmp_path / "review.jsonl"
    objects = tmp_path / "objects.jsonl"
    route.write_text("{}", encoding="utf-8")
    latest.write_text(json.dumps({"review_pack_ready": True, "delivery_missing_count": 0}), encoding="utf-8")
    offline.write_text(json.dumps({"passed": True, "git_head": "abc1234", "checks": ["python_compile"]}), encoding="utf-8")
    for path in [delivery, manual, review, objects]:
        path.write_text("ok", encoding="utf-8")
    args = type("Args", (), {
        "route_status": route,
        "latest_snapshot": latest,
        "offline_qa": offline,
        "delivery_zip": delivery,
        "manual_csv": manual,
        "review_jsonl": review,
        "long_objects": objects,
    })()

    report = module.build_report(args)

    assert report["ready_for_server_probe"] is True
    assert report["blockers"] == []
    assert report["offline_qa"]["git_head"] == "abc1234"
    assert any("resume_server_qwen_review.sh" in command for command in report["resume_commands"])


def test_resume_report_blocks_when_offline_qa_failed(tmp_path: Path):
    module = load_module()
    existing = tmp_path / "exists"
    existing.write_text("{}", encoding="utf-8")
    artifact = tmp_path / "artifact"
    artifact.write_text("ok", encoding="utf-8")
    offline = tmp_path / "offline.json"
    offline.write_text(json.dumps({"passed": False}), encoding="utf-8")
    args = type("Args", (), {
        "route_status": existing,
        "latest_snapshot": existing,
        "offline_qa": offline,
        "delivery_zip": artifact,
        "manual_csv": artifact,
        "review_jsonl": artifact,
        "long_objects": artifact,
    })()

    report = module.build_report(args)

    assert report["ready_for_server_probe"] is False
    assert "offline_qa_not_passing" in report["blockers"]
