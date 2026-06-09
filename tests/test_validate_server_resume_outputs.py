import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "validate_server_resume_outputs_for_test",
        SCRIPTS / "validate_server_resume_outputs.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def args_for(tmp_path: Path):
    return type("Args", (), {
        "qwen_report": write_json(tmp_path / "qwen.json", {"error_count": 0, "result_count": 8}),
        "reviewed_merge_qa": write_json(tmp_path / "merge_qa.json", {"passed": True, "accepted_merge_count": 2, "checks": {"ok": True}}),
        "dataset_readiness": write_json(
            tmp_path / "dataset.json",
            {"ratios": {"completion_semantic_images": 0.95, "color_ply": 0.98}},
        ),
        "target_object_qa": write_json(
            tmp_path / "target_object.json",
            {"frames": {"ok_count": 98, "count": 100}, "objects": {"ambiguous_ratio": 0.2, "count": 120}},
        ),
        "command_plan_validation": write_json(tmp_path / "command_plan.json", {"passed": True, "errors": [], "warnings": []}),
        "conceptseg_report": write_json(tmp_path / "conceptseg.md", {"note": "side"}),
        "old_route_summary": write_json(tmp_path / "old_route.json", {"note": "side"}),
        "min_semantic_ratio": 0.90,
        "min_color_ratio": 0.95,
        "min_target_frame_ok_ratio": 0.95,
        "max_ambiguous_ratio": 0.35,
    })()


def test_server_resume_outputs_pass_when_all_main_artifacts_are_ready(tmp_path: Path):
    module = load_module()

    report = module.validate(args_for(tmp_path))

    assert report["passed"] is True
    assert report["blockers"] == []
    assert report["next_gate"] == "dataset_ready_for_model_and_old_route_side_tracks"


def test_server_resume_outputs_blocks_when_qwen_report_missing(tmp_path: Path):
    module = load_module()
    args = args_for(tmp_path)
    args.qwen_report.unlink()

    report = module.validate(args)

    assert report["passed"] is False
    assert "qwen_review" in report["blockers"]


def test_server_resume_outputs_blocks_low_dataset_or_target_quality(tmp_path: Path):
    module = load_module()
    args = args_for(tmp_path)
    write_json(args.dataset_readiness, {"ratios": {"completion_semantic_images": 0.5, "color_ply": 0.98}})
    write_json(args.target_object_qa, {"frames": {"ok_count": 50, "count": 100}, "objects": {"ambiguous_ratio": 0.8}})

    report = module.validate(args)

    assert report["passed"] is False
    assert "semantic_dataset" in report["blockers"]
    assert "target_object_fusion" in report["blockers"]
