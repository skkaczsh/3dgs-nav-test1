import argparse
import importlib.util
import json
import os
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "stage_accepted_sync_anchors.py"
    spec = importlib.util.spec_from_file_location("stage_accepted_sync_anchors", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def accepted(frame, cam, video=None):
    return {
        "frame_id": frame,
        "cam_id": cam,
        "anchor_status": "accepted",
        "selected_video_idx": frame if video is None else video,
    }


def make_args(tmp_path: Path, source: Path, **kwargs):
    img_pos = tmp_path / "img_pos.txt"
    if not img_pos.exists():
        img_pos.write_text("10 1.0\n20 2.0\n", encoding="utf-8")
    base = dict(
        source=source,
        downloads_dir=tmp_path / "Downloads",
        target=tmp_path / "target" / "accepted_sync_anchors.jsonl",
        repo_root=tmp_path,
        review_name="review",
        cams=[0, 1],
        min_accepted_per_cam=1,
        img_pos_file=img_pos,
        timestamp_phase_fraction=0.0,
        expected_fps=6.0,
        max_fps_error=20.0,
        force=False,
        dry_run=False,
        run_solver=False,
        output=None,
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_stage_copies_valid_anchors(tmp_path: Path):
    module = load_module()
    source = tmp_path / "Downloads" / "accepted_sync_anchors.jsonl"
    write_jsonl(source, [accepted(10, 0, 12), accepted(10, 1, 13)])

    result = module.stage(make_args(tmp_path, source))

    assert result["passed"] is True
    assert result["staged"] is True
    assert result["run_solver"] is False
    assert result["solver_exit_code"] is None
    assert (tmp_path / "target" / "accepted_sync_anchors.jsonl").read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert "scripts/run_rtx5070_sync_anchor_solver.sh" in result["next_command"]


def test_stage_rejects_insufficient_camera_coverage(tmp_path: Path):
    module = load_module()
    source = tmp_path / "accepted_sync_anchors.jsonl"
    write_jsonl(source, [accepted(10, 0, 12)])

    result = module.stage(make_args(tmp_path, source))

    assert result["passed"] is False
    assert result["staged"] is False
    assert "accepted_anchors_cam1=0<min1" in result["errors"]


def test_stage_refuses_to_overwrite_without_force(tmp_path: Path):
    module = load_module()
    source = tmp_path / "accepted_sync_anchors.jsonl"
    target = tmp_path / "target" / "accepted_sync_anchors.jsonl"
    write_jsonl(source, [accepted(10, 0, 12), accepted(10, 1, 13)])
    write_jsonl(target, [accepted(99, 0, 99)])

    result = module.stage(make_args(tmp_path, source, target=target))

    assert result["passed"] is False
    assert result["staged"] is False
    assert result["errors"][0].startswith("target_exists=")
    assert "99" in target.read_text(encoding="utf-8")


def test_error_message_reports_missing_source_with_clear_error(tmp_path: Path):
    module = load_module()
    missing = tmp_path / "missing.jsonl"

    assert module.error_message(FileNotFoundError(str(missing))) == f"source_missing={missing}"


def test_resolve_source_discovers_latest_downloaded_export(tmp_path: Path):
    module = load_module()
    downloads = tmp_path / "Downloads"
    old = downloads / "accepted_sync_anchors.jsonl"
    new = downloads / "accepted_sync_anchors (1).jsonl"
    write_jsonl(old, [accepted(10, 0, 12)])
    write_jsonl(new, [accepted(10, 1, 13)])
    os.utime(old, (100.0, 100.0))
    os.utime(new, (200.0, 200.0))

    assert module.resolve_source(None, downloads) == new


def test_explicit_source_overrides_download_discovery(tmp_path: Path):
    module = load_module()
    downloads = tmp_path / "Downloads"
    discovered = downloads / "accepted_sync_anchors (1).jsonl"
    explicit = tmp_path / "explicit.jsonl"
    write_jsonl(discovered, [accepted(10, 0, 12)])
    write_jsonl(explicit, [accepted(10, 1, 13)])

    assert module.resolve_source(explicit, downloads) == explicit


def test_stage_uses_discovered_source_when_source_omitted(tmp_path: Path):
    module = load_module()
    source = tmp_path / "Downloads" / "accepted_sync_anchors (1).jsonl"
    write_jsonl(source, [accepted(10, 0, 12), accepted(10, 1, 13)])

    result = module.stage(make_args(tmp_path, None))

    assert result["passed"] is True
    assert result["source"] == str(source.resolve())


def test_stage_can_run_solver_after_staging(tmp_path: Path, monkeypatch):
    module = load_module()
    source = tmp_path / "Downloads" / "accepted_sync_anchors.jsonl"
    write_jsonl(source, [accepted(10, 0, 12), accepted(10, 1, 13)])
    calls = []

    def fake_run_solver(repo_root, env):
        calls.append((repo_root, env))
        return 0

    monkeypatch.setattr(module, "run_solver", fake_run_solver)
    result = module.stage(make_args(tmp_path, source, run_solver=True))

    assert result["passed"] is True
    assert result["solver_exit_code"] == 0
    assert calls[0][1]["LOCAL_ANCHORS"].endswith("accepted_sync_anchors.jsonl")
    assert calls[0][1]["REVIEW_NAME"] == "review"
