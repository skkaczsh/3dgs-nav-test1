from __future__ import annotations

from pathlib import Path

import pytest

from scripts import apply_visual_promotion_geometry_guard as module


def test_visual_promotion_guard_rejects_stride_source_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "apply_visual_promotion_geometry_guard.py",
            "--input-ply",
            str(tmp_path / "frame_object_points_stride10.ply"),
            "--input-objects-jsonl",
            str(tmp_path / "objects.jsonl"),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    with pytest.raises(ValueError, match="forbidden input path"):
        module.main()


def test_visual_promotion_guard_allows_explicit_qa_preview_source(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "out"
    monkeypatch.setattr(module, "read_jsonl", lambda path: [{"object_id": 1, "semantic_label": "wall"}])
    monkeypatch.setattr(module, "rewrite_ply", lambda source, output, objects: {"vertex_count": 1, "changed_points": 0})
    monkeypatch.setattr(
        "sys.argv",
        [
            "apply_visual_promotion_geometry_guard.py",
            "--input-ply",
            str(tmp_path / "frame_object_points_stride10.ply"),
            "--input-objects-jsonl",
            str(tmp_path / "objects.jsonl"),
            "--output-dir",
            str(output_dir),
            "--allow-qa-preview-source",
        ],
    )

    module.main()

    assert (output_dir / "full_scene_objects_visual_geometry_guard_report.json").exists()
