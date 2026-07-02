from __future__ import annotations

from pathlib import Path

import pytest

from scripts import apply_geometry_conflict_relabels as module


def test_geometry_conflict_relabel_main_rejects_stride_source(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "apply_geometry_conflict_relabels.py",
            "--input-ply",
            str(tmp_path / "frame_object_points_stride10.ply"),
            "--input-objects-jsonl",
            str(tmp_path / "objects.jsonl"),
            "--conflicts-jsonl",
            str(tmp_path / "conflicts.jsonl"),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    with pytest.raises(ValueError, match="forbidden input path"):
        module.main()


def test_geometry_conflict_relabel_main_allows_explicit_qa_preview_source(tmp_path: Path, monkeypatch) -> None:
    input_ply = tmp_path / "frame_object_points_stride10.ply"
    objects_jsonl = tmp_path / "objects.jsonl"
    conflicts_jsonl = tmp_path / "conflicts.jsonl"
    output_dir = tmp_path / "out"
    monkeypatch.setattr(module, "load_relabels", lambda path, args: {})
    monkeypatch.setattr(module, "rewrite_objects", lambda objects, output, relabels: [{"object_id": 1, "semantic_label": "wall"}])
    monkeypatch.setattr(module, "rewrite_ply", lambda source, output, relabels: {"vertex_count": 1, "changed_points": 0})
    monkeypatch.setattr(
        "sys.argv",
        [
            "apply_geometry_conflict_relabels.py",
            "--input-ply",
            str(input_ply),
            "--input-objects-jsonl",
            str(objects_jsonl),
            "--conflicts-jsonl",
            str(conflicts_jsonl),
            "--output-dir",
            str(output_dir),
            "--allow-qa-preview-source",
        ],
    )

    module.main()

    assert (output_dir / "geometry_relabel_report.json").exists()
