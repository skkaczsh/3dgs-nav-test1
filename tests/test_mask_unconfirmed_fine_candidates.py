from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts import mask_unconfirmed_fine_candidates as module


def test_mask_unconfirmed_rejects_stride_preview_source_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mask_unconfirmed_fine_candidates.py",
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


def test_mask_unconfirmed_allows_explicit_qa_preview_source(tmp_path: Path, monkeypatch) -> None:
    objects = [{"object_id": 1, "semantic_label": "car", "downstream_stage": "dino_fine_object_review"}]
    monkeypatch.setattr(module, "read_jsonl", lambda path: objects)
    monkeypatch.setattr(module, "write_jsonl", lambda *a, **k: None)
    monkeypatch.setattr(module, "rewrite_ply", lambda *a, **k: {"vertex_count": 0, "changed_points": 0})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mask_unconfirmed_fine_candidates.py",
            "--input-ply",
            str(tmp_path / "frame_object_points_stride10.ply"),
            "--input-objects-jsonl",
            str(tmp_path / "objects.jsonl"),
            "--output-dir",
            str(tmp_path / "out"),
            "--allow-qa-preview-source",
        ],
    )

    module.main()
