from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from scripts import accumulate_semantic_png_votes_to_objects as module
from scripts.geometry_input_contract import geometry_only_semantic_fields


def args(**overrides):
    base = {
        "min_votes": 3,
        "min_vote_ratio": 0.60,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_geometry_only_original_label_stays_unknown_without_votes() -> None:
    row = {
        "object_id": 1,
        "geometry_type": "horizontal",
        **geometry_only_semantic_fields("horizontal"),
    }

    assert module.normalized_original_label(row) == "unknown"
    updated, report = module.apply_votes(
        [row],
        {"votes": defaultdict(Counter), "vetoes": defaultdict(Counter)},
        args(),
    )

    assert updated[0]["semantic_label"] == "unknown"
    assert updated[0]["semantic_vote_status"] == "insufficient_sam_votes"
    assert report["label_counts"] == {"unknown": 1}


def test_geometry_only_can_accept_strong_png_votes() -> None:
    row = {
        "object_id": 1,
        "geometry_type": "horizontal",
        **geometry_only_semantic_fields("horizontal"),
    }

    updated, report = module.apply_votes(
        [row],
        {"votes": {1: Counter({"floor": 9, "wall": 1})}, "vetoes": defaultdict(Counter)},
        args(),
    )

    assert updated[0]["semantic_label_original"] == "unknown"
    assert updated[0]["semantic_label"] == "floor"
    assert updated[0]["semantic_vote_status"] == "sam_vote_applied"
    assert report["changed_object_count"] == 1


def test_legacy_geometry_label_fallback_is_preserved_for_old_artifacts() -> None:
    row = {"object_id": 1, "geometry_type": "vertical", "semantic_label": "vertical"}

    assert module.normalized_original_label(row) == "wall"


def test_png_vote_rejects_forbidden_source_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="forbidden input path"):
        module.reject_forbidden_production_input(tmp_path / "objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor/source.ply")


def test_png_vote_main_allows_explicit_qa_preview_source(tmp_path: Path, monkeypatch) -> None:
    source_ply = tmp_path / "source_stride10.ply"
    objects = tmp_path / "objects.jsonl"
    semantic_eval = tmp_path / "semantic_eval"
    output_dir = tmp_path / "out"
    source_ply.write_text("ply\nend_header\n", encoding="utf-8")
    objects.write_text('{"object_id": 1, "geometry_type": "horizontal"}\n', encoding="utf-8")
    semantic_eval.mkdir()
    monkeypatch.setattr(
        module,
        "read_ply",
        lambda path: (
            ["ply\n", "end_header\n"],
            ["x", "y", "z", "red", "green", "blue", "object", "semantic"],
            np.array([[0, 0, 0, 0, 0, 0, 1, 0]], dtype=float),
        ),
    )
    monkeypatch.setattr(module, "accumulate_votes", lambda *a, **k: {"frame_ids": [], "votes": defaultdict(Counter), "vetoes": defaultdict(Counter), "frame_reports": []})
    monkeypatch.setattr(module, "write_semantic_ply", lambda *a, **k: {"rows": 1})
    monkeypatch.setattr(
        "sys.argv",
        [
            "accumulate_semantic_png_votes_to_objects.py",
            "--source-ply",
            str(source_ply),
            "--objects-jsonl",
            str(objects),
            "--semantic-eval-dir",
            str(semantic_eval),
            "--output-dir",
            str(output_dir),
            "--allow-qa-preview-source",
        ],
    )

    assert module.main() == 0
    assert (output_dir / "objects_sam_semantic_report.json").exists()
