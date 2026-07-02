from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from scripts import transfer_teacher_semantics_to_objects as module
from scripts.current_mainline_contract import reject_forbidden_production_input
from scripts.geometry_input_contract import geometry_only_semantic_fields


def args(**overrides):
    base = {
        "min_teacher_votes": 3,
        "min_winner_ratio": 0.55,
        "min_global_winner_ratio": 0.35,
        "min_allowed_ratio": 0.35,
        "allow_surface_teacher_on_unknown": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_geometry_only_horizontal_original_label_stays_unknown() -> None:
    row = {
        "object_id": 1,
        "geometry_type": "horizontal",
        **geometry_only_semantic_fields("horizontal"),
    }

    assert module.normalized_original_label(row) == "unknown"
    label, status, confidence, allowed, vetoed = module.choose_label(row, Counter(), args())

    assert label == "unknown"
    assert status == "kept_original_insufficient_teacher_votes"
    assert confidence == 0.0
    assert allowed == Counter()
    assert vetoed == Counter()


def test_geometry_only_horizontal_can_accept_strong_teacher_votes() -> None:
    row = {
        "object_id": 1,
        "geometry_type": "horizontal",
        **geometry_only_semantic_fields("horizontal"),
    }

    label, status, confidence, allowed, vetoed = module.choose_label(
        row,
        Counter({"floor": 8, "wall": 1}),
        args(),
    )

    assert label == "floor"
    assert status == "teacher_semantic_transfer"
    assert confidence == 8 / 9
    assert allowed == Counter({"floor": 8})
    assert vetoed == Counter({"wall": 1})


def test_legacy_geometry_label_fallback_is_preserved_for_old_artifacts() -> None:
    row = {"object_id": 1, "geometry_type": "vertical", "semantic_label": "vertical"}

    assert module.normalized_original_label(row) == "wall"


def test_teacher_transfer_rejects_forbidden_source_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="forbidden input path"):
        reject_forbidden_production_input(tmp_path / "frame_object_points_stride10.ply")


def test_teacher_transfer_allows_explicit_qa_preview_source_and_teacher(tmp_path: Path, monkeypatch) -> None:
    source_ply = tmp_path / "source_stride10.ply"
    teacher_ply = tmp_path / "teacher_stride10.ply"
    objects = tmp_path / "objects.jsonl"
    output_dir = tmp_path / "out"
    args_ns = argparse.Namespace(
        source_ply=source_ply,
        source_objects_jsonl=objects,
        teacher_ply=teacher_ply,
        output_dir=output_dir,
        output_prefix="teacher",
        allow_qa_preview_source=True,
        allow_qa_preview_teacher=True,
        allow_surface_teacher_on_unknown=False,
        min_teacher_votes=3,
        min_winner_ratio=0.55,
        min_global_winner_ratio=0.35,
        min_allowed_ratio=0.35,
        max_distance=0.12,
        distance_bin=0.03,
        workers=1,
    )
    monkeypatch.setattr(
        module,
        "read_ply",
        lambda path: (
            ["ply\n", "end_header\n"],
            ["x", "y", "z", "red", "green", "blue", "object", "semantic"],
            np.array([[0, 0, 0, 0, 0, 0, 1, 0]], dtype=float),
        ),
    )
    monkeypatch.setattr(module, "read_jsonl", lambda path: [{"object_id": 1, "geometry_type": "horizontal"}])
    monkeypatch.setattr(module, "aggregate_teacher_votes", lambda *a, **k: ({}, {"matched_points": 0}))
    monkeypatch.setattr(module, "rewrite_ply", lambda *a, **k: {"rows": 1})

    report = module.run(args_ns)

    assert report["object_count"] == 1
    assert report["rows"] == 1
