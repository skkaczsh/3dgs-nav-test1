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


def test_camera_observations_are_normalized_before_multi_view_fusion() -> None:
    votes = defaultdict(Counter)
    observation_counts = Counter()

    first_weight, first_points = module.add_observation_votes(
        votes, observation_counts, 7, Counter({"floor": 1000}), min_observation_points=12
    )
    second_weight, second_points = module.add_observation_votes(
        votes, observation_counts, 7, Counter({"wall": 12}), min_observation_points=12
    )

    assert (first_weight, first_points) == (1.0, 1000)
    assert (second_weight, second_points) == (1.0, 12)
    assert votes[7] == Counter({"floor": 1.0, "wall": 1.0})
    assert observation_counts[7] == 2


def test_sparse_visual_evidence_is_retained_as_candidate_not_hard_label() -> None:
    row = {"object_id": 7, "geometry_type": "rough_mixed", **geometry_only_semantic_fields("rough_mixed")}
    updated, _report = module.apply_votes(
        [row],
        {
            "votes": {7: Counter({"person": 1.0})},
            "vetoes": defaultdict(Counter),
            "observation_counts": Counter({7: 1}),
        },
        args(min_votes=2),
    )

    assert updated[0]["semantic_label"] == "unknown"
    assert updated[0]["semantic_candidate_label"] == "person"
    assert updated[0]["semantic_observation_count"] == 1


def test_person_evidence_is_allowed_for_rough_geometry() -> None:
    assert module.label_allowed("person", "rough_mixed")
    assert not module.label_allowed("person", "horizontal")


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
