from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "build_rtx5070_parking_candidate_manifest_for_test",
    SCRIPTS / "build_rtx5070_parking_candidate_manifest.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def write_text(path: Path, text: str = "x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_args(tmp_path: Path):
    viewer_dir = tmp_path / "viewer"
    qa_dir = tmp_path / "qa"
    compare_dir = tmp_path / "compare"
    reports = tmp_path / "reports"
    viewer_ply = write_text(viewer_dir / "frame_object_points_stride10.ply")
    viewer_objects = write_text(viewer_dir / "frame_objects_viewer.jsonl")
    viewer_report = write_json(
        viewer_dir / "frame_object_viewer_export_report.json",
        {
            "input_vertices": 1000,
            "output_vertices": 100,
            "stride": 10,
            "missing_target_points": 0,
            "object_count_with_points": 10,
            "target_records": 20,
            "label_counts": {"ground": 50, "wall": 50},
        },
    )
    qa_report = write_json(
        qa_dir / "frame_local_object_qa_report.json",
        {
            "objects": 10,
            "status_counts": {"stable": 10},
            "semantic_label_counts": {"ground": 5, "wall": 5},
            "all_candidate_count": 2,
            "all_candidate_label_counts": {"wall": 1, "ground": 1},
            "all_risk_reason_counts": {"ground_has_large_height_span": 1},
        },
    )
    qa_contact = write_text(qa_dir / "frame_local_object_qa_contact.jpg")
    qa_candidates = write_text(qa_dir / "frame_local_object_qa_candidates.jsonl")
    qa_evidence = write_text(qa_dir / "frame_local_object_qa_evidence.jsonl")
    compare_report = write_json(
        compare_dir / "qa_compare.json",
        {
            "baseline": "strict_surface",
            "versions": {
                "strict_surface": {"all_candidate_count": 5},
                "ground_guard_object_relabel": {"all_candidate_count": 2},
            },
            "all_risk_deltas_from_baseline": {
                "ground_guard_object_relabel": {
                    "ground_has_large_height_span": -1,
                    "wall_too_flat_low_height": -2,
                    "wall_normal_too_up": -3,
                }
            },
        },
    )
    compare_md = write_text(compare_dir / "qa_compare.md")
    object_relabel = write_json(
        reports / "object_relabel_report.json",
        {"changed_count": 1, "changed_ratio": 0.1, "reason_counts": {"flat_wall_geometry_to_ground": 1}},
    )
    geometry_refine = write_json(
        reports / "geometry_refine_summary.json",
        {
            "input_targets": 9,
            "output_targets": 10,
            "missing_target_points": 0,
            "split_source_targets": 1,
            "relabelled_targets": 2,
            "refinement_reason_counts": {"linear_ground_artifact_to_other": 1},
        },
    )
    return argparse.Namespace(
        repo=tmp_path,
        output_json=tmp_path / "manifest.json",
        output_md=tmp_path / "manifest.md",
        candidate_name="ground_guard_object_relabel",
        source_targets=tmp_path / "source_targets",
        viewer_ply=viewer_ply,
        viewer_objects_jsonl=viewer_objects,
        viewer_report=viewer_report,
        qa_report=qa_report,
        qa_contact=qa_contact,
        qa_candidates=qa_candidates,
        qa_evidence=qa_evidence,
        compare_report=compare_report,
        compare_markdown=compare_md,
        object_relabel_report=object_relabel,
        geometry_refine_report=geometry_refine,
        viewer_url="http://viewer",
        remote_viewer_ply="/remote/viewer.ply",
        remote_viewer_objects_jsonl="/remote/objects.jsonl",
        remote_viewer_report="/remote/viewer_report.json",
        remote_qa_report="/remote/qa.json",
        remote_qa_contact="/remote/contact.jpg",
        remote_qa_candidates="/remote/candidates.jsonl",
        remote_qa_evidence="/remote/evidence.jsonl",
        remote_compare_report="/remote/compare.json",
        remote_compare_markdown="/remote/compare.md",
        remote_object_relabel_report="/remote/relabel.json",
        remote_geometry_refine_report="/remote/geometry.json",
    )


def test_build_manifest_passes_for_candidate(tmp_path: Path):
    manifest = module.build_manifest(make_args(tmp_path))

    assert manifest["passed"] is True
    assert manifest["metrics"]["comparison"]["candidate_all_candidate_count"] == 2
    assert manifest["metrics"]["comparison"]["candidate_deltas_from_baseline"]["wall_too_flat_low_height"] == -2
    assert all(row["passed"] for row in manifest["checks"])


def test_render_markdown_contains_viewer_and_commands(tmp_path: Path):
    manifest = module.build_manifest(make_args(tmp_path))
    markdown = module.render_markdown(manifest)

    assert "http://viewer" in markdown
    assert "remote_rebuild" in markdown
    assert "candidate_viewer_ply" in markdown
