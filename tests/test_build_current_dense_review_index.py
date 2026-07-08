from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import build_current_dense_review_index as module


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_current_dense_review_index.py"


def qa_fixture() -> dict:
    return {
        "schema": "current-dense-mainline-qa/v1",
        "object_refinement": {
            "metrics": {
                "v7": {
                    "candidate_count": 1,
                    "accepted_candidate_rows": 1,
                    "output_object_count": 9,
                    "mixed_object_voxel_ratio_020": 0.2,
                    "object_count_in_overlap_preview": 7,
                },
                "v8": {
                    "candidate_count": 3,
                    "accepted_candidate_rows": 2,
                    "output_object_count": 8,
                    "mixed_object_voxel_ratio_020": 0.19,
                    "object_count_in_overlap_preview": 6,
                },
                "delta_v8_minus_v7": {
                    "candidate_count": 2,
                    "accepted_candidate_rows": 1,
                    "output_object_count": -1,
                    "mixed_object_voxel_ratio_020": -0.01,
                    "object_count_in_overlap_preview": -1,
                },
            }
        },
        "surface_guard": {
            "label_point_counts": {
                "v9": {"floor": 10, "wall": 20},
                "v17": {"floor": 10, "wall": 20},
                "delta_v17_minus_v9": {"floor": 0, "wall": 0},
            }
        },
    }


def visual_fixture() -> dict:
    return {
        "schema": "current-dense-visual-acceptance/v1",
        "status": "pending",
        "accepted_candidate": "v8_object_refinement",
        "checks": [
            {
                "id": "v8_fragmentation_improves",
                "required": True,
                "status": "pending",
                "question": "v8 visibly reduces object fragmentation compared with v7.",
            },
            {
                "id": "semantic_not_promoted_from_object_view",
                "required": True,
                "status": "pending",
                "question": "Object refinement is only promoted as geometry ownership.",
            },
        ],
    }


def test_build_html_links_only_current_review_artifacts() -> None:
    html = module.build_html(qa_fixture(), visual_fixture())

    assert "Energy Attach v4" in html
    assert "SPG v4 near-bbox" in html
    assert "SPG Sonata touch-edge 0.15" in html
    assert "SPG Sonata touch-edge 0.30" in html
    assert "Promotion Review Checklist" in html
    assert "v8_fragmentation_improves" in html
    assert "update_current_dense_visual_acceptance.py" in html
    assert "plan_current_dense_promotion.py" in html
    assert "--run-gate" in html
    assert "objects_v12" not in html
    assert "objects_v15" not in html
    assert "semantic_ply_viewer.html" in html


def test_artifact_allowlist_accepts_current_review_set() -> None:
    result = module.validate_artifact_allowlist()

    assert result["passed"] is True
    assert result["errors"] == []
    assert result["artifact_ids"] == [
        "energy_attach_v4_contact_evidence",
        "superpoint_graph_v4_nearbbox_s070_e120_20260708_183437",
        "superpoint_graph_sonata_touch_edge_sample_v1_20260708",
        "superpoint_graph_sonata_touch_edge_w030_20260708",
    ]


def test_artifact_allowlist_rejects_forbidden_diagnostic_path() -> None:
    bad = [dict(module.ARTIFACTS[0])]
    bad[0]["ply"] = "/server_parking_priority_s10/objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor/bad.ply"

    result = module.validate_artifact_allowlist(bad, check_files=False)

    assert result["passed"] is False
    assert any("forbidden_artifact_reference" in error for error in result["errors"])


def test_artifact_allowlist_rejects_missing_review_files() -> None:
    bad = [dict(module.ARTIFACTS[0])]
    bad[0]["ply"] = "/server_parking_priority_s10/missing_review_file.ply"

    result = module.validate_artifact_allowlist(bad, check_files=True)

    assert result["passed"] is False
    assert any("artifact_ply_missing" in error for error in result["errors"])


def test_local_artifact_path_preserves_filesystem_absolute_paths(tmp_path: Path) -> None:
    assert module.local_artifact_path(str(tmp_path / "artifact.ply")) == tmp_path / "artifact.ply"


def test_local_artifact_path_maps_viewer_paths_to_repo() -> None:
    path = module.local_artifact_path("/server_parking_priority_s10/example/artifact.ply")

    assert path == ROOT / "server_parking_priority_s10" / "example" / "artifact.ply"


def test_artifact_file_validation_accepts_filesystem_absolute_paths(tmp_path: Path) -> None:
    ply = tmp_path / "artifact.ply"
    objects = tmp_path / "objects.jsonl"
    ply.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    objects.write_text(
        json.dumps(
            {
                "object_id": 1,
                "voxel_count": 42,
                "geometry_type": "vertical",
                "semantic_label": "wall",
                "bbox_3d": {"min": [0, 0, 0], "max": [1, 1, 1]},
                "centroid": [0.5, 0.5, 0.5],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    errors = module.validate_artifact_files(
        [{"id": "tmp_review_artifact", "ply": str(ply), "objects": str(objects)}]
    )

    assert errors == []


def test_object_jsonl_schema_validation_rejects_missing_schema(tmp_path: Path) -> None:
    objects = tmp_path / "objects.jsonl"
    objects.write_text(json.dumps({"object_id": 1, "semantic_label": "wall"}) + "\n", encoding="utf-8")

    errors = module.validate_object_jsonl_schema("v7_object_refinement", objects)

    assert any("artifact_objects_missing_schema_keys" in error for error in errors)


def test_cli_writes_review_index(tmp_path: Path) -> None:
    qa = tmp_path / "qa.json"
    visual = tmp_path / "visual.json"
    out = tmp_path / "index.html"
    qa.write_text(json.dumps(qa_fixture()), encoding="utf-8")
    visual.write_text(json.dumps(visual_fixture()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--qa-json",
            str(qa),
            "--visual-acceptance",
            str(visual),
            "--output-html",
            str(out),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    text = out.read_text(encoding="utf-8")
    assert "Current Dense Mainline Review" in text
    assert "SPG v4 near-bbox" in text
    assert "Promotion Review Checklist" in text
