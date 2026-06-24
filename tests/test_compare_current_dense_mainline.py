from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.compare_current_dense_mainline import build_report, format_md


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_current_dense_mainline.py"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def make_fixture(base: Path) -> None:
    for run, candidate_count, accepted, objects, overlap in [
        ("dense_patch_object_refinement_v7_r4_attach_v4_20260624_170126", 10, 2, 8, 0.20),
        ("dense_patch_object_refinement_v8_tiny_attach_20260624_170619", 30, 7, 3, 0.18),
    ]:
        write_json(
            base
            / run
            / "object_merge_candidates_v7_structural_multimaterial"
            / "geo_patch_object_merge_candidates_report.json",
            {
                "candidate_count": candidate_count,
                "big_mixed_attachment_count": 1,
                "merge_class_counts": {"same_material": candidate_count - 1, "structural_multimaterial": 1},
            },
        )
        write_json(
            base
            / run
            / "objects_v7_structural_multimaterial"
            / "geo_patch_objects_v7_structural_multimaterial_report.json",
            {"accepted_candidate_rows": accepted, "output_object_count": objects},
        )
        write_json(
            base / run / "objects_v7_structural_multimaterial" / "voxel_overlap_020_report.json",
            {"mixed_object_voxel_ratio": overlap, "object_count": objects},
        )

    write_json(
        base / "objects_v9_teacher_v20_semantic" / "objects_v9_teacher_v20_semantic_report.json",
        {"object_count": 100, "label_point_counts": {"wall": 50, "unknown": 10}},
    )
    write_json(
        base
        / "objects_v17_teacher_v20_surface_preserve_guard"
        / "objects_v17_teacher_v20_surface_preserve_guard_report.json",
        {"object_count": 100, "changed_object_count": 0, "label_point_counts": {"wall": 50, "unknown": 10}},
    )


def test_build_report_compares_dense_object_and_surface_guard(tmp_path: Path) -> None:
    make_fixture(tmp_path)

    report = build_report(tmp_path)

    obj = report["object_refinement"]["metrics"]
    assert obj["delta_v8_minus_v7"]["candidate_count"] == 20
    assert obj["delta_v8_minus_v7"]["accepted_candidate_rows"] == 5
    assert obj["delta_v8_minus_v7"]["mixed_object_voxel_ratio_020"] < 0
    assert report["surface_guard"]["changed_object_count"] == 0
    assert report["surface_guard"]["label_point_counts"]["delta_v17_minus_v9"] == {"unknown": 0, "wall": 0}
    assert report["surface_guard"]["unknown_point_delta_v17_minus_v9"] == 0


def test_format_md_includes_key_sections(tmp_path: Path) -> None:
    make_fixture(tmp_path)

    text = format_md(build_report(tmp_path))

    assert "# Current Dense Mainline QA" in text
    assert "| candidate_count | 10 | 30 | 20 |" in text
    assert "| wall | 50 | 50 | 0 |" in text
    assert "Unknown point delta v17-v9: `0`" in text


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    make_fixture(tmp_path / "base")
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--base-dir",
            str(tmp_path / "base"),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(out_json.read_text(encoding="utf-8"))["schema"] == "current-dense-mainline-qa/v1"
    assert out_md.read_text(encoding="utf-8").startswith("# Current Dense Mainline QA")
