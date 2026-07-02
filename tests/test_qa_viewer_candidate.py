import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "qa_viewer_candidate_for_test",
        SCRIPTS / "qa_viewer_candidate.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


qa = load_module()


def write_fixture(tmp_path: Path, semantic_for_object_2: int = 9) -> tuple[Path, Path]:
    ply = tmp_path / "candidate.ply"
    ply.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "property int object",
                "property uchar semantic",
                "property int frame",
                "end_header",
                "0 0 0 1 2 3 1 3 10",
                "1 0 0 1 2 3 1 3 10",
                f"0 1 0 4 5 6 2 {semantic_for_object_2} 20",
                f"0 2 0 4 5 6 2 {semantic_for_object_2} 20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    objects = tmp_path / "objects.jsonl"
    rows = [
        {
            "object_id": 1,
            "semantic_label": "ground",
            "status": "stable",
            "point_count": 2,
            "target_count": 1,
            "frames": [10],
        },
        {
            "object_id": 2,
            "semantic_label": "railing",
            "status": "ambiguous_object",
            "point_count": 2,
            "target_count": 2,
            "frames": [20],
        },
    ]
    objects.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return ply, objects


def write_provenance_fixture(tmp_path: Path) -> tuple[Path, Path]:
    ply, objects = write_fixture(tmp_path)
    rows = [
        {
            "object_id": 1,
            "semantic_label": "floor",
            "status": "stable",
            "point_count": 2,
            "target_count": 1,
            "semantic_fusion_status": "evidence_fusion_applied",
            "semantic_evidence_source_scores": {
                "sam": {"floor": 2},
                "teacher": {},
                "scene": {},
            },
            "conflict_flags": [],
        },
        {
            "object_id": 2,
            "semantic_label": "railing",
            "status": "stable",
            "point_count": 2,
            "target_count": 2,
            "semantic_fusion_status": "evidence_fusion_applied",
            "semantic_evidence_source_scores": {
                "sam": {},
                "teacher": {},
                "scene": {"railing": 2},
            },
            "conflict_flags": ["geometry_vetoed_some_evidence"],
        },
    ]
    objects.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return ply, objects


def args(tmp_path: Path, ply: Path, objects: Path):
    return type(
        "Args",
        (),
        {
            "ply": ply,
            "objects_jsonl": objects,
            "ambiguous_report": None,
            "consolidation_report": None,
            "output_json": tmp_path / "qa.json",
            "output_md": tmp_path / "qa.md",
            "top_n": 5,
        },
    )()


def test_build_report_accepts_ground_floor_alias_and_warns_on_ambiguous(tmp_path: Path):
    ply, objects = write_fixture(tmp_path)

    report = qa.build_report(args(tmp_path, ply, objects))

    assert report["status"] == "ok"
    assert report["ply"]["data_rows"] == 4
    assert report["consistency"]["semantic_mismatch_count"] == 0
    assert "remaining ambiguous objects: 1" in report["warnings"]
    assert "evidence provenance missing/unsupported for 100.0% of visible points" in report["warnings"]


def test_build_report_fails_on_object_semantic_mismatch(tmp_path: Path):
    ply, objects = write_fixture(tmp_path, semantic_for_object_2=8)

    report = qa.build_report(args(tmp_path, ply, objects))

    assert report["status"] == "failed"
    assert report["consistency"]["semantic_mismatch_count"] == 1
    assert report["consistency"]["top_semantic_mismatches"][0]["object_id"] == 2


def test_write_markdown_contains_chinese_labels(tmp_path: Path):
    ply, objects = write_fixture(tmp_path)
    run_args = args(tmp_path, ply, objects)
    report = qa.build_report(run_args)

    qa.write_markdown(run_args.output_md, report, top_n=5)

    text = run_args.output_md.read_text(encoding="utf-8")
    assert "地面" in text
    assert "栏杆/护栏" in text
    assert "Evidence source points" in text


def test_large_fine_warning_uses_class_aware_thresholds():
    objects = [
        {"object_id": 1, "semantic_label": "car", "status": "stable", "point_count": 10698},
        {"object_id": 2, "semantic_label": "railing", "status": "stable", "point_count": 10000},
        {"object_id": 3, "semantic_label": "car", "status": "stable", "point_count": 25000},
    ]

    summary = qa.summarize_objects(objects)

    assert [row["object_id"] for row in summary["large_fine_objects"]] == [3, 2]
    assert summary["large_fine_objects"][0]["large_fine_threshold"] == 25000
    assert summary["large_fine_objects"][1]["large_fine_threshold"] == 10000


def test_ten_k_point_car_is_not_large_fine_warning():
    summary = qa.summarize_objects(
        [{"object_id": 1, "semantic_label": "car", "status": "stable", "point_count": 10698}]
    )

    assert summary["large_fine_objects"] == []


def test_build_report_summarizes_evidence_provenance_risks(tmp_path: Path):
    ply, objects = write_provenance_fixture(tmp_path)

    report = qa.build_report(args(tmp_path, ply, objects))

    assert report["evidence"]["point_source_support_counts"] == {"sam": 2, "scene": 2}
    assert report["evidence"]["object_source_support_counts"] == {"sam": 1, "scene": 1}
    assert report["evidence"]["fusion_status_counts"] == {"evidence_fusion_applied": 2}
    assert report["evidence"]["conflict_flag_counts"] == {"geometry_vetoed_some_evidence": 1}
    assert report["evidence"]["warnings"] == [
        "scene-only support covers 50.0% of visible points",
        "scene-only support covers 50.0% of visible objects",
        "geometry veto evidence is dense: 1 flags over 2 visible objects",
    ]
    assert report["warnings"] == report["evidence"]["warnings"]
