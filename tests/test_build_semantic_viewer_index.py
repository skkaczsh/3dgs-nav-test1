import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import build_semantic_viewer_index
from scripts.build_semantic_viewer_index import build_index


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def touch_time(path: Path, mtime: int) -> None:
    os.utime(path, (mtime, mtime))


def test_build_index_sorts_by_artifact_update_time_and_builds_viewer_urls(tmp_path: Path) -> None:
    artifact_root = tmp_path / "server_parking_priority_s10"
    old_dir = artifact_root / "old_run" / "viewer"
    new_dir = artifact_root / "new_run" / "viewer_localgeom"

    write(old_dir / "frame_object_points_stride10.ply", "ply\n")
    write(old_dir / "frame_objects_viewer.jsonl", "{}\n")
    write(
        old_dir / "frame_object_viewer_export_report.json",
        json.dumps(
            {
                "output_vertices": 10,
                "object_records": 2,
                "label_counts": {"wall": 7, "car": 3},
                "point_source_support_counts": {"sam+teacher": 8, "scene": 2},
                "object_source_support_counts": {"sam+teacher": 1, "scene": 1},
                "fusion_status_counts": {"evidence_fusion_applied": 1, "kept_original_scene_only_evidence": 1},
                "conflict_flag_counts": {"geometry_vetoed_some_evidence": 2},
            }
        ),
    )
    touch_time(old_dir / "frame_object_points_stride10.ply", 100)

    write(new_dir / "frame_object_points_stride10.ply", "ply\n")
    write(new_dir / "frame_objects_viewer.jsonl", "{}\n")
    write(
        new_dir / "viewer_candidate_qa.json",
        json.dumps(
            {
                "status": "ok",
                "warnings": ["large railing object"],
                "errors": [],
                "ply": {"vertex_count": 20, "semantic_point_counts": {"floor": 12, "railing": 8}},
            }
        ),
    )
    touch_time(new_dir / "viewer_candidate_qa.json", 300)

    index = build_index(web_root=tmp_path, artifact_root=artifact_root)

    assert index["artifact_count"] == 2
    assert [entry["name"] for entry in index["entries"]] == ["viewer_localgeom", "viewer"]

    newest = index["entries"][0]
    assert newest["status"] == "ok"
    assert newest["warnings"] == ["large railing object"]
    assert newest["counts"]["vertex_count"] == 20
    assert newest["counts"]["semantic_point_counts"] == {"floor": 12, "railing": 8}
    assert newest["viewer_urls"]["semantic"].startswith("/tools/semantic_ply_viewer.html?")
    assert "mode=semantic" in newest["viewer_urls"]["semantic"]
    assert "objects=/server_parking_priority_s10/new_run/viewer_localgeom/frame_objects_viewer.jsonl" in newest["viewer_urls"]["semantic"]

    older = index["entries"][1]
    assert older["status"] == "report_only"
    assert older["counts"]["vertex_count"] == 10
    assert older["counts"]["semantic_point_counts"] == {"wall": 7, "car": 3}
    assert older["counts"]["point_source_support_counts"] == {"sam+teacher": 8, "scene": 2}
    assert older["counts"]["object_source_support_counts"] == {"sam+teacher": 1, "scene": 1}
    assert older["counts"]["fusion_status_counts"] == {
        "evidence_fusion_applied": 1,
        "kept_original_scene_only_evidence": 1,
    }
    assert older["counts"]["conflict_flag_counts"] == {"geometry_vetoed_some_evidence": 2}
    assert older["evidence_warnings"] == [
        "scene-only support covers 20.0% of visible points",
        "scene-only support covers 50.0% of visible objects",
        "geometry veto evidence is dense: 2 flags over 2 visible objects",
    ]
    assert older["warnings"] == older["evidence_warnings"]


def test_build_index_discovers_generic_stride_patch_outputs(tmp_path: Path) -> None:
    artifact_root = tmp_path / "server_parking_priority_s10"
    run_dir = artifact_root / "geo_patch_las_opt" / "energy_bucket_split_frag_attach_v5_20260702"
    write(run_dir / "geo_patches_bucket_split_frag_attach_v5_stride10.ply", "ply\n")
    write(run_dir / "geo_patches_bucket_split_frag_attach_v5.jsonl", "{}\n")
    write(
        run_dir / "geo_patches_bucket_split_frag_attach_v5_report.json",
        json.dumps({"preview_points": 123, "output_patch_count": 45, "schema": "geo-patch-energy-graph-v4"}),
    )

    index = build_index(web_root=tmp_path, artifact_root=artifact_root)

    assert index["artifact_count"] == 1
    entry = index["entries"][0]
    assert entry["name"] == "energy_bucket_split_frag_attach_v5_20260702"
    assert entry["ply"].endswith("/geo_patches_bucket_split_frag_attach_v5_stride10.ply")
    assert entry["objects"].endswith("/geo_patches_bucket_split_frag_attach_v5.jsonl")
    assert "objects=" in entry["viewer_urls"]["semantic"]
    assert entry["status"] == "report_only"
    assert entry["counts"]["vertex_count"] == 123
    assert entry["counts"]["object_count"] == 45
    assert entry["reports"]["generic"]["schema"] == "geo-patch-energy-graph-v4"


def test_build_index_skips_rejected_generic_stride_outputs(tmp_path: Path) -> None:
    artifact_root = tmp_path / "server_parking_priority_s10"
    run_dir = artifact_root / "objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor"
    write(run_dir / "objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor_stride10.ply", "ply\n")
    write(run_dir / "objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor.jsonl", "{}\n")

    index = build_index(web_root=tmp_path, artifact_root=artifact_root)

    assert index["artifact_count"] == 0


def test_evidence_risk_warnings_flags_missing_source_scores() -> None:
    warnings = build_semantic_viewer_index.evidence_risk_warnings(
        {
            "point_source_support_counts": {"sam+teacher": 90, "missing_source_scores": 10},
            "object_source_support_counts": {"sam+teacher": 9, "no_label_source_support": 1},
            "conflict_flag_counts": {},
        }
    )

    assert warnings == [
        "evidence provenance missing/unsupported for 10.0% of visible points",
        "evidence provenance missing/unsupported for 10.0% of visible objects",
    ]


def test_build_index_links_object_review_pack(tmp_path: Path) -> None:
    artifact_root = tmp_path / "work"
    viewer_dir = artifact_root / "viewer_full"
    ply = viewer_dir / "frame_object_points_stride10.ply"
    objects = viewer_dir / "frame_objects_viewer.jsonl"
    write(ply, "ply\n")
    write(objects, "{}\n")
    write(viewer_dir / "viewer_candidate_qa.json", json.dumps({"status": "ok", "warnings": [], "errors": []}))

    review_dir = artifact_root / "review_full"
    write(
        review_dir / "semantic_object_review_index.json",
        json.dumps(
            {
                "objects": [
                    {
                        "object_id": 1,
                        "semantic_url": (
                            "/tools/semantic_ply_viewer.html?"
                            "file=/work/viewer_full/frame_object_points_stride10.ply"
                            "&objects=/work/viewer_full/frame_objects_viewer.jsonl"
                            "&mode=semantic&object=1"
                        ),
                    }
                ]
            }
        ),
    )
    write(review_dir / "semantic_object_review_index.html", "<html></html>")
    write(review_dir / "manual_object_review_decisions.csv", "object_id,decision\n1,pending\n")
    write(
        review_dir / "manual_object_review_decisions.report.json",
        json.dumps({"accepted_count": 0, "error_count": 1}),
    )

    index = build_index(web_root=tmp_path, artifact_root=artifact_root)
    entry = index["entries"][0]

    assert entry["review"]["object_count"] == 1
    assert entry["review"]["review_html"] == "/work/review_full/semantic_object_review_index.html"
    assert entry["review"]["decision_csv"] == "/work/review_full/manual_object_review_decisions.csv"
    assert entry["review"]["normalize"] == {"accepted_count": 0, "error_count": 1}


def test_build_index_links_review_pack_under_symlink_artifact_root(tmp_path: Path) -> None:
    real_root = tmp_path / "work_real"
    viewer_dir = real_root / "viewer_full"
    write(viewer_dir / "frame_object_points_stride10.ply", "ply\n")
    write(viewer_dir / "frame_objects_viewer.jsonl", "{}\n")
    review_dir = real_root / "review_full"
    write(
        review_dir / "semantic_object_review_index.json",
        json.dumps(
            {
                "objects": [
                    {
                        "semantic_url": (
                            "/tools/semantic_ply_viewer.html?"
                            "file=/work/viewer_full/frame_object_points_stride10.ply"
                            "&objects=/work/viewer_full/frame_objects_viewer.jsonl"
                        )
                    }
                ]
            }
        ),
    )
    write(review_dir / "manual_object_review_decisions.csv", "object_id,decision\n1,pending\n")

    web_root = tmp_path / "repo"
    web_root.mkdir()
    link_root = web_root / "work"
    link_root.symlink_to(real_root, target_is_directory=True)

    index = build_index(web_root=web_root, artifact_root=link_root)
    entry = index["entries"][0]

    assert entry["ply"] == "/work/viewer_full/frame_object_points_stride10.ply"
    assert entry["review"]["decision_csv"] == "/work/review_full/manual_object_review_decisions.csv"


def test_index_html_uses_generated_json_and_existing_viewer() -> None:
    html = Path("tools/semantic_viewer_index.html").read_text(encoding="utf-8")
    assert "semantic_viewer_index.json" in html
    assert "viewer_urls.semantic" in html
    assert "语义点云版本索引" in html
    assert "对象审阅" in html
    assert "决策 CSV" in html
    assert "打开最新版语义" in html
    assert "updateLatestLinks" in html
    assert "point_source_support_counts" in html
    assert "object_source_support_counts" in html
    assert "fusion_status_counts" in html
    assert "conflict_flag_counts" in html
    assert "evidence_warnings" in html
    assert "Evidence 来源点数" in html
    assert "Fusion 状态" in html
    assert "Evidence QA" in html


def test_build_index_keeps_symlink_url_prefix(tmp_path: Path) -> None:
    real_root = tmp_path / "work_MT20260616-175807"
    viewer_dir = real_root / "run_a" / "viewer"
    write(viewer_dir / "frame_object_points_stride10.ply", "ply\n")
    write(viewer_dir / "frame_objects_viewer.jsonl", "{}\n")

    link_root = tmp_path / "repo" / "work_MT20260616-175807"
    link_root.parent.mkdir(parents=True)
    link_root.symlink_to(real_root, target_is_directory=True)

    index = build_index(web_root=tmp_path / "repo", artifact_root=link_root)
    entry = index["entries"][0]

    assert entry["ply"] == "/work_MT20260616-175807/run_a/viewer/frame_object_points_stride10.ply"
    assert entry["relative_dir"] == "run_a/viewer"


def test_build_index_maps_real_artifact_root_through_web_symlink(tmp_path: Path) -> None:
    real_root = tmp_path / "work_real"
    viewer_dir = real_root / "viewer_full"
    write(viewer_dir / "frame_object_points_stride10.ply", "ply\n")
    write(viewer_dir / "frame_objects_viewer.jsonl", "{}\n")
    review_dir = real_root / "review_full"
    write(
        review_dir / "semantic_object_review_index.json",
        json.dumps(
            {
                "objects": [
                    {
                        "semantic_url": (
                            "/tools/semantic_ply_viewer.html?"
                            "file=/work/viewer_full/frame_object_points_stride10.ply"
                            "&objects=/work/viewer_full/frame_objects_viewer.jsonl"
                        )
                    }
                ]
            }
        ),
    )
    write(review_dir / "semantic_object_review_index.html", "<html></html>")
    write(review_dir / "manual_object_review_decisions.csv", "object_id,decision\n1,pending\n")

    web_root = tmp_path / "repo"
    web_root.mkdir()
    (web_root / "work").symlink_to(real_root, target_is_directory=True)

    index = build_index(web_root=web_root, artifact_root=real_root)
    entry = index["entries"][0]

    assert entry["ply"] == "/work/viewer_full/frame_object_points_stride10.ply"
    assert entry["objects"] == "/work/viewer_full/frame_objects_viewer.jsonl"
    assert entry["review"]["review_html"] == "/work/review_full/semantic_object_review_index.html"
    assert entry["review"]["decision_csv"] == "/work/review_full/manual_object_review_decisions.csv"


def test_cli_keeps_symlink_url_prefix(tmp_path: Path, monkeypatch) -> None:
    real_root = tmp_path / "work"
    viewer_dir = real_root / "run_b" / "viewer"
    write(viewer_dir / "frame_object_points_stride10.ply", "ply\n")
    write(viewer_dir / "frame_objects_viewer.jsonl", "{}\n")

    web_root = tmp_path / "repo"
    link_root = web_root / "work"
    web_root.mkdir()
    link_root.symlink_to(real_root, target_is_directory=True)
    output = web_root / "tools" / "semantic_viewer_index.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_semantic_viewer_index.py",
            "--web-root",
            str(web_root),
            "--artifact-root",
            "work",
            "--output",
            str(output),
        ],
    )

    assert build_semantic_viewer_index.main() == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["entries"][0]["ply"] == "/work/run_b/viewer/frame_object_points_stride10.ply"
