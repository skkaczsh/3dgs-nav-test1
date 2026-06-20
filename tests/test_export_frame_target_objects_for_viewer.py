import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "export_frame_target_objects_for_viewer_for_test",
        SCRIPTS / "export_frame_target_objects_for_viewer.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_export_streams_target_ply_to_viewer_object_ply(tmp_path: Path):
    module = load_module()
    targets = tmp_path / "targets.jsonl"
    targets.write_text(
        json.dumps({"target_id": "pt_000000_cam0_p4_cc000", "target_index": 7}) + "\n",
        encoding="utf-8",
    )
    objects = tmp_path / "objects.jsonl"
    objects.write_text(
        json.dumps(
            {
                "object_id": "obj_000123",
                "semantic_label": "car",
                "targets": ["pt_000000_cam0_p4_cc000"],
                "centroid": [1, 2, 3],
                "label_votes": {"car": 2},
            }
        ) + "\n",
        encoding="utf-8",
    )
    ply = tmp_path / "targets.ply"
    ply.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "property int target",
                "property uchar priority",
                "property int frame",
                "property int camera",
                "property int point_index",
                "end_header",
                "1 2 3 10 20 30 7 4 0 0 42",
                "4 5 6 10 20 30 7 4 0 0 43",
            ]
        ) + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "out"
    target_index_to_id = module.load_target_index_map(targets)
    objects_by_id, target_to_object = module.load_object_maps(objects)
    report = module.export_ply(ply, out / "points.ply", target_index_to_id, target_to_object, objects_by_id, stride=1)
    module.export_objects_jsonl(objects_by_id, out / "objects.jsonl")

    assert report["output_vertices"] == 2
    assert report["label_counts"] == {"car": 2}
    text = (out / "points.ply").read_text(encoding="utf-8")
    assert "property int object" in text
    assert "1 2 3 235 90 80 123 8 0 0 7 4" in text
    meta = json.loads((out / "objects.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert meta["object_id"] == "obj_000123"
    assert meta["viewer_object_id"] == 123
    assert meta["semantic_id"] == 8
    assert meta["dominant_label_ratio"] == 1.0


def test_manual_object_review_export_rewrites_ply_semantic(tmp_path: Path, monkeypatch):
    sys.path.insert(0, str(ROOT))
    from scripts import run_manual_object_review_export as runner

    targets = tmp_path / "targets.jsonl"
    targets.write_text(
        json.dumps({"target_id": "pt_000000_cam0_p4_cc000", "target_index": 7}) + "\n",
        encoding="utf-8",
    )
    source_objects = tmp_path / "objects.jsonl"
    source_objects.write_text(
        json.dumps(
            {
                "object_id": "obj_000123",
                "viewer_object_id": 123,
                "semantic_label": "car",
                "targets": ["pt_000000_cam0_p4_cc000"],
                "point_count": 2,
                "centroid": [1, 2, 3],
                "label_votes": {"car": 2},
            }
        ) + "\n",
        encoding="utf-8",
    )
    target_ply = tmp_path / "targets.ply"
    target_ply.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "property int target",
                "property uchar priority",
                "property int frame",
                "property int camera",
                "end_header",
                "1 2 3 10 20 30 7 4 0 0",
                "4 5 6 10 20 30 7 4 0 0",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps({"objects": [{"object_id": 123, "source_object_id": "obj_000123", "label": "car"}]}),
        encoding="utf-8",
    )
    decisions = tmp_path / "manual.csv"
    decisions.write_text(
        "object_id,source_object_id,current_label,decision,new_label,confidence,reviewer,notes\n"
        "123,obj_000123,car,relabel,wall,0.9,skk,flat surface\n",
        encoding="utf-8",
    )
    out = tmp_path / "reviewed"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_manual_object_review_export.py",
            "--decisions-csv",
            str(decisions),
            "--review-index-json",
            str(review),
            "--objects-jsonl",
            str(source_objects),
            "--targets-jsonl",
            str(targets),
            "--target-ply",
            str(target_ply),
            "--output-dir",
            str(out),
            "--stride",
            "1",
        ],
    )

    assert runner.main() == 0
    summary = json.loads((out / "manual_object_review_export_report.json").read_text(encoding="utf-8"))
    reviewed_obj = json.loads((out / "frame_objects_viewer.jsonl").read_text(encoding="utf-8").splitlines()[0])
    ply_text = (out / "frame_object_points_stride10.ply").read_text(encoding="utf-8")
    qa = json.loads((out / "viewer_candidate_qa.json").read_text(encoding="utf-8"))

    assert summary["apply"]["applied_count"] == 1
    assert reviewed_obj["semantic_label"] == "wall"
    assert reviewed_obj["semantic_label_original"] == "car"
    assert "1 2 3 120 150 180 123 2 0 0 7 4" in ply_text
    assert qa["status"] == "ok"
    assert qa["consistency"]["semantic_mismatch_count"] == 0


def test_manual_object_review_export_can_rewrite_existing_viewer_ply(tmp_path: Path, monkeypatch):
    sys.path.insert(0, str(ROOT))
    from scripts import run_manual_object_review_export as runner

    source_objects = tmp_path / "viewer_objects.jsonl"
    source_objects.write_text(
        json.dumps(
            {
                "object_id": "obj_000123",
                "viewer_object_id": 123,
                "semantic_label": "car",
                "point_count": 2,
                "centroid": [1, 2, 3],
            }
        ) + "\n",
        encoding="utf-8",
    )
    source_ply = tmp_path / "viewer.ply"
    source_ply.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "property int object",
                "property uchar semantic",
                "property int frame",
                "property int camera",
                "property int target",
                "property uchar priority",
                "end_header",
                "1 2 3 235 90 80 123 8 0 0 7 4",
                "4 5 6 235 90 80 123 8 0 0 7 4",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps({"objects": [{"object_id": 123, "source_object_id": "obj_000123", "label": "car"}]}),
        encoding="utf-8",
    )
    decisions = tmp_path / "manual.csv"
    decisions.write_text(
        "object_id,source_object_id,current_label,decision,new_label,confidence,reviewer,notes\n"
        "123,obj_000123,car,relabel,wall,0.9,skk,flat surface\n",
        encoding="utf-8",
    )
    out = tmp_path / "reviewed"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_manual_object_review_export.py",
            "--decisions-csv",
            str(decisions),
            "--review-index-json",
            str(review),
            "--objects-jsonl",
            str(source_objects),
            "--source-ply",
            str(source_ply),
            "--output-dir",
            str(out),
        ],
    )

    assert runner.main() == 0
    summary = json.loads((out / "manual_object_review_export_report.json").read_text(encoding="utf-8"))
    ply_text = (out / "frame_object_points_stride10.ply").read_text(encoding="utf-8")
    qa = json.loads((out / "viewer_candidate_qa.json").read_text(encoding="utf-8"))

    assert summary["export"]["output_vertices"] == 2
    assert "1 2 3 120 150 180 123 2 0 0 7 4" in ply_text
    assert qa["status"] == "ok"
    assert qa["ply"]["semantic_point_counts"] == {"wall": 2}
