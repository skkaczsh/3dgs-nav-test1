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
    assert meta["semantic_id"] == 8
    assert meta["dominant_label_ratio"] == 1.0
