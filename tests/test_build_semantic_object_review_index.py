import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import build_semantic_object_review_index as mod


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_review_index_selects_key_objects_and_writes_links(tmp_path: Path, monkeypatch) -> None:
    objects = tmp_path / "objects.jsonl"
    write_jsonl(
        objects,
        [
            {"viewer_object_id": 1, "object_id": "obj_000001", "semantic_label": "car", "status": "stable", "point_count": 10},
            {"viewer_object_id": 2, "object_id": "obj_000002", "semantic_label": "car", "status": "stable", "point_count": 20},
            {"viewer_object_id": 3, "object_id": "obj_000003", "semantic_label": "railing", "status": "stable", "point_count": 30},
            {
                "object_id": 3000000,
                "semantic_label": "unknown",
                "status": "priority_unknown_local_geometry_child",
                "point_count": 40,
            },
        ],
    )
    out = tmp_path / "review"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_semantic_object_review_index.py",
            "--objects-jsonl",
            str(objects),
            "--output-dir",
            str(out),
            "--per-label",
            "1",
            "--ply-url",
            "/work/run/frame_object_points_stride10.ply",
            "--objects-url",
            "/work/run/frame_objects_viewer.jsonl",
        ],
    )

    assert mod.main() == 0
    report = json.loads((out / "semantic_object_review_index.json").read_text(encoding="utf-8"))
    html = (out / "semantic_object_review_index.html").read_text(encoding="utf-8")

    ids = [row["object_id"] for row in report["objects"]]
    assert 2 in ids
    assert 3 in ids
    assert 3000000 in ids
    assert "object=2" in html
    assert "semantic" in html
    assert "rgb" in html


def test_semantic_ply_viewer_supports_object_filter_links() -> None:
    html = Path("tools/semantic_ply_viewer.html").read_text(encoding="utf-8")
    assert 'params.get("object")' in html
    assert "objectKeys(row.objectId)" in html
