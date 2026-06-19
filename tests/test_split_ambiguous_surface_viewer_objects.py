import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "split_ambiguous_surface_viewer_objects_for_test",
        SCRIPTS / "split_ambiguous_surface_viewer_objects.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def target(target_id: str, index: int, label: str, points: int = 10):
    return {
        "target_id": target_id,
        "target_index": index,
        "label": label,
        "cluster_size": points,
        "bbox_3d": {"min": [0, 0, 0], "max": [1, 1, 1]},
        "centroid": [0.5, 0.5, 0.5],
    }


def test_build_split_plan_splits_surface_only_ambiguous_object():
    module = load_module()
    objects = [
        {"object_id": 1, "semantic_label": "wall", "status": "stable", "targets": ["t0"], "point_count": 5},
        {
            "object_id": 2,
            "semantic_label": "ambiguous",
            "status": "ambiguous_object",
            "targets": ["t1", "t2", "t3"],
            "point_count": 60,
        },
    ]
    targets_by_id = {
        "t1": target("t1", 11, "wall", 20),
        "t2": target("t2", 12, "ground", 30),
        "t3": target("t3", 13, "wall", 10),
    }

    rows, point_map, report = module.build_split_plan(objects, targets_by_id, min_labels=2)

    assert report["split_objects"] == 1
    assert report["split_children"] == 2
    assert {row["semantic_label"] for row in rows if row.get("parent_object_id") == 2} == {"wall", "ground"}
    assert point_map[(2, 11)][1] == "wall"
    assert point_map[(2, 12)][1] == "ground"


def test_build_split_plan_skips_non_surface_ambiguous_object():
    module = load_module()
    objects = [{"object_id": 2, "semantic_label": "ambiguous", "status": "ambiguous_object", "targets": ["t1", "t2"]}]
    targets_by_id = {"t1": target("t1", 11, "wall"), "t2": target("t2", 12, "car")}

    rows, point_map, report = module.build_split_plan(objects, targets_by_id, min_labels=2)

    assert report["split_objects"] == 0
    assert report["kept_ambiguous"] == 1
    assert point_map == {}
    assert rows[0]["semantic_label"] == "ambiguous"
    assert rows[0]["surface_split_skipped_reason"]["non_surface_labels"] == ["car"]


def test_rewrite_ply_updates_object_semantic_and_color(tmp_path: Path):
    module = load_module()
    input_ply = tmp_path / "in.ply"
    output_ply = tmp_path / "out.ply"
    input_ply.write_text(
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
                "property int target",
                "end_header",
                "0 0 0 1 2 3 2 0 11",
                "1 0 0 1 2 3 2 0 12",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = module.rewrite_ply(input_ply, output_ply, {(2, 11): (3, "wall"), (2, 12): (4, "ground")})

    text = output_ply.read_text(encoding="utf-8")
    assert "120 150 180 3 2 11" in text
    assert "196 168 112 4 3 12" in text
    assert report["changed_vertices"] == 2
