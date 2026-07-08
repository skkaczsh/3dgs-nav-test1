from argparse import Namespace
import json

from scripts.export_superpoint_risk_context import export_context


def test_export_superpoint_risk_context_filters_bbox_and_recolors(tmp_path):
    ply = tmp_path / "scene.ply"
    objects = tmp_path / "objects.jsonl"
    out = tmp_path / "out"

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
                "end_header",
                "0 0 0 1 2 3 10",
                "1 0 0 1 2 3 20",
                "0.5 0 0 9 9 9 30",
                "9 9 9 9 9 9 40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    objects.write_text(
        "\n".join(
            [
                json.dumps({"object_id": 10, "bbox_3d": {"min": [0, 0, 0], "max": [0, 0, 0]}}),
                json.dumps({"object_id": 20, "bbox_3d": {"min": [1, 0, 0], "max": [1, 0, 0]}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = export_context(
        Namespace(
            ply=ply,
            objects=objects,
            patch_ids=[10, 20],
            output_dir=out,
            output_stem="risk",
            padding=0.1,
        )
    )

    assert report["output_points"] == 3
    assert report["target_stride_points"] == {"10": 1, "20": 1}
    rows = (out / "risk.ply").read_text(encoding="utf-8").splitlines()
    assert "element vertex 3" in rows
    assert "0 0 0 255 40 40 10" in rows
    assert "1 0 0 0 220 255 20" in rows
    assert "9 9 9 9 9 9 40" not in rows
