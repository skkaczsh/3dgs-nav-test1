from __future__ import annotations

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import qa_object_voxel_overlap


def write_viewer_ply(path: Path) -> None:
    path.write_text(
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
                "end_header",
                "0.01 0.01 0.01 0 0 0 1 2",
                "0.02 0.02 0.02 0 0 0 2 4",
                "1.00 1.00 1.00 0 0 0 1 2",
                "2.00 2.00 2.00 0 0 0 3 8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_measure_overlap_reports_object_and_semantic_voxel_intersection(tmp_path: Path):
    ply = tmp_path / "viewer.ply"
    write_viewer_ply(ply)

    report = qa_object_voxel_overlap.measure_overlap(ply, voxel_size=0.10, max_pairs=10)

    assert report["mixed_object_voxels"] == 1
    assert report["mixed_semantic_voxels"] == 1
    assert report["top_object_overlaps"][0]["intersection_voxels"] == 1
    assert {report["top_semantic_overlaps"][0]["a"], report["top_semantic_overlaps"][0]["b"]} == {"wall", "ceiling"}
