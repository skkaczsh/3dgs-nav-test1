import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "split_priority_objects_by_local_geometry.py"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_split_handles_noncontiguous_object_runs(tmp_path: Path):
    input_ply = tmp_path / "input.ply"
    objects_jsonl = tmp_path / "objects.jsonl"
    conflicts_jsonl = tmp_path / "conflicts.jsonl"
    output_dir = tmp_path / "out"

    input_ply.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 5",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "property int object",
                "property uchar semantic",
                "end_header",
                "0.0 0.0 0.0 120 150 180 1 2",
                "0.0 0.0 0.0 240 210 60 2 9",
                "0.1 0.0 0.0 120 150 180 1 2",
                "1.0 0.0 0.0 240 210 60 2 9",
                "2.0 0.0 0.0 240 210 60 2 9",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_jsonl(
        objects_jsonl,
        [
            {"object_id": 1, "semantic_label": "wall", "point_count": 2},
            {"object_id": 2, "semantic_label": "railing", "point_count": 3},
        ],
    )
    write_jsonl(
        conflicts_jsonl,
        [{"object_id": 2, "suggested_action": "split", "reasons": ["railing_overmerged_extent"]}],
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-ply",
            str(input_ply),
            "--objects-jsonl",
            str(objects_jsonl),
            "--conflicts-jsonl",
            str(conflicts_jsonl),
            "--output-dir",
            str(output_dir),
            "--output-prefix",
            "split",
            "--min-split-points",
            "1",
            "--local-voxel-size",
            "10",
            "--min-cell-points",
            "1",
            "--min-child-points",
            "1",
            "--min-unknown-child-points",
            "1",
            "--railing-keep-linearity",
            "0.5",
        ],
        check=True,
    )

    rows = [json.loads(line) for line in (output_dir / "split.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["object_id"] for row in rows] == [1, 3000000]
    assert [row["semantic_label"] for row in rows] == ["wall", "railing"]

    report = json.loads((output_dir / "split_report.json").read_text(encoding="utf-8"))
    assert report["input_vertex_count"] == 5
    assert report["input_object_count"] == 2
    assert report["output_object_count"] == 2
    assert report["split_source_object_count"] == 1
    assert report["counts"]["passthrough_objects"] == 1

    data_lines = [
        line
        for line in (output_dir / "split.ply").read_text(encoding="utf-8").splitlines()
        if line and line[0].isdigit()
    ]
    assert len(data_lines) == 5
    assert sum(1 for line in data_lines if line.split()[6] == "1") == 2
    assert sum(1 for line in data_lines if line.split()[6] == "3000000") == 3
