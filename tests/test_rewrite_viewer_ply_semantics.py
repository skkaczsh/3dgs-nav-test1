from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import rewrite_viewer_ply_semantics as module


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "rewrite_viewer_ply_semantics.py"


def write_minimal_ply(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "property int object",
                "property uchar semantic",
                "end_header",
                "0 0 0 0 0 0 1 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def write_objects(path: Path) -> Path:
    path.write_text(json.dumps({"object_id": 1, "semantic_label": "floor"}) + "\n", encoding="utf-8")
    return path


def test_rewrite_ply_rejects_forbidden_source_path(tmp_path: Path) -> None:
    source = write_minimal_ply(tmp_path / "frame_object_points_stride10.ply")
    objects = write_objects(tmp_path / "objects.jsonl")

    with pytest.raises(ValueError, match="forbidden input path"):
        module.rewrite_ply(source, objects, tmp_path / "out.ply")


def test_rewrite_ply_rejects_forbidden_output_path(tmp_path: Path) -> None:
    source = write_minimal_ply(tmp_path / "source.ply")
    objects = write_objects(tmp_path / "objects.jsonl")

    with pytest.raises(ValueError, match="forbidden input path"):
        module.rewrite_ply(source, objects, tmp_path / "objects_v15_teacher_v20_grid6_geometry_guard_no_wall_to_floor.ply")


def test_rewrite_cli_still_exports_valid_input(tmp_path: Path) -> None:
    source = write_minimal_ply(tmp_path / "source.ply")
    objects = write_objects(tmp_path / "objects.jsonl")
    output = tmp_path / "semantic.ply"
    report = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-ply",
            str(source),
            "--objects-jsonl",
            str(objects),
            "--output-ply",
            str(output),
            "--report-json",
            str(report),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(report.read_text(encoding="utf-8"))["label_counts"] == {"floor": 1}
