from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "show_current_mainline.py"


def test_show_current_mainline_json_output() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--format", "json"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert data["dataset"] == "MT20260616-175807"
    assert data["dense_patch_baseline"]["id"] == "dense_las_voxel003_energy_v6_fine_gated_overlap_20260624"
    assert data["dense_object_baseline"]["id"] == "dense_las_voxel003_objects_v3_high_recall_clean_20260624"
    assert any(item["pattern"] == "frame_object_points_stride10.ply" for item in data["forbidden_inputs"])


def test_show_current_mainline_text_output() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "current dense patch baseline:" in result.stdout
    assert "dense_las_voxel003_energy_v6_fine_gated_overlap_20260624" in result.stdout
    assert "runner: scripts/run_dense_patch_object_refinement_v7.py" in result.stdout
    assert "blocker:" in result.stdout
    assert "forbidden inputs:" in result.stdout
    assert "frame_object_points_stride10.ply" in result.stdout
