from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import apply_surface_trust_guard_to_ply as module


def test_surface_trust_guard_rejects_stride_source_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "apply_surface_trust_guard_to_ply.py",
            "--drivability-pcd",
            str(tmp_path / "prior.pcd"),
            "--input-ply",
            str(tmp_path / "frame_object_points_stride10.ply"),
            "--input-objects-jsonl",
            str(tmp_path / "objects.jsonl"),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    with pytest.raises(ValueError, match="forbidden input path"):
        module.main()


def test_surface_trust_guard_allows_explicit_qa_preview_source(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "out"
    monkeypatch.setattr(module, "read_pcd_xyzrgb", lambda path: (np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.uint8)))
    monkeypatch.setattr(module, "label_from_rgb", lambda rgb: np.array([module.GEOM_GROUND], dtype=np.uint8))
    monkeypatch.setattr(module, "build_prior_voxels", lambda xyz, labels, voxel: (np.zeros((1, 3), dtype=np.int32), np.array([module.GEOM_GROUND], dtype=np.uint8), {"origin": [0, 0, 0]}))
    monkeypatch.setattr(
        module,
        "load_ascii_ply",
        lambda path: (
            ["ply\n", "end_header\n"],
            ["x", "y", "z", "red", "green", "blue", "object", "semantic"],
            [["0", "0", "0", "0", "0", "0", "1", "8"]],
            np.zeros((1, 3), dtype=np.float32),
            np.array([1], dtype=np.uint32),
            np.array([8], dtype=np.uint16),
        ),
    )
    monkeypatch.setattr(module, "vote_points", lambda *a, **k: np.array([module.GEOM_GROUND], dtype=np.uint8))
    monkeypatch.setattr(module, "read_jsonl", lambda path: [{"object_id": 1, "semantic_label": "car"}])
    monkeypatch.setattr(module, "write_ascii_ply", lambda *a, **k: None)
    monkeypatch.setattr(module, "write_jsonl", lambda *a, **k: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "apply_surface_trust_guard_to_ply.py",
            "--drivability-pcd",
            str(tmp_path / "prior.pcd"),
            "--input-ply",
            str(tmp_path / "frame_object_points_stride10.ply"),
            "--input-objects-jsonl",
            str(tmp_path / "objects.jsonl"),
            "--output-dir",
            str(output_dir),
            "--allow-qa-preview-source",
        ],
    )

    module.main()

    assert (output_dir / "full_scene_surface_trust_guard_report.json").exists()
