from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


field_mod = load_module("build_structural_region_field_for_test", SCRIPTS / "build_structural_region_field.py")
attach_mod = load_module("classify_surface_attachment_for_test", SCRIPTS / "classify_surface_attachment.py")


def test_structural_labels_are_non_semantic_regions():
    rgb = np.array([
        [255, 0, 0],
        [255, 255, 255],
        [0, 255, 0],
        [0, 0, 255],
    ], dtype=np.uint8)

    labels = field_mod.structural_labels_from_rgb(rgb)

    assert labels.tolist() == [
        field_mod.REGION_GROUND_LIKE,
        field_mod.REGION_VERTICAL_SURFACE,
        field_mod.REGION_UPPER_HORIZONTAL,
        field_mod.REGION_OTHER_STRUCTURE,
    ]
    assert field_mod.REGION_NAMES[field_mod.REGION_VERTICAL_SURFACE] == "vertical_surface_region"


def test_surface_attachment_distinguishes_wall_texture_from_railing():
    vertical_field = {
        "keys": np.array([0], dtype=np.int64),
        "labels": np.array([field_mod.REGION_VERTICAL_SURFACE], dtype=np.uint8),
        "confidence": np.array([1.0], dtype=np.float32),
        "spec": np.array([0, 0, 0, 10, 10], dtype=np.int64),
        "voxel_size": 1.0,
    }
    wall_points = np.array([[0.0, y, z] for y in np.linspace(0, 0.9, 5) for z in np.linspace(0, 0.9, 5)], dtype=np.float32)
    rail_points = np.array([[0.12, y, 0.2] for y in np.linspace(0, 0.9, 20)], dtype=np.float32)
    args = attach_mod.argparse.Namespace(
        neighbor_radius=0,
        structural_ratio_min=0.35,
        horizontal_normal_z_min=0.86,
        vertical_normal_z_max=0.42,
        surface_planarity_min=0.72,
        surface_thickness_max=0.35,
        large_surface_min_points=20,
        large_surface_min_extent=0.8,
        surface_label_agreement_ratio=0.75,
        surface_label_min_points=20,
        attached_linearity_min=0.55,
        attached_scattering_min=0.08,
        attached_thickness_min=0.10,
    )

    wall_counts, wall_conf = attach_mod.vote_structural_regions(wall_points, vertical_field, 0)
    rail_counts, rail_conf = attach_mod.vote_structural_regions(rail_points, vertical_field, 0)
    wall = attach_mod.classify_attachment({"label": "wall", "cluster_size": len(wall_points)}, wall_points, wall_counts, wall_conf, args)
    rail = attach_mod.classify_attachment({"label": "railing", "cluster_size": len(rail_points)}, rail_points, rail_counts, rail_conf, args)

    assert wall["surface_attachment_status"] == "merge_to_structural_region"
    assert wall["surface_locked"] is True
    assert rail["surface_attachment_status"] == "attached_object_candidate"
    assert rail["surface_locked"] is False


def test_surface_label_and_structural_region_can_override_sparse_pca():
    ground_field = {
        "keys": np.array([0], dtype=np.int64),
        "labels": np.array([field_mod.REGION_GROUND_LIKE], dtype=np.uint8),
        "confidence": np.array([1.0], dtype=np.float32),
        "spec": np.array([0, 0, 0, 10, 10], dtype=np.int64),
        "voxel_size": 1.0,
    }
    sparse_ground = np.array([[x / 200.0, 0.0, 0.0] for x in range(140)], dtype=np.float32)
    args = attach_mod.argparse.Namespace(
        neighbor_radius=0,
        structural_ratio_min=0.35,
        horizontal_normal_z_min=0.86,
        vertical_normal_z_max=0.42,
        surface_planarity_min=0.95,
        surface_thickness_max=0.01,
        large_surface_min_points=1000,
        large_surface_min_extent=10.0,
        surface_label_agreement_ratio=0.75,
        surface_label_min_points=120,
        attached_linearity_min=0.55,
        attached_scattering_min=0.08,
        attached_thickness_min=0.10,
    )

    counts, conf = attach_mod.vote_structural_regions(sparse_ground, ground_field, 0)
    result = attach_mod.classify_attachment({"label": "ground", "cluster_size": len(sparse_ground)}, sparse_ground, counts, conf, args)

    assert result["surface_attachment_status"] == "merge_to_structural_region"
    assert result["surface_attachment_reason"] == "surface candidate label strongly agrees with structural region"


def test_classify_surface_attachment_cli_enriches_targets(tmp_path: Path):
    field_path = tmp_path / "field.npz"
    field_mod.write_field_npz(
        field_path,
        np.array([0], dtype=np.int64),
        np.array([field_mod.REGION_VERTICAL_SURFACE], dtype=np.uint8),
        np.array([1.0], dtype=np.float32),
        np.array([[0, 0, 1, 0, 0]], dtype=np.uint32),
        np.array([0, 0, 0, 10, 10], dtype=np.int64),
        1.0,
    )
    targets = tmp_path / "targets.jsonl"
    targets.write_text(
        json.dumps({"target_index": 1, "target_id": "t1", "label": "railing", "cluster_size": 20}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    target_ply = tmp_path / "targets.ply"
    lines = [
        "ply",
        "format ascii 1.0",
        "element vertex 20",
        "property float x",
        "property float y",
        "property float z",
        "property int target",
        "end_header",
    ]
    lines.extend(f"0.12 {i / 20:.6f} 0.2 1" for i in range(20))
    target_ply.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    report = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "classify_surface_attachment.py"),
            "--targets-jsonl",
            str(targets),
            "--target-ply",
            str(target_ply),
            "--structural-field",
            str(field_path),
            "--output-jsonl",
            str(out),
            "--report",
            str(report),
            "--neighbor-radius",
            "0",
        ],
        check=True,
    )

    row = json.loads(out.read_text(encoding="utf-8").strip())
    summary = json.loads(report.read_text(encoding="utf-8"))
    assert row["surface_attachment_status"] == "attached_object_candidate"
    assert row["dominant_structural_region"] == "vertical_surface_region"
    assert summary["status_counts"]["attached_object_candidate"] == 1
