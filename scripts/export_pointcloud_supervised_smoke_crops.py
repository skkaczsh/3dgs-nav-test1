#!/usr/bin/env python3
"""Export ASCII PLY crops from the supervised smoke manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.optimize_patch_graph_energy import read_labels, read_region_input, write_labels


DEFAULT_MANIFEST = Path("docs/pointcloud_supervised_baseline_smoke_manifest_20260708.json")
DEFAULT_OUTPUT_DIR = Path("server_parking_priority_s10/pointcloud_supervised_baseline_smoke_crops_20260708")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def parse_binary_xyzrgb_header(path: Path) -> tuple[int, int, int]:
    vertex_count = 0
    props: list[tuple[str, str]] = []
    offset = 0
    in_vertex = False
    with path.open("rb") as fh:
        while True:
            raw = fh.readline()
            if not raw:
                raise ValueError(f"invalid PLY header: {path}")
            offset += len(raw)
            line = raw.decode("ascii", errors="ignore").strip()
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "format" and parts[1] != "binary_little_endian":
                raise ValueError(f"expected binary_little_endian PLY: {path}")
            if len(parts) >= 3 and parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                in_vertex = True
            elif len(parts) >= 2 and parts[0] == "element":
                in_vertex = False
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                props.append((parts[1], parts[2]))
            elif line == "end_header":
                break
    if props != [("float", "x"), ("float", "y"), ("float", "z"), ("uchar", "red"), ("uchar", "green"), ("uchar", "blue")]:
        raise ValueError(f"expected XYZRGB binary PLY properties: {props}")
    return vertex_count, offset, 15


def write_crop(path: Path, rows: list[tuple[float, float, float, int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(rows)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write("end_header\n")
        for x, y, z, r, g, b in rows:
            fh.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_from_dense_ply(dense_ply: Path, bboxes: list[tuple[list[float], list[float]]]) -> list[list[tuple[float, float, float, int, int, int]]]:
    rows: list[list[tuple[float, float, float, int, int, int]]] = [[] for _ in bboxes]
    vertex_count, header_bytes, row_size = parse_binary_xyzrgb_header(dense_ply)
    with dense_ply.open("rb") as fh:
        fh.seek(header_bytes)
        for _ in range(vertex_count):
            raw = fh.read(row_size)
            if len(raw) != row_size:
                raise ValueError(f"truncated PLY row in {dense_ply}")
            x, y, z, r, g, b = struct.unpack("<fffBBB", raw)
            for idx, (mn, mx) in enumerate(bboxes):
                if mn[0] <= x <= mx[0] and mn[1] <= y <= mx[1] and mn[2] <= z <= mx[2]:
                    rows[idx].append((x, y, z, r, g, b))
    return rows


def collect_from_region_input(
    region_input: Path,
    bboxes: list[tuple[list[float], list[float]]],
    labels_path: Path | None,
) -> tuple[list[list[tuple[float, float, float, int, int, int]]], list[list[int]] | None]:
    arrays, _, _ = read_region_input(region_input)
    xyz = arrays["xyz"]
    rgb = np.clip(arrays["rgb"], 0, 255).astype(np.uint8, copy=False)
    labels = read_labels(labels_path) if labels_path else None
    if labels is not None and len(labels) != len(xyz):
        raise ValueError(f"label count {len(labels)} != region point count {len(xyz)}")
    rows: list[list[tuple[float, float, float, int, int, int]]] = [[] for _ in bboxes]
    crop_labels: list[list[int]] | None = [[] for _ in bboxes] if labels is not None else None
    for point_idx, ((x, y, z), (r, g, b)) in enumerate(zip(xyz, rgb, strict=True)):
        for crop_idx, (mn, mx) in enumerate(bboxes):
            if mn[0] <= x <= mx[0] and mn[1] <= y <= mx[1] and mn[2] <= z <= mx[2]:
                rows[crop_idx].append((float(x), float(y), float(z), int(r), int(g), int(b)))
                if crop_labels is not None:
                    crop_labels[crop_idx].append(int(labels[point_idx]))
    return rows, crop_labels


def export_crops(
    manifest: dict[str, Any],
    output_dir: Path,
    labels_path: Path | None = None,
    region_input: Path | None = None,
) -> dict[str, Any]:
    dense_ply = Path(manifest["dense_input"]["ply"])
    crops = manifest["crops"]
    bboxes = [(crop["bbox_3d"]["min"], crop["bbox_3d"]["max"]) for crop in crops]
    if region_input:
        rows, crop_labels = collect_from_region_input(region_input, bboxes, labels_path)
    else:
        rows = collect_from_dense_ply(dense_ply, bboxes)
        crop_labels = None

    crop_reports: list[dict[str, Any]] = []
    for crop, crop_rows in zip(crops, rows):
        ply = output_dir / f"{crop['id']}.ply"
        write_crop(ply, crop_rows)
        expected = int(crop.get("dense_voxel_count_in_crop", -1))
        item = {
            "id": crop["id"],
            "geometry_type": crop["geometry_type"],
            "output_ply": str(ply),
            "sha256": sha256_file(ply),
            "point_count": len(crop_rows),
            "manifest_point_count": expected,
            "count_matches_manifest": expected == len(crop_rows),
        }
        if crop_labels is not None:
            label_path = output_dir / f"{crop['id']}_labels.bin"
            write_labels(label_path, np.array(crop_labels[len(crop_reports)], dtype=np.int32))
            item["output_labels"] = str(label_path)
        crop_reports.append(item)
    return {
        "schema": "pointcloud-supervised-smoke-crop-export/v1",
        "manifest": manifest.get("schema"),
        "dense_ply": str(dense_ply),
        "labels": str(labels_path) if labels_path else None,
        "region_input": str(region_input) if region_input else None,
        "output_dir": str(output_dir),
        "crop_count": len(crop_reports),
        "crops": crop_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--region-input", type=Path)
    args = parser.parse_args()
    report = export_crops(load_json(args.manifest), args.output_dir, args.labels, args.region_input)
    report_path = args.output_dir / "crop_export_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "crop_count": report["crop_count"]}, ensure_ascii=False, indent=2))
    if not all(crop["count_matches_manifest"] for crop in report["crops"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
