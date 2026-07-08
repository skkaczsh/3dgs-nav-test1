#!/usr/bin/env python3
"""Build fixed crops for supervised point-cloud baseline smoke tests."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any


DEFAULT_DENSE_PLY = Path("/Users/skkac/Work/SCAN/work_MT20260616-175807/dense_sources/dense_las_voxel003_local/dense_las_voxel003_binary.ply")
DEFAULT_DENSE_REPORT = Path("/Users/skkac/Work/SCAN/work_MT20260616-175807/dense_sources/dense_las_voxel003_local/report.json")
DEFAULT_PATCH_JSONL = Path("server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v7_uncertain_guard_20260708_191958/superpoint_graph_v1.jsonl")
DEFAULT_RISK_REPORT = Path("server_parking_priority_s10/geo_patch_las_opt_cpp_v2_voxel003_r4_4090d_20260623/superpoint_graph_v7_uncertain_guard_20260708_191958/risk_70503_9366_local_qa/risk_70503_9366_report.json")
DEFAULT_OUTPUT = Path("docs/pointcloud_supervised_baseline_smoke_manifest_20260708.json")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def expand_bbox(bbox: dict[str, list[float]], margin: float) -> dict[str, list[float]]:
    return {
        "min": [float(v) - margin for v in bbox["min"]],
        "max": [float(v) + margin for v in bbox["max"]],
    }


def local_bbox(row: dict[str, Any], half_xy: float, half_z: float) -> dict[str, list[float]]:
    cx, cy, cz = [float(v) for v in row["centroid"]]
    source = row["bbox_3d"]
    return {
        "min": [
            max(float(source["min"][0]), cx - half_xy),
            max(float(source["min"][1]), cy - half_xy),
            max(float(source["min"][2]), cz - half_z),
        ],
        "max": [
            min(float(source["max"][0]), cx + half_xy),
            min(float(source["max"][1]), cy + half_xy),
            min(float(source["max"][2]), cz + half_z),
        ],
    }


def select_largest_patches(path: Path, geometry_types: list[str], half_xy: float, half_z: float) -> list[dict[str, Any]]:
    wanted = set(geometry_types)
    best: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            geom = str(row.get("geometry_type"))
            if geom not in wanted:
                continue
            if geom not in best or int(row.get("voxel_count", 0)) > int(best[geom].get("voxel_count", 0)):
                best[geom] = row
    crops: list[dict[str, Any]] = []
    for geom in geometry_types:
        row = best.get(geom)
        if not row:
            continue
        patch_id = int(row.get("patch_id", row.get("object", 0)))
        crops.append(
            {
                "id": f"largest_{geom}_{patch_id}",
                "source": "superpoint_graph_v7_largest_patch",
                "geometry_type": geom,
                "patch_id": patch_id,
                "source_voxel_count": int(row.get("voxel_count", 0)),
                "centroid": row.get("centroid"),
                "bbox_3d": local_bbox(row, half_xy, half_z),
            }
        )
    return crops


def read_binary_xyz_count(path: Path, crops: list[dict[str, Any]]) -> int:
    with path.open("rb") as fh:
        vertex_count = 0
        props: list[tuple[str, str]] = []
        in_vertex = False
        while True:
            raw = fh.readline()
            if not raw:
                raise ValueError(f"invalid PLY header: {path}")
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
        if [name for _, name in props[:3]] != ["x", "y", "z"]:
            raise ValueError(f"expected x/y/z as first PLY properties: {path}")
        type_sizes = {"float": 4, "uchar": 1}
        row_size = sum(type_sizes.get(ptype, 0) for ptype, _ in props)
        if row_size <= 0:
            raise ValueError(f"unsupported PLY properties: {props}")
        counts = [0 for _ in crops]
        bboxes = [(crop["bbox_3d"]["min"], crop["bbox_3d"]["max"]) for crop in crops]
        chunk_rows = 200_000
        total = 0
        while total < vertex_count:
            take = min(chunk_rows, vertex_count - total)
            data = fh.read(take * row_size)
            for i in range(take):
                offset = i * row_size
                x, y, z = struct.unpack_from("<fff", data, offset)
                for idx, (mn, mx) in enumerate(bboxes):
                    if mn[0] <= x <= mx[0] and mn[1] <= y <= mx[1] and mn[2] <= z <= mx[2]:
                        counts[idx] += 1
            total += take
    for crop, count in zip(crops, counts):
        crop["dense_voxel_count_in_crop"] = count
    return vertex_count


def build(args: argparse.Namespace) -> dict[str, Any]:
    dense_report = load_json(args.dense_report)
    crops = select_largest_patches(args.patch_jsonl, args.geometry_types, args.crop_half_xy, args.crop_half_z)
    if args.risk_report.exists():
        risk = load_json(args.risk_report)
        crops.append(
            {
                "id": "risk_70503_9366_local",
                "source": "superpoint_graph_v7_known_overlap_risk",
                "geometry_type": "mixed_risk",
                "patch_ids": risk.get("risk_pair", [70503, 9366]),
                "bbox_3d": {"min": risk["bbox_min"], "max": risk["bbox_max"]},
            }
        )
    point_count = read_binary_xyz_count(args.dense_ply, crops) if args.count_points else None
    return {
        "schema": "pointcloud-supervised-baseline-smoke-manifest/v1",
        "status": "ready",
        "contract": "docs/pointcloud_supervised_baseline_smoke_20260708.json",
        "dense_input": {
            "id": "dense_las_voxel003_binary",
            "ply": str(args.dense_ply),
            "report": str(args.dense_report),
            "voxel_size_m": float(dense_report["voxel_size"]),
            "voxel_count": int(dense_report["voxel_count"]),
            "scanned_point_count": point_count,
        },
        "crop_count": len(crops),
        "crops": crops,
        "runner_contract": {
            "crop_input": "read dense_input.ply and keep xyz within crop.bbox_3d",
            "allowed_outputs": [
                "per_crop_predictions",
                "per_voxel_logits_or_labels",
                "per_patch_vote_summary",
                "domain_gap_report",
            ],
            "forbidden_outputs": [
                "new_patch_boundaries",
                "merged_objects",
                "overwritten_geometry_owner",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-ply", type=Path, default=DEFAULT_DENSE_PLY)
    parser.add_argument("--dense-report", type=Path, default=DEFAULT_DENSE_REPORT)
    parser.add_argument("--patch-jsonl", type=Path, default=DEFAULT_PATCH_JSONL)
    parser.add_argument("--risk-report", type=Path, default=DEFAULT_RISK_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--geometry-types", nargs="+", default=["horizontal", "vertical", "rough_mixed", "thin_linear"])
    parser.add_argument("--crop-half-xy", type=float, default=6.0)
    parser.add_argument("--crop-half-z", type=float, default=4.0)
    parser.add_argument("--count-points", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    manifest = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "crop_count": manifest["crop_count"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
