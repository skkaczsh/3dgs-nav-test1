#!/usr/bin/env python3
"""Convert LAS/LAZ XYZRGB points to the ASCII XYZRGB PLY used by this repo.

The geo patch pipeline consumes ASCII PLY and performs its own feature
construction.  For large scanner exports, writing every raw point as ASCII is
unnecessarily large, so this tool can aggregate directly to a voxel grid and
write one averaged RGB point per occupied voxel.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import laspy
import numpy as np


def color_to_u8(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    if values.size and float(values.max()) > 255.0:
        values = values / 256.0
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def aggregate_las(input_las: Path, voxel_size: float) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with laspy.open(input_las) as reader:
        header = reader.header
        mins = np.asarray(header.mins, dtype=np.float64)
        maxs = np.asarray(header.maxs, dtype=np.float64)
        origin = np.floor(mins / voxel_size).astype(np.int64)
        max_grid = np.floor(maxs / voxel_size).astype(np.int64) - origin + 2
        if np.any(max_grid <= 0):
            raise ValueError(f"invalid LAS bounds: mins={mins} maxs={maxs}")
        ny = int(max_grid[1])
        nz = int(max_grid[2])
        point_count = int(header.point_count)
        dims = set(header.point_format.dimension_names)
        if not {"red", "green", "blue"} <= dims:
            raise ValueError(f"LAS lacks RGB dimensions: {input_las}")
        points = reader.read()

    scale = np.asarray(header.scales, dtype=np.float64)
    offset = np.asarray(header.offsets, dtype=np.float64)
    gx = np.floor((points.X.astype(np.float64) * scale[0] + offset[0]) / voxel_size).astype(np.int64) - origin[0]
    gy = np.floor((points.Y.astype(np.float64) * scale[1] + offset[1]) / voxel_size).astype(np.int64) - origin[1]
    gz = np.floor((points.Z.astype(np.float64) * scale[2] + offset[2]) / voxel_size).astype(np.int64) - origin[2]
    keys = (gx * ny + gy) * nz + gz
    unique_keys, inverse = np.unique(keys, return_inverse=True)

    count = np.bincount(inverse).astype(np.float64)
    sx_raw = np.bincount(inverse, weights=points.X.astype(np.float64))
    sy_raw = np.bincount(inverse, weights=points.Y.astype(np.float64))
    sz_raw = np.bincount(inverse, weights=points.Z.astype(np.float64))
    red = color_to_u8(points.red)
    green = color_to_u8(points.green)
    blue = color_to_u8(points.blue)
    sr = np.bincount(inverse, weights=red)
    sg = np.bincount(inverse, weights=green)
    sb = np.bincount(inverse, weights=blue)
    accum = {
        "keys": unique_keys,
        "count": count,
        "x": sx_raw / count * scale[0] + offset[0],
        "y": sy_raw / count * scale[1] + offset[1],
        "z": sz_raw / count * scale[2] + offset[2],
        "red": sr / count,
        "green": sg / count,
        "blue": sb / count,
    }

    report = {
        "input_las": str(input_las),
        "point_count": point_count,
        "voxel_size": voxel_size,
        "voxel_count": int(len(unique_keys)),
        "bounds": {"min": mins.astype(float).tolist(), "max": maxs.astype(float).tolist()},
        "origin_grid": origin.astype(int).tolist(),
        "grid_shape": max_grid.astype(int).tolist(),
    }
    return accum, report


def write_ascii_ply(output_ply: Path, accum: dict[str, np.ndarray]) -> None:
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(accum["keys"], kind="stable")
    with output_ply.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(order)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uint object\nproperty uchar semantic\n")
        f.write("end_header\n")
        for i in order.tolist():
            f.write(
                f"{float(accum['x'][i]):.6f} {float(accum['y'][i]):.6f} {float(accum['z'][i]):.6f} "
                f"{int(np.clip(round(float(accum['red'][i])), 0, 255))} "
                f"{int(np.clip(round(float(accum['green'][i])), 0, 255))} "
                f"{int(np.clip(round(float(accum['blue'][i])), 0, 255))} 0 0\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-las", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.voxel_size) or args.voxel_size <= 0:
        raise ValueError("--voxel-size must be positive")
    accum, report = aggregate_las(args.input_las, args.voxel_size)
    write_ascii_ply(args.output_ply, accum)
    report["output_ply"] = str(args.output_ply)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
