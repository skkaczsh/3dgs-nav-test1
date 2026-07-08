#!/usr/bin/env python3
"""Convert LAS/LAZ XYZRGB points to the XYZRGB PLY used by this repo.

The geo patch pipeline consumes XYZRGB PLY and performs its own feature
construction.  For large scanner exports, writing every raw point is
unnecessarily large, so this tool aggregates directly to a voxel grid and
writes one averaged RGB point per occupied voxel.
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


def _aggregate_arrays(keys: np.ndarray, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    out: dict[str, np.ndarray] = {"keys": unique_keys, "count": count}
    for name, arr in values.items():
        out[name] = np.bincount(inverse, weights=arr.astype(np.float64, copy=False))
    return out


def aggregate_las(input_las: Path, voxel_size: float, chunk_size: int = 5_000_000) -> tuple[dict[str, np.ndarray], dict[str, object]]:
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
        partials: list[dict[str, np.ndarray]] = []
        for points in reader.chunk_iterator(chunk_size):
            x = np.asarray(points.x, dtype=np.float64)
            y = np.asarray(points.y, dtype=np.float64)
            z = np.asarray(points.z, dtype=np.float64)
            gx = np.floor(x / voxel_size).astype(np.int64) - origin[0]
            gy = np.floor(y / voxel_size).astype(np.int64) - origin[1]
            gz = np.floor(z / voxel_size).astype(np.int64) - origin[2]
            keys = (gx * ny + gy) * nz + gz
            partials.append(
                _aggregate_arrays(
                    keys,
                    {
                        "x": x,
                        "y": y,
                        "z": z,
                        "red": color_to_u8(points.red).astype(np.float64, copy=False),
                        "green": color_to_u8(points.green).astype(np.float64, copy=False),
                        "blue": color_to_u8(points.blue).astype(np.float64, copy=False),
                    },
                )
            )

    if not partials:
        raise ValueError(f"LAS contains no points: {input_las}")
    keys = np.concatenate([part["keys"] for part in partials])
    values = {
        name: np.concatenate([part[name] for part in partials])
        for name in ("count", "x", "y", "z", "red", "green", "blue")
    }
    merged = _aggregate_arrays(keys, values)
    count = merged["count"]
    accum = {
        "keys": merged["keys"],
        "count": count,
        "x": merged["x"] / count,
        "y": merged["y"] / count,
        "z": merged["z"] / count,
        "red": merged["red"] / count,
        "green": merged["green"] / count,
        "blue": merged["blue"] / count,
    }

    report = {
        "input_las": str(input_las),
        "point_count": point_count,
        "voxel_size": voxel_size,
        "voxel_count": int(len(merged["keys"])),
        "bounds": {"min": mins.astype(float).tolist(), "max": maxs.astype(float).tolist()},
        "origin_grid": origin.astype(int).tolist(),
        "grid_shape": max_grid.astype(int).tolist(),
        "chunk_size": int(chunk_size),
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


def write_binary_ply(output_ply: Path, accum: dict[str, np.ndarray]) -> None:
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(accum["keys"], kind="stable")
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    rows = np.empty(len(order), dtype=dtype)
    rows["x"] = accum["x"][order].astype(np.float32, copy=False)
    rows["y"] = accum["y"][order].astype(np.float32, copy=False)
    rows["z"] = accum["z"][order].astype(np.float32, copy=False)
    rows["red"] = np.clip(np.rint(accum["red"][order]), 0, 255).astype(np.uint8, copy=False)
    rows["green"] = np.clip(np.rint(accum["green"][order]), 0, 255).astype(np.uint8, copy=False)
    rows["blue"] = np.clip(np.rint(accum["blue"][order]), 0, 255).astype(np.uint8, copy=False)
    with output_ply.open("wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(rows)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        rows.tofile(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-las", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.10)
    parser.add_argument("--chunk-size", type=int, default=5_000_000)
    parser.add_argument("--output-format", choices=("ascii", "binary_little_endian"), default="ascii")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.voxel_size) or args.voxel_size <= 0:
        raise ValueError("--voxel-size must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    accum, report = aggregate_las(args.input_las, args.voxel_size, args.chunk_size)
    if args.output_format == "binary_little_endian":
        write_binary_ply(args.output_ply, accum)
    else:
        write_ascii_ply(args.output_ply, accum)
    report["output_ply"] = str(args.output_ply)
    report["output_format"] = args.output_format
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
