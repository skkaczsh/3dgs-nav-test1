#!/usr/bin/env python3
"""Convert the project viewer ASCII PLY format to LAS for PotreeConverter.

The existing semantic/object viewer PLY files are ASCII and usually contain:
  x y z red green blue object semantic

PotreeConverter 2.x consumes LAS/LAZ, not these ASCII PLY files. This script
keeps XYZ and RGB only. Object/semantic inspection should be baked into RGB
before conversion, e.g. random object-color PLY or semantic-color PLY.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import laspy
import numpy as np


def read_ply_header(path: Path) -> tuple[int, list[str], int]:
    with path.open("rb") as f:
        first = f.readline().decode("utf-8", "replace").strip()
        if first != "ply":
            raise ValueError(f"not a PLY file: {path}")
        vertex_count = None
        properties: list[str] = []
        in_vertex = False
        header_bytes = len(first) + 1
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"PLY header missing end_header: {path}")
            header_bytes += len(raw)
            line = raw.decode("utf-8", "replace").strip()
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format" and parts[1] != "ascii":
                raise ValueError(f"only ascii PLY is supported: {path}")
            if parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif in_vertex and parts[0] == "property":
                properties.append(parts[-1])
            elif line == "end_header":
                break
    if vertex_count is None:
        raise ValueError(f"PLY header missing vertex element: {path}")
    return vertex_count, properties, header_bytes


def load_ply_xyzrgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertex_count, properties, header_bytes = read_ply_header(path)
    required = ["x", "y", "z"]
    missing = [name for name in required if name not in properties]
    if missing:
        raise ValueError(f"PLY missing properties {missing}: {path}")
    idx = {name: i for i, name in enumerate(properties)}
    usecols = [idx["x"], idx["y"], idx["z"]]
    color_cols = [idx.get("red"), idx.get("green"), idx.get("blue")]
    has_rgb = all(i is not None for i in color_cols)
    if has_rgb:
        usecols.extend(int(i) for i in color_cols if i is not None)

    with path.open("rb") as f:
        f.seek(header_bytes)
        data = np.loadtxt(f, max_rows=vertex_count, dtype=np.float64, usecols=usecols, ndmin=2)

    xyz = data[:, :3].astype(np.float64, copy=False)
    if has_rgb:
        rgb = np.clip(data[:, 3:6], 0, 255).astype(np.uint16)
    else:
        rgb = np.full((xyz.shape[0], 3), 220, dtype=np.uint16)
    return xyz, rgb


def write_las(xyz: np.ndarray, rgb8: np.ndarray, output: Path, scale: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    mins = xyz.min(axis=0)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([scale, scale, scale], dtype=np.float64)
    header.offsets = mins

    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    # LAS RGB is 16-bit. Replicate 8-bit values into high/low bits.
    rgb16 = (rgb8.astype(np.uint16) << 8) | rgb8.astype(np.uint16)
    las.red = rgb16[:, 0]
    las.green = rgb16[:, 1]
    las.blue = rgb16[:, 2]
    las.write(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-las", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.001, help="LAS coordinate scale in meters")
    args = parser.parse_args()

    xyz, rgb = load_ply_xyzrgb(args.input_ply)
    write_las(xyz, rgb, args.output_las, args.scale)
    print({
        "input_ply": str(args.input_ply),
        "output_las": str(args.output_las),
        "points": int(xyz.shape[0]),
        "scale": args.scale,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
