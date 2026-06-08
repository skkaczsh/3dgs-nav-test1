#!/usr/bin/env python3
"""Fast voxel downsample for XYZRGB ASCII PLY.

The existing merge script can produce large ASCII PLY files quickly when
voxel_size=0. This tool vectorizes voxel aggregation with numpy and writes a
binary_little_endian PLY, avoiding slow Python per-voxel loops.
"""

import argparse
import os
import time

import numpy as np


def read_ascii_xyzrgb_ply(path):
    vertex_count = None
    header_lines = 0
    with open(path, "rb") as f:
        for raw in f:
            header_lines += 1
            line = raw.decode("utf-8", errors="replace").strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line == "end_header":
                break
    data = np.loadtxt(path, skiprows=header_lines, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if vertex_count is not None and len(data) != vertex_count:
        raise ValueError(f"Vertex count mismatch: header={vertex_count}, data={len(data)}")
    if data.shape[1] < 6:
        raise ValueError("Expected at least XYZRGB columns")
    points = data[:, :3].astype(np.float32, copy=False)
    colors = np.clip(data[:, 3:6], 0, 255).astype(np.float32, copy=False)
    return points, colors


def voxel_downsample(points, colors, voxel_size):
    if voxel_size <= 0:
        return points, colors.astype(np.uint8)

    mins = points.min(axis=0)
    ijk = np.floor((points - mins) / voxel_size).astype(np.int64)
    dims = ijk.max(axis=0) + 1
    keys = np.ravel_multi_index((ijk[:, 0], ijk[:, 1], ijk[:, 2]), tuple(dims))

    unique, inverse = np.unique(keys, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float32)

    out_points = np.empty((len(unique), 3), dtype=np.float32)
    out_colors = np.empty((len(unique), 3), dtype=np.float32)
    for c in range(3):
        out_points[:, c] = np.bincount(inverse, weights=points[:, c]) / counts
        out_colors[:, c] = np.bincount(inverse, weights=colors[:, c]) / counts

    return out_points, np.clip(out_colors, 0, 255).astype(np.uint8)


def write_binary_xyzrgb_ply(path, points, colors):
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    data = np.empty(len(points), dtype=dtype)
    data["x"] = points[:, 0]
    data["y"] = points[:, 1]
    data["z"] = points[:, 2]
    data["red"] = colors[:, 0]
    data["green"] = colors[:, 1]
    data["blue"] = colors[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--voxel-size", type=float, default=0.04)
    args = parser.parse_args()

    t0 = time.time()
    points, colors = read_ascii_xyzrgb_ply(args.input)
    t_read = time.time()
    out_points, out_colors = voxel_downsample(points, colors, args.voxel_size)
    t_voxel = time.time()
    write_binary_xyzrgb_ply(args.output, out_points, out_colors)
    t_write = time.time()

    print(f"input_points={len(points)}")
    print(f"output_points={len(out_points)}")
    print(f"reduction={100 * (1 - len(out_points) / max(len(points), 1)):.2f}%")
    print(f"read_sec={t_read - t0:.2f}")
    print(f"voxel_sec={t_voxel - t_read:.2f}")
    print(f"write_sec={t_write - t_voxel:.2f}")
    print(f"total_sec={t_write - t0:.2f}")
    print(f"output={args.output}")
    print(f"output_mb={os.path.getsize(args.output) / 1024 / 1024:.1f}")


if __name__ == "__main__":
    main()
