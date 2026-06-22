#!/usr/bin/env python3
"""Convert a binary XYZRGB PLY into the ASCII PLY schema used by the viewer."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_header(path: Path) -> tuple[list[str], int, str, int]:
    props: list[str] = []
    vertex_count = 0
    ply_format = ""
    header_bytes = 0
    in_vertex = False
    with path.open("rb") as f:
        for raw_line in f:
            header_bytes += len(raw_line)
            line = raw_line.decode("utf-8", errors="replace")
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0] == "format":
                ply_format = parts[1]
            elif len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                vertex_count = int(parts[2])
                in_vertex = True
            elif len(parts) >= 2 and parts[0] == "element":
                in_vertex = False
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                props.append(parts[-1])
            if line.strip() == "end_header":
                break
    return props, vertex_count, ply_format, header_bytes


def dtype_for_props(props: list[str]) -> np.dtype:
    fields: list[tuple[str, str]] = []
    for name in props:
        if name in {"x", "y", "z"}:
            fields.append((name, "<f4"))
        elif name in {"red", "green", "blue"}:
            fields.append((name, "u1"))
        else:
            raise ValueError(f"unsupported vertex property: {name}")
    return np.dtype(fields)


def export_ascii(
    input_ply: Path,
    output_ply: Path,
    stride: int,
    max_points: int | None,
    object_id: int,
    semantic_id: int,
) -> None:
    props, vertex_count, ply_format, header_bytes = parse_header(input_ply)
    if ply_format != "binary_little_endian":
        raise ValueError(f"expected binary_little_endian PLY, got {ply_format}: {input_ply}")
    for required in ("x", "y", "z", "red", "green", "blue"):
        if required not in props:
            raise ValueError(f"PLY missing {required}: {input_ply}")
    if stride < 1:
        raise ValueError("--stride must be >= 1")

    data = np.memmap(input_ply, dtype=dtype_for_props(props), mode="r", offset=header_bytes, shape=(vertex_count,))
    selected = data[::stride]
    if max_points is not None:
        selected = selected[:max_points]
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    with output_ply.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(selected)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uint object\nproperty uchar semantic\n")
        f.write("end_header\n")
        for row in selected:
            f.write(
                f"{float(row['x']):.6f} {float(row['y']):.6f} {float(row['z']):.6f} "
                f"{int(row['red'])} {int(row['green'])} {int(row['blue'])} {object_id} {semantic_id}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--object-id", type=int, default=0)
    parser.add_argument("--semantic-id", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_ascii(args.input_ply, args.output_ply, args.stride, args.max_points, args.object_id, args.semantic_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
