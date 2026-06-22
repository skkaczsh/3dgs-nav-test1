#!/usr/bin/env python3
"""Write every Nth vertex from an ASCII PLY while preserving all fields."""

from __future__ import annotations

import argparse
from pathlib import Path


def read_header(path: Path) -> tuple[list[str], int, int]:
    header: list[str] = []
    vertex_count = 0
    header_lines = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header_lines += 1
            stripped = line.strip()
            if stripped.startswith("format ") and stripped.split()[1] != "ascii":
                raise ValueError(f"Only ASCII PLY is supported: {path}")
            if stripped.startswith("element vertex"):
                vertex_count = int(stripped.split()[-1])
                header.append("element vertex __VERTEX_COUNT__\n")
            else:
                header.append(line)
            if stripped == "end_header":
                break
    if vertex_count <= 0:
        raise ValueError(f"No vertex count found: {path}")
    return header, vertex_count, header_lines


def stride_ascii_ply(input_ply: Path, output_ply: Path, stride: int) -> None:
    if stride <= 0:
        raise ValueError("--stride must be positive")
    header, vertex_count, header_lines = read_header(input_ply)
    kept = (vertex_count + stride - 1) // stride
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    with input_ply.open("r", encoding="utf-8", errors="replace") as src, output_ply.open("w", encoding="utf-8") as dst:
        for _ in range(header_lines):
            next(src)
        for line in header:
            dst.write(line.replace("__VERTEX_COUNT__", str(kept)))
        written = 0
        for i, line in enumerate(src):
            if i % stride == 0:
                dst.write(line)
                written += 1
    if written != kept:
        raise RuntimeError(f"stride count mismatch: expected={kept} written={written}")
    print(f"input_vertices={vertex_count}")
    print(f"output_vertices={written}")
    print(f"stride={stride}")
    print(f"output={output_ply}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--input-ply", type=Path)
    parser.add_argument("--output-ply", type=Path)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()
    args.input = args.input_ply or args.input
    args.output = args.output_ply or args.output
    if args.input is None or args.output is None:
        parser.error("provide input/output positional args or --input-ply/--output-ply")
    return args


def main() -> None:
    args = parse_args()
    stride_ascii_ply(args.input, args.output, args.stride)


if __name__ == "__main__":
    main()
