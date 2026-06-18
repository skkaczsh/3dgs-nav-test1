#!/usr/bin/env python3
"""Filter object PLY points by object status from JSONL metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_ply(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    with path.open("rb") as f:
        fmt = "ascii"
        props: list[str] = []
        prop_types: list[str] = []
        vertex_count = 0
        in_vertex = False
        header_lines = 0
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"Invalid PLY header: {path}")
            header_lines += 1
            line = raw.decode("ascii", errors="ignore").strip()
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "format":
                fmt = parts[1]
            elif len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                vertex_count = int(parts[2])
                in_vertex = True
            elif len(parts) >= 2 and parts[0] == "element":
                in_vertex = False
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                prop_types.append(parts[1])
                props.append(parts[-1])
            elif line == "end_header":
                break

    if fmt == "ascii":
        data = np.loadtxt(path, skiprows=header_lines, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data, props, prop_types

    if fmt != "binary_little_endian":
        raise ValueError(f"Unsupported PLY format: {fmt}")
    type_map = {
        "float": "<f4", "float32": "<f4", "double": "<f8",
        "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
        "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
        "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4",
    }
    dtype = np.dtype([(name, type_map.get(ptype, "<f4")) for ptype, name in zip(prop_types, props)])
    with path.open("rb") as f:
        while f.readline().strip() != b"end_header":
            pass
        arr = np.frombuffer(f.read(vertex_count * dtype.itemsize), dtype=dtype, count=vertex_count)
    data = np.column_stack([arr[name] for name in props]).astype(np.float64)
    return data, props, prop_types


def write_ascii_ply(path: Path, data: np.ndarray, props: list[str], prop_types: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(data)}\n")
        for ptype, name in zip(prop_types, props):
            f.write(f"property {ptype} {name}\n")
        f.write("end_header\n")
        int_like = {"uchar", "uint8", "char", "int8", "ushort", "uint16", "short", "int16", "uint", "uint32", "int", "int32"}
        for row in data:
            vals = []
            for value, ptype in zip(row, prop_types):
                if ptype in int_like:
                    vals.append(str(int(round(float(value)))))
                else:
                    vals.append(f"{float(value):.6f}")
            f.write(" ".join(vals) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--include-status", action="append", default=[])
    parser.add_argument("--exclude-status", action="append", default=[])
    args = parser.parse_args()

    data, props, prop_types = read_ply(args.input_ply)
    object_field = "object" if "object" in props else "object_id"
    object_col = props.index(object_field)

    keep_status = set(args.include_status)
    drop_status = set(args.exclude_status)
    keep_ids = set()
    kept_objects = []
    with args.objects_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            status = obj.get("status", "")
            if keep_status and status not in keep_status:
                continue
            if drop_status and status in drop_status:
                continue
            keep_ids.add(int(obj["object_id"]))
            kept_objects.append(obj)

    object_ids = data[:, object_col].astype(np.uint32)
    keep_mask = np.isin(object_ids, np.array(sorted(keep_ids), dtype=np.uint32))
    write_ascii_ply(args.output_ply, data[keep_mask], props, prop_types)

    if args.output_jsonl:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w", encoding="utf-8") as f:
            for obj in kept_objects:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(json.dumps({
        "input_points": int(len(data)),
        "output_points": int(keep_mask.sum()),
        "input_objects": int(len(np.unique(object_ids[object_ids > 0]))),
        "output_objects": int(len(keep_ids)),
        "output_ply": str(args.output_ply),
        "output_jsonl": str(args.output_jsonl) if args.output_jsonl else None,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
