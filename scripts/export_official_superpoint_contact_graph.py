#!/usr/bin/env python3
"""Export only true 6-neighbor contacts between immutable Superpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

try:
    from scripts.propose_geo_patch_object_merges import build_edge_counts, build_grid6_edges
except ModuleNotFoundError:  # Supports direct `python scripts/...` execution.
    from propose_geo_patch_object_merges import build_edge_counts, build_grid6_edges


def read_xyz(path: Path) -> np.ndarray:
    vertex = PlyData.read(str(path))["vertex"].data
    return np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32, copy=False)


def contact_rows(
    xyz: np.ndarray,
    labels: np.ndarray,
    voxel_size: float,
    min_shared_faces: int,
) -> tuple[list[dict[str, int | float]], dict[str, int]]:
    if len(xyz) != len(labels):
        raise ValueError(f"xyz/label count mismatch: {len(xyz)} != {len(labels)}")
    src, dst = build_grid6_edges({"xyz": xyz}, voxel_size)
    contacts = build_edge_counts(labels, src, dst)
    counts = np.bincount(labels.astype(np.int64, copy=False))
    rows = []
    for (object_a, object_b), shared_faces in sorted(contacts.items()):
        if shared_faces < min_shared_faces:
            continue
        smaller = max(1, min(int(counts[object_a]), int(counts[object_b])))
        rows.append({
            "object_a": object_a,
            "object_b": object_b,
            "shared_voxel_faces": shared_faces,
            "contact_ratio_min": shared_faces / smaller,
        })
    return rows, {
        "directed_voxel_edges": int(len(src)),
        "cross_superpoint_pairs": int(len(contacts)),
        "kept_contact_pairs": int(len(rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-ply", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--min-shared-faces", type=int, default=1)
    args = parser.parse_args()

    xyz = read_xyz(args.reference_ply)
    labels = np.load(args.labels).astype(np.int32, copy=False)
    rows, stats = contact_rows(xyz, labels, args.voxel_size, args.min_shared_faces)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "reference_ply": str(args.reference_ply),
        "labels": str(args.labels),
        "voxel_size": args.voxel_size,
        "min_shared_faces": args.min_shared_faces,
        "points": int(len(xyz)),
        "superpoints": int(labels.max()) + 1 if len(labels) else 0,
        **stats,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
