#!/usr/bin/env python3
"""Create a small, geometry-stratified Superpoint evidence sample."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def supported_ids(path: Path) -> set[int]:
    return {
        int(row["object_id"])
        for row in read_jsonl(path)
        if row.get("top_source_frames")
    }


def select_rows(rows: list[dict[str, Any]], support: set[int], per_geometry: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["object_id"]) in support:
            groups[str(row.get("geometry_type") or "unknown")].append(row)
    rng = np.random.default_rng(seed)
    selected = []
    for geometry in sorted(groups):
        candidates = groups[geometry]
        take = min(per_geometry, len(candidates))
        if take:
            selected.extend(candidates[i] for i in rng.choice(len(candidates), size=take, replace=False))
    return sorted(selected, key=lambda row: int(row["object_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-ply", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--source-support", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-geometry", type=int, default=20)
    parser.add_argument("--max-points-per-object", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    support = supported_ids(args.source_support)
    selected = select_rows(read_jsonl(args.objects_jsonl), support, args.per_geometry, args.seed)
    selected_ids = np.array([int(row["object_id"]) for row in selected], dtype=np.int32)
    labels = np.load(args.labels).astype(np.int32, copy=False)
    vertex = PlyData.read(str(args.reference_ply))["vertex"].data
    if len(vertex) != len(labels):
        raise ValueError(f"reference/label count mismatch: {len(vertex)} != {len(labels)}")

    rng = np.random.default_rng(args.seed)
    chunks = []
    for object_id in selected_ids:
        indices = np.flatnonzero(labels == object_id)
        if len(indices) > args.max_points_per_object:
            indices = rng.choice(indices, size=args.max_points_per_object, replace=False)
        sample = np.empty(len(indices), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("object", "i4")])
        sample["x"], sample["y"], sample["z"] = vertex["x"][indices], vertex["y"][indices], vertex["z"][indices]
        sample["object"] = object_id
        chunks.append(sample)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(np.concatenate(chunks) if chunks else np.empty(0, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("object", "i4")]), "vertex")], text=True).write(str(output / "object_samples.ply"))
    with (output / "objects.jsonl").open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"selected_objects": len(selected), "per_geometry": args.per_geometry, "supported_pool": len(support), "sample_points": int(sum(len(chunk) for chunk in chunks))}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
