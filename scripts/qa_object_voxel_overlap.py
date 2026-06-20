#!/usr/bin/env python3
"""Measure object/semantic voxel overlap in a viewer PLY.

The exported viewer PLY assigns one object and one semantic label per point, so
point rows themselves do not overlap.  Structural leakage shows up when
different objects or semantic labels occupy the same coarse voxel.  This script
reports high voxel intersections as a cheap geometry-fusion invariant.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


LABELS = {
    0: "unknown",
    1: "other",
    2: "wall",
    3: "floor",
    4: "ceiling",
    5: "grass",
    6: "tree",
    7: "person",
    8: "car",
    9: "railing",
    10: "building",
    12: "road",
    14: "furniture",
    15: "pipe",
    16: "equipment",
    17: "fine_candidate",
    18: "stair",
    19: "indoor_floor",
    20: "roof",
}


def parse_ascii_ply(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        if f.readline().strip() != "ply":
            raise ValueError(f"not a PLY file: {path}")
        props: list[str] = []
        in_vertex = False
        for line in f:
            line = line.strip()
            if line.startswith("element "):
                parts = line.split()
                in_vertex = len(parts) >= 3 and parts[1] == "vertex"
            elif in_vertex and line.startswith("property "):
                props.append(line.split()[-1])
            elif line == "end_header":
                break
        rows = [line.split() for line in f if line.strip()]
    return props, rows


def voxel_key(row: list[str], idx: dict[str, int], voxel_size: float) -> tuple[int, int, int]:
    return (
        int(float(row[idx["x"]]) // voxel_size),
        int(float(row[idx["y"]]) // voxel_size),
        int(float(row[idx["z"]]) // voxel_size),
    )


def top_pair_overlaps(
    voxel_members: dict[tuple[int, int, int], set[Any]],
    member_voxel_counts: Counter[Any],
    max_pairs: int,
) -> list[dict[str, Any]]:
    pair_counts: Counter[tuple[Any, Any]] = Counter()
    for members in voxel_members.values():
        if len(members) < 2:
            continue
        for a, b in combinations(sorted(members, key=str), 2):
            pair_counts[(a, b)] += 1

    rows: list[dict[str, Any]] = []
    for (a, b), intersection in pair_counts.most_common(max_pairs * 20):
        av = int(member_voxel_counts[a])
        bv = int(member_voxel_counts[b])
        union = av + bv - int(intersection)
        min_side = max(min(av, bv), 1)
        rows.append(
            {
                "a": a,
                "b": b,
                "intersection_voxels": int(intersection),
                "a_voxels": av,
                "b_voxels": bv,
                "jaccard": float(intersection / max(union, 1)),
                "intersection_over_min": float(intersection / min_side),
            }
        )
    rows.sort(key=lambda r: (r["intersection_over_min"], r["intersection_voxels"]), reverse=True)
    return rows[:max_pairs]


def measure_overlap(path: Path, voxel_size: float, max_pairs: int) -> dict[str, Any]:
    props, rows = parse_ascii_ply(path)
    idx = {name: i for i, name in enumerate(props)}
    for required in ("x", "y", "z", "object", "semantic"):
        if required not in idx:
            raise ValueError(f"PLY missing property {required}: {path}")

    object_voxels: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    semantic_voxels: dict[tuple[int, int, int], set[str]] = defaultdict(set)
    object_counts: Counter[int] = Counter()
    semantic_counts: Counter[str] = Counter()

    for row in rows:
        key = voxel_key(row, idx, voxel_size)
        obj = int(float(row[idx["object"]]))
        semantic_id = int(float(row[idx["semantic"]]))
        label = LABELS.get(semantic_id, f"id_{semantic_id}")
        object_voxels[key].add(obj)
        semantic_voxels[key].add(label)

    for members in object_voxels.values():
        for obj in members:
            object_counts[obj] += 1
    for members in semantic_voxels.values():
        for label in members:
            semantic_counts[label] += 1

    mixed_object_voxels = sum(1 for members in object_voxels.values() if len(members) > 1)
    mixed_semantic_voxels = sum(1 for members in semantic_voxels.values() if len(members) > 1)
    total_voxels = len(object_voxels)
    return {
        "schema": "object-voxel-overlap/v1",
        "ply": str(path),
        "voxel_size": float(voxel_size),
        "point_count": len(rows),
        "voxel_count": int(total_voxels),
        "object_count": len(object_counts),
        "semantic_count": len(semantic_counts),
        "mixed_object_voxels": int(mixed_object_voxels),
        "mixed_object_voxel_ratio": float(mixed_object_voxels / max(total_voxels, 1)),
        "mixed_semantic_voxels": int(mixed_semantic_voxels),
        "mixed_semantic_voxel_ratio": float(mixed_semantic_voxels / max(total_voxels, 1)),
        "object_voxel_counts": {str(k): int(v) for k, v in object_counts.items()},
        "semantic_voxel_counts": dict(semantic_counts),
        "top_object_overlaps": top_pair_overlaps(object_voxels, object_counts, max_pairs),
        "top_semantic_overlaps": top_pair_overlaps(semantic_voxels, semantic_counts, max_pairs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.10)
    parser.add_argument("--max-pairs", type=int, default=30)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    report = measure_overlap(args.ply, args.voxel_size, args.max_pairs)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
