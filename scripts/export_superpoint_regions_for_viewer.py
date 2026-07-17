#!/usr/bin/env python3
"""Export structural regions for the semantic viewer.

By default this is a strict, high-confidence review export. The
--include-unassigned flag retains the complete reference cloud: unassigned
points have object=0 and semantic=unknown. This lets the viewer show coverage
honestly instead of making a partial structural export look like a whole scene.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


SEMANTIC = {"unknown": 0, "wall": 2, "floor": 3, "ceiling": 4, "grass": 5, "stair": 18}
COLOR = {
    "unknown": (90, 90, 90), "wall": (160, 170, 180), "floor": (190, 172, 135),
    "ceiling": (180, 180, 210), "grass": (70, 150, 80), "stair": (210, 150, 80),
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def region_lookup(assignments: list[dict], regions: list[dict]) -> tuple[dict[int, int], dict[int, dict]]:
    labels = {str(row["region_id"]): str(row["region_label"]) for row in regions}
    ordered = sorted(labels)
    region_ids = {name: index + 1 for index, name in enumerate(ordered)}
    point_regions = {int(row["superpoint_id"]): region_ids[str(row["region_id"])] for row in assignments}
    metadata = {
        region_ids[str(row["region_id"])]: {
            "object_id": region_ids[str(row["region_id"])],
            "semantic_label": str(row["region_label"]),
            "description": f"{row['region_id']} from source anchors {row.get('source_anchor_ids', [])}",
            "point_count": 0,
            "superpoint_ids": row.get("superpoint_ids", []),
            "source_anchor_ids": row.get("source_anchor_ids", []),
        }
        for row in regions
    }
    return point_regions, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-ply", type=Path, required=True)
    parser.add_argument("--superpoint-labels", type=Path, required=True)
    parser.add_argument("--assignments-jsonl", type=Path, required=True)
    parser.add_argument("--regions-jsonl", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--output-objects-jsonl", type=Path, required=True)
    parser.add_argument(
        "--include-unassigned",
        action="store_true",
        help="Keep every reference point; points outside strict regions become unknown.",
    )
    parser.add_argument(
        "--point-stride",
        type=int,
        default=1,
        help="Keep every Nth source point after coverage selection (viewer export only).",
    )
    args = parser.parse_args()

    assignments, regions = read_jsonl(args.assignments_jsonl), read_jsonl(args.regions_jsonl)
    lookup, objects = region_lookup(assignments, regions)
    labels = np.load(args.superpoint_labels).astype(np.int32, copy=False)
    vertex = PlyData.read(str(args.reference_ply))["vertex"].data
    if len(vertex) != len(labels):
        raise SystemExit(f"reference/label count mismatch: {len(vertex)} != {len(labels)}")
    if args.point_stride < 1:
        raise SystemExit("--point-stride must be positive")
    region_for_point = np.fromiter((lookup.get(int(label), 0) for label in labels), dtype=np.uint32, count=len(labels))
    keep = np.ones(len(vertex), dtype=bool) if args.include_unassigned else region_for_point > 0
    if args.point_stride > 1:
        keep &= np.arange(len(vertex)) % args.point_stride == 0
    out = np.empty(int(keep.sum()), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("object", "u4"), ("semantic", "u1")])
    out["x"], out["y"], out["z"] = vertex["x"][keep], vertex["y"][keep], vertex["z"][keep]
    out["object"] = region_for_point[keep]
    unknown = out["object"] == 0
    out["semantic"][unknown] = SEMANTIC["unknown"]
    out["red"][unknown], out["green"][unknown], out["blue"][unknown] = COLOR["unknown"]
    for region_id, obj in objects.items():
        mask = out["object"] == region_id
        label = obj["semantic_label"]
        out["semantic"][mask] = SEMANTIC.get(label, 0)
        out["red"][mask], out["green"][mask], out["blue"][mask] = COLOR.get(label, COLOR["unknown"])
        obj["point_count"] = int(mask.sum())
    args.output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(out, "vertex")], text=True).write(str(args.output_ply))
    args.output_objects_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_objects_jsonl.write_text(
        "".join(json.dumps(objects[key], ensure_ascii=False) + "\n" for key in sorted(objects)), encoding="utf-8"
    )
    print(json.dumps({
        "regions": len(objects),
        "points": len(out),
        "assigned_points": int((region_for_point > 0).sum()),
        "unassigned_points": int((region_for_point == 0).sum()),
        "include_unassigned": args.include_unassigned,
        "point_stride": args.point_stride,
        "output_ply": str(args.output_ply),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
