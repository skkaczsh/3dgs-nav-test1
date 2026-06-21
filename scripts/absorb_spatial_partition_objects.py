#!/usr/bin/env python3
"""Absorb small spatial-partition components into nearby compatible anchors.

This is a post-partition cleanup stage.  It never creates overlapping geometry:
each input voxel keeps one owner, but small owner ids may be remapped to a
nearby anchor object when the merge is spatially adjacent and semantically
compatible.  Components that do not pass the gates remain residual objects.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from build_spatial_partition_objects import COLORS, LABELS, LABEL_TO_SEMANTIC, parse_header

SEMANTIC_TO_LABEL = {v: k for k, v in LABELS.items()}


def read_objects(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[int(row["object_id"])] = row
    return out


def read_partition_ply(path: Path, voxel_size: float) -> tuple[list[dict[str, Any]], dict[tuple[int, int, int], int]]:
    _header, props, vertex_count, header_lines = parse_header(path)
    idx = {name: i for i, name in enumerate(props)}
    for required in ("x", "y", "z", "red", "green", "blue", "object", "semantic"):
        if required not in idx:
            raise ValueError(f"PLY missing property {required}: {path}")
    rows: list[dict[str, Any]] = []
    voxel_owner: dict[tuple[int, int, int], int] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(header_lines):
            next(f)
        for i, line in enumerate(f):
            if i >= vertex_count:
                break
            if not line.strip():
                continue
            parts = line.split()
            xyz = np.array([float(parts[idx["x"]]), float(parts[idx["y"]]), float(parts[idx["z"]])], dtype=np.float64)
            rgb = np.array([float(parts[idx["red"]]), float(parts[idx["green"]]), float(parts[idx["blue"]])], dtype=np.float64)
            object_id = int(float(parts[idx["object"]]))
            semantic = int(float(parts[idx["semantic"]]))
            label = SEMANTIC_TO_LABEL.get(semantic, "unknown")
            key = tuple(math.floor(float(v) / voxel_size) for v in xyz)
            rows.append({"xyz": xyz, "rgb": rgb, "object_id": object_id, "semantic": semantic, "label": label, "key": key})
            voxel_owner[key] = object_id
    return rows, voxel_owner


def label_compatible(source_label: str, anchor_label: str, absorb_unknown: bool, allow_groups: bool) -> bool:
    if source_label == anchor_label:
        return True
    if absorb_unknown and source_label == "unknown" and anchor_label not in {"unknown", "ignore"}:
        return True
    if not allow_groups:
        return False
    groups = [
        {"floor", "ground", "indoor_floor", "roof"},
        {"wall", "building"},
        {"grass", "tree"},
    ]
    return any(source_label in group and anchor_label in group for group in groups)


def normal_angle_deg(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b:
        return None
    av = np.array(a, dtype=np.float64)
    bv = np.array(b, dtype=np.float64)
    an = np.linalg.norm(av)
    bn = np.linalg.norm(bv)
    if an < 1e-9 or bn < 1e-9:
        return None
    cos = abs(float(np.dot(av, bv) / (an * bn)))
    return math.degrees(math.acos(max(min(cos, 1.0), -1.0)))


def compute_stats(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    pts_by_object: dict[int, list[np.ndarray]] = defaultdict(list)
    rgb_by_object: dict[int, list[np.ndarray]] = defaultdict(list)
    label_by_object: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        oid = int(row["object_id"])
        pts_by_object[oid].append(row["xyz"])
        rgb_by_object[oid].append(row["rgb"])
        label_by_object[oid][row["label"]] += 1
    stats: dict[int, dict[str, Any]] = {}
    for oid, pts in pts_by_object.items():
        arr = np.vstack(pts)
        rgb = np.vstack(rgb_by_object[oid])
        label, _ = label_by_object[oid].most_common(1)[0]
        stats[oid] = {
            "object_id": oid,
            "label": label,
            "voxel_count": int(len(pts)),
            "centroid": arr.mean(axis=0),
            "bbox_min": arr.min(axis=0),
            "bbox_max": arr.max(axis=0),
            "mean_rgb": rgb.mean(axis=0),
        }
    return stats


def build_offsets(radius_voxels: int) -> list[tuple[int, int, int]]:
    offsets: list[tuple[int, int, int]] = []
    r2 = radius_voxels * radius_voxels
    for dx in range(-radius_voxels, radius_voxels + 1):
        for dy in range(-radius_voxels, radius_voxels + 1):
            for dz in range(-radius_voxels, radius_voxels + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                if dx * dx + dy * dy + dz * dz <= r2:
                    offsets.append((dx, dy, dz))
    offsets.sort(key=lambda item: item[0] * item[0] + item[1] * item[1] + item[2] * item[2])
    return offsets


def choose_anchor(
    small_id: int,
    small_keys: list[tuple[int, int, int]],
    objects: dict[int, dict[str, Any]],
    stats: dict[int, dict[str, Any]],
    voxel_owner: dict[tuple[int, int, int], int],
    anchor_ids: set[int],
    offsets: list[tuple[int, int, int]],
    max_rgb_distance: float,
    max_normal_angle: float,
    absorb_unknown: bool,
    allow_groups: bool,
) -> tuple[int | None, str]:
    small_obj = objects[small_id]
    small_label = str(small_obj["semantic_label"])
    candidates: Counter[int] = Counter()
    for key in small_keys:
        for dx, dy, dz in offsets:
            oid = voxel_owner.get((key[0] + dx, key[1] + dy, key[2] + dz))
            if oid is None or oid == small_id or oid not in anchor_ids:
                continue
            candidates[oid] += 1
    if not candidates:
        return None, "no_neighbor_anchor"

    best: tuple[float, int] | None = None
    best_reason = "no_compatible_anchor"
    for anchor_id, contacts in candidates.items():
        anchor_obj = objects[anchor_id]
        anchor_label = str(anchor_obj["semantic_label"])
        if not label_compatible(small_label, anchor_label, absorb_unknown, allow_groups):
            continue
        rgb_dist = float(np.linalg.norm(stats[small_id]["mean_rgb"] - stats[anchor_id]["mean_rgb"]))
        if rgb_dist > max_rgb_distance:
            best_reason = "rgb_gate"
            continue
        angle = normal_angle_deg(small_obj.get("pca_normal"), anchor_obj.get("pca_normal"))
        if angle is not None and angle > max_normal_angle and small_label not in {"unknown", "railing", "car"}:
            best_reason = "normal_gate"
            continue
        centroid_dist = float(np.linalg.norm(stats[small_id]["centroid"] - stats[anchor_id]["centroid"]))
        same_label_bonus = 2.0 if small_label == anchor_label else 1.0
        score = contacts * same_label_bonus - 0.25 * centroid_dist - 0.01 * rgb_dist
        if best is None or score > best[0]:
            best = (score, anchor_id)
    if best is None:
        return None, best_reason
    return best[1], "absorbed"


def absorb_components(
    rows: list[dict[str, Any]],
    objects: dict[int, dict[str, Any]],
    voxel_owner: dict[tuple[int, int, int], int],
    radius_voxels: int,
    max_rgb_distance: float,
    max_normal_angle: float,
    absorb_unknown: bool,
    allow_groups: bool,
) -> tuple[dict[int, int], dict[str, Any]]:
    stats = compute_stats(rows)
    keys_by_object: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for row in rows:
        keys_by_object[int(row["object_id"])].append(row["key"])

    anchor_ids = {oid for oid, obj in objects.items() if obj.get("status") != "small_component"}
    small_ids = [oid for oid, obj in objects.items() if obj.get("status") == "small_component"]
    remap = {oid: oid for oid in objects}
    offsets = build_offsets(radius_voxels)
    reason_counts: Counter[str] = Counter()
    absorbed_by_label: Counter[str] = Counter()
    absorbed_to_label: Counter[str] = Counter()

    # Process larger fragments first so coherent near-anchor fragments stabilize early.
    small_ids.sort(key=lambda oid: int(objects[oid].get("voxel_count", 0)), reverse=True)
    for small_id in small_ids:
        anchor_id, reason = choose_anchor(
            small_id,
            keys_by_object[small_id],
            objects,
            stats,
            voxel_owner,
            anchor_ids,
            offsets,
            max_rgb_distance,
            max_normal_angle,
            absorb_unknown,
            allow_groups,
        )
        reason_counts[reason] += 1
        if anchor_id is None:
            continue
        remap[small_id] = anchor_id
        absorbed_by_label[str(objects[small_id]["semantic_label"])] += int(objects[small_id].get("voxel_count", 0))
        absorbed_to_label[str(objects[anchor_id]["semantic_label"])] += int(objects[small_id].get("voxel_count", 0))

    report = {
        "input_object_count": len(objects),
        "anchor_count": len(anchor_ids),
        "small_component_count": len(small_ids),
        "absorbed_component_count": sum(1 for oid in small_ids if remap[oid] != oid),
        "residual_small_component_count": sum(1 for oid in small_ids if remap[oid] == oid),
        "absorb_reason_counts": dict(reason_counts),
        "absorbed_voxels_by_source_label": dict(absorbed_by_label),
        "absorbed_voxels_to_anchor_label": dict(absorbed_to_label),
    }
    return remap, report


def write_absorbed_outputs(
    rows: list[dict[str, Any]],
    objects: dict[int, dict[str, Any]],
    remap: dict[int, int],
    output_dir: Path,
    voxel_size: float,
) -> tuple[Path, Path, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = output_dir / "spatial_absorbed_objects.ply"
    jsonl_path = output_dir / "spatial_absorbed_objects.jsonl"

    final_rows = []
    aggregate: dict[int, dict[str, Any]] = {}
    for row in rows:
        source_id = int(row["object_id"])
        final_id = int(remap[source_id])
        final_obj = objects[final_id]
        final_label = str(final_obj["semantic_label"])
        semantic = LABEL_TO_SEMANTIC.get(final_label, 0)
        color = COLORS.get(final_label, tuple(int(x) for x in np.clip(row["rgb"], 0, 255)))
        final_rows.append((row["xyz"], color, final_id, semantic))
        item = aggregate.setdefault(
            final_id,
            {
                "object_id": final_id,
                "semantic_label": final_label,
                "source_object_ids": set(),
                "voxel_count": 0,
                "xyz": [],
            },
        )
        item["source_object_ids"].add(source_id)
        item["voxel_count"] += 1
        item["xyz"].append(row["xyz"])

    with ply_path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(final_rows)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uint object\nproperty uchar semantic\n")
        f.write("end_header\n")
        for xyz, color, object_id, semantic in final_rows:
            f.write(
                f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} {object_id} {semantic}\n"
            )

    final_object_count = 0
    absorbed_object_count = 0
    with jsonl_path.open("w", encoding="utf-8") as f:
        for object_id in sorted(aggregate):
            item = aggregate[object_id]
            arr = np.vstack(item.pop("xyz"))
            source_ids = sorted(int(x) for x in item.pop("source_object_ids"))
            row = {
                **item,
                "status": "absorbed_anchor" if len(source_ids) > 1 else objects[object_id].get("status", "residual"),
                "source_object_ids": source_ids,
                "absorbed_source_count": len(source_ids),
                "centroid": arr.mean(axis=0).astype(float).tolist(),
                "bbox_3d": {"min": arr.min(axis=0).astype(float).tolist(), "max": arr.max(axis=0).astype(float).tolist()},
                "voxel_size": voxel_size,
                "description": f"spatial absorption object: {item['semantic_label']}",
            }
            final_object_count += 1
            if len(source_ids) > 1:
                absorbed_object_count += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"output_ply": str(ply_path), "output_jsonl": str(jsonl_path), "final_object_count": final_object_count, "absorbed_anchor_object_count": absorbed_object_count}
    return ply_path, jsonl_path, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-ply", type=Path, required=True)
    parser.add_argument("--partition-objects", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.10)
    parser.add_argument("--radius-voxels", type=int, default=2)
    parser.add_argument("--max-rgb-distance", type=float, default=95.0)
    parser.add_argument("--max-normal-angle", type=float, default=35.0)
    parser.add_argument("--absorb-unknown", action="store_true")
    parser.add_argument("--allow-compatible-label-groups", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    objects = read_objects(args.partition_objects)
    rows, voxel_owner = read_partition_ply(args.partition_ply, args.voxel_size)
    remap, absorb_report = absorb_components(
        rows,
        objects,
        voxel_owner,
        args.radius_voxels,
        args.max_rgb_distance,
        args.max_normal_angle,
        args.absorb_unknown,
        args.allow_compatible_label_groups,
    )
    _ply, _jsonl, output_report = write_absorbed_outputs(rows, objects, remap, args.output_dir, args.voxel_size)
    report = {
        "schema": "spatial-absorption/v1",
        "partition_ply": str(args.partition_ply),
        "partition_objects": str(args.partition_objects),
        "voxel_size": args.voxel_size,
        "radius_voxels": args.radius_voxels,
        "max_rgb_distance": args.max_rgb_distance,
        "max_normal_angle": args.max_normal_angle,
        "absorb_unknown": args.absorb_unknown,
        "allow_compatible_label_groups": args.allow_compatible_label_groups,
        **absorb_report,
        **output_report,
    }
    report_path = args.output_dir / "spatial_absorption_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
