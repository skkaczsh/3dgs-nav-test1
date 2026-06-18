#!/usr/bin/env python3
"""Apply drivability ground/wall prior to a full-scene object viewer bundle.

This is the full-scene counterpart of apply_drivability_prior_to_residual.py.
It uses the red/white/blue PCD from drivability_cpp as a coarse geometry prior:

- red: drivable ground/floor
- white: wall
- blue: other obstacle/non-drivable

The pass is intentionally conservative. It only relabels objects when both the
drivability vote and object PCA geometry agree. The goal is to stop large
ground/wall surfaces from being shown as fine objects while keeping ambiguous
objects available for later DINO/GroundingDINO review.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from apply_drivability_prior_to_residual import (
    GEOM_GROUND,
    GEOM_NAMES,
    GEOM_OTHER,
    GEOM_UNKNOWN,
    GEOM_WALL,
    build_prior_voxels,
    label_from_rgb,
    pca_normal_and_planarity,
    read_pcd_xyzrgb,
    read_ply_xyz_object,
    vote_points,
)


LABEL_TO_SEMANTIC = {
    "unknown": 0,
    "other": 1,
    "wall": 2,
    "floor": 3,
    "ceiling": 4,
    "grass": 5,
    "tree": 6,
    "person": 7,
    "car": 8,
    "railing": 9,
    "building": 10,
    "sky": 11,
    "road": 12,
    "water": 13,
    "furniture": 14,
    "pipe": 15,
    "equipment": 16,
    "fine_candidate": 17,
    "ignore": 255,
}

SURFACE_LABELS = {"floor", "wall", "ceiling", "building", "road"}
FINE_LABELS = {"car", "railing", "fine_candidate"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_ply_header(path: Path) -> tuple[list[str], int]:
    props: list[str] = []
    vertex_count = 0
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                vertex_count = int(parts[2])
                in_vertex = True
            elif len(parts) >= 2 and parts[0] == "element":
                in_vertex = False
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                props.append(parts[-1])
            elif line.strip() == "end_header":
                break
    if vertex_count <= 0:
        raise ValueError(f"No vertex count in PLY: {path}")
    return props, vertex_count


def object_max_extent(obj: dict[str, Any], points: np.ndarray) -> float:
    if "max_extent" in obj:
        try:
            return float(obj["max_extent"])
        except (TypeError, ValueError):
            pass
    if len(points):
        return float((points.max(axis=0) - points.min(axis=0)).max())
    return 0.0


def point_count_for_rules(obj: dict[str, Any], display_points: int) -> int:
    try:
        return int(obj.get("point_count") or display_points)
    except (TypeError, ValueError):
        return int(display_points)


def relabel_decision(
    obj: dict[str, Any],
    counts: Counter,
    total: int,
    normal_abs_z: float,
    thickness: float,
    planarity: float,
    max_extent: float,
    args: argparse.Namespace,
) -> tuple[str, str, str]:
    old = str(obj.get("semantic_label") or "unknown")
    ground_ratio = counts[GEOM_GROUND] / max(total, 1)
    wall_ratio = counts[GEOM_WALL] / max(total, 1)
    other_ratio = counts[GEOM_OTHER] / max(total, 1)
    unknown_ratio = counts[GEOM_UNKNOWN] / max(total, 1)
    point_count = point_count_for_rules(obj, total)
    horizontal = normal_abs_z >= args.ground_normal_z_min
    vertical = normal_abs_z <= args.wall_normal_z_max
    planar = planarity >= args.planarity_min and thickness <= args.thickness_max
    big_enough = point_count >= args.min_relabel_points or max_extent >= args.min_relabel_extent

    if not big_enough:
        return old, "kept_small_object", "object too small for drivability full-scene relabel"

    if ground_ratio >= args.ground_ratio and horizontal and planar:
        if old != "floor":
            return "floor", f"{old}_to_floor_by_drivability_prior", "dominant red prior vote and horizontal planar PCA"
        return old, "kept_floor_confirmed_by_drivability_prior", "floor confirmed by red prior vote"

    if wall_ratio >= args.wall_ratio and vertical and planar:
        if old != "wall":
            return "wall", f"{old}_to_wall_by_drivability_prior", "dominant white prior vote and vertical planar PCA"
        return old, "kept_wall_confirmed_by_drivability_prior", "wall confirmed by white prior vote"

    if old in FINE_LABELS:
        if ground_ratio >= args.fine_surface_ratio and horizontal and planarity >= args.fine_surface_planarity_min:
            return "floor", f"{old}_to_floor_by_geometry_prior", "fine object overlaps ground prior and is horizontally planar"
        if wall_ratio >= args.fine_surface_ratio and vertical and planarity >= args.fine_surface_planarity_min:
            return "wall", f"{old}_to_wall_by_geometry_prior", "fine object overlaps wall prior and is vertically planar"
        if max(ground_ratio, wall_ratio) >= args.fine_ambiguous_surface_ratio and planar:
            return "fine_candidate", f"{old}_surface_ambiguous", "fine object has strong surface prior but insufficient class-safe relabel"

    if old == "wall" and horizontal and ground_ratio >= args.loose_surface_ratio and other_ratio <= args.loose_other_ratio_max:
        return "floor", "wall_to_floor_by_loose_drivability_prior", "horizontal wall object with loose red prior vote"
    if old == "floor" and vertical and wall_ratio >= args.loose_surface_ratio and other_ratio <= args.loose_other_ratio_max:
        return "wall", "floor_to_wall_by_loose_drivability_prior", "vertical floor object with loose white prior vote"

    return old, "kept_no_confident_drivability_relabel", f"prior ratios ground={ground_ratio:.3f} wall={wall_ratio:.3f} other={other_ratio:.3f} unknown={unknown_ratio:.3f}"


def rewrite_ply(input_ply: Path, output_ply: Path, objects_by_id: dict[int, dict[str, Any]]) -> dict[str, Any]:
    props, vertex_count = parse_ply_header(input_ply)
    idx = {name: i for i, name in enumerate(props)}
    object_col = idx.get("object", idx.get("object_id"))
    semantic_col = idx.get("semantic")
    if object_col is None or semantic_col is None:
        raise ValueError(f"PLY needs object and semantic fields: {input_ply}")

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    changed_points = 0
    changed_objects: set[int] = set()
    semantic_counts = Counter()
    with input_ply.open("r", encoding="utf-8", errors="replace") as src, output_ply.open("w", encoding="utf-8") as dst:
        for line in src:
            dst.write(line)
            if line.strip() == "end_header":
                break
        for _ in range(vertex_count):
            line = src.readline()
            if not line:
                break
            parts = line.strip().split()
            if len(parts) <= max(object_col, semantic_col):
                dst.write(line)
                continue
            object_id = int(round(float(parts[object_col])))
            obj = objects_by_id.get(object_id)
            if obj:
                label = str(obj.get("semantic_label") or "unknown")
                semantic = LABEL_TO_SEMANTIC.get(label, 0)
                old_semantic = int(round(float(parts[semantic_col])))
                if semantic != old_semantic:
                    parts[semantic_col] = str(semantic)
                    changed_points += 1
                    changed_objects.add(object_id)
                semantic_counts[label] += 1
                dst.write(" ".join(parts) + "\n")
            else:
                dst.write(line)
    return {
        "vertex_count": vertex_count,
        "changed_points": changed_points,
        "changed_object_count": len(changed_objects),
        "semantic_counts_after": dict(semantic_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drivability-pcd", type=Path, required=True)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--input-objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="full_scene_drivability_prior")
    parser.add_argument("--prior-voxel-size", type=float, default=0.10)
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument("--ground-ratio", type=float, default=0.58)
    parser.add_argument("--wall-ratio", type=float, default=0.50)
    parser.add_argument("--loose-surface-ratio", type=float, default=0.42)
    parser.add_argument("--loose-other-ratio-max", type=float, default=0.35)
    parser.add_argument("--fine-surface-ratio", type=float, default=0.45)
    parser.add_argument("--fine-ambiguous-surface-ratio", type=float, default=0.35)
    parser.add_argument("--fine-surface-planarity-min", type=float, default=0.78)
    parser.add_argument("--min-relabel-points", type=int, default=600)
    parser.add_argument("--min-relabel-extent", type=float, default=1.5)
    parser.add_argument("--ground-normal-z-min", type=float, default=0.86)
    parser.add_argument("--wall-normal-z-max", type=float, default=0.42)
    parser.add_argument("--planarity-min", type=float, default=0.82)
    parser.add_argument("--thickness-max", type=float, default=0.45)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prior_xyz, prior_rgb = read_pcd_xyzrgb(args.drivability_pcd)
    prior_labels = label_from_rgb(prior_rgb)
    prior_keys, prior_voxel_labels, spec = build_prior_voxels(prior_xyz, prior_labels, args.prior_voxel_size)

    scene_xyz, object_ids = read_ply_xyz_object(args.input_ply)
    point_prior = vote_points(
        scene_xyz,
        prior_keys,
        prior_voxel_labels,
        spec,
        args.prior_voxel_size,
        args.neighbor_radius,
    )
    objects = {int(row["object_id"]): row for row in read_jsonl(args.input_objects_jsonl)}
    by_object_indices: dict[int, list[int]] = defaultdict(list)
    for i, oid in enumerate(object_ids.tolist()):
        if oid:
            by_object_indices[int(oid)].append(i)

    out_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    decision_counts = Counter()
    label_counts = Counter()
    point_label_counts = Counter()
    for oid in sorted(objects):
        obj = objects[oid]
        idx = np.asarray(by_object_indices.get(oid, []), dtype=np.int64)
        counts = Counter()
        if len(idx):
            counts.update(int(x) for x in point_prior[idx].tolist())
            pts = scene_xyz[idx]
            normal, thickness, planarity = pca_normal_and_planarity(pts)
            max_extent = object_max_extent(obj, pts)
        else:
            normal, thickness, planarity, max_extent = [0.0, 0.0, 1.0], 0.0, 0.0, object_max_extent(obj, np.empty((0, 3), dtype=np.float32))
        total = int(sum(counts.values()))
        old_label = str(obj.get("semantic_label") or "unknown")
        new_label, decision, reason = relabel_decision(
            obj,
            counts,
            total,
            abs(float(normal[2])),
            thickness,
            planarity,
            max_extent,
            args,
        )
        out = dict(obj)
        out["semantic_label_original"] = out.get("semantic_label_original") or old_label
        out["semantic_label"] = new_label
        out["drivability_full_scene_decision"] = decision
        out["drivability_full_scene_reason"] = reason
        out["drivability_prior_counts"] = {GEOM_NAMES[k]: int(v) for k, v in sorted(counts.items())}
        out["drivability_prior_ratios"] = {GEOM_NAMES[k]: float(v) / max(total, 1) for k, v in sorted(counts.items())}
        out["pca_normal_abs_z_drivability_guard"] = abs(float(normal[2]))
        out["pca_thickness_rms_drivability_guard"] = thickness
        out["pca_planarity_drivability_guard"] = planarity
        if new_label != old_label:
            out["status"] = f"drivability_full_scene_{decision}"
            if new_label in SURFACE_LABELS:
                out["downstream_stage"] = "stable_surface"
                out["stable_surface"] = True
            elif new_label == "fine_candidate":
                out["downstream_stage"] = "dino_fine_object_review"
                out["stable_surface"] = False
        out_rows.append(out)
        decision_counts[decision] += 1
        label_counts[new_label] += 1
        point_label_counts[new_label] += int(obj.get("point_count") or total)
        decisions.append({
            "object_id": oid,
            "old_label": old_label,
            "new_label": new_label,
            "decision": decision,
            "reason": reason,
            "display_point_count": int(total),
            "object_point_count": int(obj.get("point_count") or total),
            "prior_counts": {GEOM_NAMES[k]: int(v) for k, v in sorted(counts.items())},
            "normal_abs_z": round(abs(float(normal[2])), 6),
            "thickness": round(float(thickness), 6),
            "planarity": round(float(planarity), 6),
            "max_extent": round(float(max_extent), 6),
        })

    output_jsonl = args.output_dir / f"{args.output_prefix}.jsonl"
    output_ply = args.output_dir / f"{args.output_prefix}.ply"
    output_decisions = args.output_dir / f"{args.output_prefix}_decisions.jsonl"
    write_jsonl(output_jsonl, out_rows)
    write_jsonl(output_decisions, decisions)
    ply_report = rewrite_ply(args.input_ply, output_ply, {int(row["object_id"]): row for row in out_rows})
    report = {
        "drivability_pcd": str(args.drivability_pcd),
        "input_ply": str(args.input_ply),
        "input_objects_jsonl": str(args.input_objects_jsonl),
        "output_ply": str(output_ply),
        "output_jsonl": str(output_jsonl),
        "output_decisions": str(output_decisions),
        "prior_voxel_size": args.prior_voxel_size,
        "neighbor_radius": args.neighbor_radius,
        "prior_point_count": int(len(prior_xyz)),
        "prior_voxel_count": int(len(prior_keys)),
        "scene_point_count": int(len(scene_xyz)),
        "object_count": len(out_rows),
        "point_prior_counts": {GEOM_NAMES[k]: int(v) for k, v in sorted(Counter(point_prior.tolist()).items())},
        "decision_counts": dict(decision_counts),
        "semantic_label_counts": dict(label_counts),
        "semantic_label_points": dict(point_label_counts),
        **ply_report,
    }
    report_path = args.output_dir / f"{args.output_prefix}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
