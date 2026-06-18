#!/usr/bin/env python3
"""Refine frame-local priority targets with conservative 3D geometry rules.

This script sits between build_frame_targets_from_priority.py and
fuse_targets_to_objects.py. It does not reproject images. It reads the target
JSONL plus frame_targets.ply, optionally splits large targets with a finer 3D
connectivity pass, and corrects only obvious geometry/label contradictions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from build_frame_targets_from_priority import connected_components, pca_stats
from export_frame_target_objects_for_viewer import read_ply_header
from project_priority_masks_to_lx import PRIORITY_COLORS, PRIORITY_NAMES


LABEL_TO_PRIORITY = {v: k for k, v in PRIORITY_NAMES.items()}
PARENT_BY_LABEL = {
    "ground": "surface",
    "wall": "surface",
    "building": "surface",
    "floor": "surface",
    "ceiling": "surface",
    "grass": "vegetation",
    "car": "object",
    "railing": "structure",
    "other": "other",
    "unknown": "other",
}
FINE_LABELS = {"railing", "car"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extent(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(3, dtype=np.float64)
    return points.max(axis=0) - points.min(axis=0)


def surface_label_from_normal(normal: list[float]) -> str:
    nz = abs(float(normal[2])) if normal else 1.0
    return "ground" if nz >= 0.68 else "wall"


def target_should_split(target: dict[str, Any], args: argparse.Namespace) -> bool:
    label = str(target.get("label") or "unknown")
    n = int(target.get("cluster_size") or 0)
    if n < int(args.min_split_points):
        return False
    dims = np.array(target.get("bbox_3d", {}).get("max", [0, 0, 0]), dtype=np.float64) - np.array(
        target.get("bbox_3d", {}).get("min", [0, 0, 0]), dtype=np.float64
    )
    pca = target.get("pca") or {}
    linearity = float(pca.get("linearity") or 0.0)
    planarity = float(pca.get("planarity") or 0.0)
    normal = pca.get("normal") or [0.0, 0.0, 1.0]
    normal_z = abs(float(normal[2])) if len(normal) >= 3 else 1.0
    if label == "railing":
        return linearity < args.railing_min_linearity or float(dims.max()) > args.railing_max_extent
    if label == "car":
        return float(dims.max()) > args.car_max_extent or (planarity > args.surface_planarity and linearity < args.car_surface_max_linearity)
    if label in {"ground", "ceiling"} and float(dims[2]) > args.surface_height_split_threshold:
        return n >= int(args.surface_min_split_points)
    if (
        label == "wall"
        and bool(getattr(args, "split_horizontal_wall_by_height", False))
        and normal_z >= args.wall_max_normal_z
        and planarity >= args.surface_planarity
        and float(dims[2]) > args.surface_height_split_threshold
    ):
        return n >= int(args.surface_min_split_points)
    if label in {"ground", "wall", "ceiling"}:
        return float(dims.max()) > args.surface_max_extent and n >= int(args.surface_min_split_points)
    return False


def refined_label(label: str, points: np.ndarray, args: argparse.Namespace) -> tuple[str, list[str]]:
    pca = pca_stats(points)
    dims = extent(points)
    linearity = float(pca.get("linearity") or 0.0)
    planarity = float(pca.get("planarity") or 0.0)
    normal_z = abs(float((pca.get("normal") or [0, 0, 1])[2]))
    reasons: list[str] = []
    out = label

    if label == "railing":
        broad = float(dims.max()) > args.railing_max_extent or linearity < args.railing_min_linearity
        if broad and planarity >= args.surface_planarity:
            out = surface_label_from_normal(pca["normal"])
            reasons.append("broad_planar_railing_to_surface")
        elif broad:
            reasons.append("split_broad_railing")
    elif label == "car":
        if planarity >= args.surface_planarity and linearity <= args.car_surface_max_linearity:
            out = surface_label_from_normal(pca["normal"])
            reasons.append("planar_car_to_surface")
    elif label == "ground":
        if normal_z <= args.ground_min_normal_z and planarity >= args.surface_planarity:
            out = "wall"
            reasons.append("ground_normal_to_wall")
        elif (
            bool(getattr(args, "guard_linear_ground_artifacts", False))
            and float(dims[2]) >= args.ground_artifact_min_height_span
            and linearity >= args.ground_artifact_min_linearity
            and planarity <= args.ground_artifact_max_planarity
        ):
            out = "wall" if normal_z <= args.ground_artifact_wall_max_normal_z else "other"
            reasons.append(f"linear_ground_artifact_to_{out}")
    elif label == "wall":
        if normal_z >= args.wall_max_normal_z and planarity >= args.surface_planarity:
            centroid_z = float(points[:, 2].mean())
            if bool(args.enable_ceiling_label) and centroid_z >= args.ceiling_min_z:
                out = "ceiling"
                reasons.append("wall_normal_to_ceiling")
            else:
                out = "ground"
                reasons.append("wall_normal_to_ground")

    return out, reasons


def read_target_ply_points(path: Path) -> dict[int, dict[str, Any]]:
    _header, props, _count, header_lines = read_ply_header(path)
    idx = {name: i for i, name in enumerate(props)}
    required = {"x", "y", "z", "target", "frame", "camera", "point_index"}
    if not required.issubset(idx):
        raise ValueError(f"Target PLY missing fields: {sorted(required - set(idx))}")
    buckets: dict[int, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(header_lines):
            next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) < len(props):
                continue
            target = int(round(float(parts[idx["target"]])))
            buckets[target]["points"].append([float(parts[idx["x"]]), float(parts[idx["y"]]), float(parts[idx["z"]])])
            buckets[target]["point_indices"].append(int(round(float(parts[idx["point_index"]]))))
            buckets[target]["frame"].append(int(round(float(parts[idx["frame"]]))))
            buckets[target]["camera"].append(int(round(float(parts[idx["camera"]]))))
    return {
        target: {
            "points": np.asarray(values["points"], dtype=np.float32),
            "point_indices": np.asarray(values["point_indices"], dtype=np.int64),
            "frame": int(values["frame"][0]) if values["frame"] else 0,
            "camera": int(values["camera"][0]) if values["camera"] else 0,
        }
        for target, values in buckets.items()
    }


def summarize(points: np.ndarray, base: dict[str, Any]) -> dict[str, Any]:
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    return {
        "bbox_3d": {"min": [float(x) for x in bbox_min], "max": [float(x) for x in bbox_max]},
        "centroid": [float(x) for x in points.mean(axis=0)],
        "pca": pca_stats(points),
        "cluster_size": int(len(points)),
        "bbox_2d": base.get("bbox_2d", {"xyxy": [0, 0, 0, 0], "area": 0}),
        "mean_color": base.get("mean_color", [128.0, 128.0, 128.0]),
    }


def make_child_target(
    base: dict[str, Any],
    points: np.ndarray,
    point_indices: np.ndarray,
    target_index: int,
    child_id: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    original_label = str(base.get("label") or "unknown")
    label, reasons = refined_label(original_label, points, args)
    priority_id = int(LABEL_TO_PRIORITY.get(label, base.get("priority_label_id", base.get("mask_id", 0))))
    child = dict(base)
    child["target_id"] = f"{base['target_id']}_g{child_id:03d}"
    child["target_index"] = int(target_index)
    child["label"] = label
    child["raw_label"] = str(base.get("raw_label") or original_label)
    child["refined_from_label"] = original_label
    child["refined_from_target_id"] = str(base["target_id"])
    child["refinement_reasons"] = reasons
    child["parent_class"] = PARENT_BY_LABEL.get(label, base.get("parent_class", "other"))
    child["priority_label_id"] = priority_id
    child["mask_id"] = priority_id
    child["point_indices"] = [int(x) for x in point_indices.tolist()]
    child.update(summarize(points, base))
    return child


def split_target(base: dict[str, Any], points: np.ndarray, point_indices: np.ndarray, args: argparse.Namespace) -> list[tuple[np.ndarray, np.ndarray]]:
    label = str(base.get("label") or "unknown")
    if not target_should_split(base, args):
        return [(points, point_indices)]
    horizontal_wall = False
    if label == "wall" and bool(getattr(args, "split_horizontal_wall_by_height", False)):
        pca = pca_stats(points)
        normal_z = abs(float((pca.get("normal") or [0, 0, 1])[2]))
        horizontal_wall = normal_z >= args.wall_max_normal_z and float(pca.get("planarity") or 0.0) >= args.surface_planarity
    if label in {"ground", "ceiling"} or horizontal_wall:
        if float(points[:, 2].max() - points[:, 2].min()) > args.surface_height_split_threshold:
            return split_horizontal_surface_by_height(base, points, point_indices, args)
    voxel = {
        "railing": args.railing_split_voxel,
        "car": args.car_split_voxel,
        "ground": args.surface_split_voxel,
        "wall": args.surface_split_voxel,
        "ceiling": args.surface_split_voxel,
    }.get(label, args.split_voxel)
    min_points = int(args.surface_split_min_points if label in {"ground", "wall", "ceiling"} else args.split_min_points)
    comps, residual = connected_components(points, voxel, min_points)
    out = [(points[comp], point_indices[comp]) for comp in comps]
    if bool(args.keep_residual) and residual.any():
        residual_idx = np.where(residual)[0]
        out.append((points[residual_idx], point_indices[residual_idx]))
    return out or [(points, point_indices)]


def split_horizontal_surface_by_height(
    base: dict[str, Any],
    points: np.ndarray,
    point_indices: np.ndarray,
    args: argparse.Namespace,
) -> list[tuple[np.ndarray, np.ndarray]]:
    z = points[:, 2]
    z_min = float(z.min())
    bins = np.floor((z - z_min) / max(float(args.surface_height_bin), 1e-6)).astype(np.int64)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    residual_points: list[np.ndarray] = []
    for bin_id in sorted(set(int(x) for x in bins.tolist())):
        local = np.where(bins == bin_id)[0]
        if len(local) < int(args.surface_split_min_points):
            residual_points.append(local)
            continue
        comps, residual = connected_components(points[local], args.surface_split_voxel, args.surface_split_min_points)
        for comp in comps:
            selected = local[comp]
            out.append((points[selected], point_indices[selected]))
        if residual.any():
            residual_points.append(local[np.where(residual)[0]])
    if bool(args.keep_residual) and residual_points:
        residual = np.concatenate(residual_points)
        if len(residual) >= int(args.min_output_points):
            out.append((points[residual], point_indices[residual]))
    if len(out) <= 1:
        return split_target_without_height(base, points, point_indices, args)
    return sorted(out, key=lambda pair: len(pair[0]), reverse=True)


def split_target_without_height(base: dict[str, Any], points: np.ndarray, point_indices: np.ndarray, args: argparse.Namespace) -> list[tuple[np.ndarray, np.ndarray]]:
    label = str(base.get("label") or "unknown")
    voxel = {
        "railing": args.railing_split_voxel,
        "car": args.car_split_voxel,
        "ground": args.surface_split_voxel,
        "wall": args.surface_split_voxel,
        "ceiling": args.surface_split_voxel,
    }.get(label, args.split_voxel)
    min_points = int(args.surface_split_min_points if label in {"ground", "wall", "ceiling"} else args.split_min_points)
    comps, residual = connected_components(points, voxel, min_points)
    out = [(points[comp], point_indices[comp]) for comp in comps]
    if bool(args.keep_residual) and residual.any():
        residual_idx = np.where(residual)[0]
        out.append((points[residual_idx], point_indices[residual_idx]))
    return out or [(points, point_indices)]


def write_ply(path: Path, rows: list[dict[str, Any]], points_by_target: dict[str, np.ndarray]) -> None:
    total = sum(len(points_by_target[row["target_id"]]) for row in rows)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {total}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property int target\nproperty uchar priority\nproperty int frame\nproperty int camera\nproperty int point_index\n")
        f.write("end_header\n")
        for row in rows:
            points = points_by_target[row["target_id"]]
            color = PRIORITY_COLORS.get(int(row.get("priority_label_id", 0)), (200, 200, 200))
            point_indices = row["point_indices"]
            for point, point_index in zip(points, point_indices):
                f.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{color[0]} {color[1]} {color[2]} {int(row['target_index'])} "
                    f"{int(row.get('priority_label_id', 0))} {int(row['frame_id'])} {int(row['cam_id'])} {int(point_index)}\n"
                )


def refine_targets(targets: list[dict[str, Any]], ply_points: dict[int, dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    points_by_target: dict[str, np.ndarray] = {}
    next_index = 0
    split_count = 0
    relabel_count = 0
    missing = 0
    reason_counts = Counter()
    for base in targets:
        data = ply_points.get(int(base["target_index"]))
        if data is None or len(data["points"]) == 0:
            missing += 1
            continue
        chunks = split_target(base, data["points"], data["point_indices"], args)
        if len(chunks) > 1:
            split_count += 1
        for child_id, (points, point_indices) in enumerate(chunks):
            if len(points) < int(args.min_output_points):
                continue
            child = make_child_target(base, points, point_indices, next_index, child_id, args)
            if child["label"] != base.get("label"):
                relabel_count += 1
            for reason in child.get("refinement_reasons", []):
                reason_counts[reason] += 1
            rows.append(child)
            points_by_target[child["target_id"]] = points
            next_index += 1
    summary = {
        "input_targets": len(targets),
        "output_targets": len(rows),
        "missing_target_points": missing,
        "split_source_targets": split_count,
        "relabelled_targets": relabel_count,
        "input_label_counts": dict(Counter(str(t.get("label") or "unknown") for t in targets)),
        "output_label_counts": dict(Counter(str(t.get("label") or "unknown") for t in rows)),
        "refinement_reason_counts": dict(reason_counts),
        "output_points": int(sum(int(r.get("cluster_size") or 0) for r in rows)),
    }
    return rows, points_by_target, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets-jsonl", type=Path, required=True)
    parser.add_argument("--target-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-voxel", type=float, default=0.12)
    parser.add_argument("--surface-split-voxel", type=float, default=0.18)
    parser.add_argument("--railing-split-voxel", type=float, default=0.06)
    parser.add_argument("--car-split-voxel", type=float, default=0.12)
    parser.add_argument("--split-min-points", type=int, default=40)
    parser.add_argument("--surface-split-min-points", type=int, default=120)
    parser.add_argument("--min-output-points", type=int, default=20)
    parser.add_argument("--min-split-points", type=int, default=400)
    parser.add_argument("--surface-min-split-points", type=int, default=3000)
    parser.add_argument("--surface-max-extent", type=float, default=12.0)
    parser.add_argument("--surface-height-split-threshold", type=float, default=1.2)
    parser.add_argument("--surface-height-bin", type=float, default=0.7)
    parser.add_argument("--split-horizontal-wall-by-height", action="store_true")
    parser.add_argument("--surface-planarity", type=float, default=0.55)
    parser.add_argument("--railing-min-linearity", type=float, default=0.45)
    parser.add_argument("--railing-max-extent", type=float, default=6.0)
    parser.add_argument("--car-max-extent", type=float, default=8.0)
    parser.add_argument("--car-surface-max-linearity", type=float, default=0.20)
    parser.add_argument("--ground-min-normal-z", type=float, default=0.55)
    parser.add_argument("--guard-linear-ground-artifacts", action="store_true")
    parser.add_argument("--ground-artifact-min-height-span", type=float, default=1.2)
    parser.add_argument("--ground-artifact-min-linearity", type=float, default=0.75)
    parser.add_argument("--ground-artifact-max-planarity", type=float, default=0.25)
    parser.add_argument("--ground-artifact-wall-max-normal-z", type=float, default=0.72)
    parser.add_argument("--wall-max-normal-z", type=float, default=0.72)
    parser.add_argument("--enable-ceiling-label", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ceiling-min-z", type=float, default=2.5)
    parser.add_argument("--keep-residual", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = read_jsonl(args.targets_jsonl)
    ply_points = read_target_ply_points(args.target_ply)
    rows, points_by_target, summary = refine_targets(targets, ply_points, args)
    targets_out = args.output_dir / "frame_targets_refined.jsonl"
    ply_out = args.output_dir / "frame_targets_refined.ply"
    with targets_out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_ply(ply_out, rows, points_by_target)
    summary.update({
        "targets_jsonl": str(targets_out),
        "targets_ply": str(ply_out),
        "source_targets_jsonl": str(args.targets_jsonl),
        "source_target_ply": str(args.target_ply),
    })
    (args.output_dir / "geometry_refine_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
