#!/usr/bin/env python3
"""Classify whether frame-local targets are surfaces or attached objects.

This is the first clean fusion point between:

- first-touch visibility targets, which provide view-valid 3D evidence;
- structural regions from drivability_cpp, which provide non-semantic geometry;
- target PCA/shape cues, which decide whether something is part of a large
  surface or an object attached to that surface.

The output intentionally does not overwrite semantic labels.  It adds
attachment metadata that later object fusion can use.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_drivability_prior_to_residual import pack_query
from build_structural_region_field import (
    REGION_GROUND_LIKE,
    REGION_NAMES,
    REGION_OTHER_STRUCTURE,
    REGION_UNKNOWN,
    REGION_UPPER_HORIZONTAL,
    REGION_VERTICAL_SURFACE,
)


FINE_LABELS = {"car", "railing", "pipe", "equipment", "fine_candidate", "person", "vehicle"}
SURFACE_LABELS = {"ground", "floor", "wall", "building", "ceiling", "road", "grass"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def read_target_points_ply(path: Path) -> dict[int, np.ndarray]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        props: list[str] = []
        vertex_count = 0
        in_vertex = False
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
        idx = {name: i for i, name in enumerate(props)}
        target_col = idx.get("target", idx.get("target_index"))
        for required in ("x", "y", "z"):
            if required not in idx:
                raise ValueError(f"Target PLY missing {required}: {path}")
        if target_col is None:
            raise ValueError(f"Target PLY missing target/target_index: {path}")
        grouped: dict[int, list[list[float]]] = defaultdict(list)
        for _ in range(vertex_count):
            line = f.readline()
            if not line:
                break
            parts = line.strip().split()
            if len(parts) <= max(idx["x"], idx["y"], idx["z"], target_col):
                continue
            target = int(round(float(parts[target_col])))
            grouped[target].append([float(parts[idx["x"]]), float(parts[idx["y"]]), float(parts[idx["z"]])])
    return {target: np.asarray(points, dtype=np.float32) for target, points in grouped.items()}


def load_structural_field(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {
        "keys": data["keys"].astype(np.int64),
        "labels": data["labels"].astype(np.uint8),
        "confidence": data["confidence"].astype(np.float32),
        "spec": data["spec"].astype(np.int64),
        "voxel_size": float(data["voxel_size"][0]),
    }


def vote_structural_regions(points: np.ndarray, field: dict[str, Any], neighbor_radius: int) -> tuple[Counter, float]:
    counts: Counter[int] = Counter()
    if len(points) == 0:
        return counts, 0.0
    voxel_size = float(field["voxel_size"])
    coords0 = np.floor(points / voxel_size).astype(np.int32)
    offsets = [
        (dx, dy, dz)
        for dx in range(-neighbor_radius, neighbor_radius + 1)
        for dy in range(-neighbor_radius, neighbor_radius + 1)
        for dz in range(-neighbor_radius, neighbor_radius + 1)
    ]
    offsets.sort(key=lambda item: item[0] * item[0] + item[1] * item[1] + item[2] * item[2])
    keys_ref = field["keys"]
    labels_ref = field["labels"]
    confidence_ref = field["confidence"]
    confidence_sum = 0.0
    hit_points = 0
    assigned = np.full(len(points), REGION_UNKNOWN, dtype=np.uint8)
    assigned_conf = np.zeros(len(points), dtype=np.float32)
    for dx, dy, dz in offsets:
        unresolved = assigned == REGION_UNKNOWN
        if not np.any(unresolved):
            break
        coords = coords0[unresolved] + np.array([dx, dy, dz], dtype=np.int32)
        keys = pack_query(coords, field["spec"])
        pos = np.searchsorted(keys_ref, keys)
        pos_clip = np.clip(pos, 0, len(keys_ref) - 1)
        hit = (pos < len(keys_ref)) & (keys_ref[pos_clip] == keys)
        if not np.any(hit):
            continue
        unresolved_rows = np.where(unresolved)[0]
        rows = unresolved_rows[hit]
        assigned[rows] = labels_ref[pos_clip[hit]]
        assigned_conf[rows] = confidence_ref[pos_clip[hit]]
    counts.update(int(x) for x in assigned.tolist())
    hit_mask = assigned != REGION_UNKNOWN
    if np.any(hit_mask):
        hit_points = int(np.count_nonzero(hit_mask))
        confidence_sum = float(assigned_conf[hit_mask].sum())
    return counts, confidence_sum / max(hit_points, 1)


def pca_from_points(points: np.ndarray) -> dict[str, Any]:
    if len(points) < 3:
        return {"normal": [0.0, 0.0, 1.0], "linearity": 0.0, "planarity": 0.0, "scattering": 0.0, "thickness": 0.0}
    centered = points.astype(np.float64) - points.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(len(points) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    denom = max(float(vals[0]), 1e-12)
    normal = vecs[:, -1]
    if normal[2] < 0:
        normal = -normal
    return {
        "normal": [float(x) for x in normal],
        "linearity": float((vals[0] - vals[1]) / denom),
        "planarity": float((vals[1] - vals[2]) / denom),
        "scattering": float(vals[2] / denom),
        "thickness": float(np.sqrt(vals[-1])),
    }


def target_extent(target: dict[str, Any], points: np.ndarray) -> float:
    if len(points):
        return float((points.max(axis=0) - points.min(axis=0)).max())
    bbox = target.get("bbox_3d") or {}
    try:
        lo = np.asarray(bbox["min"], dtype=np.float64)
        hi = np.asarray(bbox["max"], dtype=np.float64)
        return float((hi - lo).max())
    except Exception:
        return 0.0


def classify_attachment(
    target: dict[str, Any],
    points: np.ndarray,
    counts: Counter,
    structural_confidence: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    total = max(sum(counts.values()), 1)
    ratios = {REGION_NAMES[k]: float(v / total) for k, v in sorted(counts.items())}
    dominant_region = max(counts, key=counts.get) if counts else REGION_UNKNOWN
    dominant_ratio = counts[dominant_region] / total
    pca = pca_from_points(points) if len(points) else target.get("pca", {})
    normal = np.asarray(pca.get("normal", [0.0, 0.0, 1.0]), dtype=np.float64)
    normal_abs_z = abs(float(normal[2])) if len(normal) >= 3 else 1.0
    linearity = float(pca.get("linearity", 0.0))
    planarity = float(pca.get("planarity", 0.0))
    scattering = float(pca.get("scattering", 0.0))
    thickness = float(pca.get("thickness", 0.0))
    extent = target_extent(target, points)
    label = str(target.get("label") or target.get("raw_label") or "unknown").lower()
    cluster_size = int(target.get("cluster_size") or len(points))

    horizontal = normal_abs_z >= args.horizontal_normal_z_min
    vertical = normal_abs_z <= args.vertical_normal_z_max
    planar = planarity >= args.surface_planarity_min and thickness <= args.surface_thickness_max
    large_surface = cluster_size >= args.large_surface_min_points or extent >= args.large_surface_min_extent
    fine_candidate = label in FINE_LABELS
    surface_candidate = label in SURFACE_LABELS
    line_like = linearity >= args.attached_linearity_min
    bulky_or_offset_like = scattering >= args.attached_scattering_min or thickness >= args.attached_thickness_min
    region_name = REGION_NAMES.get(int(dominant_region), "unknown")
    surface_label_agrees = (
        surface_candidate
        and dominant_region in {REGION_GROUND_LIKE, REGION_VERTICAL_SURFACE, REGION_UPPER_HORIZONTAL}
        and dominant_ratio >= args.surface_label_agreement_ratio
        and cluster_size >= args.surface_label_min_points
    )

    status = "ambiguous_surface_attachment"
    reason = "insufficient independent object or merge evidence"
    if dominant_ratio < args.structural_ratio_min:
        status = "independent_object_candidate" if fine_candidate else "unstructured_target"
        reason = "weak structural-region support"
    elif surface_label_agrees:
        status = "merge_to_structural_region"
        reason = "surface candidate label strongly agrees with structural region"
    elif dominant_region == REGION_GROUND_LIKE and horizontal and planar and large_surface and not fine_candidate:
        status = "merge_to_structural_region"
        reason = "large horizontal target compatible with ground-like region"
    elif dominant_region == REGION_VERTICAL_SURFACE and vertical and planar and large_surface and not line_like:
        status = "merge_to_structural_region"
        reason = "large vertical planar target compatible with vertical structural region"
    elif dominant_region == REGION_UPPER_HORIZONTAL and horizontal and planar and large_surface and not fine_candidate:
        status = "merge_to_structural_region"
        reason = "large upper-horizontal target compatible with structural region"
    elif dominant_region in {REGION_GROUND_LIKE, REGION_VERTICAL_SURFACE, REGION_UPPER_HORIZONTAL} and fine_candidate and (line_like or bulky_or_offset_like):
        status = "attached_object_candidate"
        reason = "fine target has independent geometry inside a structural region"
    elif dominant_region in {REGION_GROUND_LIKE, REGION_VERTICAL_SURFACE, REGION_UPPER_HORIZONTAL} and planar and not fine_candidate:
        status = "merge_to_structural_region"
        reason = "target geometry is surface-compatible even if not large"
    elif dominant_region == REGION_OTHER_STRUCTURE:
        status = "independent_object_candidate"
        reason = "dominant structural prior is other/non-surface"

    return {
        "surface_attachment_status": status,
        "surface_attachment_reason": reason,
        "dominant_structural_region": region_name,
        "dominant_structural_region_ratio": float(dominant_ratio),
        "structural_region_ratios": ratios,
        "structural_confidence_mean": float(structural_confidence),
        "attachment_geometry": {
            "normal_abs_z": normal_abs_z,
            "linearity": linearity,
            "planarity": planarity,
            "scattering": scattering,
            "thickness": thickness,
            "extent": extent,
        },
        "surface_locked": status == "merge_to_structural_region",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-jsonl", type=Path, required=True)
    parser.add_argument("--target-ply", type=Path, required=True)
    parser.add_argument("--structural-field", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument("--structural-ratio-min", type=float, default=0.35)
    parser.add_argument("--horizontal-normal-z-min", type=float, default=0.86)
    parser.add_argument("--vertical-normal-z-max", type=float, default=0.42)
    parser.add_argument("--surface-planarity-min", type=float, default=0.72)
    parser.add_argument("--surface-thickness-max", type=float, default=0.35)
    parser.add_argument("--large-surface-min-points", type=int, default=300)
    parser.add_argument("--large-surface-min-extent", type=float, default=1.2)
    parser.add_argument("--surface-label-agreement-ratio", type=float, default=0.75)
    parser.add_argument("--surface-label-min-points", type=int, default=120)
    parser.add_argument("--attached-linearity-min", type=float, default=0.55)
    parser.add_argument("--attached-scattering-min", type=float, default=0.08)
    parser.add_argument("--attached-thickness-min", type=float, default=0.10)
    args = parser.parse_args()

    targets = read_jsonl(args.targets_jsonl)
    points_by_target = read_target_points_ply(args.target_ply)
    field = load_structural_field(args.structural_field)

    out_rows = []
    status_counts: Counter[str] = Counter()
    region_counts: Counter[str] = Counter()
    missing_points = 0
    for target in targets:
        target_index = int(target.get("target_index", -1))
        points = points_by_target.get(target_index, np.empty((0, 3), dtype=np.float32))
        if len(points) == 0:
            missing_points += 1
        counts, confidence = vote_structural_regions(points, field, args.neighbor_radius)
        attachment = classify_attachment(target, points, counts, confidence, args)
        row = dict(target)
        row.update(attachment)
        out_rows.append(row)
        status_counts[str(attachment["surface_attachment_status"])] += 1
        region_counts[str(attachment["dominant_structural_region"])] += 1

    write_jsonl(args.output_jsonl, out_rows)
    report = {
        "targets_jsonl": str(args.targets_jsonl),
        "target_ply": str(args.target_ply),
        "structural_field": str(args.structural_field),
        "output_jsonl": str(args.output_jsonl),
        "target_count": len(targets),
        "missing_target_points": missing_points,
        "status_counts": dict(status_counts),
        "dominant_region_counts": dict(region_counts),
        "parameters": {
            "neighbor_radius": args.neighbor_radius,
            "structural_ratio_min": args.structural_ratio_min,
            "surface_planarity_min": args.surface_planarity_min,
            "surface_thickness_max": args.surface_thickness_max,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
