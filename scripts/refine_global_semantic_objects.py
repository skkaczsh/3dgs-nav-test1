#!/usr/bin/env python3
"""Refine global semantic voxel objects with text and geometry guards.

This script takes the global voxel vote output produced by
``build_global_semantic_votes.py`` and re-labels coarse objects using:

1. free-form identity / description votes from the VLM,
2. object-level PCA geometry from voxel centroids,
3. conservative fine-object guards for railings, pipes, and equipment.

It intentionally operates after target/object aggregation. The goal is to fix
systematic coarse-label failures such as rooftop floor being voted as wall, or
thin railings being absorbed into large surfaces, without rerunning SAM/VLM.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from project_semantic import LABEL_COLORS, LABEL_NAMES
except ImportError:  # pragma: no cover - allows running from copied script dirs
    LABEL_NAMES = {
        0: "unknown", 1: "other", 2: "wall", 3: "floor", 4: "ceiling",
        5: "grass", 6: "tree", 7: "person", 8: "car", 9: "railing",
        10: "building", 11: "sky", 12: "road", 13: "water",
        14: "furniture", 15: "pipe", 16: "equipment", 255: "ignore",
    }
    LABEL_COLORS = {
        0: (120, 120, 120), 1: (210, 210, 210), 2: (160, 160, 165),
        3: (139, 100, 60), 4: (180, 180, 210), 5: (70, 150, 80),
        6: (40, 120, 60), 7: (230, 80, 80), 8: (80, 110, 230),
        9: (245, 210, 50), 10: (190, 170, 140), 11: (70, 150, 220),
        12: (90, 90, 90), 13: (50, 120, 200), 14: (180, 100, 200),
        15: (240, 140, 40), 16: (30, 210, 190), 255: (20, 20, 20),
    }


LABEL_IDS = {name: idx for idx, name in LABEL_NAMES.items()}
SURFACE_LABELS = {"floor", "wall", "ceiling", "building", "other", "ambiguous", "unknown"}
FINE_LABELS = {"railing", "pipe", "equipment", "furniture", "person", "car", "tree", "grass"}

TERM_PATTERNS = {
    "ceiling": re.compile(r"\b(ceiling|overhead|underside|soffit|roof underside)\b", re.I),
    "floor": re.compile(
        r"\b(floor|ground|rooftop floor|roof floor|roof surface|rooftop surface|"
        r"walkable|platform|pavement|horizontal plane|horizontal surface|concrete surface)\b",
        re.I,
    ),
    "wall": re.compile(
        r"\b(wall|facade|façade|parapet|partition|vertical plane|vertical surface|"
        r"corrugated metal wall|side wall)\b",
        re.I,
    ),
    "railing": re.compile(r"\b(railing|guardrail|handrail|fence|barrier|balustrade)\b", re.I),
    "pipe": re.compile(r"\b(pipe|conduit|cable|duct|tube|hose|wire)\b", re.I),
    "equipment": re.compile(
        r"\b(equipment|hvac|air[- ]?conditioning|outdoor unit|unit|machine|cabinet|"
        r"box|device|fixture|sensor|antenna|container|case)\b",
        re.I,
    ),
}


def semantic_id(label: str) -> int:
    return int(LABEL_IDS.get(label, 0))


def semantic_color(label: str) -> tuple[int, int, int]:
    return tuple(int(x) for x in LABEL_COLORS.get(semantic_id(label), LABEL_COLORS[0]))


def dominant(votes: dict[str, float], default: str = "unknown") -> tuple[str, float, float]:
    if not votes:
        return default, 0.0, 0.0
    total = float(sum(float(v) for v in votes.values()))
    key, value = max(votes.items(), key=lambda kv: float(kv[1]))
    return str(key), float(value), float(value) / max(total, 1e-9)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_votes(votes: Any) -> dict[str, float]:
    if not isinstance(votes, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in votes.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def combined_text(obj: dict[str, Any]) -> str:
    parts = [
        obj.get("display_identity", ""),
        obj.get("description", ""),
    ]
    for field in ("identity_votes", "description_votes"):
        votes = normalize_votes(obj.get(field))
        for text, _ in sorted(votes.items(), key=lambda kv: kv[1], reverse=True)[:8]:
            parts.append(text)
    return " ".join(str(p) for p in parts if p).lower()


def text_hits(text: str) -> set[str]:
    return {label for label, pattern in TERM_PATTERNS.items() if pattern.search(text)}


def weighted_text_ratio(obj: dict[str, Any], category: str) -> float:
    pattern = TERM_PATTERNS[category]
    total = 0.0
    matched = 0.0
    for field in ("identity_votes", "description_votes"):
        for text, weight in normalize_votes(obj.get(field)).items():
            total += weight
            if pattern.search(text):
                matched += weight
    return matched / max(total, 1e-9)


def geometry_stats(points: np.ndarray, fallback_bbox: dict[str, Any] | None = None) -> dict[str, Any]:
    if points.size == 0:
        points = np.zeros((1, 3), dtype=np.float64)
    centroid = points.mean(axis=0)
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    if len(points) < 3 and fallback_bbox:
        try:
            pmin = np.array(fallback_bbox["min"], dtype=np.float64)
            pmax = np.array(fallback_bbox["max"], dtype=np.float64)
        except Exception:
            pass
    extent = np.maximum(pmax - pmin, 1e-9)

    if len(points) >= 3:
        cov = np.cov((points - centroid).T)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals = np.maximum(vals[order], 0.0)
        vecs = vecs[:, order]
    else:
        vals = np.array([extent.max() ** 2, np.median(extent) ** 2, extent.min() ** 2], dtype=np.float64)
        vecs = np.eye(3, dtype=np.float64)

    l1, l2, l3 = [float(x) for x in np.maximum(vals, 1e-12)]
    normal = vecs[:, 2]
    normal_abs_z = float(abs(normal[2]))
    linearity = float((l1 - l2) / max(l1, 1e-12))
    planarity = float((l2 - l3) / max(l1, 1e-12))
    scatter = float(l3 / max(l1, 1e-12))
    sorted_extent = sorted((float(x) for x in extent), reverse=True)
    slenderness = sorted_extent[0] / max(sorted_extent[1], 1e-6)
    return {
        "centroid": [float(x) for x in centroid],
        "bbox_3d": {"min": [float(x) for x in pmin], "max": [float(x) for x in pmax]},
        "extent": [float(x) for x in extent],
        "normal_abs_z": normal_abs_z,
        "linearity": linearity,
        "planarity": planarity,
        "scatter": scatter,
        "slenderness": float(slenderness),
        "max_extent": float(sorted_extent[0]),
        "mid_extent": float(sorted_extent[1]),
        "min_extent": float(sorted_extent[2]),
        "is_horizontal": normal_abs_z >= 0.70 and planarity >= 0.03,
        "is_vertical": normal_abs_z <= 0.45 and planarity >= 0.03,
        "is_linear": linearity >= 0.70 and slenderness >= 2.5,
        "is_large_planar": sorted_extent[0] >= 0.9 and sorted_extent[1] >= 0.28 and planarity >= 0.03,
    }


def vote_ratio(obj: dict[str, Any], label: str) -> float:
    votes = normalize_votes(obj.get("label_votes"))
    total = sum(votes.values())
    return float(votes.get(label, 0.0) / max(total, 1e-9))


def choose_label(obj: dict[str, Any], geom: dict[str, Any], context: dict[str, Any] | None = None) -> tuple[str, str]:
    original = str(obj.get("semantic_label") or "unknown")
    text = combined_text(obj)
    primary_text = " ".join(str(obj.get(k) or "") for k in ("display_identity", "description")).lower()
    hits = text_hits(text)
    primary_hits = text_hits(primary_text)
    label_votes = normalize_votes(obj.get("label_votes"))
    dom_label, _, dom_ratio = dominant(label_votes, original)
    ratios = {name: weighted_text_ratio(obj, name) for name in TERM_PATTERNS}
    context = context or {}

    # Fine object protection comes first. Large planar objects are allowed to
    # stay surface-like unless the text is explicitly fine-object oriented.
    railing_geometry = geom["linearity"] >= 0.84 and geom["max_extent"] >= 0.6
    if original == "railing" and "railing" not in primary_hits and not railing_geometry:
        if geom["is_vertical"] or "wall" in primary_hits:
            return "wall", "railing_rejected_non_linear_wall"
        if geom["is_horizontal"] or "floor" in primary_hits:
            return "floor", "railing_rejected_non_linear_floor"
        return "other", "railing_rejected_non_linear_other"
    if "railing" in hits and (
        "railing" in primary_hits
        or ratios["railing"] >= 0.30
        or vote_ratio(obj, "railing") >= 0.08
        or (railing_geometry and ratios["railing"] >= 0.015)
    ):
        return "railing", "text_or_linear_railing_guard"
    if "pipe" in hits and (
        ratios["pipe"] >= 0.18 or vote_ratio(obj, "pipe") >= 0.05 or (geom["is_linear"] and ratios["pipe"] >= 0.04)
    ):
        return "pipe", "text_or_linear_pipe_guard"
    surface_text_ratio = max(ratios["floor"], ratios["wall"], ratios["ceiling"])
    equipment_supported = ratios["equipment"] >= 0.30 or vote_ratio(obj, "equipment") >= 0.18 or dom_label == "equipment"
    if "equipment" in hits and equipment_supported and surface_text_ratio < 0.55 and not (
        geom["is_large_planar"] and ("floor" in hits or "wall" in hits)
    ):
        return "equipment", "text_equipment_guard"

    # Explicit ceiling text should override floor when geometry is horizontal.
    if "ceiling" in hits and geom["is_horizontal"]:
        return "ceiling", "text_ceiling_horizontal"
    if context.get("height_layer_ceiling") and geom["is_horizontal"] and original in SURFACE_LABELS | {"floor", "wall"}:
        return "ceiling", "height_layer_ceiling"

    # Surface relabeling: use free text plus PCA orientation to resolve common
    # floor/wall inversions caused by forcing VLM into a finite label set.
    if "wall" in hits and geom["is_vertical"]:
        return "wall", "text_wall_vertical"
    if "floor" in hits and geom["is_horizontal"]:
        return "floor", "text_floor_horizontal"

    if dom_label in SURFACE_LABELS or original in SURFACE_LABELS:
        if geom["is_vertical"] and (vote_ratio(obj, "wall") >= 0.15 or "wall" in hits or "building" in hits):
            return "wall", "geometry_vertical_surface"
        if geom["is_horizontal"] and (vote_ratio(obj, "floor") >= 0.12 or "floor" in hits):
            return "floor", "geometry_horizontal_surface"

    if dom_ratio >= 0.8:
        return dom_label, "dominant_vote_high_confidence"
    return original, "kept_original_ambiguous"


def xy_overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    amin = np.array(a["bbox_3d"]["min"][:2], dtype=np.float64)
    amax = np.array(a["bbox_3d"]["max"][:2], dtype=np.float64)
    bmin = np.array(b["bbox_3d"]["min"][:2], dtype=np.float64)
    bmax = np.array(b["bbox_3d"]["max"][:2], dtype=np.float64)
    inter = np.maximum(0.0, np.minimum(amax, bmax) - np.maximum(amin, bmin))
    inter_area = float(inter[0] * inter[1])
    a_area = float(np.prod(np.maximum(amax - amin, 1e-6)))
    b_area = float(np.prod(np.maximum(bmax - bmin, 1e-6)))
    return inter_area / max(min(a_area, b_area), 1e-9)


def build_height_layer_context(
    objects: list[dict[str, Any]],
    geoms: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[int, dict[str, Any]]:
    if not bool(getattr(args, "enable_height_layer_ceiling", False)):
        return {}
    candidates: dict[int, dict[str, Any]] = {}
    horizontal = []
    for obj in objects:
        obj_num = int(obj.get("object_number") or 0)
        geom = geoms.get(obj_num, {})
        label = str(obj.get("semantic_label") or "unknown")
        if not geom.get("is_horizontal") or label in FINE_LABELS:
            continue
        horizontal.append(obj)
    for upper in horizontal:
        upper_num = int(upper.get("object_number") or 0)
        upper_z = float(upper.get("centroid", [0, 0, 0])[2])
        upper_text = combined_text(upper)
        text_support = bool(
            TERM_PATTERNS["ceiling"].search(upper_text)
            or re.search(r"\b(interior|indoor|roof panel|metal roof|underside|overhead)\b", upper_text, re.I)
        )
        for lower in horizontal:
            if lower is upper:
                continue
            lower_z = float(lower.get("centroid", [0, 0, 0])[2])
            dz = upper_z - lower_z
            if dz < args.ceiling_min_z_gap or dz > args.ceiling_max_z_gap:
                continue
            if xy_overlap_ratio(upper, lower) < args.ceiling_min_xy_overlap:
                continue
            lower_label = str(lower.get("semantic_label") or "unknown")
            if lower_label not in {"floor", "other", "ambiguous", "wall"}:
                continue
            # Without any text support, only convert small/medium upper layers;
            # large roof decks are too easy to misclassify as ceilings.
            if not text_support and int(upper.get("voxel_count") or 0) > args.ceiling_max_no_text_voxels:
                continue
            candidates[upper_num] = {
                "height_layer_ceiling": True,
                "height_layer_lower_object": int(lower.get("object_number") or 0),
                "height_layer_z_gap": float(dz),
                "height_layer_xy_overlap": float(xy_overlap_ratio(upper, lower)),
            }
            break
    return candidates


def first_existing(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return ""


def load_voxel_groups(voxels_path: Path) -> tuple[dict[int, list[np.ndarray]], Counter[str]]:
    groups: dict[int, list[np.ndarray]] = defaultdict(list)
    voxel_labels: Counter[str] = Counter()
    with voxels_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            obj_num = int(row.get("object_number") or 0)
            if obj_num:
                groups[obj_num].append(np.array(row["centroid"], dtype=np.float64))
            voxel_labels[str(row.get("label") or "unknown")] += 1
    return groups, voxel_labels


def refine_objects(
    objects: list[dict[str, Any]],
    groups: dict[int, list[np.ndarray]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[int, str], dict[str, Any]]:
    refined: list[dict[str, Any]] = []
    label_by_object: dict[int, str] = {}
    reason_counts: Counter[str] = Counter()
    object_label_before: Counter[str] = Counter()
    object_label_after: Counter[str] = Counter()
    changed: list[dict[str, Any]] = []
    geoms: dict[int, dict[str, Any]] = {}
    for obj in objects:
        obj_num = int(obj.get("object_number") or 0)
        points = np.array(groups.get(obj_num, []), dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,):
            points = np.empty((0, 3), dtype=np.float64)
        geoms[obj_num] = geometry_stats(points, obj.get("bbox_3d"))
    height_context = build_height_layer_context(objects, geoms, args)

    for obj in objects:
        obj_num = int(obj.get("object_number") or 0)
        geom = geoms[obj_num]
        original = str(obj.get("semantic_label") or "unknown")
        context = height_context.get(obj_num, {})
        new_label, reason = choose_label(obj, geom, context)

        out = dict(obj)
        out["object_id"] = f"global_obj_{obj_num:06d}" if obj_num else str(obj.get("object_id") or "")
        out["semantic_label_original"] = original
        out["semantic_label"] = new_label
        out["refined_from"] = original
        out["refine_reason"] = reason
        out["geometry_stats"] = geom
        if context:
            out["height_layer_context"] = context
        out["display_identity"] = first_existing(obj.get("display_identity"), obj.get("identity_hint"), obj.get("description"), new_label)
        out["description"] = first_existing(obj.get("description"), obj.get("display_identity"), new_label)
        if original != new_label:
            out["status"] = "refined_" + str(obj.get("status") or "object")
            changed.append({
                "object_number": obj_num,
                "from": original,
                "to": new_label,
                "reason": reason,
                "voxel_count": obj.get("voxel_count"),
                "point_count": obj.get("point_count"),
                "identity": out["display_identity"],
            })
        refined.append(out)
        label_by_object[obj_num] = new_label
        reason_counts[reason] += 1
        object_label_before[original] += 1
        object_label_after[new_label] += 1

    report = {
        "object_count": len(objects),
        "changed_count": len(changed),
        "height_layer_ceiling_candidates": len(height_context),
        "reason_counts": dict(reason_counts),
        "object_label_counts_before": dict(object_label_before),
        "object_label_counts_after": dict(object_label_after),
        "sample_changes": changed[:50],
    }
    return refined, label_by_object, report


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_refined_voxel_ply(
    voxels_path: Path,
    output_path: Path,
    label_by_object: dict[int, str],
    min_voxel_points: int,
) -> Counter[str]:
    kept_rows: list[tuple[list[float], list[float], int, str, float, int]] = []
    label_counts: Counter[str] = Counter()
    with voxels_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            point_count = int(row.get("point_count") or 0)
            if point_count < min_voxel_points:
                continue
            obj_num = int(row.get("object_number") or 0)
            label = label_by_object.get(obj_num, str(row.get("label") or "unknown"))
            label_counts[label] += 1
            kept_rows.append((
                [float(x) for x in row["centroid"]],
                [float(x) for x in row.get("mean_color", [120, 120, 120])],
                obj_num,
                label,
                float(row.get("label_purity") or 0.0),
                point_count,
            ))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(kept_rows)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property int object\nproperty uchar semantic\n")
        f.write("property float purity\n")
        f.write("property int votes\n")
        f.write("end_header\n")
        for point, _rgb, obj_num, label, purity, votes in kept_rows:
            color = semantic_color(label)
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{color[0]} {color[1]} {color[2]} "
                f"{obj_num} {semantic_id(label)} {purity:.6f} {votes}\n"
            )
    return label_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voxels-jsonl", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-voxel-points", type=int, default=1)
    parser.add_argument("--enable-height-layer-ceiling", action="store_true")
    parser.add_argument("--ceiling-min-z-gap", type=float, default=1.4)
    parser.add_argument("--ceiling-max-z-gap", type=float, default=4.0)
    parser.add_argument("--ceiling-min-xy-overlap", type=float, default=0.20)
    parser.add_argument("--ceiling-max-no-text-voxels", type=int, default=80)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups, voxel_label_before = load_voxel_groups(args.voxels_jsonl)
    objects = load_jsonl(args.objects_jsonl)
    refined, label_by_object, report = refine_objects(objects, groups, args)
    voxel_label_after = write_refined_voxel_ply(
        args.voxels_jsonl,
        args.output_dir / "global_semantic_voxels_refined.ply",
        label_by_object,
        args.min_voxel_points,
    )
    write_jsonl(args.output_dir / "global_semantic_objects_refined.jsonl", refined)
    report.update({
        "voxels_jsonl": str(args.voxels_jsonl),
        "objects_jsonl": str(args.objects_jsonl),
        "output_dir": str(args.output_dir),
        "voxel_label_counts_before": dict(voxel_label_before),
        "voxel_label_counts_after": dict(voxel_label_after),
        "params": {
            "min_voxel_points": args.min_voxel_points,
        },
    })
    (args.output_dir / "global_semantic_refine_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
