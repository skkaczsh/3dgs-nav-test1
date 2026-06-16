#!/usr/bin/env python3
"""Refine target-fusion objects with text and geometry guards.

This operates on ``fuse_targets_to_objects.py`` output and focuses on the
coarse surface labels that remain noisy after target-level geometry guards:
``floor``, ``wall``, ``ceiling``, and ``building``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


SURFACE = {"floor", "wall", "ceiling", "building", "other", "ambiguous"}
FINE = {"railing", "pipe", "equipment", "person", "car", "tree", "grass"}

PATTERNS = {
    "ceiling": re.compile(r"\b(ceiling|overhead|underside|soffit|roof underside|roof panel underside)\b", re.I),
    "floor": re.compile(
        r"\b(floor|ground|rooftop floor|roof floor|roof surface|rooftop surface|"
        r"walkable|platform|pavement|concrete surface|deck)\b",
        re.I,
    ),
    "wall": re.compile(
        r"\b(wall|facade|façade|parapet|partition|vertical plane|vertical surface|"
        r"side wall|retaining wall|building facade)\b",
        re.I,
    ),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def vote_ratio(obj: dict[str, Any], label: str) -> float:
    votes = normalize_votes(obj.get("label_votes"))
    total = sum(votes.values())
    return float(votes.get(label, 0.0) / max(total, 1e-9))


def text_blob(obj: dict[str, Any]) -> str:
    parts = [
        obj.get("description", ""),
        obj.get("display_identity", ""),
        obj.get("object_identity", ""),
    ]
    for field in ("description_votes", "freeform_label_votes"):
        values = obj.get(field) or {}
        if isinstance(values, dict):
            parts.extend(str(k) for k in values.keys())
    attrs = obj.get("dominant_attributes") or {}
    if isinstance(attrs, dict):
        for value in attrs.values():
            if isinstance(value, dict):
                parts.append(str(value.get("value", "")))
    return " ".join(str(p) for p in parts if p).lower()


def geometry(obj: dict[str, Any]) -> dict[str, float | bool]:
    normal = obj.get("normal") or [0.0, 0.0, 1.0]
    try:
        nz = abs(float(normal[2]))
    except (TypeError, ValueError, IndexError):
        nz = 1.0
    bbox = obj.get("bbox_3d") or {}
    bmin = np.array(bbox.get("min", [0.0, 0.0, 0.0]), dtype=np.float64)
    bmax = np.array(bbox.get("max", [0.0, 0.0, 0.0]), dtype=np.float64)
    extent = np.maximum(bmax - bmin, 1e-9)
    max_extent = float(extent.max())
    min_extent = float(extent.min())
    horiz_area = float(extent[0] * extent[1])
    return {
        "normal_abs_z": nz,
        "max_extent": max_extent,
        "min_extent": min_extent,
        "horiz_area": horiz_area,
        "is_horizontal": nz >= 0.72,
        "is_vertical": nz <= 0.40,
        "z_span": float(extent[2]),
        "min_z": float(bmin[2]),
        "max_z": float(bmax[2]),
    }


def xy_overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    amin = np.array((a.get("bbox_3d") or {}).get("min", [0, 0, 0])[:2], dtype=np.float64)
    amax = np.array((a.get("bbox_3d") or {}).get("max", [0, 0, 0])[:2], dtype=np.float64)
    bmin = np.array((b.get("bbox_3d") or {}).get("min", [0, 0, 0])[:2], dtype=np.float64)
    bmax = np.array((b.get("bbox_3d") or {}).get("max", [0, 0, 0])[:2], dtype=np.float64)
    inter = np.maximum(0.0, np.minimum(amax, bmax) - np.maximum(amin, bmin))
    inter_area = float(inter[0] * inter[1])
    a_area = float(np.prod(np.maximum(amax - amin, 1e-6)))
    b_area = float(np.prod(np.maximum(bmax - bmin, 1e-6)))
    return inter_area / max(min(a_area, b_area), 1e-9)


def build_ceiling_context(objects: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    horizontal = [(obj, geometry(obj)) for obj in objects if str(obj.get("semantic_label") or "unknown") in SURFACE]
    horizontal = [(obj, geom) for obj, geom in horizontal if bool(geom["is_horizontal"])]
    context: dict[str, dict[str, Any]] = {}
    for upper, upper_geom in horizontal:
        upper_text = text_blob(upper)
        has_ceiling_text = bool(PATTERNS["ceiling"].search(upper_text))
        has_wall_text = bool(PATTERNS["wall"].search(upper_text))
        upper_voxels = int(upper.get("point_count") or upper.get("voxel_count") or 0)
        upper_label = str(upper.get("semantic_label") or "unknown")
        for lower, lower_geom in horizontal:
            if lower is upper:
                continue
            dz = float(upper_geom["min_z"] - lower_geom["min_z"])
            if dz < args.ceiling_min_z_gap or dz > args.ceiling_max_z_gap:
                continue
            if xy_overlap_ratio(upper, lower) < args.ceiling_min_xy_overlap:
                continue
            lower_label = str(lower.get("semantic_label") or "unknown")
            if lower_label not in {"floor", "ambiguous", "wall"}:
                continue
            if upper_label not in {"floor", "wall"}:
                continue
            if upper_label == "floor" and not has_ceiling_text:
                continue
            if has_wall_text and not has_ceiling_text:
                continue
            if upper_voxels < args.ceiling_min_points:
                continue
            if not has_ceiling_text and upper_voxels > args.ceiling_max_no_text_points:
                continue
            context[str(upper.get("object_id"))] = {
                "height_layer_ceiling": True,
                "lower_object_id": str(lower.get("object_id")),
                "z_gap": dz,
                "xy_overlap": xy_overlap_ratio(upper, lower),
            }
            break
    return context


def choose_label(obj: dict[str, Any], context: dict[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    original = str(obj.get("semantic_label") or "unknown")
    if original in FINE:
        return original, "fine_passthrough"

    geom = geometry(obj)
    text = text_blob(obj)
    has_floor = bool(PATTERNS["floor"].search(text))
    has_wall = bool(PATTERNS["wall"].search(text))
    has_ceiling = bool(PATTERNS["ceiling"].search(text))

    if context.get("height_layer_ceiling") and bool(geom["is_horizontal"]):
        return "ceiling", "height_layer_ceiling"
    if has_ceiling and bool(geom["is_horizontal"]):
        return "ceiling", "text_ceiling_horizontal"
    if original in {"floor", "wall", "building", "ambiguous"} and has_floor and bool(geom["is_horizontal"]):
        return "floor", "text_floor_horizontal"
    if has_wall and bool(geom["is_vertical"]):
        return "wall", "text_wall_vertical"

    # Strong correction for "rooftop floor" text trapped inside a tall merged wall object.
    if original == "wall" and has_floor and vote_ratio(obj, "ceiling") < args.wall_floor_max_ceiling_ratio:
        if geom["horiz_area"] >= args.wall_floor_min_area and geom["max_extent"] >= args.wall_floor_min_extent:
            return "floor", "wall_text_floor_large_surface"

    # Building is often used as a fallback surface bucket. Resolve by geometry + text.
    if original == "building":
        if has_floor and bool(geom["is_horizontal"]):
            return "floor", "building_text_floor_horizontal"
        if has_wall and bool(geom["is_vertical"]):
            return "wall", "building_text_wall_vertical"

    # Conservative geometry fallback.
    if original in {"floor", "wall", "building", "other", "ambiguous"}:
        if bool(geom["is_horizontal"]) and vote_ratio(obj, "floor") >= 0.15:
            return "floor", "geometry_horizontal_surface"
        if bool(geom["is_vertical"]) and (vote_ratio(obj, "wall") >= 0.12 or vote_ratio(obj, "building") >= 0.10):
            return "wall", "geometry_vertical_surface"

    return original, "kept_original"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ceiling-min-z-gap", type=float, default=1.4)
    parser.add_argument("--ceiling-max-z-gap", type=float, default=4.0)
    parser.add_argument("--ceiling-min-xy-overlap", type=float, default=0.20)
    parser.add_argument("--ceiling-min-points", type=int, default=120)
    parser.add_argument("--ceiling-max-no-text-points", type=int, default=800)
    parser.add_argument("--wall-floor-max-ceiling-ratio", type=float, default=0.10)
    parser.add_argument("--wall-floor-min-area", type=float, default=4.0)
    parser.add_argument("--wall-floor-min-extent", type=float, default=2.0)
    args = parser.parse_args()

    objects = load_jsonl(args.objects_jsonl)
    ceiling_context = build_ceiling_context(objects, args)
    before = Counter()
    after = Counter()
    reasons = Counter()
    changed: list[dict[str, Any]] = []
    refined: list[dict[str, Any]] = []

    for obj in objects:
        original = str(obj.get("semantic_label") or "unknown")
        context = ceiling_context.get(str(obj.get("object_id")), {})
        new_label, reason = choose_label(obj, context, args)
        out = dict(obj)
        out["semantic_label_original"] = original
        out["semantic_label"] = new_label
        out["refine_reason"] = reason
        if context:
            out["height_layer_context"] = context
        refined.append(out)
        before[original] += 1
        after[new_label] += 1
        reasons[reason] += 1
        if new_label != original:
            changed.append({
                "object_id": obj.get("object_id"),
                "from": original,
                "to": new_label,
                "reason": reason,
                "point_count": obj.get("point_count"),
                "description": obj.get("description", ""),
            })

    write_jsonl(args.output_jsonl, refined)
    report = {
        "objects_jsonl": str(args.objects_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "object_count": len(objects),
        "changed_count": len(changed),
        "changed_ratio": float(len(changed) / max(len(objects), 1)),
        "height_layer_ceiling_candidates": len(ceiling_context),
        "label_counts_before": dict(before),
        "label_counts_after": dict(after),
        "reason_counts": dict(reasons),
        "sample_changes": changed[:100],
        "params": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "changed_count": report["changed_count"],
        "changed_ratio": report["changed_ratio"],
        "height_layer_ceiling_candidates": report["height_layer_ceiling_candidates"],
        "label_counts_after": report["label_counts_after"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
