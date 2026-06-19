#!/usr/bin/env python3
"""Absorb weak fine-object targets into nearby trusted surface targets.

The parking route currently produces many low-point ``car`` / ``railing``
fragments. Keeping each fragment as an independent target makes object fusion
look noisy. This script is a conservative JSONL preprocessor that only relabels
fine targets when both conditions hold:

1. the fine target itself is weak or surface-like;
2. a same-frame/camera surface target is geometrically nearby.

It does not touch the target PLY. Downstream fusion uses the relabelled JSONL,
while ``raw_label`` and absorption metadata preserve provenance.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


FINE_LABELS = {"car", "railing"}
SURFACE_LABELS = {"ground", "wall", "ceiling", "grass"}
PARENT_BY_LABEL = {
    "ground": "surface",
    "wall": "surface",
    "ceiling": "surface",
    "grass": "vegetation",
    "car": "object",
    "railing": "structure",
    "other": "other",
}
PRIORITY_ID_BY_LABEL = {
    "ground": 1,
    "wall": 2,
    "grass": 3,
    "car": 4,
    "railing": 5,
}


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
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def bbox_arrays(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    bbox = row.get("bbox_3d") or {}
    lo = np.array(bbox.get("min", [0.0, 0.0, 0.0]), dtype=np.float64)
    hi = np.array(bbox.get("max", [0.0, 0.0, 0.0]), dtype=np.float64)
    return lo, hi


def bbox_gap(a: dict[str, Any], b: dict[str, Any]) -> float:
    amin, amax = bbox_arrays(a)
    bmin, bmax = bbox_arrays(b)
    gap = np.maximum(0.0, np.maximum(bmin - amax, amin - bmax))
    return float(np.linalg.norm(gap))


def centroid_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    ac = np.array(a.get("centroid", [0.0, 0.0, 0.0]), dtype=np.float64)
    bc = np.array(b.get("centroid", [0.0, 0.0, 0.0]), dtype=np.float64)
    return float(np.linalg.norm(ac - bc))


def color_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    ac = np.array(a.get("mean_color", [0.0, 0.0, 0.0]), dtype=np.float64)
    bc = np.array(b.get("mean_color", [0.0, 0.0, 0.0]), dtype=np.float64)
    return float(np.linalg.norm(ac - bc))


def pca_value(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    pca = row.get("pca") or {}
    try:
        return float(pca.get(key, default))
    except (TypeError, ValueError):
        return default


def normal_z(row: dict[str, Any]) -> float:
    normal = (row.get("pca") or {}).get("normal") or [0.0, 0.0, 1.0]
    try:
        return abs(float(normal[2]))
    except (TypeError, ValueError, IndexError):
        return 1.0


def dims(row: dict[str, Any]) -> np.ndarray:
    lo, hi = bbox_arrays(row)
    return hi - lo


def is_surface_like_fine(row: dict[str, Any], args: argparse.Namespace) -> bool:
    d = dims(row)
    label = str(row.get("label") or "unknown")
    nz = normal_z(row)
    linearity = pca_value(row, "linearity")
    planarity = pca_value(row, "planarity")
    if label == "car":
        return nz >= args.surface_like_normal_z and float(d[2]) <= args.car_surface_like_max_z_span
    if label == "railing":
        return (
            nz >= args.surface_like_normal_z
            and float(d[2]) <= args.railing_surface_like_max_z_span
            and (planarity >= args.railing_surface_like_min_planarity or linearity < args.railing_keep_linearity)
        )
    return False


def is_weak_fine(row: dict[str, Any], args: argparse.Namespace) -> bool:
    label = str(row.get("label") or "unknown")
    if label not in FINE_LABELS:
        return False
    size = int(row.get("cluster_size") or 0)
    if is_surface_like_fine(row, args):
        return True
    if label == "car" and size <= args.car_max_absorb_points:
        return True
    if label == "railing" and size <= args.railing_max_absorb_points:
        return True
    return False


def is_horizontal_surface_artifact(row: dict[str, Any], args: argparse.Namespace) -> bool:
    d = dims(row)
    return (
        normal_z(row) >= args.surface_like_normal_z
        and float(d[2]) <= args.horizontal_absorb_max_z_span
        and pca_value(row, "linearity") < args.railing_keep_linearity
    )


def is_wall_surface_artifact(row: dict[str, Any], args: argparse.Namespace) -> bool:
    d = dims(row)
    return (
        normal_z(row) <= args.wall_absorb_max_normal_z
        and float(d[2]) >= args.wall_absorb_min_z_span
        and pca_value(row, "planarity") >= args.wall_absorb_min_planarity
        and pca_value(row, "linearity") < args.railing_keep_linearity
    )


def is_surface_compatible(source: dict[str, Any], surface: dict[str, Any], args: argparse.Namespace) -> bool:
    surface_label = str(surface.get("label") or "unknown")
    if surface_label in {"ground", "grass", "ceiling"}:
        return is_horizontal_surface_artifact(source, args)
    if surface_label == "wall":
        return is_wall_surface_artifact(source, args)
    return False


def can_absorb(source: dict[str, Any], surface: dict[str, Any], args: argparse.Namespace) -> tuple[bool, dict[str, float]]:
    metrics = {
        "bbox_gap": bbox_gap(source, surface),
        "centroid_distance": centroid_distance(source, surface),
        "color_distance": color_distance(source, surface),
    }
    near = metrics["bbox_gap"] <= args.max_bbox_gap or metrics["centroid_distance"] <= args.max_centroid_distance
    color_ok = metrics["color_distance"] <= args.max_color_distance
    compatible = is_surface_compatible(source, surface, args)
    return bool(near and color_ok and compatible), metrics


def choose_surface(source: dict[str, Any], surfaces: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, Any] | None, dict[str, float]]:
    best = None
    best_metrics: dict[str, float] = {}
    best_score = float("inf")
    for surface in surfaces:
        ok, metrics = can_absorb(source, surface, args)
        if not ok:
            continue
        score = metrics["bbox_gap"] * 3.0 + metrics["centroid_distance"] + metrics["color_distance"] / 100.0
        if score < best_score:
            best = surface
            best_metrics = metrics
            best_score = score
    return best, best_metrics


def absorb_targets(targets: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_frame_cam: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in targets:
        key = (int(row.get("frame_id") or 0), int(row.get("cam_id") or 0))
        by_frame_cam.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    reason_counts = Counter()
    label_flow = Counter()
    absorbed_examples = []
    demoted_examples = []
    unabsorbed_weak = 0

    for row in targets:
        label = str(row.get("label") or "unknown")
        if label not in FINE_LABELS or not is_weak_fine(row, args):
            output.append(row)
            continue
        key = (int(row.get("frame_id") or 0), int(row.get("cam_id") or 0))
        surfaces = [
            candidate
            for candidate in by_frame_cam.get(key, [])
            if str(candidate.get("label") or "unknown") in SURFACE_LABELS
            and int(candidate.get("cluster_size") or 0) >= int(args.min_surface_points)
        ]
        surface, metrics = choose_surface(row, surfaces, args)
        if surface is None:
            unabsorbed_weak += 1
            if args.demote_unabsorbed_weak_label:
                out = dict(row)
                new_label = str(args.demote_unabsorbed_weak_label)
                out["label"] = new_label
                out["raw_label"] = str(row.get("raw_label") or label)
                out["parent_class"] = PARENT_BY_LABEL.get(new_label, "other")
                out["demoted_from_label"] = label
                out["demotion_reason"] = "unabsorbed_weak_fine"
                out["priority_label_id"] = int(PRIORITY_ID_BY_LABEL.get(new_label, 0))
                out["mask_id"] = int(PRIORITY_ID_BY_LABEL.get(new_label, 0))
                output.append(out)
                reason_counts["demoted_unabsorbed_weak_fine"] += 1
                label_flow[(label, new_label)] += 1
                if len(demoted_examples) < args.example_limit:
                    demoted_examples.append({
                        "target_id": row.get("target_id"),
                        "from": label,
                        "to": new_label,
                        "frame_id": row.get("frame_id"),
                        "cam_id": row.get("cam_id"),
                        "cluster_size": row.get("cluster_size"),
                    })
            else:
                output.append(row)
            continue

        new_label = str(surface.get("label") or "ground")
        out = dict(row)
        out["label"] = new_label
        out["raw_label"] = str(row.get("raw_label") or label)
        out["parent_class"] = PARENT_BY_LABEL.get(new_label, "other")
        out["absorbed_from_label"] = label
        out["absorbed_into_surface_target_id"] = surface.get("target_id")
        out["absorption_reason"] = "surface_like_fine" if is_surface_like_fine(row, args) else "low_point_fine_near_surface"
        out["absorption_metrics"] = {key: round(float(value), 4) for key, value in metrics.items()}
        out["priority_label_id"] = int(PRIORITY_ID_BY_LABEL.get(new_label, out.get("priority_label_id", 0)))
        out["mask_id"] = int(PRIORITY_ID_BY_LABEL.get(new_label, out.get("mask_id", 0)))
        output.append(out)

        reason_counts[out["absorption_reason"]] += 1
        label_flow[(label, new_label)] += 1
        if len(absorbed_examples) < args.example_limit:
            absorbed_examples.append({
                "target_id": row.get("target_id"),
                "from": label,
                "to": new_label,
                "frame_id": row.get("frame_id"),
                "cam_id": row.get("cam_id"),
                "cluster_size": row.get("cluster_size"),
                "surface_target_id": surface.get("target_id"),
                "metrics": out["absorption_metrics"],
            })

    report = {
        "input_targets": len(targets),
        "output_targets": len(output),
        "absorbed_targets": int(sum(reason_counts.values())),
        "unabsorbed_weak_fine_targets": int(unabsorbed_weak),
        "reason_counts": dict(reason_counts.most_common()),
        "label_flow_counts": {f"{src}->{dst}": count for (src, dst), count in sorted(label_flow.items())},
        "input_label_counts": dict(Counter(str(row.get("label") or "unknown") for row in targets).most_common()),
        "output_label_counts": dict(Counter(str(row.get("label") or "unknown") for row in output).most_common()),
        "examples": absorbed_examples,
        "demoted_examples": demoted_examples,
        "params": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool, list, type(None)))
        },
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--car-max-absorb-points", type=int, default=80)
    parser.add_argument("--railing-max-absorb-points", type=int, default=80)
    parser.add_argument("--min-surface-points", type=int, default=120)
    parser.add_argument("--max-bbox-gap", type=float, default=0.35)
    parser.add_argument("--max-centroid-distance", type=float, default=1.2)
    parser.add_argument("--max-color-distance", type=float, default=90.0)
    parser.add_argument("--surface-like-normal-z", type=float, default=0.92)
    parser.add_argument("--car-surface-like-max-z-span", type=float, default=0.25)
    parser.add_argument("--railing-surface-like-max-z-span", type=float, default=0.25)
    parser.add_argument("--railing-surface-like-min-planarity", type=float, default=0.35)
    parser.add_argument("--railing-keep-linearity", type=float, default=0.82)
    parser.add_argument("--horizontal-absorb-max-z-span", type=float, default=0.25)
    parser.add_argument("--wall-absorb-max-normal-z", type=float, default=0.35)
    parser.add_argument("--wall-absorb-min-z-span", type=float, default=0.4)
    parser.add_argument("--wall-absorb-min-planarity", type=float, default=0.35)
    parser.add_argument("--demote-unabsorbed-weak-label", default=None)
    parser.add_argument("--example-limit", type=int, default=50)
    args = parser.parse_args()

    rows, report = absorb_targets(read_jsonl(args.targets_jsonl), args)
    write_jsonl(args.output_jsonl, rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
