#!/usr/bin/env python3
"""Add scene context and downstream routing fields to object JSONL.

This is a lightweight metadata stage after priority/residual object assembly.
It does not relabel points. It adds reusable context for QA, DINO-style
detectors, and later LLM/object merging:

- height-layer context for floor/wall surfaces
- parking-scene descriptions for stable priority classes
- DINO/fine-review prompt groups for cars, railings, and residual objects
- geometry-quality warnings for overmerged priority surfaces
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


FINE_PROMPTS = {
    "car": ["car", "parked car", "vehicle", "truck", "van"],
    "railing": ["railing", "guardrail", "handrail", "metal fence", "fence"],
    "unknown": ["equipment", "pipe", "cable", "traffic cone", "sign", "small object"],
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
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def bbox_min_max(obj: dict[str, Any]) -> tuple[list[float], list[float]]:
    if isinstance(obj.get("bbox_3d"), dict):
        b = obj["bbox_3d"]
        if "min" in b and "max" in b:
            return [float(x) for x in b["min"]], [float(x) for x in b["max"]]
    return (
        [float(x) for x in obj.get("bbox_min", obj.get("centroid", [0, 0, 0]))],
        [float(x) for x in obj.get("bbox_max", obj.get("centroid", [0, 0, 0]))],
    )


def centroid_z(obj: dict[str, Any]) -> float:
    c = obj.get("centroid") or [0.0, 0.0, 0.0]
    return float(c[2])


def cluster_height_layers(z_values: list[float], gap: float) -> list[dict[str, Any]]:
    if not z_values:
        return []
    values = sorted(z_values)
    groups: list[list[float]] = [[values[0]]]
    for z in values[1:]:
        if z - groups[-1][-1] > gap:
            groups.append([z])
        else:
            groups[-1].append(z)
    layers = []
    for i, group in enumerate(groups):
        layers.append({
            "index": i,
            "name": "ground_level" if i == 0 else f"upper_level_{i}",
            "z_min": float(min(group)),
            "z_max": float(max(group)),
            "z_median": float(median(group)),
            "object_count": len(group),
        })
    return layers


def nearest_layer(z: float, layers: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not layers:
        return None
    return min(layers, key=lambda layer: abs(z - float(layer["z_median"])))


def geometry_quality(obj: dict[str, Any], label: str) -> tuple[str, list[str]]:
    planarity = float(obj.get("planarity", obj.get("pca_planarity_recomputed", 0.0)) or 0.0)
    thickness = float(obj.get("thickness_rms", obj.get("pca_thickness_rms_recomputed", 0.0)) or 0.0)
    max_extent = float(obj.get("max_extent", 0.0) or 0.0)
    bbox_min, bbox_max = bbox_min_max(obj)
    z_extent = float(bbox_max[2] - bbox_min[2])
    flags: list[str] = []
    quality = "clean"

    if label in {"wall", "grass"} and (planarity < 0.70 or thickness > 1.5):
        quality = "mixed_or_overmerged"
        flags.append("low_planarity_surface")
    if label == "ceiling" and (planarity < 0.70 or thickness > 1.0):
        quality = "mixed_or_overmerged"
        flags.append("low_planarity_ceiling_surface")
    if label == "floor" and z_extent > 5.0:
        quality = "mixed_or_overmerged"
        flags.append("large_vertical_extent_on_floor")
    if label in {"car", "railing"} and max_extent > 25.0:
        quality = "mixed_or_overmerged"
        flags.append("large_priority_object_component")
    if label == "unknown" and obj.get("status") == "needs_semantic_review":
        quality = "unclassified_candidate"
    return quality, flags


def height_zone(z: float, args: argparse.Namespace) -> dict[str, Any]:
    if z <= args.ground_zone_z_max:
        return {
            "name": "ground_zone",
            "description": "parking-lot ground / low indoor level",
        }
    if z >= args.upper_zone_z_min:
        return {
            "name": "upper_zone",
            "description": "upper parking deck / overhead structure zone",
        }
    return {
        "name": "transition_zone",
        "description": "ramp / intermediate transition level",
    }


def assign_context(obj: dict[str, Any], layers: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    label = str(obj.get("semantic_label", "unknown"))
    status = str(obj.get("status", ""))
    z = centroid_z(obj)
    layer = nearest_layer(z, layers)
    layer_name = layer["name"] if layer else "unknown_layer"
    zone = height_zone(z, args)
    zone_name = str(zone["name"])
    quality, flags = geometry_quality(obj, label)

    context = "residual_fine_object_candidate"
    description = "unclassified residual object candidate"
    downstream_stage = "fine_semantic_review"
    review_priority = "high" if label == "unknown" else "medium"
    prompt_group = label if label in FINE_PROMPTS else ""
    stable_surface = False

    if label == "floor":
        stable_surface = quality == "clean"
        if zone_name == "ground_zone":
            context = "outdoor_parking_ground_or_pavement"
            description = "parking-lot ground / pavement surface"
        elif zone_name == "upper_zone":
            context = "upper_parking_deck_floor"
            description = "upper parking deck / elevated floor surface"
        else:
            context = "parking_ramp_or_transition_floor"
            description = "parking ramp / intermediate transition floor surface"
        downstream_stage = "stable_surface" if stable_surface else "geometry_review"
        review_priority = "low" if stable_surface else "medium"
    elif label == "wall":
        stable_surface = quality == "clean"
        context = f"{zone_name}_building_or_indoor_wall_surface"
        description = f"{zone['description']} wall / vertical building surface"
        downstream_stage = "stable_surface" if stable_surface else "geometry_review"
        review_priority = "low" if stable_surface else "medium"
    elif label == "ceiling":
        stable_surface = quality == "clean"
        context = "ceiling_or_overhead_deck_surface"
        description = "ceiling / overhead deck horizontal surface in the parking-scene scan"
        downstream_stage = "stable_surface" if stable_surface else "geometry_review"
        review_priority = "low" if stable_surface else "medium"
    elif label == "grass":
        context = "parking_lot_vegetation"
        description = "grass / vegetation area around the parking lot"
        downstream_stage = "stable_context_object" if quality == "clean" else "geometry_review"
        review_priority = "low" if quality == "clean" else "medium"
    elif label == "car":
        context = "parked_vehicle_candidate"
        description = "parked vehicle candidate"
        downstream_stage = "dino_fine_object_review"
        review_priority = "high"
        prompt_group = "car"
    elif label == "railing":
        context = "guardrail_or_fence_candidate"
        description = "guardrail / railing / fence candidate"
        downstream_stage = "dino_fine_object_review"
        review_priority = "high"
        prompt_group = "railing"
    elif label == "unknown":
        context = "residual_object_candidate_after_surface_removal"
        description = "remaining residual object after sky, surfaces, vegetation, car, and railing priority removal"
        downstream_stage = "fine_semantic_review"
        review_priority = "high" if int(obj.get("point_count", 0)) >= args.large_residual_points else "medium"
        prompt_group = "unknown"

    out = dict(obj)
    out["scene_id"] = args.scene_id
    out["scene_name"] = args.scene_name
    out["scene_context"] = context
    out["scene_description"] = description
    out["height_layer"] = {
        "name": layer_name,
        "z_median": layer.get("z_median") if layer else None,
        "object_z": z,
    }
    out["height_zone"] = {
        "name": zone["name"],
        "description": zone["description"],
        "object_z": z,
    }
    out["geometry_quality"] = quality
    out["geometry_flags"] = flags
    out["downstream_stage"] = downstream_stage
    out["review_priority"] = review_priority
    out["stable_surface"] = stable_surface
    if prompt_group:
        out["dino_prompt_group"] = prompt_group
        out["dino_prompts"] = FINE_PROMPTS[prompt_group]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path)
    parser.add_argument("--dino-jsonl", type=Path)
    parser.add_argument("--scene-id", default="MT20260616-175807")
    parser.add_argument("--scene-name", default="outdoor parking lot with connected indoor/upper-level areas")
    parser.add_argument("--height-layer-gap", type=float, default=3.0)
    parser.add_argument("--ground-zone-z-max", type=float, default=1.5)
    parser.add_argument("--upper-zone-z-min", type=float, default=6.0)
    parser.add_argument("--large-residual-points", type=int, default=1000)
    args = parser.parse_args()

    objects = read_jsonl(args.input_jsonl)
    floor_z = [centroid_z(obj) for obj in objects if obj.get("semantic_label") == "floor"]
    layers = cluster_height_layers(floor_z, args.height_layer_gap)
    enriched = [assign_context(obj, layers, args) for obj in objects]
    write_jsonl(args.output_jsonl, enriched)

    if args.candidate_jsonl:
        candidates = [
            obj for obj in enriched
            if obj.get("downstream_stage") in {"dino_fine_object_review", "fine_semantic_review", "geometry_review"}
        ]
        write_jsonl(args.candidate_jsonl, candidates)
    if args.dino_jsonl:
        dino_candidates = [
            obj for obj in enriched
            if obj.get("downstream_stage") == "dino_fine_object_review"
        ]
        write_jsonl(args.dino_jsonl, dino_candidates)

    report = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "candidate_jsonl": str(args.candidate_jsonl) if args.candidate_jsonl else None,
        "dino_jsonl": str(args.dino_jsonl) if args.dino_jsonl else None,
        "scene_id": args.scene_id,
        "scene_name": args.scene_name,
        "object_count": len(enriched),
        "height_layers": layers,
        "semantic_label_counts": dict(Counter(str(obj.get("semantic_label", "")) for obj in enriched)),
        "scene_context_counts": dict(Counter(str(obj.get("scene_context", "")) for obj in enriched)),
        "downstream_stage_counts": dict(Counter(str(obj.get("downstream_stage", "")) for obj in enriched)),
        "review_priority_counts": dict(Counter(str(obj.get("review_priority", "")) for obj in enriched)),
        "geometry_quality_counts": dict(Counter(str(obj.get("geometry_quality", "")) for obj in enriched)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
