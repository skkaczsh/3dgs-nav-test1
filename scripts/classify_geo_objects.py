#!/usr/bin/env python3
"""Classify geometry-first patches into viewer-ready semantic objects."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


LABEL_TO_SEMANTIC = {
    "unknown": 0,
    "ambiguous": 0,
    "other": 1,
    "wall": 2,
    "floor": 3,
    "ground": 3,
    "ceiling": 4,
    "grass": 5,
    "tree": 6,
    "person": 7,
    "car": 8,
    "railing": 9,
    "building": 10,
    "road": 12,
    "furniture": 14,
    "pipe": 15,
    "equipment": 16,
    "fine_candidate": 17,
    "stair": 18,
    "indoor_floor": 19,
    "roof": 20,
}

SEMANTIC_COLORS = {
    0: (150, 150, 150),
    1: (180, 180, 180),
    2: (120, 150, 180),
    3: (196, 168, 112),
    4: (170, 170, 210),
    5: (80, 160, 80),
    6: (50, 130, 70),
    8: (235, 90, 80),
    9: (240, 210, 60),
    10: (145, 145, 160),
    12: (120, 120, 120),
    15: (220, 160, 60),
    16: (210, 90, 210),
    17: (245, 150, 40),
    18: (245, 125, 60),
    19: (105, 180, 210),
    20: (165, 145, 210),
}

GROUND_SUBTYPE_LABELS = {
    "ordinary_ground": "ground",
    "outdoor_ground": "ground",
    "parking_ground": "ground",
    "indoor_floor": "indoor_floor",
    "stair": "stair",
    "stairs": "stair",
    "grass": "grass",
    "roof": "roof",
    "rooftop": "roof",
}

SURFACE_GEOMETRY = {"horizontal_surface", "vertical_surface", "upper_surface"}
FINE_LABELS = {"car", "railing", "pipe", "equipment", "person", "fine_candidate"}
INDOOR_AREAS = {"indoor_lobby", "indoor_corridor", "stairwell"}


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


def parse_ascii_ply_header(path: Path) -> tuple[list[str], int, int]:
    props: list[str] = []
    vertex_count = 0
    header_lines = 0
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header_lines += 1
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
        raise ValueError(f"No vertex count found: {path}")
    return props, vertex_count, header_lines


def evidence(patch: dict[str, Any]) -> dict[str, Any]:
    value = patch.get("evidence")
    return value if isinstance(value, dict) else {}


def dominant_scene_area(patch: dict[str, Any]) -> str:
    scene = evidence(patch).get("scene_prior") if isinstance(evidence(patch).get("scene_prior"), dict) else {}
    return str(scene.get("dominant_scene_area_type") or "unknown")


def dominant_ground_subtype(patch: dict[str, Any]) -> str:
    scene = evidence(patch).get("scene_prior") if isinstance(evidence(patch).get("scene_prior"), dict) else {}
    return str(scene.get("dominant_scene_ground_subtype") or "")


def candidate_votes(patch: dict[str, Any]) -> Counter[str]:
    ev = evidence(patch)
    votes: Counter[str] = Counter()
    for bucket, weight in (("semantic_votes", 1.0), ("priority_votes", 1.0)):
        values = ev.get(bucket) if isinstance(ev.get(bucket), dict) else {}
        for label, count in values.items():
            label = str(label)
            if label in {"residual", "sky", "ignore"}:
                continue
            votes[label] += float(count) * weight
    return votes


def vote_ratio(votes: Counter[str], label: str) -> float:
    return float(votes.get(label, 0.0) / max(sum(votes.values()), 1.0))


def color_green_score(patch: dict[str, Any]) -> float:
    mean = ((patch.get("color_stats") or {}).get("mean_rgb") or [0.0, 0.0, 0.0])
    if len(mean) < 3:
        return 0.0
    r, g, b = [float(x) for x in mean[:3]]
    return float((g - max(r, b)) / 255.0)


def normal_angle(a: list[float], b: list[float]) -> float:
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    an = np.linalg.norm(av)
    bn = np.linalg.norm(bv)
    if an < 1e-9 or bn < 1e-9:
        return 0.0
    dot = abs(float(np.dot(av / an, bv / bn)))
    return float(math.degrees(math.acos(np.clip(dot, -1.0, 1.0))))


def bbox_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    amin = np.asarray(a["bbox_3d"]["min"], dtype=np.float64)
    amax = np.asarray(a["bbox_3d"]["max"], dtype=np.float64)
    bmin = np.asarray(b["bbox_3d"]["min"], dtype=np.float64)
    bmax = np.asarray(b["bbox_3d"]["max"], dtype=np.float64)
    gap = np.maximum(0.0, np.maximum(bmin - amax, amin - bmax))
    return float(np.linalg.norm(gap))


def classify_patch(patch: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    gtype = str(patch.get("geometry_type") or "unknown")
    votes = candidate_votes(patch)
    area = dominant_scene_area(patch)
    subtype = dominant_ground_subtype(patch)
    conflicts: list[str] = []
    label = "unknown"
    confidence = 0.0

    if gtype == "horizontal_surface":
        label = GROUND_SUBTYPE_LABELS.get(subtype, "ground")
        if vote_ratio(votes, "grass") >= args.grass_vote_min or (area == "grass_landscape" and color_green_score(patch) >= args.grass_green_min):
            label = "grass"
        confidence = max(vote_ratio(votes, label), 0.65)
        if votes and any(v in votes for v in FINE_LABELS):
            conflicts.append("fine_label_on_horizontal_surface_vetoed")
    elif gtype == "upper_surface":
        label = "roof" if area == "rooftop" or subtype in {"roof", "rooftop"} else "ceiling"
        confidence = max(vote_ratio(votes, label), 0.60)
        if votes and any(v in votes for v in {"car", "railing"}):
            conflicts.append("fine_label_on_upper_surface_vetoed")
    elif gtype == "vertical_surface":
        label = "wall"
        confidence = max(vote_ratio(votes, "wall"), vote_ratio(votes, "building"), 0.70)
        if vote_ratio(votes, "car") >= args.fine_vote_warn_ratio:
            conflicts.append("car_vote_on_vertical_surface_vetoed")
        if vote_ratio(votes, "railing") >= args.railing_vote_warn_ratio:
            conflicts.append("railing_vote_on_vertical_surface_requires_split")
    elif gtype == "vegetation_like":
        label = "grass" if vote_ratio(votes, "grass") >= args.grass_vote_min or area == "grass_landscape" else "unknown"
        confidence = max(vote_ratio(votes, "grass"), 0.45 if label == "grass" else 0.2)
    elif gtype == "linear_thin":
        label = "railing" if vote_ratio(votes, "railing") >= args.railing_vote_min else "fine_candidate"
        confidence = max(vote_ratio(votes, "railing"), 0.55 if label == "railing" else 0.35)
    elif gtype == "bulky_object":
        if vote_ratio(votes, "car") >= args.car_vote_min and area not in INDOOR_AREAS:
            label = "car"
            confidence = vote_ratio(votes, "car")
        elif vote_ratio(votes, "equipment") >= args.equipment_vote_min:
            label = "equipment"
            confidence = vote_ratio(votes, "equipment")
        else:
            label = "unknown"
            confidence = max(vote_ratio(votes, "car"), vote_ratio(votes, "equipment"), 0.25)
        if vote_ratio(votes, "car") >= args.car_vote_min and area in INDOOR_AREAS:
            conflicts.append("indoor_car_vetoed")
            label = "unknown"
    elif gtype == "mixed":
        label = "ambiguous"
        confidence = 0.0
        conflicts.append("mixed_geometry_requires_split")
    else:
        winner, value = votes.most_common(1)[0] if votes else ("unknown", 0.0)
        label = winner if value / max(sum(votes.values()), 1.0) >= args.unknown_vote_accept_ratio else "unknown"
        confidence = value / max(sum(votes.values()), 1.0) if votes else 0.0

    if confidence < args.min_stable_confidence or conflicts:
        status = "ambiguous_object" if label not in {"wall", "ground", "grass", "roof", "indoor_floor", "stair"} else "geometry_guarded"
    else:
        status = "stable"

    return {
        "canonical_label": label,
        "confidence": float(confidence),
        "conflict_flags": conflicts,
        "status": status,
        "label_votes": dict(votes),
    }


def compatible_patch_object(obj: dict[str, Any], patch: dict[str, Any], classification: dict[str, Any], args: argparse.Namespace) -> bool:
    if obj["semantic_label"] != classification["canonical_label"]:
        return False
    if obj["object_type_geometry"] != str(patch.get("geometry_type") or "unknown"):
        return False
    if obj["status"] != "stable" or classification["status"] != "stable":
        return False
    if bbox_distance(obj, patch) > args.merge_bbox_distance:
        return False
    if normal_angle(obj.get("normal", [0, 0, 1]), patch.get("normal", [0, 0, 1])) > args.merge_normal_angle:
        return False
    return True


def create_object(object_number: int, patch: dict[str, Any], classification: dict[str, Any]) -> dict[str, Any]:
    label = classification["canonical_label"]
    return {
        "object_id": f"obj_{object_number:06d}",
        "viewer_object_id": object_number,
        "patch_ids": [patch["patch_id"]],
        "patch_count": 1,
        "semantic_label": label,
        "canonical_label": label,
        "object_type_geometry": patch.get("geometry_type"),
        "semantic_description": patch.get("description") or "",
        "status": classification["status"],
        "point_count": int(patch.get("point_count") or 0),
        "bbox_3d": patch["bbox_3d"],
        "centroid": patch["centroid"],
        "normal": patch.get("normal", [0, 0, 1]),
        "geometry_stats": {
            "planarity_mean": float(patch.get("planarity") or 0.0),
            "linearity_mean": float(patch.get("linearity") or 0.0),
            "roughness_mean": float(patch.get("roughness") or 0.0),
            "thickness_mean": float(patch.get("thickness") or 0.0),
        },
        "label_votes": classification["label_votes"],
        "description_votes": {},
        "scene_prior_votes": (evidence(patch).get("scene_prior") or {}),
        "conflict_flags": classification["conflict_flags"],
        "classification_confidence": classification["confidence"],
        "dominant_structural_region": evidence(patch).get("dominant_structural_region"),
        "dominant_structural_region_ratio": evidence(patch).get("dominant_structural_region_ratio"),
        "_patch_records": [patch],
    }


def merge_object(obj: dict[str, Any], patch: dict[str, Any], classification: dict[str, Any]) -> None:
    old = max(int(obj["point_count"]), 1)
    new = max(int(patch.get("point_count") or 0), 1)
    total = old + new
    obj["patch_ids"].append(patch["patch_id"])
    obj["patch_count"] = len(obj["patch_ids"])
    obj["point_count"] = int(total)
    obj["centroid"] = [
        float(x)
        for x in ((np.asarray(obj["centroid"]) * old + np.asarray(patch["centroid"]) * new) / total).tolist()
    ]
    omin = np.minimum(np.asarray(obj["bbox_3d"]["min"], dtype=np.float64), np.asarray(patch["bbox_3d"]["min"], dtype=np.float64))
    omax = np.maximum(np.asarray(obj["bbox_3d"]["max"], dtype=np.float64), np.asarray(patch["bbox_3d"]["max"], dtype=np.float64))
    obj["bbox_3d"] = {"min": [float(x) for x in omin.tolist()], "max": [float(x) for x in omax.tolist()]}
    obj["_patch_records"].append(patch)
    obj["conflict_flags"] = sorted(set(obj.get("conflict_flags", []) + classification["conflict_flags"]))
    for label, count in classification["label_votes"].items():
        obj["label_votes"][label] = float(obj["label_votes"].get(label, 0.0) + count)
    normals = [np.asarray(p.get("normal", [0, 0, 1]), dtype=np.float64) for p in obj["_patch_records"]]
    normal = np.mean(normals, axis=0)
    norm = np.linalg.norm(normal)
    obj["normal"] = [float(x) for x in (normal / norm if norm > 1e-9 else normal).tolist()]


def finalize_object(obj: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in obj.items() if not k.startswith("_")}
    out["semantic_id"] = LABEL_TO_SEMANTIC.get(str(out.get("semantic_label") or "unknown"), 0)
    return out


def build_objects(patches: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[int, int], dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    patch_to_object: dict[int, int] = {}
    classifications: dict[int, dict[str, Any]] = {}
    for patch in sorted(patches, key=lambda p: int(p.get("point_count") or 0), reverse=True):
        cls = classify_patch(patch, args)
        classifications[int(patch["patch_index"])] = cls
        best = None
        if args.merge_compatible_patches:
            for idx, obj in enumerate(objects):
                if compatible_patch_object(obj, patch, cls, args):
                    best = idx
                    break
        if best is None:
            object_number = len(objects) + 1
            objects.append(create_object(object_number, patch, cls))
            patch_to_object[int(patch["patch_index"])] = object_number
        else:
            merge_object(objects[best], patch, cls)
            patch_to_object[int(patch["patch_index"])] = int(objects[best]["viewer_object_id"])
    finalized = [finalize_object(obj) for obj in objects]
    report = {
        "schema": "geo-object-classification/v1",
        "patch_count": len(patches),
        "object_count": len(finalized),
        "label_counts": dict(Counter(obj["semantic_label"] for obj in finalized)),
        "geometry_type_counts": dict(Counter(obj["object_type_geometry"] for obj in finalized)),
        "status_counts": dict(Counter(obj["status"] for obj in finalized)),
        "conflict_counts": dict(Counter(flag for obj in finalized for flag in obj.get("conflict_flags", []))),
    }
    return finalized, patch_to_object, report


def write_viewer_ply(input_patch_ply: Path, output_ply: Path, patch_to_object: dict[int, int], objects: list[dict[str, Any]]) -> dict[str, Any]:
    props, vertex_count, header_lines = parse_ascii_ply_header(input_patch_ply)
    idx = {name: i for i, name in enumerate(props)}
    for name in ("x", "y", "z", "patch"):
        if name not in idx:
            raise ValueError(f"Patch PLY missing {name}: {input_patch_ply}")
    objects_by_number = {int(obj["viewer_object_id"]): obj for obj in objects}
    label_counts = Counter()
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    with output_ply.open("w", encoding="utf-8") as dst:
        dst.write("ply\nformat ascii 1.0\n")
        dst.write(f"element vertex {vertex_count}\n")
        dst.write("property float x\nproperty float y\nproperty float z\n")
        dst.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        dst.write("property int object\nproperty uchar semantic\n")
        dst.write("property int patch\n")
        dst.write("end_header\n")
        with input_patch_ply.open("r", encoding="utf-8", errors="replace") as src:
            for _ in range(header_lines):
                next(src)
            for line in src:
                parts = line.strip().split()
                if len(parts) < len(props):
                    continue
                patch = int(round(float(parts[idx["patch"]])))
                object_number = patch_to_object.get(patch, 0)
                obj = objects_by_number.get(object_number, {})
                label = str(obj.get("semantic_label") or "unknown")
                semantic = LABEL_TO_SEMANTIC.get(label, 0)
                color = SEMANTIC_COLORS.get(semantic, SEMANTIC_COLORS[0])
                label_counts[label] += 1
                dst.write(
                    f"{parts[idx['x']]} {parts[idx['y']]} {parts[idx['z']]} "
                    f"{color[0]} {color[1]} {color[2]} {object_number} {semantic} {patch}\n"
                )
    return {"vertex_count": vertex_count, "semantic_point_counts": dict(label_counts)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geo-patches", type=Path, required=True)
    parser.add_argument("--geo-patch-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merge-compatible-patches", action="store_true")
    parser.add_argument("--merge-bbox-distance", type=float, default=0.20)
    parser.add_argument("--merge-normal-angle", type=float, default=12.0)
    parser.add_argument("--min-stable-confidence", type=float, default=0.55)
    parser.add_argument("--grass-vote-min", type=float, default=0.25)
    parser.add_argument("--grass-green-min", type=float, default=0.05)
    parser.add_argument("--railing-vote-min", type=float, default=0.20)
    parser.add_argument("--railing-vote-warn-ratio", type=float, default=0.15)
    parser.add_argument("--car-vote-min", type=float, default=0.25)
    parser.add_argument("--fine-vote-warn-ratio", type=float, default=0.15)
    parser.add_argument("--equipment-vote-min", type=float, default=0.35)
    parser.add_argument("--unknown-vote-accept-ratio", type=float, default=0.75)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patches = read_jsonl(args.geo_patches)
    objects, patch_to_object, report = build_objects(patches, args)
    write_jsonl(args.output_dir / "frame_objects_viewer.jsonl", objects)
    ply_report = write_viewer_ply(args.geo_patch_ply, args.output_dir / "frame_object_points_stride10.ply", patch_to_object, objects)
    report["ply"] = ply_report
    (args.output_dir / "geo_object_classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
