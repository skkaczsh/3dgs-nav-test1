#!/usr/bin/env python3
"""Fuse Target JSONL records into incremental Object and Zone records."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from build_targets_from_masks import read_colored_ply
from project_semantic import LABEL_COLORS, LABEL_NAMES


SURFACE_PARENT_CLASSES = {"surface", "structure"}
SURFACE_LABELS = {"floor", "wall", "building"}
DEFAULT_MIN_MERGE_CONFIDENCE = 0.5
IDENTITY_GATE_LABELS = {"equipment", "pipe", "railing", "vehicle", "furniture", "person", "tree", "grass", "other"}
IDENTITY_STOPWORDS = {
    "a",
    "an",
    "and",
    "around",
    "at",
    "near",
    "object",
    "on",
    "roof",
    "rooftop",
    "surface",
    "the",
    "unit",
}


def target_confidence(target: dict) -> float:
    value = target.get("confidence", 1.0)
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 1.0


def target_vote_weight(target: dict) -> float:
    return float(max(int(target.get("cluster_size", 1)), 1)) * target_confidence(target)


def target_description(target: dict) -> str:
    return str(
        target.get("identity_text")
        or target.get("freeform_label")
        or target.get("identity_hint")
        or target.get("description")
        or ""
    ).strip()


def identity_terms(text: str) -> set[str]:
    terms = set()
    for token in re.findall(r"[a-zA-Z0-9]+", text.lower()):
        if len(token) < 3 or token in IDENTITY_STOPWORDS:
            continue
        terms.add(token)
    return terms


def object_identity_terms(obj: dict) -> set[str]:
    terms: set[str] = set()
    for description in (obj.get("description_votes") or {}).keys():
        terms.update(identity_terms(str(description)))
    if not terms:
        terms.update(identity_terms(str(obj.get("description", ""))))
    for values in (obj.get("attribute_votes") or {}).values():
        if isinstance(values, dict):
            for value in values.keys():
                terms.update(identity_terms(str(value)))
    return terms


def target_identity_terms(target: dict) -> set[str]:
    terms = identity_terms(target_description(target))
    attrs = target.get("attributes") or {}
    if isinstance(attrs, dict):
        for value in attrs.values():
            terms.update(identity_terms(str(value)))
    return terms


def identity_similarity(obj: dict, target: dict) -> float | None:
    obj_terms = object_identity_terms(obj)
    target_terms = target_identity_terms(target)
    if not obj_terms or not target_terms:
        return None
    return float(len(obj_terms & target_terms) / max(len(obj_terms | target_terms), 1))


def update_description_votes(votes: dict, target: dict) -> None:
    description = target_description(target)
    if not description:
        return
    votes[description] = float(votes.get(description, 0.0) + target_vote_weight(target))


def merge_attribute_votes(votes: dict, target: dict) -> None:
    attrs = target.get("attributes") or {}
    if not isinstance(attrs, dict):
        return
    weight = target_vote_weight(target)
    for key, value in attrs.items():
        value = str(value).strip()
        if not value:
            continue
        bucket = votes.setdefault(str(key), {})
        bucket[value] = float(bucket.get(value, 0.0) + weight)


def target_quality(target: dict, min_merge_confidence: float = DEFAULT_MIN_MERGE_CONFIDENCE) -> dict:
    confidence = target_confidence(target)
    mixed = bool(target.get("vlm_mixed", False))
    can_merge_to_surface = bool(target.get("vlm_can_merge_to_surface", False))
    is_large_surface = bool(target.get("vlm_is_large_surface", False))
    low_confidence = confidence < min_merge_confidence
    mixed_blocks_merge = mixed and not can_merge_to_surface
    return {
        "confidence": confidence,
        "mixed": mixed,
        "is_large_surface": is_large_surface,
        "can_merge_to_surface": can_merge_to_surface,
        "low_confidence": low_confidence,
        "mixed_blocks_merge": mixed_blocks_merge,
        "merge_quality_ok": not low_confidence and not mixed_blocks_merge,
    }


def load_targets(inputs: list[Path]) -> list[dict]:
    rows = []
    for path in inputs:
        from_directory = path.is_dir()
        files = sorted(path.glob("targets_frame_*.jsonl")) if from_directory else [path]
        for file_path in files:
            if from_directory and file_path.name == "targets_all.jsonl":
                continue
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return sorted(rows, key=lambda r: (int(r.get("frame_id", 0)), str(r.get("target_id", ""))))


def bbox_distance(a: dict, b: dict) -> float:
    amin = np.array(a["bbox_3d"]["min"], dtype=np.float64)
    amax = np.array(a["bbox_3d"]["max"], dtype=np.float64)
    bmin = np.array(b["bbox_3d"]["min"], dtype=np.float64)
    bmax = np.array(b["bbox_3d"]["max"], dtype=np.float64)
    gap = np.maximum(0.0, np.maximum(bmin - amax, amin - bmax))
    return float(np.linalg.norm(gap))


def angle_degrees(a: list[float], b: list[float]) -> float:
    av = np.array(a, dtype=np.float64)
    bv = np.array(b, dtype=np.float64)
    an = np.linalg.norm(av)
    bn = np.linalg.norm(bv)
    if an < 1e-9 or bn < 1e-9:
        return 0.0
    dot = abs(float(np.dot(av / an, bv / bn)))
    return float(math.degrees(math.acos(np.clip(dot, -1.0, 1.0))))


def target_point_indices(target: dict) -> set[int]:
    frame = int(target.get("frame_id", 0))
    return {frame * 100000000 + int(i) for i in target.get("point_indices", [])}


def bbox_cells(bbox: dict, cell_size: float, padding: float = 0.0) -> set[tuple[int, int, int]]:
    bmin = np.array(bbox["min"], dtype=np.float64) - padding
    bmax = np.array(bbox["max"], dtype=np.float64) + padding
    lo = np.floor(bmin / cell_size).astype(int)
    hi = np.floor(bmax / cell_size).astype(int)
    return {
        (int(x), int(y), int(z))
        for x in range(lo[0], hi[0] + 1)
        for y in range(lo[1], hi[1] + 1)
        for z in range(lo[2], hi[2] + 1)
    }


def create_object(object_id: str, target: dict) -> dict:
    point_ids = target_point_indices(target)
    vote_weight = target_vote_weight(target)
    quality = target_quality(target)
    description_votes: dict[str, float] = {}
    attribute_votes: dict[str, dict[str, float]] = {}
    update_description_votes(description_votes, target)
    merge_attribute_votes(attribute_votes, target)
    return {
        "object_id": object_id,
        "semantic_label": target["label"],
        "raw_label_votes": {str(target.get("raw_label", target["label"])): int(target.get("cluster_size", 1))},
        "freeform_label_votes": {
            str(target.get("freeform_label", "")).strip(): float(vote_weight)
        } if str(target.get("freeform_label", "")).strip() else {},
        "status": "single_target",
        "targets": [target["target_id"]],
        "target_count": 1,
        "frames": [int(target["frame_id"])],
        "merged_point_indices": sorted(point_ids),
        "point_count": int(target.get("cluster_size", len(point_ids))),
        "bbox_3d": target["bbox_3d"],
        "centroid": target["centroid"],
        "label_votes": {target["label"]: int(target.get("cluster_size", 1))},
        "label_vote_weights": {target["label"]: float(vote_weight)},
        "description": target_description(target),
        "object_identity": target_description(target),
        "description_votes": description_votes,
        "attribute_votes": attribute_votes,
        "parent_class_votes": {target.get("parent_class", "other"): 1},
        "mean_color": target["mean_color"],
        "color_sum": (np.array(target["mean_color"], dtype=np.float64) * max(int(target.get("cluster_size", 1)), 1)).tolist(),
        "normal": target.get("pca", {}).get("normal", [0.0, 0.0, 1.0]),
        "geometry_stats": {
            "planarity_mean": float(target.get("pca", {}).get("planarity", 0.0)),
            "linearity_mean": float(target.get("pca", {}).get("linearity", 0.0)),
        },
        "color_stats": {"mean_rgb": target["mean_color"], "target_rgb_variance": 0.0},
        "quality_stats": {
            "confidence_mean": float(quality["confidence"]),
            "low_confidence_targets": int(quality["low_confidence"]),
            "mixed_targets": int(quality["mixed"]),
            "surface_mergeable_targets": int(quality["can_merge_to_surface"]),
        },
        "zone_id": f"zone_{int(target['frame_id']) // 100:03d}",
        "_target_records": [target],
        "_point_id_set": point_ids,
    }


def update_object(obj: dict, target: dict) -> None:
    old_count = max(int(obj["point_count"]), 1)
    new_count = max(int(target.get("cluster_size", 1)), 1)
    total = old_count + new_count
    obj["targets"].append(target["target_id"])
    obj["target_count"] = len(obj["targets"])
    obj["frames"] = sorted(set(obj["frames"] + [int(target["frame_id"])]))
    new_point_ids = target_point_indices(target)
    obj["_point_id_set"].update(new_point_ids)
    obj["merged_point_indices"] = sorted(obj["_point_id_set"])
    obj["point_count"] = int(total)

    omin = np.minimum(np.array(obj["bbox_3d"]["min"], dtype=np.float64), np.array(target["bbox_3d"]["min"], dtype=np.float64))
    omax = np.maximum(np.array(obj["bbox_3d"]["max"], dtype=np.float64), np.array(target["bbox_3d"]["max"], dtype=np.float64))
    obj["bbox_3d"] = {"min": [float(x) for x in omin], "max": [float(x) for x in omax]}
    centroid = (np.array(obj["centroid"]) * old_count + np.array(target["centroid"]) * new_count) / total
    obj["centroid"] = [float(x) for x in centroid]
    color_sum = np.array(obj["color_sum"], dtype=np.float64) + np.array(target["mean_color"], dtype=np.float64) * new_count
    obj["color_sum"] = [float(x) for x in color_sum]
    obj["mean_color"] = [float(x) for x in color_sum / total]

    obj["label_votes"][target["label"]] = int(obj["label_votes"].get(target["label"], 0) + new_count)
    obj["label_vote_weights"][target["label"]] = float(obj["label_vote_weights"].get(target["label"], 0.0) + target_vote_weight(target))
    raw_label = str(target.get("raw_label", target["label"])).strip()
    if raw_label:
        obj.setdefault("raw_label_votes", {})[raw_label] = int(obj.setdefault("raw_label_votes", {}).get(raw_label, 0) + new_count)
    freeform_label = str(target.get("freeform_label", "")).strip()
    if freeform_label:
        obj.setdefault("freeform_label_votes", {})[freeform_label] = float(
            obj.setdefault("freeform_label_votes", {}).get(freeform_label, 0.0) + target_vote_weight(target)
        )
    update_description_votes(obj.setdefault("description_votes", {}), target)
    merge_attribute_votes(obj.setdefault("attribute_votes", {}), target)
    parent = target.get("parent_class", "other")
    obj["parent_class_votes"][parent] = int(obj["parent_class_votes"].get(parent, 0) + 1)
    obj["_target_records"].append(target)
    normals = [np.array(t.get("pca", {}).get("normal", [0.0, 0.0, 1.0]), dtype=np.float64) for t in obj["_target_records"]]
    normal = np.mean(normals, axis=0)
    norm = np.linalg.norm(normal)
    obj["normal"] = [float(x) for x in (normal / norm if norm > 1e-9 else normal)]
    obj["geometry_stats"] = {
        "planarity_mean": float(np.mean([t.get("pca", {}).get("planarity", 0.0) for t in obj["_target_records"]])),
        "linearity_mean": float(np.mean([t.get("pca", {}).get("linearity", 0.0) for t in obj["_target_records"]])),
    }
    colors = np.array([t["mean_color"] for t in obj["_target_records"]], dtype=np.float64)
    obj["color_stats"] = {
        "mean_rgb": obj["mean_color"],
        "target_rgb_variance": float(np.mean(np.var(colors, axis=0))) if len(colors) > 1 else 0.0,
    }
    qualities = [target_quality(t) for t in obj["_target_records"]]
    obj["quality_stats"] = {
        "confidence_mean": float(np.mean([q["confidence"] for q in qualities])) if qualities else 1.0,
        "low_confidence_targets": int(sum(q["low_confidence"] for q in qualities)),
        "mixed_targets": int(sum(q["mixed"] for q in qualities)),
        "surface_mergeable_targets": int(sum(q["can_merge_to_surface"] for q in qualities)),
    }
    votes = Counter(obj["label_vote_weights"])
    winner, winner_votes = votes.most_common(1)[0]
    vote_total = float(sum(votes.values()))
    obj["semantic_label"] = winner if winner_votes / max(vote_total, 1) >= 0.8 else "ambiguous"
    if obj["semantic_label"] == "ambiguous" or len(votes) > 1:
        obj["status"] = "ambiguous_object"
    elif len(obj["targets"]) > 1:
        obj["status"] = "stable"
    else:
        obj["status"] = "single_target"


def match_score(obj: dict, target: dict, args: argparse.Namespace, target_point_ids: set[int] | None = None) -> tuple[bool, dict]:
    centroid_dist = float(np.linalg.norm(np.array(obj["centroid"]) - np.array(target["centroid"])))
    bbox_dist = bbox_distance(obj, target)
    color_dist = float(np.linalg.norm(np.array(obj["mean_color"]) - np.array(target["mean_color"])))
    normal_angle = angle_degrees(obj.get("normal", [0, 0, 1]), target.get("pca", {}).get("normal", [0, 0, 1]))
    same_label = obj["semantic_label"] == target["label"] or target["label"] in obj["label_votes"]
    parent = target.get("parent_class", "other")
    obj_parents = set(obj.get("parent_class_votes", {}).keys())
    same_parent = parent in obj_parents
    identity_sim = identity_similarity(obj, target)
    identity_threshold = float(getattr(args, "identity_similarity_threshold", 0.2))
    use_identity_gate = (
        not bool(getattr(args, "disable_identity_gate", False))
        and target.get("label") in IDENTITY_GATE_LABELS
        and bool(set(obj.get("label_votes", {}).keys()) & IDENTITY_GATE_LABELS)
        and identity_sim is not None
    )
    identity_ok = (not use_identity_gate) or identity_sim >= identity_threshold
    strict_surface_labels = bool(getattr(args, "strict_surface_labels", False))
    target_is_surface_label = target.get("label") in SURFACE_LABELS
    obj_surface_labels = set(obj.get("label_votes", {}).keys()) & SURFACE_LABELS
    surface_cross_label = strict_surface_labels and target_is_surface_label and bool(obj_surface_labels) and not same_label
    surface_same_label = same_label and target_is_surface_label and bool(obj_surface_labels)
    surface_centroid_distance = float(getattr(args, "surface_centroid_distance", getattr(args, "centroid_distance", 0.35)))
    surface_bbox_distance = float(getattr(args, "surface_bbox_distance", getattr(args, "bbox_distance", 0.35)))
    surface_color_distance = float(getattr(args, "surface_color_distance", getattr(args, "color_distance", 70.0)))
    surface_normal_angle = float(getattr(args, "surface_normal_angle", getattr(args, "normal_angle", 25.0)))
    centroid_distance = float(getattr(args, "centroid_distance", 0.35))
    bbox_distance_limit = float(getattr(args, "bbox_distance", 0.35))
    color_distance_limit = float(getattr(args, "color_distance", 70.0))
    normal_angle_limit = float(getattr(args, "normal_angle", 25.0))
    surface_near = centroid_dist <= surface_centroid_distance or bbox_dist <= surface_bbox_distance
    surface_color_ok = color_dist <= surface_color_distance
    surface_normal_ok = normal_angle <= surface_normal_angle
    near = centroid_dist <= centroid_distance or bbox_dist <= bbox_distance_limit
    color_ok = color_dist <= color_distance_limit
    normal_ok = normal_angle <= normal_angle_limit
    quality = target_quality(target, float(getattr(args, "min_merge_confidence", DEFAULT_MIN_MERGE_CONFIDENCE)))
    quality_ok = bool(quality["merge_quality_ok"])
    overlap = False
    if int(target["frame_id"]) in set(obj.get("frames", [])):
        if target_point_ids is None:
            target_point_ids = target_point_indices(target)
        overlap = bool(obj.get("_point_id_set", set()) & target_point_ids)
    merge = False
    reason = "no_match"
    if same_label and near and color_ok and normal_ok and quality_ok and identity_ok:
        merge = True
        reason = "same_label_geometry_color"
    elif surface_same_label and surface_near and surface_color_ok and surface_normal_ok and quality_ok:
        merge = True
        reason = "same_surface_label_relaxed"
    elif overlap and (same_label or (same_parent and not surface_cross_label)) and color_ok and not quality["mixed_blocks_merge"] and identity_ok:
        merge = True
        reason = "point_overlap"
    elif same_parent and parent in SURFACE_PARENT_CLASSES and not surface_cross_label and near and color_ok and normal_ok and quality_ok and identity_ok:
        merge = True
        reason = "same_parent_surface_review"
    elif use_identity_gate and not identity_ok and (same_label or same_parent) and near and color_ok and normal_ok:
        reason = "identity_gate"
    elif (same_label or same_parent) and near and color_ok and normal_ok and not quality_ok:
        reason = "quality_gate"
    return merge, {
        "reason": reason,
        "centroid_distance": centroid_dist,
        "bbox_distance": bbox_dist,
        "color_distance": color_dist,
        "normal_angle": normal_angle,
        "same_label": same_label,
        "same_parent": same_parent,
        "identity_similarity": identity_sim,
        "identity_gate_used": use_identity_gate,
        "identity_ok": identity_ok,
        "surface_cross_label_blocked": surface_cross_label,
        "surface_same_label": surface_same_label,
        "surface_near": surface_near,
        "overlap": overlap,
        "target_confidence": quality["confidence"],
        "target_mixed": quality["mixed"],
        "target_is_large_surface": quality["is_large_surface"],
        "target_can_merge_to_surface": quality["can_merge_to_surface"],
        "target_low_confidence": quality["low_confidence"],
        "target_mixed_blocks_merge": quality["mixed_blocks_merge"],
    }


def finalize_object(obj: dict) -> dict:
    out = {k: v for k, v in obj.items() if not k.startswith("_") and k != "color_sum"}
    votes = Counter(out.get("label_vote_weights") or out["label_votes"])
    if votes:
        winner, count = votes.most_common(1)[0]
        ratio = count / max(sum(votes.values()), 1)
        out["dominant_label"] = winner
        out["dominant_label_ratio"] = float(ratio)
        out["semantic_label"] = winner if ratio >= 0.8 else "ambiguous"
        if ratio < 0.8:
            out["status"] = "ambiguous_object"
        elif int(out.get("target_count", len(out.get("targets", [])))) > 1:
            out["status"] = "stable"
        else:
            out["status"] = "single_target"
    description_votes = Counter(out.get("description_votes", {}))
    if description_votes:
        description, weight = description_votes.most_common(1)[0]
        out["description"] = description
        out["object_identity"] = description
        out["description_vote_ratio"] = float(weight / max(sum(description_votes.values()), 1.0))
    freeform_votes = Counter(out.get("freeform_label_votes", {}))
    if freeform_votes:
        freeform, weight = freeform_votes.most_common(1)[0]
        out["dominant_freeform_label"] = freeform
        out["freeform_label_vote_ratio"] = float(weight / max(sum(freeform_votes.values()), 1.0))
        out.setdefault("object_identity", freeform)
    raw_votes = Counter(out.get("raw_label_votes", {}))
    if raw_votes:
        raw, weight = raw_votes.most_common(1)[0]
        out["dominant_raw_label"] = raw
        out["raw_label_vote_ratio"] = float(weight / max(sum(raw_votes.values()), 1.0))
    attr_summary = {}
    for key, values in (out.get("attribute_votes") or {}).items():
        value_votes = Counter(values)
        if value_votes:
            value, weight = value_votes.most_common(1)[0]
            attr_summary[key] = {
                "value": value,
                "vote_ratio": float(weight / max(sum(value_votes.values()), 1.0)),
            }
    if attr_summary:
        out["dominant_attributes"] = attr_summary
    return out


def semantic_color(label: str, object_id: int) -> tuple[int, int, int]:
    label_to_id = {v: k for k, v in LABEL_NAMES.items()}
    label_id = label_to_id.get(label, 0)
    if label_id in LABEL_COLORS:
        return LABEL_COLORS[label_id]
    rng = np.random.default_rng(object_id)
    return tuple(int(x) for x in rng.integers(60, 240, 3))


def write_object_ply(path: Path, objects: list[dict]) -> None:
    ply_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    target_points = []
    for object_number, obj in enumerate(objects, start=1):
        label = finalize_object(obj)["semantic_label"]
        color = semantic_color(label, object_number)
        sem = int({v: k for k, v in LABEL_NAMES.items()}.get(label, 0))
        for target in obj.get("_target_records", []):
            frame_ply = target.get("colored_frame_ply", "")
            indices = np.array(target.get("point_indices", []), dtype=np.int64)
            points = None
            if frame_ply and indices.size:
                if frame_ply not in ply_cache:
                    ply_cache[frame_ply] = read_colored_ply(Path(frame_ply))
                frame_points, _ = ply_cache[frame_ply]
                valid = indices[(indices >= 0) & (indices < len(frame_points))]
                if valid.size:
                    points = frame_points[valid]
            if points is None or len(points) == 0:
                points = np.array([target["centroid"]], dtype=np.float32)
            target_points.append((points, color, sem, object_number, int(target["frame_id"])))

    total = sum(len(points) for points, _, _, _, _ in target_points)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {total}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property int object\nproperty uchar semantic\n")
        f.write("property int frame\nend_header\n")
        for points, color, sem, object_number, frame_id in target_points:
            for p in points:
                f.write(
                    f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{color[0]} {color[1]} {color[2]} {object_number} {sem} {frame_id}\n"
                )


def fuse_targets(targets: list[dict], args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    objects: list[dict] = []
    objects_by_zone: dict[int, list[int]] = defaultdict(list)
    objects_by_zone_cell: dict[tuple[int, tuple[int, int, int]], list[int]] = defaultdict(list)
    object_cells: list[set[tuple[int, int, int]]] = []
    decisions = []
    for target in targets:
        best_idx = None
        best_meta = None
        best_dist = float("inf")
        best_reject_meta = None
        best_reject_dist = float("inf")
        current_zone = int(target["frame_id"]) // args.zone_size
        spatial_cell_size = float(getattr(args, "spatial_cell_size", 1.0))
        fallback_zone_scan = bool(getattr(args, "fallback_zone_scan", False))
        if args.active_zone_window >= 0:
            candidate_indices_set: set[int] = set()
            target_cells = bbox_cells(target["bbox_3d"], spatial_cell_size, args.bbox_distance)
            for zone in range(current_zone - args.active_zone_window, current_zone + args.active_zone_window + 1):
                for cell in target_cells:
                    candidate_indices_set.update(objects_by_zone_cell.get((zone, cell), []))
            candidate_indices = list(candidate_indices_set)
            if not candidate_indices and fallback_zone_scan:
                for zone in range(current_zone - args.active_zone_window, current_zone + args.active_zone_window + 1):
                    candidate_indices.extend(objects_by_zone.get(zone, []))
        else:
            candidate_indices = list(range(len(objects)))
        target_point_ids = target_point_indices(target)
        for idx in candidate_indices:
            obj = objects[idx]
            ok, meta = match_score(obj, target, args, target_point_ids)
            if ok and meta["centroid_distance"] < best_dist:
                best_idx = idx
                best_meta = meta
                best_dist = meta["centroid_distance"]
            elif not ok and meta["centroid_distance"] < best_reject_dist:
                best_reject_meta = meta
                best_reject_dist = meta["centroid_distance"]
        if best_idx is None:
            object_id = f"obj_{len(objects) + 1:06d}"
            objects.append(create_object(object_id, target))
            obj_idx = len(objects) - 1
            objects_by_zone[current_zone].append(obj_idx)
            cells = bbox_cells(objects[obj_idx]["bbox_3d"], spatial_cell_size, args.bbox_distance)
            object_cells.append(cells)
            for cell in cells:
                objects_by_zone_cell[(current_zone, cell)].append(obj_idx)
            decision = {"target_id": target["target_id"], "object_id": object_id, "action": "new_object"}
            if best_reject_meta is not None:
                decision.update({f"nearest_{key}": value for key, value in best_reject_meta.items()})
            decisions.append(decision)
        else:
            obj = objects[best_idx]
            update_object(obj, target)
            obj_zone = int(obj["zone_id"].rsplit("_", 1)[1])
            cells = bbox_cells(obj["bbox_3d"], spatial_cell_size, args.bbox_distance)
            new_cells = cells - object_cells[best_idx]
            if new_cells:
                object_cells[best_idx].update(new_cells)
                for cell in new_cells:
                    objects_by_zone_cell[(obj_zone, cell)].append(best_idx)
            decisions.append({"target_id": target["target_id"], "object_id": obj["object_id"], "action": "merge", **(best_meta or {})})
    return objects, decisions


def build_zones(objects: list[dict], zone_size: int) -> list[dict]:
    zones: dict[str, dict] = {}
    for obj in objects:
        out = finalize_object(obj)
        first_frame = min(out["frames"]) if out["frames"] else 0
        zone_id = f"zone_{first_frame // zone_size:03d}"
        zone = zones.setdefault(zone_id, {"zone_id": zone_id, "objects": [], "bbox_3d": None, "frame_min": first_frame, "frame_max": first_frame})
        zone["objects"].append(out["object_id"])
        zone["frame_min"] = min(zone["frame_min"], min(out["frames"]))
        zone["frame_max"] = max(zone["frame_max"], max(out["frames"]))
        if zone["bbox_3d"] is None:
            zone["bbox_3d"] = out["bbox_3d"]
        else:
            zone["bbox_3d"] = {
                "min": [float(x) for x in np.minimum(zone["bbox_3d"]["min"], out["bbox_3d"]["min"])],
                "max": [float(x) for x in np.maximum(zone["bbox_3d"]["max"], out["bbox_3d"]["max"])],
            }
    return sorted(zones.values(), key=lambda x: x["zone_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--centroid-distance", type=float, default=0.35)
    parser.add_argument("--bbox-distance", type=float, default=0.35)
    parser.add_argument("--color-distance", type=float, default=70.0)
    parser.add_argument("--normal-angle", type=float, default=25.0)
    parser.add_argument("--surface-centroid-distance", type=float, default=0.8)
    parser.add_argument("--surface-bbox-distance", type=float, default=0.65)
    parser.add_argument("--surface-color-distance", type=float, default=95.0)
    parser.add_argument("--surface-normal-angle", type=float, default=18.0)
    parser.add_argument("--min-merge-confidence", type=float, default=0.5)
    parser.add_argument("--identity-similarity-threshold", type=float, default=0.2)
    parser.add_argument("--disable-identity-gate", action="store_true")
    parser.add_argument("--zone-size", type=int, default=100)
    parser.add_argument("--active-zone-window", type=int, default=1)
    parser.add_argument("--spatial-cell-size", type=float, default=1.0)
    parser.add_argument("--fallback-zone-scan", action="store_true")
    parser.add_argument(
        "--strict-surface-labels",
        action="store_true",
        help="Do not merge floor/wall/building targets across labels via parent-class surface rules.",
    )
    parser.add_argument("--write-ply", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = load_targets(args.targets)
    objects, decisions = fuse_targets(targets, args)
    finalized = [finalize_object(obj) for obj in objects]
    zones = build_zones(objects, args.zone_size)

    objects_path = args.output_dir / "objects.jsonl"
    with objects_path.open("w", encoding="utf-8") as f:
        for obj in finalized:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    zones_path = args.output_dir / "zones.json"
    zones_path.write_text(json.dumps({"zones": zones}, ensure_ascii=False, indent=2), encoding="utf-8")
    decisions_path = args.output_dir / "fusion_decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8") as f:
        for row in decisions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.write_ply:
        write_object_ply(args.output_dir / "object_centroids.ply", objects)

    statuses = Counter(obj["status"] for obj in finalized)
    report = {
        "targets": len(targets),
        "objects": len(finalized),
        "zones": len(zones),
        "merge_ratio": float(1.0 - len(finalized) / max(len(targets), 1)),
        "status_counts": dict(statuses),
        "ambiguous_objects": int(statuses.get("ambiguous_object", 0)),
        "objects_jsonl": str(objects_path),
        "zones_json": str(zones_path),
        "decisions_jsonl": str(decisions_path),
    }
    report_path = args.output_dir / "fusion_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"targets={len(targets)} objects={len(finalized)} merge_ratio={report['merge_ratio']:.3f}")
    print(f"wrote={report_path}")


if __name__ == "__main__":
    main()
