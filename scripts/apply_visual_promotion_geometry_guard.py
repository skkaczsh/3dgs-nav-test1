#!/usr/bin/env python3
"""Demote visually promoted fine-object labels that fail 3D shape checks.

Crop-level detection is useful, but it can confirm the context around a wall or
surface fragment rather than the point-cloud object itself. This pass keeps a
candidate as final `car` / `railing` only when the associated 3D object also
looks geometrically plausible for that class. Rejected objects are not deleted;
they are demoted back to `fine_candidate` for later review.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


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


def float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return []
    return out


def geometry_features(obj: dict[str, Any]) -> dict[str, float]:
    extent = sorted(float_list(obj.get("extent")), reverse=True)
    if len(extent) != 3:
        extent = [0.0, 0.0, 0.0]
    eig = float_list(obj.get("pca_eigenvalues"))
    if len(eig) == 3 and eig[0] > 1e-12:
        linearity = (eig[0] - eig[1]) / eig[0]
        planarity = (eig[1] - eig[2]) / eig[0]
        scattering = eig[2] / eig[0]
    else:
        linearity = 0.0
        planarity = float(obj.get("planarity") or 0.0)
        scattering = 0.0
    normal = float_list(obj.get("pca_normal"))
    normal_z_abs = abs(normal[2]) if len(normal) == 3 else 0.0
    volume = max(extent[0] * extent[1] * extent[2], 1e-6)
    point_count = float(obj.get("point_count") or 0.0)
    return {
        "extent_max": extent[0],
        "extent_mid": extent[1],
        "extent_min": extent[2],
        "extent_mid_ratio": extent[1] / extent[0] if extent[0] > 0 else 0.0,
        "extent_min_ratio": extent[2] / extent[0] if extent[0] > 0 else 0.0,
        "bbox_density": point_count / volume,
        "linearity": linearity,
        "planarity_pca": planarity,
        "scattering": scattering,
        "normal_z_abs": normal_z_abs,
        "thickness_rms": float(obj.get("thickness_rms") or 0.0),
        "point_count": point_count,
    }


def check_car(obj: dict[str, Any], f: dict[str, float]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if f["point_count"] < 180:
        reasons.append("too_few_points_for_car")
    if f["extent_max"] > 8.5:
        reasons.append("car_extent_too_large")
    if f["extent_max"] < 0.35:
        reasons.append("car_extent_too_small")
    if f["extent_min"] < 0.08 and f["extent_max"] > 1.2:
        reasons.append("car_thin_sheet")
    if f["linearity"] > 0.96 and f["extent_mid_ratio"] < 0.35:
        reasons.append("car_too_linear")
    if f["normal_z_abs"] > 0.92 and f["extent_min"] < 0.35 and f["extent_max"] > 2.0:
        reasons.append("car_horizontal_surface_fragment")
    return not reasons, reasons


def check_railing(obj: dict[str, Any], f: dict[str, float]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if f["point_count"] < 120:
        reasons.append("too_few_points_for_railing")
    if f["extent_max"] < 0.6:
        reasons.append("railing_extent_too_small")
    if f["normal_z_abs"] > 0.72:
        reasons.append("railing_horizontal_surface_normal")
    if f["extent_max"] > 8.5 and f["extent_min_ratio"] > 0.28:
        reasons.append("railing_broad_volume")
    if f["point_count"] > 8000 and f["thickness_rms"] > 0.25:
        reasons.append("railing_dense_thick_cluster")
    if f["linearity"] < 0.50 and f["planarity_pca"] > 0.45 and f["extent_mid_ratio"] > 0.75:
        reasons.append("railing_wall_like_plane")
    return not reasons, reasons


def check_label(obj: dict[str, Any], label: str) -> tuple[bool, list[str], dict[str, float]]:
    f = geometry_features(obj)
    if label == "car":
        ok, reasons = check_car(obj, f)
    elif label == "railing":
        ok, reasons = check_railing(obj, f)
    else:
        ok, reasons = True, []
    return ok, reasons, f


def transform_objects(objects: list[dict[str, Any]], labels: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    demoted: dict[int, dict[str, Any]] = {}
    checked_counts = Counter()
    demoted_label_counts = Counter()
    reason_counts = Counter()

    for obj in objects:
        out = dict(obj)
        object_id = int(out["object_id"])
        label = str(out.get("semantic_label") or "unknown")
        if label in labels and str(out.get("visual_review_status") or "") == "visual_confirmed":
            checked_counts[label] += 1
            ok, reasons, features = check_label(out, label)
            out["geometry_guard_checked"] = True
            out["geometry_guard_features"] = {k: round(v, 6) for k, v in features.items()}
            out["geometry_guard_reasons"] = reasons
            if not ok:
                out["candidate_label"] = out.get("candidate_label") or label
                out["semantic_label_original"] = out.get("semantic_label_original") or label
                out["semantic_label"] = "fine_candidate"
                out["candidate_status"] = f"geometry_rejected_visual_confirmed_{label}"
                out["status"] = f"geometry_rejected_visual_confirmed_{label}"
                out["scene_context"] = "geometry_rejected_fine_object_candidate"
                out["scene_description"] = f"visual confirmed {label}, but 3D geometry guard rejected final promotion"
                out["review_priority"] = "high"
                demoted[object_id] = {"old_label": label, "reasons": reasons, "features": features}
                demoted_label_counts[label] += 1
                reason_counts.update(reasons)
            else:
                out["candidate_status"] = f"geometry_confirmed_visual_{label}"
                out["status"] = f"geometry_confirmed_visual_{label}"
        out_rows.append(out)

    return out_rows, {
        "checked_label_counts": dict(checked_counts),
        "demoted_object_count": len(demoted),
        "demoted_label_counts": dict(demoted_label_counts),
        "demotion_reason_counts": dict(reason_counts),
        "demoted_objects": [
            {
                "object_id": object_id,
                "old_label": info["old_label"],
                "reasons": info["reasons"],
                "features": {k: round(v, 6) for k, v in info["features"].items()},
            }
            for object_id, info in sorted(demoted.items())
        ],
    }


def rewrite_ply(input_ply: Path, output_ply: Path, objects_by_id: dict[int, dict[str, Any]]) -> dict[str, Any]:
    props, vertex_count = parse_ply_header(input_ply)
    idx = {name: i for i, name in enumerate(props)}
    object_col = idx.get("object", idx.get("object_id"))
    semantic_col = idx.get("semantic")
    if object_col is None or semantic_col is None:
        raise ValueError(f"PLY needs object and semantic fields: {input_ply}")

    changed_points = 0
    changed_objects: set[int] = set()
    semantic_counts = Counter()
    output_ply.parent.mkdir(parents=True, exist_ok=True)
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
            if not obj:
                dst.write(line)
                continue
            label = str(obj.get("semantic_label") or "unknown")
            semantic = LABEL_TO_SEMANTIC.get(label, 0)
            old_semantic = int(round(float(parts[semantic_col])))
            if semantic != old_semantic:
                parts[semantic_col] = str(semantic)
                changed_points += 1
                changed_objects.add(object_id)
            semantic_counts[label] += 1
            dst.write(" ".join(parts) + "\n")
    return {
        "vertex_count": vertex_count,
        "changed_points": changed_points,
        "changed_object_count": len(changed_objects),
        "semantic_counts_after": dict(semantic_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--input-objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="full_scene_objects_visual_geometry_guard")
    parser.add_argument("--labels", nargs="+", default=["car", "railing"])
    args = parser.parse_args()

    objects = read_jsonl(args.input_objects_jsonl)
    transformed, transform_report = transform_objects(objects, set(args.labels))
    objects_by_id = {int(obj["object_id"]): obj for obj in transformed}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.output_dir / f"{args.output_prefix}.jsonl"
    out_ply = args.output_dir / f"{args.output_prefix}.ply"
    write_jsonl(out_jsonl, transformed)
    ply_report = rewrite_ply(args.input_ply, out_ply, objects_by_id)

    report = {
        "input_ply": str(args.input_ply),
        "input_objects_jsonl": str(args.input_objects_jsonl),
        "output_ply": str(out_ply),
        "output_jsonl": str(out_jsonl),
        "object_count": len(transformed),
        "object_label_counts_after": dict(Counter(str(obj.get("semantic_label") or "unknown") for obj in transformed)),
        **transform_report,
        **ply_report,
    }
    (args.output_dir / f"{args.output_prefix}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
