#!/usr/bin/env python3
"""Build a global semantic voxel vote cloud from per-frame Targets.

Inputs are Target JSONL files from the validated projection route. Each target
already represents one 2D mask projected to one frame point cloud. This script
aggregates those target-level semantic/identity observations into global
voxels, then clusters voxels into coarse objects.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from build_targets_from_masks import connected_components, read_colored_ply
from project_semantic import LABEL_COLORS, LABEL_NAMES


LABEL_IDS = {v: k for k, v in LABEL_NAMES.items()}


def iter_target_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.glob("targets_frame_*.jsonl") if p.name != "targets_all.jsonl")
    return [path]


def target_identity(target: dict[str, Any]) -> str:
    return str(
        target.get("identity_text")
        or target.get("freeform_label")
        or target.get("identity_hint")
        or target.get("description")
        or target.get("label")
        or "unknown"
    ).strip()


def target_description(target: dict[str, Any]) -> str:
    return str(
        target.get("description")
        or target.get("identity_text")
        or target.get("freeform_label")
        or target.get("label")
        or "unknown"
    ).strip()


def target_weight(target: dict[str, Any], args: argparse.Namespace) -> float:
    try:
        confidence = float(target.get("confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 1.0
    confidence = max(args.min_vote_confidence, min(1.0, confidence))
    try:
        cluster_size = max(int(target.get("cluster_size", 1)), 1)
    except (TypeError, ValueError):
        cluster_size = 1
    size_weight = min(math.sqrt(cluster_size), args.max_size_weight)
    return confidence * size_weight


def semantic_id(label: str) -> int:
    return int(LABEL_IDS.get(label, 0))


def semantic_color(label: str) -> tuple[int, int, int]:
    return tuple(int(x) for x in LABEL_COLORS.get(semantic_id(label), LABEL_COLORS.get(0, (120, 120, 120))))


def add_counter(dst: dict[str, float], key: str, weight: float) -> None:
    key = str(key or "").strip()
    if not key:
        return
    dst[key] = float(dst.get(key, 0.0) + weight)


def dominant(votes: dict[str, float], default: str = "unknown") -> tuple[str, float, float]:
    if not votes:
        return default, 0.0, 0.0
    total = float(sum(votes.values()))
    key, value = max(votes.items(), key=lambda kv: kv[1])
    return key, float(value), float(value / max(total, 1e-9))


def voxel_key(point: np.ndarray, voxel_size: float) -> tuple[int, int, int]:
    return tuple(int(x) for x in np.floor(point / voxel_size).astype(np.int64))


def load_votes(args: argparse.Namespace) -> dict[tuple[int, int, int], dict[str, Any]]:
    frame_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    voxels: dict[tuple[int, int, int], dict[str, Any]] = {}
    for file_path in iter_target_files(args.targets):
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                target = json.loads(line)
                label = str(target.get("label") or "unknown")
                if label in args.skip_labels:
                    continue
                frame_ply = str(target.get("colored_frame_ply") or "")
                indices = np.array(target.get("point_indices") or [], dtype=np.int64)
                if not frame_ply or indices.size == 0:
                    continue
                if frame_ply not in frame_cache:
                    frame_cache[frame_ply] = read_colored_ply(Path(frame_ply))
                frame_points, frame_colors = frame_cache[frame_ply]
                valid = indices[(indices >= 0) & (indices < len(frame_points))]
                if valid.size == 0:
                    continue
                points = frame_points[valid]
                colors = frame_colors[valid] if len(frame_colors) else np.zeros((len(points), 3), dtype=np.uint8)
                weight = target_weight(target, args)
                identity = target_identity(target)
                description = target_description(target)
                for point, color in zip(points, colors):
                    key = voxel_key(point, args.voxel_size)
                    row = voxels.get(key)
                    if row is None:
                        row = {
                            "key": key,
                            "count": 0,
                            "point_sum": np.zeros(3, dtype=np.float64),
                            "color_sum": np.zeros(3, dtype=np.float64),
                            "label_votes": {},
                            "identity_votes": {},
                            "description_votes": {},
                            "source_targets": set(),
                            "source_frames": set(),
                        }
                        voxels[key] = row
                    row["count"] += 1
                    row["point_sum"] += point.astype(np.float64)
                    row["color_sum"] += color.astype(np.float64)
                    add_counter(row["label_votes"], label, weight)
                    add_counter(row["identity_votes"], identity, weight)
                    add_counter(row["description_votes"], description, weight)
                    row["source_targets"].add(str(target.get("target_id", "")))
                    row["source_frames"].add(int(target.get("frame_id", 0)))
    return voxels


def voxel_rows(voxels: dict[tuple[int, int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for i, row in enumerate(voxels.values()):
        count = max(int(row["count"]), 1)
        label, label_weight, label_purity = dominant(row["label_votes"])
        identity, identity_weight, identity_purity = dominant(row["identity_votes"], label)
        description, _, _ = dominant(row["description_votes"], identity)
        rows.append({
            "voxel_id": i,
            "key": row["key"],
            "centroid": (row["point_sum"] / count).astype(np.float64),
            "mean_color": np.clip(row["color_sum"] / count, 0, 255).astype(np.float64),
            "point_count": count,
            "label": label,
            "semantic_id": semantic_id(label),
            "label_purity": label_purity,
            "label_weight": label_weight,
            "identity": identity,
            "identity_purity": identity_purity,
            "description": description,
            "label_votes": row["label_votes"],
            "identity_votes": row["identity_votes"],
            "description_votes": row["description_votes"],
            "source_target_count": len(row["source_targets"]),
            "source_frame_count": len(row["source_frames"]),
            "object_number": 0,
        })
    return rows


def cluster_voxels(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        if row["point_count"] < args.min_voxel_points:
            continue
        if row["label_purity"] < args.min_label_purity:
            row["label"] = "ambiguous"
            row["semantic_id"] = 0
        by_label[row["label"]].append(i)

    objects = []
    object_number = 0
    for label, ids in sorted(by_label.items()):
        if not ids:
            continue
        pts = np.array([rows[i]["centroid"] for i in ids], dtype=np.float32)
        comps, residual = connected_components(pts, args.object_voxel_size, args.min_object_voxels)
        for comp in comps:
            object_number += 1
            source_ids = [ids[int(j)] for j in comp]
            for row_id in source_ids:
                rows[row_id]["object_number"] = object_number
            objects.append(make_object(object_number, label, [rows[row_id] for row_id in source_ids]))
        residual_ids = [ids[int(j)] for j in np.where(residual)[0]]
        for row_id in residual_ids:
            object_number += 1
            rows[row_id]["object_number"] = object_number
            objects.append(make_object(object_number, rows[row_id]["label"], [rows[row_id]]))
    return objects


def make_object(object_number: int, label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    points = np.array([r["centroid"] for r in rows], dtype=np.float64)
    label_votes: Counter[str] = Counter()
    identity_votes: Counter[str] = Counter()
    description_votes: Counter[str] = Counter()
    for row in rows:
        label_votes.update(row["label_votes"])
        identity_votes.update(row["identity_votes"])
        description_votes.update(row["description_votes"])
    dominant_label, _, label_ratio = dominant(dict(label_votes), label)
    identity, _, identity_ratio = dominant(dict(identity_votes), dominant_label)
    description, _, _ = dominant(dict(description_votes), identity)
    return {
        "object_id": f"global_obj_{object_number:06d}",
        "object_number": object_number,
        "semantic_label": dominant_label,
        "display_identity": identity,
        "description": description,
        "voxel_count": len(rows),
        "point_count": int(sum(int(r["point_count"]) for r in rows)),
        "centroid": [float(x) for x in points.mean(axis=0)],
        "bbox_3d": {
            "min": [float(x) for x in points.min(axis=0)],
            "max": [float(x) for x in points.max(axis=0)],
        },
        "label_purity": float(label_ratio),
        "identity_purity": float(identity_ratio),
        "label_votes": dict(label_votes),
        "identity_votes": dict(identity_votes),
        "description_votes": dict(description_votes),
        "status": "ambiguous_object" if label_ratio < 0.8 else ("stable" if len(rows) > 1 else "single_voxel"),
    }


def write_voxel_ply(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    kept = [r for r in rows if r["point_count"] >= args.min_voxel_points]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(kept)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property int object\nproperty uchar semantic\n")
        f.write("property float purity\n")
        f.write("property int votes\n")
        f.write("end_header\n")
        for row in kept:
            color = row["mean_color"] if args.color_mode == "rgb" else np.array(semantic_color(row["label"]))
            p = row["centroid"]
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} "
                f"{int(row['object_number'])} {int(row['semantic_id'])} "
                f"{float(row['label_purity']):.6f} {int(row['point_count'])}\n"
            )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            clean = {}
            for key, value in row.items():
                if isinstance(value, np.ndarray):
                    clean[key] = [float(x) for x in value.tolist()]
                elif isinstance(value, set):
                    clean[key] = sorted(value)
                elif key == "key":
                    clean[key] = [int(x) for x in value]
                else:
                    clean[key] = value
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.06)
    parser.add_argument("--object-voxel-size", type=float, default=0.16)
    parser.add_argument("--min-voxel-points", type=int, default=1)
    parser.add_argument("--min-object-voxels", type=int, default=8)
    parser.add_argument("--min-label-purity", type=float, default=0.55)
    parser.add_argument("--min-vote-confidence", type=float, default=0.2)
    parser.add_argument("--max-size-weight", type=float, default=30.0)
    parser.add_argument("--color-mode", choices=["rgb", "semantic"], default="semantic")
    parser.add_argument("--skip-label", dest="skip_labels", action="append", default=["ignore", "sky"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    voxels = load_votes(args)
    rows = voxel_rows(voxels)
    objects = cluster_voxels(rows, args)
    write_voxel_ply(args.output_dir / "global_semantic_voxels.ply", rows, args)
    write_jsonl(args.output_dir / "global_semantic_voxels.jsonl", rows)
    write_jsonl(args.output_dir / "global_semantic_objects.jsonl", objects)
    status_counts = Counter(obj["status"] for obj in objects)
    label_counts = Counter(row["label"] for row in rows)
    report = {
        "targets": str(args.targets),
        "output_dir": str(args.output_dir),
        "voxel_count": len(rows),
        "object_count": len(objects),
        "status_counts": dict(status_counts),
        "label_counts": dict(label_counts),
        "params": {
            k: (
                str(v)
                if isinstance(v, Path)
                else (list(v) if isinstance(v, list) else v)
            )
            for k, v in vars(args).items()
        },
    }
    (args.output_dir / "global_semantic_vote_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
