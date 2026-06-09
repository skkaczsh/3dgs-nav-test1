#!/usr/bin/env python3
"""Estimate whether small target residual points can be absorbed into stable surfaces.

This is a diagnostic step for the target/object route. It does not mutate the
dataset. It reads residual PLY files exported by build_targets_from_masks.py and
stable objects from fuse_targets_to_objects.py, then checks whether each
residual point is close to a stable surface object in 3D, PCA plane distance,
and visual color.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SURFACE_LABELS = {"floor", "wall", "building", "road"}
SEMANTIC_NAMES = {
    0: "unknown",
    1: "other",
    2: "wall",
    3: "floor",
    4: "ceiling",
    5: "grass",
    6: "tree",
    7: "person",
    8: "car",
    9: "railing",
    10: "building",
    11: "sky",
    12: "road",
    13: "water",
    14: "furniture",
    15: "pipe",
    16: "equipment",
    254: "ambiguous",
    255: "ignore",
}
SEMANTIC_IDS = {name: value for value, name in SEMANTIC_NAMES.items()}
LABEL_COLORS = {
    0: (128, 128, 128),
    1: (160, 160, 160),
    2: (200, 200, 200),
    3: (139, 100, 60),
    4: (240, 240, 240),
    5: (80, 180, 80),
    6: (20, 120, 40),
    7: (255, 80, 80),
    8: (60, 120, 255),
    9: (255, 210, 40),
    10: (190, 170, 140),
    11: (135, 206, 250),
    12: (80, 80, 80),
    13: (30, 160, 220),
    14: (120, 80, 200),
    15: (255, 165, 0),
    16: (255, 0, 255),
    254: (120, 60, 255),
    255: (30, 30, 30),
}


def read_ascii_ply(path: Path) -> tuple[list[str], np.ndarray]:
    props: list[str] = []
    vertex_count = None
    header_lines = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header_lines += 1
            s = line.strip()
            if s.startswith("element vertex"):
                vertex_count = int(s.split()[-1])
            elif s.startswith("property"):
                props.append(s.split()[-1])
            elif s == "end_header":
                break
    if vertex_count is None:
        raise ValueError(f"missing vertex count: {path}")
    if vertex_count == 0:
        return props, np.empty((0, len(props)), dtype=np.float32)
    data = np.loadtxt(path, skiprows=header_lines, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return props, data


def bbox_distance(point: np.ndarray, bbox: dict) -> float:
    bmin = np.array(bbox["min"], dtype=np.float32)
    bmax = np.array(bbox["max"], dtype=np.float32)
    gap = np.maximum(0.0, np.maximum(bmin - point, point - bmax))
    return float(np.linalg.norm(gap))


def plane_distance(point: np.ndarray, obj: dict) -> float:
    normal = np.array(obj.get("normal", [0.0, 0.0, 1.0]), dtype=np.float32)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        return 0.0
    normal = normal / norm
    centroid = np.array(obj["centroid"], dtype=np.float32)
    return float(abs((point - centroid) @ normal))


def label_compatible(residual_label: str, object_label: str) -> bool:
    if residual_label == object_label:
        return True
    if residual_label in {"floor", "road"} and object_label in {"floor", "road"}:
        return True
    if residual_label in {"wall", "building"} and object_label in {"wall", "building"}:
        return True
    return False


def load_surface_objects(path: Path, min_targets: int, min_points: int) -> list[dict]:
    objects = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("status") != "stable":
                continue
            if obj.get("semantic_label") not in SURFACE_LABELS:
                continue
            if int(obj.get("target_count", 0)) < min_targets:
                continue
            if int(obj.get("point_count", 0)) < min_points:
                continue
            objects.append(obj)
    return objects


def cell_key(point: np.ndarray, cell_size: float) -> tuple[int, int, int]:
    cell = np.floor(point / cell_size).astype(int)
    return int(cell[0]), int(cell[1]), int(cell[2])


def object_cells(obj: dict, cell_size: float, padding: float) -> set[tuple[int, int, int]]:
    bmin = np.array(obj["bbox_3d"]["min"], dtype=np.float32) - padding
    bmax = np.array(obj["bbox_3d"]["max"], dtype=np.float32) + padding
    lo = np.floor(bmin / cell_size).astype(int)
    hi = np.floor(bmax / cell_size).astype(int)
    return {
        (int(x), int(y), int(z))
        for x in range(lo[0], hi[0] + 1)
        for y in range(lo[1], hi[1] + 1)
        for z in range(lo[2], hi[2] + 1)
    }


def build_index(objects: list[dict], cell_size: float, padding: float) -> dict[tuple[int, int, int], list[int]]:
    index: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i, obj in enumerate(objects):
        for cell in object_cells(obj, cell_size, padding):
            index[cell].append(i)
    return index


def analyze(args: argparse.Namespace) -> dict:
    objects = load_surface_objects(args.objects_jsonl, args.min_object_targets, args.min_object_points)
    index = build_index(objects, args.cell_size, args.bbox_padding)
    total = 0
    absorbable = 0
    by_label = Counter()
    absorbable_by_label = Counter()
    reason_counts = Counter()
    examples = []

    files = sorted(args.residual_dir.glob("residuals_frame_*.ply"))
    if args.limit_frames:
        files = files[: args.limit_frames]
    for path in files:
        props, data = read_ascii_ply(path)
        if len(data) == 0:
            continue
        idx = {name: i for i, name in enumerate(props)}
        points = data[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float32)
        colors = data[:, [idx["visual_red"], idx["visual_green"], idx["visual_blue"]]].astype(np.float32)
        labels = data[:, idx["semantic"]].astype(np.int32)
        for point, color, sem in zip(points, colors, labels):
            total += 1
            residual_label = SEMANTIC_NAMES.get(int(sem), "unknown")
            by_label[residual_label] += 1
            candidate_ids = index.get(cell_key(point, args.cell_size), [])
            best = None
            best_meta = None
            for object_idx in candidate_ids:
                obj = objects[object_idx]
                object_label = obj.get("semantic_label", "unknown")
                if not label_compatible(residual_label, object_label):
                    continue
                bd = bbox_distance(point, obj["bbox_3d"])
                if bd > args.bbox_padding:
                    continue
                pd = plane_distance(point, obj)
                if pd > args.max_plane_distance:
                    continue
                cd = float(np.linalg.norm(color - np.array(obj.get("mean_color", [0, 0, 0]), dtype=np.float32)))
                if cd > args.max_color_distance:
                    continue
                score = bd + pd + cd / 255.0
                if best is None or score < best:
                    best = score
                    best_meta = {
                        "object_id": obj["object_id"],
                        "object_label": object_label,
                        "bbox_distance": bd,
                        "plane_distance": pd,
                        "color_distance": cd,
                    }
            if best_meta:
                absorbable += 1
                absorbable_by_label[residual_label] += 1
                reason_counts["matched_surface"] += 1
                if len(examples) < args.example_limit:
                    examples.append({"residual_label": residual_label, **best_meta})
            else:
                reason_counts["no_surface_match"] += 1

    return {
        "residual_dir": str(args.residual_dir),
        "objects_jsonl": str(args.objects_jsonl),
        "surface_objects": len(objects),
        "residual_points": int(total),
        "absorbable_points": int(absorbable),
        "absorbable_ratio": float(absorbable / max(total, 1)),
        "by_label": dict(by_label),
        "absorbable_by_label": dict(absorbable_by_label),
        "reason_counts": dict(reason_counts),
        "params": {
            "min_object_targets": args.min_object_targets,
            "min_object_points": args.min_object_points,
            "cell_size": args.cell_size,
            "bbox_padding": args.bbox_padding,
            "max_plane_distance": args.max_plane_distance,
            "max_color_distance": args.max_color_distance,
        },
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-dir", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-object-targets", type=int, default=5)
    parser.add_argument("--min-object-points", type=int, default=1000)
    parser.add_argument("--cell-size", type=float, default=1.0)
    parser.add_argument("--bbox-padding", type=float, default=0.35)
    parser.add_argument("--max-plane-distance", type=float, default=0.12)
    parser.add_argument("--max-color-distance", type=float, default=70.0)
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--example-limit", type=int, default=30)
    args = parser.parse_args()

    report = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
