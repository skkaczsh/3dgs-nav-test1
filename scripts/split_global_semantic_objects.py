#!/usr/bin/env python3
"""Split mixed global voxel objects into semantic/spatial sub-objects.

``build_global_semantic_votes.py`` first clusters voxels by coarse label. In
practice, some resulting objects still contain mixed surfaces or fine objects
because the upstream target fusion occasionally connects distinct structures.

This script works one level later:

1. read global voxel JSONL and object JSONL,
2. assign each voxel a conservative split label from its own label + VLM text,
3. split each original object by split label and 3D connected components,
4. write a new global voxel/object set for downstream refinement.

It does not rerun SAM/VLM or touch original target artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

try:
    from project_semantic import LABEL_COLORS, LABEL_NAMES
except ImportError:  # pragma: no cover
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

PATTERNS = {
    "ceiling": re.compile(r"\b(ceiling|overhead|underside|soffit|roof underside)\b", re.I),
    "floor": re.compile(
        r"\b(floor|ground|rooftop floor|roof floor|roof surface|rooftop surface|"
        r"walkable|platform|pavement|stair landing|tiled floor|roof deck)\b",
        re.I,
    ),
    "wall": re.compile(r"\b(wall|facade|façade|parapet|partition|vertical plane|vertical surface|panel)\b", re.I),
    "railing": re.compile(r"\b(railing|guardrail|handrail|fence|barrier|balustrade)\b", re.I),
    "pipe": re.compile(r"\b(pipe|conduit|cable|duct|tube|hose|wire)\b", re.I),
    "equipment": re.compile(
        r"\b(equipment|hvac|air[- ]?conditioning|outdoor unit|machine|cabinet|device|fixture|sensor|antenna)\b",
        re.I,
    ),
}


def semantic_id(label: str) -> int:
    return int(LABEL_IDS.get(label, 0))


def semantic_color(label: str) -> tuple[int, int, int]:
    return tuple(int(x) for x in LABEL_COLORS.get(semantic_id(label), LABEL_COLORS[0]))


def normalize_votes(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for key, weight in value.items():
        try:
            out[str(key)] = float(weight)
        except (TypeError, ValueError):
            continue
    return out


def add_votes(dst: Counter[str], src: Any) -> None:
    for key, weight in normalize_votes(src).items():
        dst[key] += float(weight)


def dominant(votes: Counter[str] | dict[str, float], default: str = "unknown") -> tuple[str, float, float]:
    if not votes:
        return default, 0.0, 0.0
    total = float(sum(float(v) for v in votes.values()))
    key, value = max(votes.items(), key=lambda kv: float(kv[1]))
    return str(key), float(value), float(value) / max(total, 1e-9)


def row_text(row: dict[str, Any]) -> str:
    chunks = [
        row.get("identity", ""),
        row.get("description", ""),
    ]
    for field in ("identity_votes", "description_votes"):
        votes = normalize_votes(row.get(field))
        chunks.extend(text for text, _ in sorted(votes.items(), key=lambda kv: kv[1], reverse=True)[:4])
    return " ".join(str(x) for x in chunks if x).lower()


def hits(text: str) -> set[str]:
    return {label for label, pattern in PATTERNS.items() if pattern.search(text)}


def geometry_stats(points: list[np.ndarray]) -> dict[str, float]:
    if len(points) < 3:
        return {"normal_abs_z": 1.0, "planarity": 0.0, "linearity": 0.0}
    pts = np.array(points, dtype=np.float64)
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 1e-12)
    vecs = vecs[:, order]
    l1, l2, l3 = [float(x) for x in vals]
    normal_abs_z = float(abs(vecs[:, 2][2]))
    return {
        "normal_abs_z": normal_abs_z,
        "planarity": float((l2 - l3) / max(l1, 1e-12)),
        "linearity": float((l1 - l2) / max(l1, 1e-12)),
    }


def split_label_for_row(row: dict[str, Any], source_geom: dict[str, float]) -> tuple[str, str]:
    label = str(row.get("label") or "unknown")
    text = row_text(row)
    h = hits(text)
    label_votes = normalize_votes(row.get("label_votes"))
    total = sum(label_votes.values())
    def ratio(name: str) -> float:
        return label_votes.get(name, 0.0) / max(total, 1e-9)

    # Fine labels need either an existing coarse vote or explicit text on the
    # voxel. This prevents broad rooftop/floor text with one noisy word from
    # becoming equipment.
    if "railing" in h and (label == "railing" or ratio("railing") >= 0.08):
        return "railing", "row_railing"
    if "pipe" in h and (label == "pipe" or ratio("pipe") >= 0.05):
        return "pipe", "row_pipe"
    if "equipment" in h and (label == "equipment" or ratio("equipment") >= 0.18):
        return "equipment", "row_equipment"

    normal_abs_z = float(source_geom.get("normal_abs_z", 1.0))
    horizontal_object = normal_abs_z >= 0.70
    vertical_object = normal_abs_z <= 0.45

    if "ceiling" in h and (label == "ceiling" or ratio("ceiling") >= 0.08 or (label == "floor" and horizontal_object)):
        return "ceiling", "text_ceiling"
    if "floor" in h and (
        label == "floor"
        or ratio("floor") >= 0.20
        or (label in {"wall", "building", "other", "ambiguous", "unknown"} and horizontal_object and ratio("wall") < 0.90)
    ):
        return "floor", "text_floor"
    if "wall" in h and (
        label == "wall"
        or ratio("wall") >= 0.20
        or (label in {"floor", "building", "other", "ambiguous", "unknown"} and vertical_object and ratio("floor") < 0.90)
    ):
        return "wall", "text_wall"
    return label, "keep_label"


def cell_key(point: np.ndarray, size: float) -> tuple[int, int, int]:
    return tuple(int(x) for x in np.floor(point / size).astype(np.int64))


def connected_components(rows: list[dict[str, Any]], cell_size: float) -> list[list[int]]:
    if not rows:
        return []
    cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        cells[cell_key(np.array(row["centroid"], dtype=np.float64), cell_size)].append(idx)
    remaining = set(range(len(rows)))
    comps: list[list[int]] = []
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]
    while remaining:
        start = remaining.pop()
        comp = [start]
        queue = deque([start])
        while queue:
            current = queue.popleft()
            c = cell_key(np.array(rows[current]["centroid"], dtype=np.float64), cell_size)
            for dx, dy, dz in offsets:
                for candidate in cells.get((c[0] + dx, c[1] + dy, c[2] + dz), []):
                    if candidate in remaining:
                        remaining.remove(candidate)
                        queue.append(candidate)
                        comp.append(candidate)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def make_object(object_number: int, rows: list[dict[str, Any]], split_label: str, source_object: int) -> dict[str, Any]:
    points = np.array([r["centroid"] for r in rows], dtype=np.float64)
    label_votes: Counter[str] = Counter()
    identity_votes: Counter[str] = Counter()
    description_votes: Counter[str] = Counter()
    for row in rows:
        add_votes(label_votes, row.get("label_votes"))
        add_votes(identity_votes, row.get("identity_votes"))
        add_votes(description_votes, row.get("description_votes"))
        label_votes[str(row.get("label") or "unknown")] += float(row.get("label_weight") or row.get("point_count") or 1)
    dominant_label, _, label_ratio = dominant(label_votes, split_label)
    identity, _, identity_ratio = dominant(identity_votes, dominant_label)
    description, _, _ = dominant(description_votes, identity)
    return {
        "object_id": f"global_obj_{object_number:06d}",
        "object_number": object_number,
        "source_object_number": source_object,
        "semantic_label": split_label,
        "pre_split_dominant_label": dominant_label,
        "display_identity": identity,
        "description": description,
        "voxel_count": len(rows),
        "point_count": int(sum(int(r.get("point_count") or 0) for r in rows)),
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
        "status": "global_split_object" if len(rows) > 1 else "global_split_single_voxel",
    }


def read_objects(path: Path) -> dict[int, dict[str, Any]]:
    objects = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                objects[int(row.get("object_number") or 0)] = row
    return objects


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_ply(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(rows)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property int object\nproperty uchar semantic\n")
        f.write("property float purity\n")
        f.write("property int votes\n")
        f.write("end_header\n")
        for row in rows:
            point = row["centroid"]
            label = str(row.get("label") or "unknown")
            color = semantic_color(label)
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{color[0]} {color[1]} {color[2]} "
                f"{int(row.get('object_number') or 0)} {semantic_id(label)} "
                f"{float(row.get('label_purity') or 0.0):.6f} {int(row.get('point_count') or 0)}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voxels-jsonl", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--component-cell-size", type=float, default=0.18)
    parser.add_argument("--min-component-voxels", type=int, default=3)
    parser.add_argument("--split-small-components", action="store_true")
    args = parser.parse_args()

    source_objects = read_objects(args.objects_jsonl)
    by_source_object: dict[int, list[dict[str, Any]]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    with args.voxels_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            source_object = int(row.get("object_number") or 0)
            by_source_object[source_object].append(row)
            source_counts[str(row.get("label") or "unknown")] += 1

    by_object_label: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for source_object, rows in by_source_object.items():
        source_geom = geometry_stats([np.array(row["centroid"], dtype=np.float64) for row in rows])
        for row in rows:
            label, reason = split_label_for_row(row, source_geom)
            row["pre_global_split_label"] = row.get("label")
            row["label"] = label
            row["semantic_id"] = semantic_id(label)
            row["global_split_reason"] = reason
            row["source_object_normal_abs_z"] = source_geom["normal_abs_z"]
            by_object_label[(source_object, label)].append(row)
            reason_counts[reason] += 1

    output_rows: list[dict[str, Any]] = []
    output_objects: list[dict[str, Any]] = []
    object_number = 0
    split_source_objects = 0
    source_object_child_counts: Counter[int] = Counter()
    for (source_object, label), rows in sorted(by_object_label.items()):
        comps = connected_components(rows, args.component_cell_size)
        if not args.split_small_components:
            large: list[list[int]] = []
            small: list[int] = []
            for comp in comps:
                if len(comp) >= args.min_component_voxels:
                    large.append(comp)
                else:
                    small.extend(comp)
            if small:
                large.append(small)
            comps = large
        for comp in comps:
            object_number += 1
            comp_rows = [dict(rows[i]) for i in comp]
            for row in comp_rows:
                row["source_object_number"] = source_object
                row["object_number"] = object_number
                output_rows.append(row)
            output_objects.append(make_object(object_number, comp_rows, label, source_object))
            source_object_child_counts[source_object] += 1

    split_source_objects = sum(1 for count in source_object_child_counts.values() if count > 1)
    write_jsonl(args.output_dir / "global_semantic_voxels_split.jsonl", output_rows)
    write_jsonl(args.output_dir / "global_semantic_objects_split.jsonl", output_objects)
    write_ply(args.output_dir / "global_semantic_voxels_split.ply", output_rows)
    label_counts = Counter(str(row.get("label") or "unknown") for row in output_rows)
    report = {
        "voxels_jsonl": str(args.voxels_jsonl),
        "objects_jsonl": str(args.objects_jsonl),
        "output_dir": str(args.output_dir),
        "source_object_count": len(source_objects),
        "output_object_count": len(output_objects),
        "split_source_object_count": split_source_objects,
        "source_label_counts": dict(source_counts),
        "output_label_counts": dict(label_counts),
        "reason_counts": dict(reason_counts),
        "params": {
            "component_cell_size": args.component_cell_size,
            "min_component_voxels": args.min_component_voxels,
            "split_small_components": args.split_small_components,
        },
    }
    (args.output_dir / "global_semantic_split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
