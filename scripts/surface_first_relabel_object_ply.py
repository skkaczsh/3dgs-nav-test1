#!/usr/bin/env python3
"""Relabel an object-level semantic PLY with surface-first geometry rules.

This is a local QA tool for the target/object fusion output. It does not need
per-frame masks or residual files: it groups points by the existing `object`
property, estimates object geometry from XYZ, and rewrites only the semantic
label/color for objects whose shape strongly contradicts the current label.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


LABEL_NAMES = {
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
    255: "ignore",
}

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
    255: (30, 30, 30),
}


LABEL_IDS = {name: idx for idx, name in LABEL_NAMES.items()}
SURFACE_LABELS = {"floor", "wall", "building", "road", "ceiling"}
FINE_LABELS = {"equipment", "railing", "pipe", "furniture"}


def read_ascii_ply(path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    props: list[str] = []
    header: list[str] = []
    header_lines = 0
    fmt = ""
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header.append(line)
            header_lines += 1
            s = line.strip()
            if s.startswith("format "):
                fmt = s.split()[1]
            elif s.startswith("element vertex"):
                in_vertex = True
            elif s.startswith("element "):
                in_vertex = False
            elif in_vertex and s.startswith("property"):
                props.append(s.split()[-1])
            elif s == "end_header":
                break
    if fmt != "ascii":
        raise ValueError(f"Only ASCII PLY is supported, got format={fmt}")
    data = np.loadtxt(path, skiprows=header_lines, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return props, data, header


def pca_shape(points: np.ndarray) -> dict:
    if len(points) < 3:
        return {
            "normal": [0.0, 0.0, 1.0],
            "linearity": 0.0,
            "planarity": 0.0,
            "scattering": 1.0,
            "eigenvalues": [0.0, 0.0, 0.0],
        }
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    denom = max(float(vals[0]), 1e-12)
    normal = vecs[:, -1]
    if normal[2] < 0:
        normal = -normal
    return {
        "normal": [float(x) for x in normal.tolist()],
        "linearity": float((vals[0] - vals[1]) / denom),
        "planarity": float((vals[1] - vals[2]) / denom),
        "scattering": float(vals[2] / denom),
        "eigenvalues": [float(x) for x in vals.tolist()],
    }


def bbox_stats(points: np.ndarray) -> dict:
    bmin = points.min(axis=0)
    bmax = points.max(axis=0)
    extent = bmax - bmin
    extent_sorted = np.sort(extent)[::-1]
    return {
        "min": [float(x) for x in bmin],
        "max": [float(x) for x in bmax],
        "extent": [float(x) for x in extent],
        "max_extent": float(extent_sorted[0]),
        "mid_extent": float(extent_sorted[1]),
        "min_extent": float(extent_sorted[2]),
        "xy_span": float(np.linalg.norm(extent[:2])),
        "z_span": float(extent[2]),
    }


def dominant_semantic(values: np.ndarray) -> tuple[int, float, dict[str, int]]:
    counts = Counter(int(x) for x in values.tolist())
    label_id, count = counts.most_common(1)[0]
    total = max(sum(counts.values()), 1)
    named = {LABEL_NAMES.get(k, str(k)): int(v) for k, v in counts.items()}
    return label_id, float(count / total), named


def classify_object(current_label: str, point_count: int, shape: dict, bbox: dict, args: argparse.Namespace) -> tuple[str, str]:
    normal = np.array(shape["normal"], dtype=np.float64)
    abs_z = abs(float(normal[2]))
    planarity = float(shape["planarity"])
    linearity = float(shape["linearity"])
    scattering = float(shape["scattering"])
    max_extent = float(bbox["max_extent"])
    mid_extent = float(bbox["mid_extent"])
    min_extent = float(bbox["min_extent"])
    z_span = float(bbox["z_span"])

    is_large = point_count >= args.min_surface_points and max_extent >= args.min_surface_extent
    planar = planarity >= args.min_surface_planarity and scattering <= args.max_surface_scattering
    sheet_like = is_large and planar and mid_extent >= args.min_surface_mid_extent
    line_like = linearity >= args.min_railing_linearity and max_extent >= args.min_railing_extent and mid_extent <= args.max_railing_mid_extent
    compact = max_extent <= args.max_equipment_extent and z_span >= args.min_equipment_z_span

    if current_label in {"railing", "pipe"} and line_like:
        return current_label, "preserve_linear_fine_object"
    if current_label == "equipment" and compact and not sheet_like:
        return current_label, "preserve_compact_equipment"

    if sheet_like:
        if abs_z >= args.floor_normal_z:
            return "floor", "large_horizontal_planar_surface"
        if abs_z <= args.wall_normal_z:
            return "wall" if current_label in {"floor", "equipment", "unknown", "other"} else "building", "large_vertical_planar_surface"
        if current_label in FINE_LABELS:
            return "building", "large_oblique_planar_fine_label"
        if current_label in SURFACE_LABELS:
            return current_label, "preserve_oblique_surface"

    if current_label == "equipment" and is_large and (planar or mid_extent >= args.min_surface_mid_extent):
        return "building", "large_equipment_surface_like"
    if current_label == "floor" and is_large and abs_z <= args.wall_normal_z and planar:
        return "wall", "floor_with_vertical_planar_geometry"
    if current_label == "wall" and is_large and abs_z >= args.floor_normal_z and planar:
        return "floor", "wall_with_horizontal_planar_geometry"

    return current_label, "unchanged"


def write_ply(path: Path, header: list[str], props: list[str], data: np.ndarray) -> None:
    vertex_idx = None
    for i, line in enumerate(header):
        if line.startswith("element vertex"):
            vertex_idx = i
            break
    if vertex_idx is None:
        raise ValueError("PLY header missing element vertex")
    header_out = list(header)
    header_out[vertex_idx] = f"element vertex {len(data)}\n"
    fmt_by_prop = []
    for prop in props:
        if prop in {"red", "green", "blue", "semantic"}:
            fmt_by_prop.append("{:d}")
        elif prop in {"object", "frame", "target", "camera", "mask"}:
            fmt_by_prop.append("{:d}")
        else:
            fmt_by_prop.append("{:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.writelines(header_out)
        for row in data:
            values = []
            for value, fmt in zip(row, fmt_by_prop):
                if fmt == "{:d}":
                    values.append(fmt.format(int(round(float(value)))))
                else:
                    values.append(fmt.format(float(value)))
            f.write(" ".join(values) + "\n")


def process(args: argparse.Namespace) -> dict:
    props, data, header = read_ascii_ply(args.input_ply)
    idx = {name: i for i, name in enumerate(props)}
    required = {"x", "y", "z", "red", "green", "blue", "object", "semantic"}
    if not required.issubset(idx):
        raise ValueError(f"missing required fields: {sorted(required - set(idx))}; available={props}")

    out = data.copy()
    objects = sorted(set(int(x) for x in data[:, idx["object"]].tolist()))
    report_rows = []
    reason_counts = Counter()
    before_counts = Counter()
    after_counts = Counter()
    changed_points = 0

    for object_id in objects:
        mask = data[:, idx["object"]].astype(np.int64) == object_id
        rows = data[mask]
        points = rows[:, [idx["x"], idx["y"], idx["z"]]]
        label_id, label_ratio, semantic_counts = dominant_semantic(rows[:, idx["semantic"]])
        current_label = LABEL_NAMES.get(label_id, "unknown")
        shape = pca_shape(points)
        bbox = bbox_stats(points)
        new_label, reason = classify_object(current_label, len(rows), shape, bbox, args)
        new_id = LABEL_IDS.get(new_label, label_id)
        before_counts[current_label] += int(len(rows))
        after_counts[new_label] += int(len(rows))
        reason_counts[reason] += int(len(rows))
        if new_label != current_label:
            changed_points += int(len(rows))
            color = LABEL_COLORS.get(new_id, LABEL_COLORS[0])
            out[mask, idx["semantic"]] = new_id
            out[mask, idx["red"]] = color[0]
            out[mask, idx["green"]] = color[1]
            out[mask, idx["blue"]] = color[2]
        report_rows.append(
            {
                "object": int(object_id),
                "points": int(len(rows)),
                "before": current_label,
                "after": new_label,
                "reason": reason,
                "dominant_label_ratio": label_ratio,
                "semantic_counts": semantic_counts,
                "bbox": bbox,
                "pca": shape,
            }
        )

    write_ply(args.output_ply, header, props, out)
    params = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    summary = {
        "input_ply": str(args.input_ply),
        "output_ply": str(args.output_ply),
        "objects": len(objects),
        "points": int(len(data)),
        "changed_points": int(changed_points),
        "changed_ratio": float(changed_points / max(len(data), 1)),
        "before_counts": dict(before_counts),
        "after_counts": dict(after_counts),
        "reason_counts": dict(reason_counts),
        "params": params,
        "objects_detail": sorted(report_rows, key=lambda r: (-r["points"], r["object"])),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-surface-points", type=int, default=800)
    parser.add_argument("--min-surface-extent", type=float, default=2.0)
    parser.add_argument("--min-surface-mid-extent", type=float, default=0.8)
    parser.add_argument("--min-surface-planarity", type=float, default=0.18)
    parser.add_argument("--max-surface-scattering", type=float, default=0.16)
    parser.add_argument("--floor-normal-z", type=float, default=0.72)
    parser.add_argument("--wall-normal-z", type=float, default=0.35)
    parser.add_argument("--min-railing-linearity", type=float, default=0.72)
    parser.add_argument("--min-railing-extent", type=float, default=1.2)
    parser.add_argument("--max-railing-mid-extent", type=float, default=0.45)
    parser.add_argument("--max-equipment-extent", type=float, default=2.2)
    parser.add_argument("--min-equipment-z-span", type=float, default=0.25)
    args = parser.parse_args()
    summary = process(args)
    print(json.dumps({k: summary[k] for k in ["objects", "points", "changed_points", "changed_ratio", "before_counts", "after_counts", "reason_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
