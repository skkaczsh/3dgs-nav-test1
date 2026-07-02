#!/usr/bin/env python3
"""Build non-overlapping spatial objects from voxel-level semantic votes.

This is the new spatial-partition mainline prototype.  It treats semantic
point-cloud generation as a spatial segmentation problem:

- each voxel has exactly one winning semantic label
- each object is a connected component of voxels with the same label
- no two objects can own the same voxel
- semantic votes are pluggable teachers; current teachers are semantic PLY files
  such as V2/V8, while later mask observations can be added as another vote
  source without changing the spatial partition invariant.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.semantic_label_contract import LABEL_TO_SEMANTIC, SEMANTIC_COLORS, SEMANTIC_TO_LABEL
from scripts.current_mainline_contract import reject_forbidden_production_input


LABELS = SEMANTIC_TO_LABEL
COLORS = {label: SEMANTIC_COLORS[semantic] for semantic, label in LABELS.items()}
COLORS["ground"] = SEMANTIC_COLORS[LABEL_TO_SEMANTIC["ground"]]


def parse_header(path: Path) -> tuple[list[str], list[str], int, int]:
    header: list[str] = []
    props: list[str] = []
    vertex_count = 0
    header_lines = 0
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header.append(line)
            header_lines += 1
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                vertex_count = int(parts[2])
                in_vertex = True
            elif len(parts) >= 2 and parts[0] == "element":
                in_vertex = False
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                props.append(parts[-1])
            if line.strip() == "end_header":
                break
    return header, props, vertex_count, header_lines


def semantic_label_from_id(value: int) -> str:
    return LABELS.get(int(value), "unknown")


def read_ply_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    _header, props, vertex_count, header_lines = parse_header(path)
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(header_lines):
            next(f)
        for i, line in enumerate(f):
            if i >= vertex_count:
                break
            if line.strip():
                rows.append(line.split())
    return props, rows


def key_from_row(row: list[str], idx: dict[str, int], voxel_size: float) -> tuple[int, int, int]:
    return (
        math.floor(float(row[idx["x"]]) / voxel_size),
        math.floor(float(row[idx["y"]]) / voxel_size),
        math.floor(float(row[idx["z"]]) / voxel_size),
    )


def centroid_of_key(key: tuple[int, int, int], voxel_size: float) -> tuple[float, float, float]:
    return tuple((float(v) + 0.5) * voxel_size for v in key)


def parse_teacher(raw: str) -> tuple[str, Path, float]:
    parts = raw.split(":", 2)
    if len(parts) == 1:
        path = Path(parts[0])
        return path.parent.name or path.stem, path, 1.0
    if len(parts) == 2:
        name, path = parts
        return name, Path(path), 1.0
    name, path, weight = parts
    return name, Path(path), float(weight)


def add_teacher_votes(
    votes: dict[tuple[int, int, int], Counter[str]],
    teacher_path: Path,
    weight: float,
    voxel_size: float,
    accepted_labels: set[str] | None,
) -> dict[str, Any]:
    props, rows = read_ply_rows(teacher_path)
    idx = {name: i for i, name in enumerate(props)}
    for required in ("x", "y", "z", "semantic"):
        if required not in idx:
            raise ValueError(f"teacher PLY missing {required}: {teacher_path}")
    label_counts = Counter()
    for row in rows:
        label = semantic_label_from_id(int(round(float(row[idx["semantic"]]))))
        if accepted_labels and label not in accepted_labels:
            continue
        key = key_from_row(row, idx, voxel_size)
        votes[key][label] += float(weight)
        label_counts[label] += 1
    return {
        "teacher_path": str(teacher_path),
        "weight": weight,
        "rows": len(rows),
        "label_counts": dict(label_counts),
    }


def build_voxel_points_from_base(
    base_ply: Path,
    voxel_size: float,
) -> tuple[dict[tuple[int, int, int], dict[str, Any]], dict[str, Any]]:
    props, rows = read_ply_rows(base_ply)
    idx = {name: i for i, name in enumerate(props)}
    for required in ("x", "y", "z", "red", "green", "blue"):
        if required not in idx:
            raise ValueError(f"base PLY missing {required}: {base_ply}")
    accum: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in rows:
        key = key_from_row(row, idx, voxel_size)
        item = accum.get(key)
        xyz = np.array([float(row[idx["x"]]), float(row[idx["y"]]), float(row[idx["z"]])], dtype=np.float64)
        rgb = np.array([float(row[idx["red"]]), float(row[idx["green"]]), float(row[idx["blue"]])], dtype=np.float64)
        if item is None:
            accum[key] = {"count": 1, "xyz_sum": xyz, "rgb_sum": rgb}
        else:
            item["count"] += 1
            item["xyz_sum"] += xyz
            item["rgb_sum"] += rgb
    report = {"base_ply": str(base_ply), "base_rows": len(rows), "base_voxels": len(accum)}
    return accum, report


def select_voxel_labels(
    voxel_points: dict[tuple[int, int, int], dict[str, Any]],
    votes: dict[tuple[int, int, int], Counter[str]],
    default_label: str,
) -> dict[tuple[int, int, int], str]:
    labels: dict[tuple[int, int, int], str] = {}
    for key in voxel_points:
        counter = votes.get(key)
        if counter:
            label, _score = counter.most_common(1)[0]
            labels[key] = label
        else:
            labels[key] = default_label
    return labels


def connected_components_by_label(
    voxel_labels: dict[tuple[int, int, int], str],
    min_voxels_by_label: dict[str, int],
    small_component_policy: str,
) -> tuple[dict[tuple[int, int, int], int], list[dict[str, Any]], dict[str, Any]]:
    offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    unvisited = set(voxel_labels)
    object_for_voxel: dict[tuple[int, int, int], int] = {}
    objects: list[dict[str, Any]] = []
    next_object_id = 1
    dropped = Counter()
    while unvisited:
        start = unvisited.pop()
        label = voxel_labels[start]
        component = [start]
        queue = deque([start])
        while queue:
            key = queue.popleft()
            for dx, dy, dz in offsets:
                nbr = (key[0] + dx, key[1] + dy, key[2] + dz)
                if nbr not in unvisited:
                    continue
                if voxel_labels[nbr] != label:
                    continue
                unvisited.remove(nbr)
                queue.append(nbr)
                component.append(nbr)
        min_voxels = min_voxels_by_label.get(label, min_voxels_by_label.get("*", 1))
        is_small = len(component) < min_voxels
        if is_small and small_component_policy == "drop":
            dropped[label] += len(component)
            continue
        object_id = next_object_id
        next_object_id += 1
        for key in component:
            object_for_voxel[key] = object_id
        pts = np.array([centroid_of_key(key, 1.0) for key in component], dtype=np.float64)
        objects.append(
            {
                "object_id": object_id,
                "semantic_label": label,
                "status": "small_component" if is_small else "spatial_connected_component",
                "voxel_count": int(len(component)),
                "min_voxels_for_label": int(min_voxels),
                "bbox_voxel_min": [int(v) for v in np.min(np.array(component), axis=0).tolist()],
                "bbox_voxel_max": [int(v) for v in np.max(np.array(component), axis=0).tolist()],
            }
        )
    return object_for_voxel, objects, {"dropped_small_voxels_by_label": dict(dropped)}


def enrich_objects_geometry(
    objects: list[dict[str, Any]],
    object_for_voxel: dict[tuple[int, int, int], int],
    voxel_points: dict[tuple[int, int, int], dict[str, Any]],
) -> None:
    points_by_object: dict[int, list[np.ndarray]] = defaultdict(list)
    point_counts_by_object: Counter[int] = Counter()
    for key, object_id in object_for_voxel.items():
        item = voxel_points[key]
        points_by_object[object_id].append(item["xyz_sum"] / max(float(item["count"]), 1.0))
        point_counts_by_object[object_id] += int(item["count"])
    by_id = {int(obj["object_id"]): obj for obj in objects}
    for object_id, points in points_by_object.items():
        arr = np.vstack(points).astype(np.float64)
        obj = by_id[object_id]
        obj["point_count"] = int(point_counts_by_object[object_id])
        obj["centroid"] = arr.mean(axis=0).astype(float).tolist()
        obj["bbox_3d"] = {"min": arr.min(axis=0).astype(float).tolist(), "max": arr.max(axis=0).astype(float).tolist()}
        extent = arr.max(axis=0) - arr.min(axis=0)
        obj["extent"] = extent.astype(float).tolist()
        if len(arr) >= 3:
            centered = arr - arr.mean(axis=0, keepdims=True)
            cov = (centered.T @ centered) / max(len(arr) - 1, 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            obj["pca_eigenvalues"] = eigvals.astype(float).tolist()
            obj["pca_normal"] = eigvecs[:, -1].astype(float).tolist()
        else:
            obj["pca_eigenvalues"] = [0.0, 0.0, 0.0]
            obj["pca_normal"] = [0.0, 0.0, 1.0]


def write_outputs(
    output_dir: Path,
    voxel_size: float,
    voxel_points: dict[tuple[int, int, int], dict[str, Any]],
    voxel_labels: dict[tuple[int, int, int], str],
    object_for_voxel: dict[tuple[int, int, int], int],
    objects: list[dict[str, Any]],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = output_dir / "spatial_partition_objects.ply"
    jsonl_path = output_dir / "spatial_partition_objects.jsonl"
    with ply_path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(object_for_voxel)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uint object\nproperty uchar semantic\n")
        f.write("end_header\n")
        for key, object_id in sorted(object_for_voxel.items(), key=lambda item: item[1]):
            item = voxel_points[key]
            xyz = item["xyz_sum"] / max(float(item["count"]), 1.0)
            rgb_mean = item["rgb_sum"] / max(float(item["count"]), 1.0)
            label = voxel_labels[key]
            color = COLORS.get(label, tuple(int(x) for x in np.clip(rgb_mean, 0, 255)))
            semantic = LABEL_TO_SEMANTIC.get(label, 0)
            f.write(
                f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} {object_id} {semantic}\n"
            )
    with jsonl_path.open("w", encoding="utf-8") as f:
        for obj in objects:
            row = dict(obj)
            row["description"] = f"spatially connected {row['semantic_label']} component"
            row["voxel_size"] = voxel_size
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return ply_path, jsonl_path


def parse_min_voxels(raw: str) -> dict[str, int]:
    out = {"*": 1}
    for item in raw.split(","):
        if not item.strip():
            continue
        key, value = item.split(":", 1)
        out[key.strip()] = int(value)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ply", type=Path, required=True, help="Geometry/RGB source PLY.")
    parser.add_argument("--teacher", action="append", required=True, help="name:/path/to/semantic.ply:weight")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.10)
    parser.add_argument("--default-label", default="unknown")
    parser.add_argument("--accepted-labels", default="", help="Comma-separated labels to accept from teachers.")
    parser.add_argument("--min-voxels-by-label", default="*:4,floor:20,wall:20,grass:10,car:8,railing:4,unknown:20")
    parser.add_argument(
        "--small-component-policy",
        choices=("keep", "drop"),
        default="keep",
        help="keep preserves a true voxel partition; drop reproduces filtering-style previews.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reject_forbidden_production_input(args.base_ply)
    for raw in args.teacher:
        _name, path, _weight = parse_teacher(raw)
        reject_forbidden_production_input(path)
    reject_forbidden_production_input(args.output_dir)

    accepted = {x.strip() for x in args.accepted_labels.split(",") if x.strip()} or None
    min_voxels = parse_min_voxels(args.min_voxels_by_label)
    voxel_points, base_report = build_voxel_points_from_base(args.base_ply, args.voxel_size)
    votes: dict[tuple[int, int, int], Counter[str]] = defaultdict(Counter)
    teacher_reports = []
    for raw in args.teacher:
        name, path, weight = parse_teacher(raw)
        report = add_teacher_votes(votes, path, weight, args.voxel_size, accepted)
        report["name"] = name
        teacher_reports.append(report)
    voxel_labels = select_voxel_labels(voxel_points, votes, args.default_label)
    object_for_voxel, objects, component_report = connected_components_by_label(
        voxel_labels, min_voxels, args.small_component_policy
    )
    enrich_objects_geometry(objects, object_for_voxel, voxel_points)
    ply_path, jsonl_path = write_outputs(
        args.output_dir, args.voxel_size, voxel_points, voxel_labels, object_for_voxel, objects
    )
    label_counts = Counter(voxel_labels[key] for key in object_for_voxel)
    report = {
        "schema": "spatial-partition-mainline/v1",
        "base": base_report,
        "teachers": teacher_reports,
        "output_ply": str(ply_path),
        "output_jsonl": str(jsonl_path),
        "voxel_size": args.voxel_size,
        "assigned_voxels": len(object_for_voxel),
        "unassigned_voxels": len(voxel_points) - len(object_for_voxel),
        "object_count": len(objects),
        "small_component_policy": args.small_component_policy,
        "label_voxel_counts": dict(label_counts),
        **component_report,
    }
    report_path = args.output_dir / "spatial_partition_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
