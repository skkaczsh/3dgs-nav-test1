#!/usr/bin/env python3
"""Build a consolidated object QA PLY from fused targets and absorbed residuals.

The output is intended for CloudCompare inspection. It combines:
- all target points, recolored by their fused Object semantic label/status
- residual points assigned to stable surface Objects

It does not modify target/object JSONL. It produces a derived QA artifact and a
small report.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from analyze_residual_absorbability import LABEL_COLORS, SEMANTIC_IDS


STATUS_IDS = {
    "stable": 1,
    "ambiguous_object": 2,
    "single_target": 3,
    "absorbed_residual": 4,
    "unassigned_residual": 5,
}
SOURCE_IDS = {"target": 1, "residual": 2}


def semantic_color(label: str) -> tuple[int, int, int]:
    sem = SEMANTIC_IDS.get(label, 0)
    return LABEL_COLORS.get(sem, LABEL_COLORS[0])


def object_number(object_id: str) -> int:
    match = re.search(r"(\d+)$", object_id or "")
    return int(match.group(1)) if match else 0


def read_ply_header(path: Path) -> tuple[list[str], int, int]:
    props: list[str] = []
    vertex_count = 0
    header_lines = 0
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header_lines += 1
            s = line.strip()
            if s.startswith("element vertex"):
                vertex_count = int(s.split()[-1])
                in_vertex = True
            elif s.startswith("element "):
                in_vertex = False
            elif in_vertex and s.startswith("property"):
                props.append(s.split()[-1])
            elif s == "end_header":
                break
    return props, vertex_count, header_lines


def load_target_maps(targets_dir: Path, objects_jsonl: Path) -> tuple[dict[int, dict], dict[int, dict], Counter]:
    target_to_object: dict[str, dict] = {}
    object_by_number: dict[int, dict] = {}
    status_counts = Counter()
    with objects_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            num = object_number(obj.get("object_id", ""))
            label = obj.get("semantic_label") or obj.get("dominant_label") or "unknown"
            status = obj.get("status", "single_target")
            meta = {
                "object_number": num,
                "object_id": obj.get("object_id", ""),
                "semantic_label": label,
                "semantic_id": SEMANTIC_IDS.get(label, 0),
                "status": status,
            }
            object_by_number[num] = meta
            status_counts[status] += 1
            for target_id in obj.get("targets", []):
                target_to_object[str(target_id)] = meta

    target_index_map: dict[int, dict] = {}
    for path in sorted(targets_dir.glob("targets_frame_*.jsonl")):
        if path.name == "targets_all.jsonl":
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                target = json.loads(line)
                meta = target_to_object.get(str(target.get("target_id")))
                if meta is None:
                    continue
                target_index_map[int(target["target_index"])] = meta
    return target_index_map, object_by_number, status_counts


def count_target_vertices(targets_dir: Path) -> int:
    total = 0
    for path in sorted(targets_dir.glob("targets_frame_*.ply")):
        _, vertex_count, _ = read_ply_header(path)
        total += vertex_count
    return total


def count_residual_vertices(path: Path, include_unassigned: bool) -> int:
    props, vertex_count, header_lines = read_ply_header(path)
    if include_unassigned:
        return vertex_count
    idx = {name: i for i, name in enumerate(props)}
    status_idx = idx.get("assignment_status")
    if status_idx is None:
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(header_lines):
            next(f)
        for line in f:
            parts = line.split()
            if len(parts) > status_idx and int(float(parts[status_idx])) == 1:
                count += 1
    return count


def write_header(f, total_vertices: int) -> None:
    f.write("ply\nformat ascii 1.0\n")
    f.write(f"element vertex {total_vertices}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    f.write("property uchar semantic\n")
    f.write("property int object\n")
    f.write("property uchar object_status\n")
    f.write("property uchar source\n")
    f.write("property uchar original_semantic\n")
    f.write("property int frame\n")
    f.write("end_header\n")


def append_target_points(out, targets_dir: Path, target_index_map: dict[int, dict]) -> Counter:
    counts = Counter()
    for path in sorted(targets_dir.glob("targets_frame_*.ply")):
        props, _, header_lines = read_ply_header(path)
        idx = {name: i for i, name in enumerate(props)}
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for _ in range(header_lines):
                next(f)
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                target_index = int(float(parts[idx["target"]]))
                meta = target_index_map.get(target_index)
                if meta is None:
                    label = "unknown"
                    sem = 0
                    color = LABEL_COLORS[0]
                    obj_num = 0
                    status_id = 0
                else:
                    label = meta["semantic_label"]
                    sem = int(meta["semantic_id"])
                    color = semantic_color(label)
                    obj_num = int(meta["object_number"])
                    status_id = STATUS_IDS.get(meta["status"], 0)
                original_sem = int(float(parts[idx["semantic"]]))
                frame = int(float(parts[idx["frame"]])) if "frame" in idx else -1
                out.write(
                    f"{float(parts[idx['x']]):.6f} {float(parts[idx['y']]):.6f} {float(parts[idx['z']]):.6f} "
                    f"{color[0]} {color[1]} {color[2]} {sem} {obj_num} {status_id} "
                    f"{SOURCE_IDS['target']} {original_sem} {frame}\n"
                )
                counts[f"target_{label}"] += 1
                counts[f"status_{meta['status'] if meta else 'missing_object'}"] += 1
    return counts


def append_residual_points(
    out,
    residual_ply: Path,
    surface_objects: list[dict],
    include_unassigned: bool,
) -> Counter:
    counts = Counter()
    props, _, header_lines = read_ply_header(residual_ply)
    idx = {name: i for i, name in enumerate(props)}
    with residual_ply.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(header_lines):
            next(f)
        for line in f:
            parts = line.split()
            if not parts:
                continue
            assigned = int(float(parts[idx["assignment_status"]])) == 1
            if not assigned and not include_unassigned:
                continue
            assigned_sem = int(float(parts[idx["assigned_semantic"]]))
            original_sem = int(float(parts[idx["original_semantic"]]))
            object_idx = int(float(parts[idx["object_index"]]))
            obj_num = 0
            if assigned and 0 <= object_idx < len(surface_objects):
                obj_num = object_number(surface_objects[object_idx].get("object_id", ""))
            color = LABEL_COLORS.get(assigned_sem, LABEL_COLORS[0])
            status_id = STATUS_IDS["absorbed_residual"] if assigned else STATUS_IDS["unassigned_residual"]
            out.write(
                f"{float(parts[idx['x']]):.6f} {float(parts[idx['y']]):.6f} {float(parts[idx['z']]):.6f} "
                f"{color[0]} {color[1]} {color[2]} {assigned_sem} {obj_num} {status_id} "
                f"{SOURCE_IDS['residual']} {original_sem} -1\n"
            )
            counts["absorbed_residual" if assigned else "unassigned_residual"] += 1
            counts[f"residual_semantic_{assigned_sem}"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets-dir", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--residual-assignment-ply", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--include-unassigned-residual", action="store_true")
    parser.add_argument("--min-object-targets", type=int, default=5)
    parser.add_argument("--min-object-points", type=int, default=1000)
    args = parser.parse_args()

    from analyze_residual_absorbability import load_surface_objects

    target_index_map, _, object_status_counts = load_target_maps(args.targets_dir, args.objects_jsonl)
    surface_objects = load_surface_objects(args.objects_jsonl, args.min_object_targets, args.min_object_points)
    target_vertices = count_target_vertices(args.targets_dir)
    residual_vertices = count_residual_vertices(args.residual_assignment_ply, args.include_unassigned_residual)
    total_vertices = target_vertices + residual_vertices

    args.output_ply.parent.mkdir(parents=True, exist_ok=True)
    with args.output_ply.open("w", encoding="utf-8") as out:
        write_header(out, total_vertices)
        target_counts = append_target_points(out, args.targets_dir, target_index_map)
        residual_counts = append_residual_points(
            out,
            args.residual_assignment_ply,
            surface_objects,
            args.include_unassigned_residual,
        )

    report = {
        "targets_dir": str(args.targets_dir),
        "objects_jsonl": str(args.objects_jsonl),
        "residual_assignment_ply": str(args.residual_assignment_ply),
        "output_ply": str(args.output_ply),
        "target_index_mapped": len(target_index_map),
        "surface_objects": len(surface_objects),
        "target_vertices": int(target_vertices),
        "residual_vertices": int(residual_vertices),
        "total_vertices": int(total_vertices),
        "object_status_counts": dict(object_status_counts),
        "point_counts": dict(target_counts + residual_counts),
        "status_ids": STATUS_IDS,
        "source_ids": SOURCE_IDS,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["target_vertices", "residual_vertices", "total_vertices"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
