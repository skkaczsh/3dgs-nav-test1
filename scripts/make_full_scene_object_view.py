#!/usr/bin/env python3
"""Build a unified ASCII PLY/JSONL view from priority points and residual objects.

The priority route intentionally removes large known classes before residual
object clustering. Reviewing only residual candidates can therefore hide useful
context such as cars, grass, railing, or absorbed floor/wall. This script
combines:

- priority points with pseudo object ids per priority class
- residual object points with object ids and semantic/status metadata

The output is intended for `tools/semantic_ply_viewer.html`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PRIORITY_TO_VIEWER = {
    1: ("floor", 3, 900001, "priority_ground"),
    2: ("wall", 2, 900002, "priority_wall"),
    3: ("grass", 5, 900003, "priority_grass"),
    4: ("car", 8, 900004, "priority_car"),
    5: ("railing", 9, 900005, "priority_railing"),
}

RESIDUAL_LABEL_TO_SEMANTIC = {
    "wall_surface_prior": ("wall", 2),
    "ground_surface_prior": ("floor", 3),
    "unlabeled_residual": ("unknown", 0),
    "residual_surface_candidate": ("unknown", 0),
}


def read_ascii_ply(path: Path) -> tuple[np.ndarray, list[str], int]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        props: list[str] = []
        vertex_count = 0
        header_lines = 0
        in_vertex = False
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
    data = np.loadtxt(path, skiprows=header_lines, dtype=np.float64, max_rows=vertex_count)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data, props, header_lines


def load_objects(path: Path) -> dict[int, dict]:
    objects: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            objects[int(obj["object_id"])] = obj
    return objects


def mean_color_from_rows(rows: list[np.ndarray]) -> list[float]:
    if not rows:
        return [128.0, 128.0, 128.0]
    arr = np.vstack(rows)
    return [float(x) for x in arr[:, 3:6].mean(axis=0)]


def write_outputs(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    priority, priority_props, _ = read_ascii_ply(args.priority_ply)
    residual, residual_props, _ = read_ascii_ply(args.residual_ply)
    residual_objects = load_objects(args.objects_jsonl)

    pri_idx = {name: i for i, name in enumerate(priority_props)}
    res_idx = {name: i for i, name in enumerate(residual_props)}
    out_ply = args.output_dir / "full_scene_objects_ascii.ply"
    out_jsonl = args.output_dir / "full_scene_objects.jsonl"

    priority_counts: Counter[int] = Counter()
    priority_xyz_by_object: dict[int, list[np.ndarray]] = defaultdict(list)
    priority_rgb_by_object: dict[int, list[np.ndarray]] = defaultdict(list)
    residual_counts: Counter[int] = Counter()

    with out_ply.open("w", encoding="utf-8") as f:
        total = int(len(priority) + len(residual))
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {total}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uint object\nproperty uchar semantic\n")
        f.write("end_header\n")

        for row in priority:
            priority_id = int(round(row[pri_idx["priority"]]))
            if priority_id not in PRIORITY_TO_VIEWER:
                continue
            label, semantic, object_id, _status = PRIORITY_TO_VIEWER[priority_id]
            x, y, z = (float(row[pri_idx[k]]) for k in ("x", "y", "z"))
            r, g, b = (int(round(row[pri_idx[k]])) for k in ("red", "green", "blue"))
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {object_id} {semantic}\n")
            priority_counts[priority_id] += 1
            priority_xyz_by_object[object_id].append(np.array([x, y, z], dtype=np.float64))
            priority_rgb_by_object[object_id].append(np.array([r, g, b], dtype=np.float64))

        object_col = res_idx.get("object", res_idx.get("object_id"))
        for row in residual:
            object_id = int(round(row[object_col]))
            obj = residual_objects.get(object_id, {})
            label = obj.get("semantic_label", "unlabeled_residual")
            viewer_label, semantic = RESIDUAL_LABEL_TO_SEMANTIC.get(label, ("unknown", 0))
            x, y, z = (float(row[res_idx[k]]) for k in ("x", "y", "z"))
            r, g, b = (int(round(row[res_idx[k]])) for k in ("red", "green", "blue"))
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {object_id} {semantic}\n")
            residual_counts[object_id] += 1

    with out_jsonl.open("w", encoding="utf-8") as f:
        for priority_id, (label, _semantic, object_id, status) in PRIORITY_TO_VIEWER.items():
            pts = priority_xyz_by_object.get(object_id, [])
            if not pts:
                continue
            xyz = np.vstack(pts)
            rgb = np.vstack(priority_rgb_by_object[object_id])
            obj = {
                "object_id": object_id,
                "semantic_label": label,
                "description": f"priority layer {label}",
                "status": status,
                "point_count": int(priority_counts[priority_id]),
                "centroid": [float(x) for x in xyz.mean(axis=0)],
                "bbox_3d": {
                    "min": [float(x) for x in xyz.min(axis=0)],
                    "max": [float(x) for x in xyz.max(axis=0)],
                },
                "mean_color": [float(x) for x in rgb.mean(axis=0)],
                "source": "priority_projection",
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        for object_id in sorted(residual_objects):
            obj = dict(residual_objects[object_id])
            label = obj.get("semantic_label", "unlabeled_residual")
            viewer_label, _semantic = RESIDUAL_LABEL_TO_SEMANTIC.get(label, ("unknown", 0))
            obj["semantic_label_raw"] = label
            obj["semantic_label"] = viewer_label
            obj["point_count"] = int(residual_counts.get(object_id, obj.get("point_count", 0)))
            obj["source"] = "residual_objects_drivability_prior"
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    report = {
        "output_ply": str(out_ply),
        "output_jsonl": str(out_jsonl),
        "priority_points": int(sum(priority_counts.values())),
        "residual_object_points": int(sum(residual_counts.values())),
        "total_points": int(sum(priority_counts.values()) + sum(residual_counts.values())),
        "priority_counts": {
            PRIORITY_TO_VIEWER[k][0]: int(v) for k, v in sorted(priority_counts.items())
        },
        "residual_object_count": int(len(residual_objects)),
    }
    (args.output_dir / "full_scene_objects_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--priority-ply", type=Path, required=True)
    parser.add_argument("--residual-ply", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    write_outputs(parser.parse_args())


if __name__ == "__main__":
    main()
