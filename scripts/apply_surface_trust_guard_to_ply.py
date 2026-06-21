#!/usr/bin/env python3
"""Apply point-level trusted surface prior to a semantic object PLY.

The parking route treats the C++ drivability output as the strongest available
geometry prior for large ground/wall surfaces.  Object-level review can still
leave surface points inside fine-object candidates.  This pass fixes that at
point granularity:

- only labels in `--guard-labels` are eligible for overwrite
- prior ground/wall points are rewritten to floor/wall
- prior other/unknown points are left unchanged

The JSONL metadata is updated with per-object guard counts and only relabeled at
object level when the guarded point majority is strong.  The PLY remains the
authoritative point-level semantic artifact.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from apply_drivability_prior_to_residual import (
    GEOM_GROUND,
    GEOM_NAMES,
    GEOM_WALL,
    build_prior_voxels,
    label_from_rgb,
    read_pcd_xyzrgb,
    vote_points,
)


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
SEMANTIC_TO_LABEL = {value: key for key, value in LABEL_TO_SEMANTIC.items()}

LABEL_COLORS = {
    "unknown": (90, 90, 90),
    "wall": (160, 170, 180),
    "floor": (190, 172, 135),
    "ceiling": (180, 180, 210),
    "grass": (70, 150, 80),
    "car": (235, 90, 80),
    "railing": (245, 200, 35),
    "fine_candidate": (230, 55, 220),
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


def parse_ascii_ply_header(path: Path) -> tuple[list[str], list[str], int]:
    header: list[str] = []
    props: list[str] = []
    vertex_count = 0
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header.append(line)
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == "format" and parts[1] != "ascii":
                raise ValueError(f"Only ascii PLY is supported: {path}")
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                vertex_count = int(parts[2])
                in_vertex = True
            elif len(parts) >= 2 and parts[0] == "element":
                in_vertex = False
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                props.append(parts[-1])
            elif line.strip() == "end_header":
                break
    return header, props, vertex_count


def load_ascii_ply(path: Path) -> tuple[list[str], list[str], list[list[str]], np.ndarray, np.ndarray, np.ndarray]:
    header, props, vertex_count = parse_ascii_ply_header(path)
    idx = {name: i for i, name in enumerate(props)}
    required = {"x", "y", "z", "semantic"}
    if not required.issubset(idx):
        raise ValueError(f"PLY missing required fields {sorted(required - set(idx))}: {path}")
    object_col = idx.get("object", idx.get("object_id"))
    if object_col is None:
        raise ValueError(f"PLY missing object/object_id field: {path}")

    rows: list[list[str]] = []
    xyz = np.empty((vertex_count, 3), dtype=np.float32)
    object_ids = np.empty(vertex_count, dtype=np.uint32)
    semantics = np.empty(vertex_count, dtype=np.uint16)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(len(header)):
            next(f)
        for i, line in enumerate(f):
            if i >= vertex_count:
                break
            parts = line.strip().split()
            rows.append(parts)
            xyz[i] = [float(parts[idx["x"]]), float(parts[idx["y"]]), float(parts[idx["z"]])]
            object_ids[i] = int(round(float(parts[object_col])))
            semantics[i] = int(round(float(parts[idx["semantic"]])))
    if len(rows) != vertex_count:
        xyz = xyz[: len(rows)]
        object_ids = object_ids[: len(rows)]
        semantics = semantics[: len(rows)]
    return header, props, rows, xyz, object_ids, semantics


def write_ascii_ply(path: Path, header: list[str], props: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz_idx = [props.index("x"), props.index("y"), props.index("z")]
    with path.open("w", encoding="utf-8") as f:
        for line in header:
            f.write(line)
        for parts in rows:
            for col in xyz_idx:
                parts[col] = f"{float(parts[col]):.6f}"
            f.write(" ".join(parts) + "\n")


def guarded_label(old_label: str, prior_label: int, guard_labels: set[str], trusted_prior_labels: set[str]) -> str:
    if old_label not in guard_labels:
        return old_label
    if prior_label == GEOM_GROUND and "ground" in trusted_prior_labels:
        return "floor"
    if prior_label == GEOM_WALL and "wall" in trusted_prior_labels:
        return "wall"
    return old_label


def update_rows(
    rows: list[list[str]],
    props: list[str],
    semantics: np.ndarray,
    object_ids: np.ndarray,
    prior_labels: np.ndarray,
    guard_labels: set[str],
    trusted_prior_labels: set[str],
    recolor: bool,
) -> tuple[Counter, Counter, dict[int, Counter], int]:
    idx = {name: i for i, name in enumerate(props)}
    semantic_col = idx["semantic"]
    red_col = idx.get("red")
    green_col = idx.get("green")
    blue_col = idx.get("blue")
    before = Counter()
    after = Counter()
    by_object: dict[int, Counter] = defaultdict(Counter)
    changed = 0
    for i, parts in enumerate(rows):
        old_label = SEMANTIC_TO_LABEL.get(int(semantics[i]), "unknown")
        before[old_label] += 1
        new_label = guarded_label(old_label, int(prior_labels[i]), guard_labels, trusted_prior_labels)
        after[new_label] += 1
        by_object[int(object_ids[i])][new_label] += 1
        if new_label != old_label:
            parts[semantic_col] = str(LABEL_TO_SEMANTIC[new_label])
            if recolor and red_col is not None and green_col is not None and blue_col is not None:
                r, g, b = LABEL_COLORS[new_label]
                parts[red_col] = str(r)
                parts[green_col] = str(g)
                parts[blue_col] = str(b)
            changed += 1
    return before, after, by_object, changed


def update_objects(
    objects: list[dict[str, Any]],
    object_point_counts: dict[int, Counter],
    relabel_majority: float,
) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for obj in objects:
        out = dict(obj)
        oid = int(out["object_id"])
        counts = object_point_counts.get(oid, Counter())
        total = int(sum(counts.values()))
        if total:
            label, count = counts.most_common(1)[0]
            ratio = float(count) / total
            out["surface_trust_guard_point_labels"] = dict(counts)
            out["surface_trust_guard_majority_label"] = label
            out["surface_trust_guard_majority_ratio"] = ratio
            old_label = str(out.get("semantic_label") or "unknown")
            if label in {"floor", "wall"} and old_label in {"car", "railing", "fine_candidate", "unknown"} and ratio >= relabel_majority:
                out["semantic_label_original"] = out.get("semantic_label_original") or old_label
                out["semantic_label"] = label
                out["surface_trust_guard_status"] = f"{old_label}_to_{label}_by_point_surface_prior"
                out["downstream_stage"] = "stable_surface"
                out["stable_surface"] = True
            elif old_label in {"car", "railing", "fine_candidate"} and counts.get("floor", 0) + counts.get("wall", 0):
                out["surface_trust_guard_status"] = "mixed_fine_candidate_with_surface_points"
        out_rows.append(out)
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drivability-pcd", type=Path, required=True)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--input-objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="full_scene_surface_trust_guard")
    parser.add_argument("--prior-voxel-size", type=float, default=0.10)
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument("--guard-labels", default="car,railing,fine_candidate,unknown")
    parser.add_argument(
        "--trusted-prior-labels",
        default="ground,wall",
        help="Comma-separated drivability prior labels allowed to overwrite guarded point labels.",
    )
    parser.add_argument("--object-relabel-majority", type=float, default=0.80)
    parser.add_argument("--no-recolor", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    guard_labels = {item.strip() for item in args.guard_labels.split(",") if item.strip()}
    trusted_prior_labels = {item.strip() for item in args.trusted_prior_labels.split(",") if item.strip()}

    prior_xyz, prior_rgb = read_pcd_xyzrgb(args.drivability_pcd)
    prior_labels = label_from_rgb(prior_rgb)
    prior_keys, prior_voxel_labels, spec = build_prior_voxels(prior_xyz, prior_labels, args.prior_voxel_size)

    header, props, rows, xyz, object_ids, semantics = load_ascii_ply(args.input_ply)
    point_prior = vote_points(
        xyz,
        prior_keys,
        prior_voxel_labels,
        spec,
        args.prior_voxel_size,
        args.neighbor_radius,
    )
    before, after, by_object, changed_points = update_rows(
        rows,
        props,
        semantics,
        object_ids,
        point_prior,
        guard_labels,
        trusted_prior_labels,
        not args.no_recolor,
    )

    objects = read_jsonl(args.input_objects_jsonl)
    out_objects = update_objects(objects, by_object, args.object_relabel_majority)

    out_ply = args.output_dir / f"{args.output_prefix}.ply"
    out_jsonl = args.output_dir / f"{args.output_prefix}.jsonl"
    out_report = args.output_dir / f"{args.output_prefix}_report.json"
    write_ascii_ply(out_ply, header, props, rows)
    write_jsonl(out_jsonl, out_objects)

    point_prior_counts = Counter(int(x) for x in point_prior.tolist())
    object_status_counts = Counter(str(row.get("surface_trust_guard_status") or "") for row in out_objects)
    object_status_counts.pop("", None)
    report = {
        "drivability_pcd": str(args.drivability_pcd),
        "input_ply": str(args.input_ply),
        "input_objects_jsonl": str(args.input_objects_jsonl),
        "output_ply": str(out_ply),
        "output_jsonl": str(out_jsonl),
        "guard_labels": sorted(guard_labels),
        "trusted_prior_labels": sorted(trusted_prior_labels),
        "prior_voxel_size": args.prior_voxel_size,
        "neighbor_radius": args.neighbor_radius,
        "prior_point_count": int(len(prior_xyz)),
        "prior_voxel_count": int(len(prior_keys)),
        "scene_point_count": int(len(xyz)),
        "changed_points": int(changed_points),
        "semantic_counts_before": dict(before),
        "semantic_counts_after": dict(after),
        "point_prior_counts": {GEOM_NAMES[k]: int(v) for k, v in sorted(point_prior_counts.items())},
        "object_status_counts": dict(object_status_counts),
        "object_label_counts_after": dict(Counter(str(row.get("semantic_label") or "unknown") for row in out_objects)),
    }
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
