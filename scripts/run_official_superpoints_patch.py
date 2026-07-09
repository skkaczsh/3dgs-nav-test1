#!/usr/bin/env python3
"""Run the official Superpoint Graph partition on one PLY file."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def read_ply_xyz_rgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype("float32")
    if all(name in vertex.dtype.names for name in ("red", "green", "blue")):
        rgb = np.vstack([vertex["red"], vertex["green"], vertex["blue"]]).T.astype("uint8")
    else:
        rgb = np.zeros((len(xyz), 3), dtype="uint8")
    return np.ascontiguousarray(xyz), rgb


def write_random_color_ply(path: Path, xyz: np.ndarray, labels: np.ndarray) -> None:
    rng = random.Random(0)
    unique = np.unique(labels)
    colors = {int(label): [rng.randrange(256), rng.randrange(256), rng.randrange(256)] for label in unique}
    rgb = np.zeros((len(labels), 3), dtype="uint8")
    for label, color in colors.items():
        rgb[labels == label] = color

    vertex = np.empty(
        len(xyz),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=True).write(str(path))


def write_objects_jsonl(path: Path, xyz: np.ndarray, labels: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        for label in np.unique(labels):
            idx = np.flatnonzero(labels == label)
            pts = xyz[idx]
            row = {
                "object_id": int(label),
                "label": "official_superpoint",
                "count": int(len(idx)),
                "bbox_min": pts.min(axis=0).round(4).tolist(),
                "bbox_max": pts.max(axis=0).round(4).tolist(),
                "centroid": pts.mean(axis=0).round(4).tolist(),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--k-nn-adj", type=int, default=10)
    parser.add_argument("--k-nn-geof", type=int, default=45)
    parser.add_argument("--reg-strength", type=float, default=0.1)
    parser.add_argument("--lambda-edge-weight", type=float, default=1.0)
    parser.add_argument("--stride-preview", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from graphs import compute_graph_nn_2
    import libcp
    import libply_c

    xyz, _rgb = read_ply_xyz_rgb(args.input)
    graph_nn, target_fea = compute_graph_nn_2(xyz, args.k_nn_adj, args.k_nn_geof)
    geof = libply_c.compute_geof(xyz, target_fea, args.k_nn_geof).astype("float32")
    geof[:, 3] = 2.0 * geof[:, 3]
    edge_weight = np.array(
        1.0 / (args.lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])),
        dtype="float32",
    )
    components, in_component = libcp.cutpursuit(
        geof,
        graph_nn["source"],
        graph_nn["target"],
        edge_weight,
        args.reg_strength,
    )
    labels = np.asarray(in_component, dtype=np.uint32)

    labels_path = args.output_dir / "official_superpoints_labels.npy"
    np.save(labels_path, labels)

    full_ply = args.output_dir / "official_superpoints_random_color.ply"
    write_random_color_ply(full_ply, xyz, labels)

    stride = max(1, args.stride_preview)
    preview_ply = args.output_dir / f"official_superpoints_random_color_stride{stride}.ply"
    write_random_color_ply(preview_ply, xyz[::stride], labels[::stride])
    write_objects_jsonl(args.output_dir / "official_superpoints_objects.jsonl", xyz, labels)

    counts = np.bincount(labels)
    report = {
        "input": str(args.input),
        "points": int(len(xyz)),
        "superpoints": int(labels.max() + 1) if len(labels) else 0,
        "nonempty_superpoints": int((counts > 0).sum()),
        "median_points_per_superpoint": float(np.median(counts[counts > 0])) if np.any(counts > 0) else 0.0,
        "largest_superpoints": sorted([int(x) for x in counts], reverse=True)[:20],
        "params": {
            "k_nn_adj": args.k_nn_adj,
            "k_nn_geof": args.k_nn_geof,
            "reg_strength": args.reg_strength,
            "lambda_edge_weight": args.lambda_edge_weight,
        },
        "outputs": {
            "labels": str(labels_path),
            "full_ply": str(full_ply),
            "preview_ply": str(preview_ply),
        },
    }
    (args.output_dir / "official_superpoints_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
