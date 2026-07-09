#!/usr/bin/env python3
"""Materialize a labeled superpoint preview PLY with an object field."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from plyfile import PlyData


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()

    labels = np.load(args.labels)
    ply = PlyData.read(str(args.input), mmap=True)
    vertex = ply["vertex"].data
    stride = max(1, args.stride)
    idx = np.arange(0, len(labels), stride, dtype=np.int64)

    rng = random.Random(0)
    counts = np.bincount(labels)
    colors = {
        int(label): (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        for label in np.flatnonzero(counts)
    }

    preview = args.output_dir / f"official_superpoints_random_color_stride{stride}.ply"
    with preview.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(idx)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uint object\nend_header\n")
        for i in idx:
            label = int(labels[i])
            red, green, blue = colors[label]
            row = vertex[i]
            f.write(
                f"{float(row['x']):.4f} {float(row['y']):.4f} {float(row['z']):.4f} "
                f"{red} {green} {blue} {label}\n"
            )

    with (args.output_dir / "official_superpoints_objects.jsonl").open("w", encoding="utf-8") as f:
        for object_id, count in enumerate(counts):
            if count:
                f.write(
                    json.dumps(
                        {"object_id": object_id, "label": "official_superpoint", "count": int(count)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    report = {
        "input": str(args.input),
        "points": int(len(labels)),
        "preview_stride": stride,
        "preview_points": int(len(idx)),
        "superpoints": int(len(counts)),
        "nonempty_superpoints": int((counts > 0).sum()),
        "median_points_per_superpoint": float(np.median(counts[counts > 0])),
        "largest_superpoints": sorted([int(x) for x in counts if x], reverse=True)[:20],
        "params": {
            "k_nn_adj": 10,
            "k_nn_geof": 45,
            "reg_strength": 0.1,
            "lambda_edge_weight": 1.0,
        },
        "outputs": {
            "labels": str(args.labels),
            "preview_ply": str(preview),
            "objects": str(args.output_dir / "official_superpoints_objects.jsonl"),
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
