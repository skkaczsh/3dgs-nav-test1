#!/usr/bin/env python3
"""Export a region-aligned sample covering SPG touch-edge endpoint patches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.optimize_patch_graph_energy import build_edge_counts, read_labels, read_region_input, write_labels


def choose_indices(labels: np.ndarray, target_patches: set[int], max_per_patch: int) -> np.ndarray:
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_labels)) + 1]
    ends = np.r_[starts[1:], len(sorted_labels)]
    chosen: list[np.ndarray] = []
    for start, end in zip(starts, ends, strict=True):
        patch_id = int(sorted_labels[start])
        if patch_id not in target_patches:
            continue
        count = int(end - start)
        take = min(max_per_patch, count)
        offsets = np.linspace(0, count - 1, take, dtype=np.int64)
        chosen.append(order[start + offsets])
    if not chosen:
        return np.array([], dtype=np.int64)
    return np.sort(np.concatenate(chosen).astype(np.int64, copy=False))


def write_ascii_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8, copy=False)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(xyz)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb, strict=True):
            fh.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-input", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-patch", type=int, default=8)
    parser.add_argument("--stem", default="touch_edge_patch_sample")
    args = parser.parse_args()

    arrays, src, dst = read_region_input(args.region_input)
    labels = read_labels(args.labels)
    if len(labels) != len(arrays["xyz"]):
        raise ValueError(f"label count {len(labels)} != region point count {len(arrays['xyz'])}")
    edge_counts = build_edge_counts(labels, src, dst)
    target_patches = {int(v) for edge in edge_counts for v in edge}
    idx = choose_indices(labels, target_patches, args.max_per_patch)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ply = args.output_dir / f"{args.stem}.ply"
    label_path = args.output_dir / f"{args.stem}_labels.bin"
    index_path = args.output_dir / f"{args.stem}_indices.npy"
    write_ascii_ply(ply, arrays["xyz"][idx], arrays["rgb"][idx])
    write_labels(label_path, labels[idx].astype(np.int32, copy=False))
    np.save(index_path, idx)
    report = {
        "schema": "touch-edge-patch-sample/v1",
        "output_ply": str(ply),
        "output_labels": str(label_path),
        "output_indices": str(index_path),
        "touch_edge_count": int(len(edge_counts)),
        "target_patch_count": int(len(target_patches)),
        "sample_point_count": int(len(idx)),
        "max_per_patch": int(args.max_per_patch),
    }
    (args.output_dir / f"{args.stem}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
