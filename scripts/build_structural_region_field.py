#!/usr/bin/env python3
"""Build a non-semantic structural region field from drivability_cpp output.

The drivability PCD is a geometry prior, not a semantic ground truth.  This
script deliberately names its output as structural regions:

- red   -> ground_like_region
- white -> vertical_surface_region
- green -> upper_horizontal_region
- blue/other -> other_structure_region

Downstream target/object code can use these regions as compatibility evidence,
but must not treat them as final labels such as floor, wall, or ceiling.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_drivability_prior_to_residual import pack_keys, read_pcd_xyzrgb


REGION_UNKNOWN = 0
REGION_GROUND_LIKE = 1
REGION_VERTICAL_SURFACE = 2
REGION_UPPER_HORIZONTAL = 3
REGION_OTHER_STRUCTURE = 4

REGION_NAMES = {
    REGION_UNKNOWN: "unknown",
    REGION_GROUND_LIKE: "ground_like_region",
    REGION_VERTICAL_SURFACE: "vertical_surface_region",
    REGION_UPPER_HORIZONTAL: "upper_horizontal_region",
    REGION_OTHER_STRUCTURE: "other_structure_region",
}


def structural_labels_from_rgb(rgb: np.ndarray) -> np.ndarray:
    labels = np.full(len(rgb), REGION_OTHER_STRUCTURE, dtype=np.uint8)
    red = (rgb[:, 0] > 200) & (rgb[:, 1] < 90) & (rgb[:, 2] < 90)
    white = (rgb[:, 0] > 200) & (rgb[:, 1] > 200) & (rgb[:, 2] > 200)
    green = (rgb[:, 1] > 180) & (rgb[:, 0] < 120) & (rgb[:, 2] < 120)
    labels[red] = REGION_GROUND_LIKE
    labels[white] = REGION_VERTICAL_SURFACE
    labels[green] = REGION_UPPER_HORIZONTAL
    return labels


def build_region_voxels(
    xyz: np.ndarray,
    labels: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords = np.floor(xyz / float(voxel_size)).astype(np.int32)
    keys, spec = pack_keys(coords)
    order = np.argsort(keys)
    sorted_keys = keys[order]
    sorted_labels = labels[order]
    unique_keys, start, counts = np.unique(sorted_keys, return_index=True, return_counts=True)
    out_labels = np.empty(len(unique_keys), dtype=np.uint8)
    confidence = np.empty(len(unique_keys), dtype=np.float32)
    histograms = np.zeros((len(unique_keys), len(REGION_NAMES)), dtype=np.uint32)
    for i, (s, c) in enumerate(zip(start, counts)):
        hist = np.bincount(sorted_labels[s:s + c], minlength=len(REGION_NAMES))
        # Prefer explicit structural surfaces over generic "other" when close.
        surface = hist[[REGION_GROUND_LIKE, REGION_VERTICAL_SURFACE, REGION_UPPER_HORIZONTAL]]
        surface_best = int(np.argmax(surface)) + REGION_GROUND_LIKE
        if hist[surface_best] >= max(1, hist[REGION_OTHER_STRUCTURE] * 0.75):
            label = surface_best
        else:
            label = REGION_OTHER_STRUCTURE
        out_labels[i] = label
        confidence[i] = float(hist[label] / max(int(c), 1))
        histograms[i, : len(hist)] = hist[: len(REGION_NAMES)]
    return unique_keys.astype(np.int64), out_labels, confidence, histograms, spec


def write_field_npz(
    path: Path,
    keys: np.ndarray,
    labels: np.ndarray,
    confidence: np.ndarray,
    histograms: np.ndarray,
    spec: np.ndarray,
    voxel_size: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        keys=keys,
        labels=labels,
        confidence=confidence,
        histograms=histograms,
        spec=spec,
        voxel_size=np.asarray([float(voxel_size)], dtype=np.float32),
        region_names=np.asarray([REGION_NAMES[i] for i in range(len(REGION_NAMES))]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drivability-pcd", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.10)
    args = parser.parse_args()

    xyz, rgb = read_pcd_xyzrgb(args.drivability_pcd)
    point_labels = structural_labels_from_rgb(rgb)
    keys, labels, confidence, histograms, spec = build_region_voxels(xyz, point_labels, args.voxel_size)
    write_field_npz(args.output_npz, keys, labels, confidence, histograms, spec, args.voxel_size)

    point_counts = Counter(int(x) for x in point_labels.tolist())
    voxel_counts = Counter(int(x) for x in labels.tolist())
    report: dict[str, Any] = {
        "drivability_pcd": str(args.drivability_pcd),
        "output_npz": str(args.output_npz),
        "voxel_size": float(args.voxel_size),
        "point_count": int(len(xyz)),
        "voxel_count": int(len(keys)),
        "region_names": REGION_NAMES,
        "point_region_counts": {REGION_NAMES[k]: int(v) for k, v in sorted(point_counts.items())},
        "voxel_region_counts": {REGION_NAMES[k]: int(v) for k, v in sorted(voxel_counts.items())},
        "mean_voxel_confidence": float(confidence.mean()) if len(confidence) else 0.0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
