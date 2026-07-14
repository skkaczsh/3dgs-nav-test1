#!/usr/bin/env python3
"""Attach non-semantic drivability region votes to official Superpoints."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData

try:
    from .apply_drivability_prior_to_residual import pack_query
    from .build_structural_region_field import REGION_NAMES
    from .classify_surface_attachment import load_structural_field
except ImportError:  # Direct script execution keeps scripts/ on sys.path.
    from apply_drivability_prior_to_residual import pack_query
    from build_structural_region_field import REGION_NAMES
    from classify_surface_attachment import load_structural_field


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def exact_region_labels(xyz: np.ndarray, field: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    coords = np.floor(xyz / float(field["voxel_size"])).astype(np.int32)
    keys = pack_query(coords, field["spec"])
    positions = np.searchsorted(field["keys"], keys)
    valid = positions < len(field["keys"])
    hit = np.zeros(len(keys), dtype=bool)
    hit[valid] = field["keys"][positions[valid]] == keys[valid]
    labels = np.zeros(len(keys), dtype=np.uint8)
    labels[hit] = field["labels"][positions[hit]]
    return labels, hit


def aggregate_region_votes(object_ids: np.ndarray, region_labels: np.ndarray) -> np.ndarray:
    if len(object_ids) != len(region_labels):
        raise ValueError("object_ids and region_labels must share point order")
    object_count = int(object_ids.max()) + 1 if len(object_ids) else 0
    width = len(REGION_NAMES)
    flat = object_ids.astype(np.int64, copy=False) * width + region_labels.astype(np.int64, copy=False)
    return np.bincount(flat, minlength=object_count * width).reshape(object_count, width)


def enrich_rows(rows: list[dict[str, Any]], votes: np.ndarray, voxel_size: float) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        object_id = int(row["object_id"])
        counts = votes[object_id] if object_id < len(votes) else np.zeros(len(REGION_NAMES), dtype=np.int64)
        total = int(counts.sum())
        non_unknown = counts[1:]
        dominant_index = int(np.argmax(non_unknown)) + 1 if int(non_unknown.sum()) else 0
        enriched = dict(row)
        enriched.update({
            "structural_region_votes": {REGION_NAMES[index]: int(count) for index, count in enumerate(counts) if count},
            "structural_region_dominant": REGION_NAMES[dominant_index],
            "structural_region_dominant_ratio": float(counts[dominant_index] / max(total, 1)),
            "structural_field_voxel_size": float(voxel_size),
            "structural_region_policy": "non_semantic_evidence_only",
        })
        out.append(enriched)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-ply", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--structural-field", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    vertex = PlyData.read(str(args.reference_ply))["vertex"].data
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32, copy=False)
    object_ids = np.load(args.labels).astype(np.int64, copy=False)
    if len(xyz) != len(object_ids):
        raise SystemExit(f"reference PLY / labels count mismatch: {len(xyz)} != {len(object_ids)}")
    field = load_structural_field(args.structural_field)
    regions, hit = exact_region_labels(xyz, field)
    votes = aggregate_region_votes(object_ids, regions)
    rows = enrich_rows(read_jsonl(args.objects_jsonl), votes, float(field["voxel_size"]))

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    dominant_counts = Counter(str(row["structural_region_dominant"]) for row in rows)
    report = {
        "reference_points": int(len(xyz)),
        "official_superpoints": len(rows),
        "field_hit_points": int(hit.sum()),
        "field_hit_ratio": float(hit.mean()),
        "field_voxel_size": float(field["voxel_size"]),
        "dominant_region_counts": dict(dominant_counts),
        "policy": "non_semantic_evidence_only",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
