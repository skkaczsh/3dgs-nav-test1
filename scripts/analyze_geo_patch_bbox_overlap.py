#!/usr/bin/env python3
"""Analyze AABB overlap among the largest geometry patches."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


SIZE_BINS = [0, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 10**18]


def size_bin_label(value: int) -> str:
    for lo, hi in zip(SIZE_BINS[:-1], SIZE_BINS[1:]):
        if lo <= value < hi:
            return f"[{lo},{hi})" if hi < 10**18 else f"[{lo},inf)"
    return "unknown"


def ratio_bin_label(value: float) -> str:
    if value <= 0:
        return "0"
    bins = [
        (0, 0.001, "(0,0.001)"),
        (0.001, 0.01, "[0.001,0.01)"),
        (0.01, 0.05, "[0.01,0.05)"),
        (0.05, 0.1, "[0.05,0.1)"),
        (0.1, 0.25, "[0.1,0.25)"),
        (0.25, 0.5, "[0.25,0.5)"),
        (0.5, 0.75, "[0.5,0.75)"),
        (0.75, 0.95, "[0.75,0.95)"),
        (0.95, 1.01, "[0.95,1.01)"),
    ]
    for lo, hi, label in bins:
        if lo < value < hi or (lo >= 0.001 and lo <= value < hi):
            return label
    return ">=1.01"


def load_top_patches(path: Path, top_n: int, bbox_pad: float) -> list[dict]:
    patches: list[dict] = []
    with path.open() as src:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            bbox = row.get("bbox_3d") or {}
            bmin = bbox.get("min")
            bmax = bbox.get("max")
            if not bmin or not bmax:
                continue

            padded_min = [float(v) - bbox_pad for v in bmin]
            padded_max = [float(v) + bbox_pad for v in bmax]
            extent = [max(0.0, padded_max[i] - padded_min[i]) for i in range(3)]
            volume = max(1e-9, extent[0] * extent[1] * extent[2])
            patches.append(
                {
                    "patch_id": row["patch_id"],
                    "voxel_count": int(row.get("voxel_count", 0)),
                    "geometry_type": row.get("geometry_type", ""),
                    "centroid": row.get("centroid", [None, None, None]),
                    "min": padded_min,
                    "max": padded_max,
                    "extent": extent,
                    "volume": volume,
                }
            )

    patches.sort(key=lambda item: item["voxel_count"], reverse=True)
    return patches[:top_n]


def bbox_overlap(a: dict, b: dict) -> tuple[float, float, float, float, list[float], float | None] | None:
    dims = [max(0.0, min(a["max"][i], b["max"][i]) - max(a["min"][i], b["min"][i])) for i in range(3)]
    overlap_volume = dims[0] * dims[1] * dims[2]
    if overlap_volume <= 0:
        return None

    min_volume = min(a["volume"], b["volume"])
    max_volume = max(a["volume"], b["volume"])
    union = a["volume"] + b["volume"] - overlap_volume
    centroid_distance = None
    if a["centroid"][0] is not None and b["centroid"][0] is not None:
        centroid_distance = math.sqrt(sum((a["centroid"][i] - b["centroid"][i]) ** 2 for i in range(3)))
    return (
        overlap_volume,
        overlap_volume / min_volume,
        overlap_volume / max_volume,
        overlap_volume / union if union > 0 else 0.0,
        dims,
        centroid_distance,
    )


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(input_jsonl: Path, output_dir: Path, top_n: int, bbox_pad: float) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    top = load_top_patches(input_jsonl, top_n=top_n, bbox_pad=bbox_pad)
    if len(top) < 2:
        raise ValueError(f"Need at least two patches, got {len(top)}")

    pair_rows: list[dict] = []
    ratio_hist: Counter[str] = Counter()
    geometry_hist: Counter[tuple[str, str]] = Counter()
    size_hist: Counter[tuple[str, str]] = Counter()
    patch_stats = {
        patch["patch_id"]: {
            "patch_id": patch["patch_id"],
            "voxel_count": patch["voxel_count"],
            "geometry_type": patch["geometry_type"],
            "overlap_pairs": 0,
            "high_pairs_50": 0,
            "high_pairs_95": 0,
            "max_ratio_min": 0.0,
            "sum_ratio_min": 0.0,
            "sum_overlap_volume": 0.0,
        }
        for patch in top
    }

    for i, a in enumerate(top):
        for b in top[i + 1 :]:
            overlap = bbox_overlap(a, b)
            if not overlap:
                continue
            overlap_volume, ratio_min, ratio_max, bbox_iou, dims, centroid_distance = overlap
            ratio_bin = ratio_bin_label(ratio_min)
            ratio_hist[ratio_bin] += 1
            geometry_key = " + ".join(sorted([a["geometry_type"], b["geometry_type"]]))
            size_key = " + ".join(sorted([size_bin_label(a["voxel_count"]), size_bin_label(b["voxel_count"])]))
            geometry_hist[(geometry_key, ratio_bin)] += 1
            size_hist[(size_key, ratio_bin)] += 1

            pair_rows.append(
                {
                    "patch_a": a["patch_id"],
                    "patch_b": b["patch_id"],
                    "voxels_a": a["voxel_count"],
                    "voxels_b": b["voxel_count"],
                    "geom_a": a["geometry_type"],
                    "geom_b": b["geometry_type"],
                    "overlap_volume": overlap_volume,
                    "ratio_min_volume": ratio_min,
                    "ratio_max_volume": ratio_max,
                    "bbox_iou": bbox_iou,
                    "centroid_distance": centroid_distance,
                    "overlap_dx": dims[0],
                    "overlap_dy": dims[1],
                    "overlap_dz": dims[2],
                    "volume_a": a["volume"],
                    "volume_b": b["volume"],
                }
            )

            for patch in (a, b):
                stats = patch_stats[patch["patch_id"]]
                stats["overlap_pairs"] += 1
                stats["max_ratio_min"] = max(stats["max_ratio_min"], ratio_min)
                stats["sum_ratio_min"] += ratio_min
                stats["sum_overlap_volume"] += overlap_volume
                if ratio_min >= 0.5:
                    stats["high_pairs_50"] += 1
                if ratio_min >= 0.95:
                    stats["high_pairs_95"] += 1

    pair_rows.sort(key=lambda row: (row["ratio_min_volume"], row["overlap_volume"]), reverse=True)
    patch_rows = list(patch_stats.values())
    for row in patch_rows:
        row["avg_ratio_min"] = row["sum_ratio_min"] / row["overlap_pairs"] if row["overlap_pairs"] else 0.0
    patch_rows.sort(key=lambda row: (row["high_pairs_95"], row["overlap_pairs"], row["voxel_count"]), reverse=True)

    patch_size_summary = []
    for lo, hi in zip(SIZE_BINS[:-1], SIZE_BINS[1:]):
        label = f"[{lo},{hi})" if hi < 10**18 else f"[{lo},inf)"
        rows = [row for row in patch_rows if size_bin_label(row["voxel_count"]) == label]
        if not rows:
            continue
        patch_size_summary.append(
            {
                "size_bin": label,
                "patch_count": len(rows),
                "patches_with_overlap": sum(1 for row in rows if row["overlap_pairs"] > 0),
                "avg_overlap_pairs": sum(row["overlap_pairs"] for row in rows) / len(rows),
                "avg_high_pairs_95": sum(row["high_pairs_95"] for row in rows) / len(rows),
                "avg_max_ratio_min": sum(row["max_ratio_min"] for row in rows) / len(rows),
                "max_overlap_pairs": max(row["overlap_pairs"] for row in rows),
            }
        )

    pair_csv = output_dir / f"bbox_overlap_top{top_n}_pairs.csv"
    patch_csv = output_dir / f"bbox_overlap_top{top_n}_patch_summary.csv"
    report_path = output_dir / f"bbox_overlap_top{top_n}_report.json"
    write_csv(
        pair_csv,
        pair_rows,
        [
            "patch_a",
            "patch_b",
            "voxels_a",
            "voxels_b",
            "geom_a",
            "geom_b",
            "overlap_volume",
            "ratio_min_volume",
            "ratio_max_volume",
            "bbox_iou",
            "centroid_distance",
            "overlap_dx",
            "overlap_dy",
            "overlap_dz",
            "volume_a",
            "volume_b",
        ],
    )
    write_csv(
        patch_csv,
        patch_rows,
        [
            "patch_id",
            "voxel_count",
            "geometry_type",
            "overlap_pairs",
            "high_pairs_50",
            "high_pairs_95",
            "max_ratio_min",
            "avg_ratio_min",
            "sum_ratio_min",
            "sum_overlap_volume",
        ],
    )

    report = {
        "input_jsonl": str(input_jsonl),
        "top_n": top_n,
        "bbox_pad": bbox_pad,
        "top_patch_count": len(top),
        "top_patch_voxel_min": top[-1]["voxel_count"],
        "top_patch_voxel_max": top[0]["voxel_count"],
        "overlap_pair_count": len(pair_rows),
        "possible_pair_count": top_n * (top_n - 1) // 2,
        "overlap_pair_ratio": len(pair_rows) / (top_n * (top_n - 1) // 2),
        "ratio_min_volume_hist": dict(ratio_hist),
        "patch_size_summary": patch_size_summary,
        "top_geometry_ratio_bins": [
            {"geometry_pair": key[0], "ratio_bin": key[1], "count": count}
            for key, count in geometry_hist.most_common(30)
        ],
        "top_size_ratio_bins": [
            {"size_pair": key[0], "ratio_bin": key[1], "count": count}
            for key, count in size_hist.most_common(30)
        ],
        "top_pairs": pair_rows[:100],
        "top_patch_overlap_summary": patch_rows[:100],
        "output_pairs_csv": str(pair_csv),
        "output_patch_csv": str(patch_csv),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=1000)
    parser.add_argument("--bbox-pad", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        top_n=args.top_n,
        bbox_pad=args.bbox_pad,
    )
    print(
        json.dumps(
            {
                "top_patch_count": report["top_patch_count"],
                "top_patch_voxel_min": report["top_patch_voxel_min"],
                "top_patch_voxel_max": report["top_patch_voxel_max"],
                "overlap_pair_count": report["overlap_pair_count"],
                "possible_pair_count": report["possible_pair_count"],
                "overlap_pair_ratio": report["overlap_pair_ratio"],
                "ratio_min_volume_hist": report["ratio_min_volume_hist"],
                "output_pairs_csv": report["output_pairs_csv"],
                "output_patch_csv": report["output_patch_csv"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
