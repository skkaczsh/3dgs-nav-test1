#!/usr/bin/env python3
"""Build a review set for fine residual clusters.

The goal is to separate likely real fine objects from likely mixed/projection
contamination using simple geometric heuristics. It writes a JSON/CSV review
table and, optionally, a PLY containing the top suspicious clusters.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def bbox_span(row: dict) -> list[float]:
    b = row.get("bbox_3d", {})
    mn = b.get("min", [0, 0, 0])
    mx = b.get("max", [0, 0, 0])
    return [float(mx[i] - mn[i]) for i in range(3)]


def classify(row: dict, args: argparse.Namespace) -> tuple[str, float, list[str]]:
    points = int(row.get("points", 0))
    sx, sy, sz = bbox_span(row)
    max_xy = max(sx, sy)
    linearity = float(row.get("linearity", 0.0))
    planarity = float(row.get("planarity", 0.0))
    label = row.get("label", "unknown")
    reasons = []
    score = 0.0

    if points >= args.large_points:
        reasons.append("large_points")
        score += min(points / max(args.large_points, 1), 5.0)
    if max_xy >= args.large_xy_span:
        reasons.append("large_xy_span")
        score += min(max_xy / max(args.large_xy_span, 1e-6), 5.0)
    if sz >= args.large_z_span:
        reasons.append("large_z_span")
        score += min(sz / max(args.large_z_span, 1e-6), 3.0)
    if label == "equipment" and (linearity >= args.linear_threshold or planarity >= args.planar_threshold) and max_xy >= args.equipment_surface_xy_span:
        reasons.append("equipment_surface_like")
        score += 2.0
    if label == "railing" and linearity >= args.railing_linear_threshold:
        reasons.append("railing_linear")
        score -= 0.5
    if not reasons:
        reasons.append("compact_candidate")
        score -= 1.0

    if "equipment_surface_like" in reasons or "large_xy_span" in reasons or "large_points" in reasons:
        status = "review_suspicious"
    elif label == "railing" and ("railing_linear" in reasons or max_xy >= args.railing_expected_span):
        status = "likely_fine_object"
    else:
        status = "review_candidate"
    return status, float(score), reasons


def read_ply_header(path: Path) -> tuple[list[str], int, int]:
    props = []
    n = 0
    header_lines = 0
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header_lines += 1
            s = line.strip()
            if s.startswith("element vertex"):
                n = int(s.split()[-1])
                in_vertex = True
            elif s.startswith("element "):
                in_vertex = False
            elif in_vertex and s.startswith("property"):
                props.append(s.split()[-1])
            elif s == "end_header":
                break
    return props, n, header_lines


def write_filtered_ply(input_ply: Path, output_ply: Path, keep_clusters: set[int]) -> int:
    props, _, header_lines = read_ply_header(input_ply)
    idx = {name: i for i, name in enumerate(props)}
    cluster_idx = idx["cluster"]
    kept_lines = []
    with input_ply.open("r", encoding="utf-8", errors="replace") as f:
        header = [next(f) for _ in range(header_lines)]
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if int(float(parts[cluster_idx])) in keep_clusters:
                kept_lines.append(line)
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    with output_ply.open("w", encoding="utf-8") as f:
        for line in header:
            if line.startswith("element vertex"):
                f.write(f"element vertex {len(kept_lines)}\n")
            else:
                f.write(line)
        for line in kept_lines:
            f.write(line)
    return len(kept_lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-report", type=Path, required=True)
    parser.add_argument("--cluster-ply", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-suspicious-ply", type=Path, required=True)
    parser.add_argument("--top-suspicious", type=int, default=30)
    parser.add_argument("--large-points", type=int, default=8000)
    parser.add_argument("--large-xy-span", type=float, default=10.0)
    parser.add_argument("--large-z-span", type=float, default=3.0)
    parser.add_argument("--equipment-surface-xy-span", type=float, default=6.0)
    parser.add_argument("--linear-threshold", type=float, default=0.75)
    parser.add_argument("--planar-threshold", type=float, default=0.45)
    parser.add_argument("--railing-linear-threshold", type=float, default=0.75)
    parser.add_argument("--railing-expected-span", type=float, default=6.0)
    args = parser.parse_args()

    report = json.loads(args.cluster_report.read_text(encoding="utf-8"))
    rows = []
    for row in report.get("top_clusters", []):
        status, score, reasons = classify(row, args)
        sx, sy, sz = bbox_span(row)
        rows.append(
            {
                "cluster_id": int(row["cluster_id"]),
                "label": row.get("label", "unknown"),
                "points": int(row.get("points", 0)),
                "status": status,
                "suspicious_score": score,
                "reasons": reasons,
                "span_x": sx,
                "span_y": sy,
                "span_z": sz,
                "linearity": float(row.get("linearity", 0.0)),
                "planarity": float(row.get("planarity", 0.0)),
                "mean_visual_color": row.get("mean_visual_color", []),
                "centroid": row.get("centroid", []),
                "bbox_3d": row.get("bbox_3d", {}),
            }
        )
    rows.sort(key=lambda r: (r["status"] != "review_suspicious", -r["suspicious_score"], -r["points"]))
    suspicious = [r for r in rows if r["status"] == "review_suspicious"]
    keep = {int(r["cluster_id"]) for r in suspicious[: args.top_suspicious]}
    kept_points = write_filtered_ply(args.cluster_ply, args.output_suspicious_ply, keep)

    summary = {
        "cluster_report": str(args.cluster_report),
        "cluster_ply": str(args.cluster_ply),
        "output_suspicious_ply": str(args.output_suspicious_ply),
        "review_rows": rows,
        "counts": {
            "total_review_rows": len(rows),
            "review_suspicious": sum(1 for r in rows if r["status"] == "review_suspicious"),
            "review_candidate": sum(1 for r in rows if r["status"] == "review_candidate"),
            "likely_fine_object": sum(1 for r in rows if r["status"] == "likely_fine_object"),
            "suspicious_ply_clusters": len(keep),
            "suspicious_ply_points": int(kept_points),
        },
        "params": {
            "large_points": args.large_points,
            "large_xy_span": args.large_xy_span,
            "large_z_span": args.large_z_span,
            "equipment_surface_xy_span": args.equipment_surface_xy_span,
            "linear_threshold": args.linear_threshold,
            "planar_threshold": args.planar_threshold,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cluster_id",
                "label",
                "points",
                "status",
                "suspicious_score",
                "reasons",
                "span_x",
                "span_y",
                "span_z",
                "linearity",
                "planarity",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
