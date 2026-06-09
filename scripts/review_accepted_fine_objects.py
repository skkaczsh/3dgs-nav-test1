#!/usr/bin/env python3
"""Apply stricter geometry review to accepted fine-object candidates."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


STATUS_KEEP = 1
STATUS_REVIEW = 2
STATUS_DEMOTE = 3


def read_ascii_ply(path: Path) -> tuple[list[str], int, np.ndarray]:
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
    if vertex_count == 0:
        return props, header_lines, np.empty((0, len(props)), dtype=np.float32)
    data = np.loadtxt(path, skiprows=header_lines, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return props, header_lines, data


def bbox_span(row: dict) -> list[float]:
    bbox = row["bbox_3d"]
    return [float(bbox["max"][i] - bbox["min"][i]) for i in range(3)]


def classify(row: dict, args: argparse.Namespace) -> tuple[str, int, list[str]]:
    sx, sy, sz = bbox_span(row)
    max_xy = max(sx, sy)
    max_span = max(sx, sy, sz)
    linearity = float(row.get("linearity", 0.0))
    planarity = float(row.get("planarity", 0.0))
    points = int(row.get("points", 0))
    reasons: list[str] = []

    if max_xy >= args.long_xy_span and linearity >= args.linear_threshold:
        reasons.append("long_linear")
    if sz >= args.tall_z_span and linearity >= args.linear_threshold:
        reasons.append("tall_linear")
    if max_span >= args.large_span:
        reasons.append("large_span")
    if points >= args.large_points:
        reasons.append("large_points")
    if max_xy >= args.large_xy_span and planarity >= args.planar_threshold:
        reasons.append("large_planar")

    if "long_linear" in reasons or "tall_linear" in reasons:
        return "demote_line_like", STATUS_DEMOTE, reasons
    if "large_span" in reasons or "large_points" in reasons or "large_planar" in reasons:
        return "review_large_or_planar", STATUS_REVIEW, reasons
    return "keep_strict", STATUS_KEEP, reasons or ["compact"]


def status_color(status: int) -> tuple[int, int, int]:
    if status == STATUS_KEEP:
        return (80, 210, 120)
    if status == STATUS_REVIEW:
        return (255, 210, 40)
    if status == STATUS_DEMOTE:
        return (255, 100, 40)
    return (160, 160, 160)


def write_status_ply(path: Path, props: list[str], data: np.ndarray, candidate_status: dict[int, int]) -> None:
    idx = {name: i for i, name in enumerate(props)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(data)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uchar semantic\n")
        f.write("property int accepted_candidate\n")
        f.write("property uchar source_type\n")
        f.write("property int source_cluster\n")
        f.write("property int subcluster\n")
        f.write("property uchar visual_red\nproperty uchar visual_green\nproperty uchar visual_blue\n")
        f.write("property uchar strict_status\n")
        f.write("end_header\n")
        for row in data:
            candidate_id = int(row[idx["accepted_candidate"]])
            status = candidate_status.get(candidate_id, 0)
            color = status_color(status)
            f.write(
                f"{row[idx['x']]:.6f} {row[idx['y']]:.6f} {row[idx['z']]:.6f} "
                f"{color[0]} {color[1]} {color[2]} {int(row[idx['semantic']])} "
                f"{candidate_id} {int(row[idx['source_type']])} {int(row[idx['source_cluster']])} "
                f"{int(row[idx['subcluster']])} {int(row[idx['visual_red']])} "
                f"{int(row[idx['visual_green']])} {int(row[idx['visual_blue']])} {status}\n"
            )


def write_filtered_ply(path: Path, props: list[str], data: np.ndarray, candidate_status: dict[int, int]) -> int:
    idx = {name: i for i, name in enumerate(props)}
    keep = np.array(
        [candidate_status.get(int(row[idx["accepted_candidate"]]), 0) == STATUS_KEEP for row in data],
        dtype=bool,
    )
    kept = data[keep]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(kept)}\n")
        for name in props:
            if name in {"x", "y", "z"}:
                f.write(f"property float {name}\n")
            elif name in {"accepted_candidate", "source_cluster", "subcluster"}:
                f.write(f"property int {name}\n")
            else:
                f.write(f"property uchar {name}\n")
        f.write("end_header\n")
        for row in kept:
            f.write(" ".join(str(int(x)) if i >= 3 else f"{x:.6f}" for i, x in enumerate(row)) + "\n")
    return int(len(kept))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-report", type=Path, required=True)
    parser.add_argument("--accepted-ply", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-status-ply", type=Path, required=True)
    parser.add_argument("--output-filtered-ply", type=Path, required=True)
    parser.add_argument("--long-xy-span", type=float, default=4.0)
    parser.add_argument("--large-xy-span", type=float, default=4.0)
    parser.add_argument("--large-span", type=float, default=6.0)
    parser.add_argument("--tall-z-span", type=float, default=2.5)
    parser.add_argument("--large-points", type=int, default=5000)
    parser.add_argument("--linear-threshold", type=float, default=0.90)
    parser.add_argument("--planar-threshold", type=float, default=0.55)
    args = parser.parse_args()

    accepted = json.loads(args.accepted_report.read_text(encoding="utf-8"))
    rows = []
    action_counts = Counter()
    point_counts = Counter()
    candidate_status = {}
    for row in accepted.get("top_candidates", []):
        action, status, reasons = classify(row, args)
        candidate_id = int(row["candidate_id"])
        candidate_status[candidate_id] = status
        sx, sy, sz = bbox_span(row)
        out = {
            "candidate_id": candidate_id,
            "source_type": row["source_type"],
            "source_cluster": int(row["source_cluster"]),
            "subcluster": int(row["subcluster"]),
            "points": int(row["points"]),
            "span_x": sx,
            "span_y": sy,
            "span_z": sz,
            "linearity": float(row.get("linearity", 0.0)),
            "planarity": float(row.get("planarity", 0.0)),
            "strict_action": action,
            "strict_status": status,
            "reasons": reasons,
        }
        rows.append(out)
        action_counts[action] += 1
        point_counts[action] += int(row["points"])
    props, _, data = read_ascii_ply(args.accepted_ply)
    kept_points = write_filtered_ply(args.output_filtered_ply, props, data, candidate_status)
    write_status_ply(args.output_status_ply, props, data, candidate_status)

    summary = {
        "accepted_report": str(args.accepted_report),
        "accepted_ply": str(args.accepted_ply),
        "output_status_ply": str(args.output_status_ply),
        "output_filtered_ply": str(args.output_filtered_ply),
        "params": {
            "long_xy_span": args.long_xy_span,
            "large_xy_span": args.large_xy_span,
            "large_span": args.large_span,
            "tall_z_span": args.tall_z_span,
            "large_points": args.large_points,
            "linear_threshold": args.linear_threshold,
            "planar_threshold": args.planar_threshold,
        },
        "candidate_count": len(rows),
        "input_points": int(len(data)),
        "kept_points": int(kept_points),
        "action_counts": dict(action_counts),
        "point_counts": dict(point_counts),
        "candidates": rows,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_id",
        "source_type",
        "source_cluster",
        "subcluster",
        "points",
        "span_x",
        "span_y",
        "span_z",
        "linearity",
        "planarity",
        "strict_action",
        "strict_status",
        "reasons",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})
    print(json.dumps({k: summary[k] for k in ["candidate_count", "input_points", "kept_points", "action_counts", "point_counts"]}, indent=2))


if __name__ == "__main__":
    main()
