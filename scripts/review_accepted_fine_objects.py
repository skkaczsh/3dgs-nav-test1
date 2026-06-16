#!/usr/bin/env python3
"""Apply geometry/surface-aware guard to projected fine-object candidates."""

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


def focus_of(row: dict) -> str:
    focus = str(row.get("focus", "")).strip().lower()
    if focus:
        return focus
    phrase = str(row.get("phrase", "")).lower()
    if any(token in phrase for token in ("railing", "guardrail", "handrail", "fence")):
        return "railing"
    if "pipe" in phrase or "cable" in phrase:
        return "pipe"
    if any(token in phrase for token in ("hvac", "air conditioning", "outdoor unit", "equipment")):
        return "equipment"
    return "generic"


def classify_equipment(row: dict, args: argparse.Namespace) -> tuple[str, int, list[str]]:
    sx, sy, sz = bbox_span(row)
    max_xy = max(sx, sy)
    max_span = max(sx, sy, sz)
    min_xy = min(sx, sy)
    aspect_xy = max_xy / max(min_xy, 1e-6)
    linearity = float(row.get("linearity", 0.0))
    planarity = float(row.get("planarity", 0.0))
    points = int(row.get("points", 0))
    reasons: list[str] = []

    if max_span >= args.equipment_demote_large_span:
        reasons.append("equipment_large_span")
    if max_xy >= args.equipment_demote_long_xy and linearity >= args.equipment_linear_demote:
        reasons.append("equipment_long_linear")
    if max_xy >= args.equipment_demote_planar_xy and planarity >= args.equipment_planar_demote:
        reasons.append("equipment_large_planar")
    if aspect_xy >= args.equipment_demote_aspect_xy and linearity >= args.equipment_linear_demote:
        reasons.append("equipment_aspect_linear")
    if points >= args.equipment_review_large_points:
        reasons.append("equipment_many_points")
    if max_xy >= args.equipment_review_xy:
        reasons.append("equipment_large_xy")
    if sz >= args.equipment_review_z:
        reasons.append("equipment_tall")

    if any(tag in reasons for tag in ("equipment_large_span", "equipment_long_linear", "equipment_large_planar", "equipment_aspect_linear")):
        return "demote_surface_like_equipment", STATUS_DEMOTE, reasons
    if any(tag in reasons for tag in ("equipment_many_points", "equipment_large_xy", "equipment_tall")):
        return "review_large_equipment", STATUS_REVIEW, reasons
    return "keep_compact_equipment", STATUS_KEEP, reasons or ["compact_equipment"]


def classify_railing_or_pipe(row: dict, args: argparse.Namespace, *, focus: str) -> tuple[str, int, list[str]]:
    sx, sy, sz = bbox_span(row)
    sorted_span = sorted([sx, sy, sz], reverse=True)
    max_span = sorted_span[0]
    mid_span = sorted_span[1]
    min_span = sorted_span[2]
    linearity = float(row.get("linearity", 0.0))
    planarity = float(row.get("planarity", 0.0))
    reasons: list[str] = []

    if max_span >= args.linear_object_min_extent and linearity >= args.linear_object_min_linearity:
        reasons.append("linear_object")
    if mid_span >= args.linear_object_max_mid_extent and planarity >= args.linear_object_surface_planarity:
        reasons.append("broad_planar_blob")
    if min_span >= args.linear_object_max_thickness and planarity >= args.linear_object_surface_planarity:
        reasons.append("thick_surface_blob")

    if "broad_planar_blob" in reasons or "thick_surface_blob" in reasons:
        return f"demote_surface_like_{focus}", STATUS_DEMOTE, reasons
    if "linear_object" in reasons:
        return f"keep_linear_{focus}", STATUS_KEEP, reasons
    return f"review_ambiguous_{focus}", STATUS_REVIEW, reasons or ["weak_linearity"]


def classify_generic(row: dict, args: argparse.Namespace) -> tuple[str, int, list[str]]:
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
    return "keep_generic", STATUS_KEEP, reasons or ["compact"]


def classify(row: dict, args: argparse.Namespace) -> tuple[str, int, list[str], str]:
    focus = focus_of(row)
    if focus == "equipment":
        action, status, reasons = classify_equipment(row, args)
    elif focus in {"railing", "pipe"}:
        action, status, reasons = classify_railing_or_pipe(row, args, focus=focus)
    else:
        action, status, reasons = classify_generic(row, args)
    return action, status, reasons, focus


def build_guarded_report(
    accepted: dict,
    rows: list[dict],
    kept_candidates: list[dict],
    kept_points: int,
    output_filtered_ply: Path,
    input_accepted_report: Path,
) -> dict:
    guarded = dict(accepted)
    guarded["input_accepted_report"] = str(input_accepted_report)
    guarded["output_ply"] = str(output_filtered_ply)
    guarded["candidate_count_before"] = int(len(accepted.get("top_candidates", [])))
    guarded["accepted_points_before"] = int(accepted.get("accepted_points", 0))
    guarded["candidate_count"] = int(len(kept_candidates))
    guarded["accepted_points"] = int(kept_points)
    guarded["candidate_counts"] = {"guarded_keep": int(len(kept_candidates))}
    guarded["point_counts"] = {"guarded_keep": int(kept_points)}
    guarded["top_candidates"] = sorted(kept_candidates, key=lambda x: int(x.get("points", 0)), reverse=True)
    guarded["guard_summary"] = {
        "keep": int(sum(1 for row in rows if int(row["strict_status"]) == STATUS_KEEP)),
        "review": int(sum(1 for row in rows if int(row["strict_status"]) == STATUS_REVIEW)),
        "demote": int(sum(1 for row in rows if int(row["strict_status"]) == STATUS_DEMOTE)),
    }
    return guarded


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
    parser.add_argument("--output-accepted-report", type=Path, required=True)
    parser.add_argument("--long-xy-span", type=float, default=4.0)
    parser.add_argument("--large-xy-span", type=float, default=4.0)
    parser.add_argument("--large-span", type=float, default=6.0)
    parser.add_argument("--tall-z-span", type=float, default=2.5)
    parser.add_argument("--large-points", type=int, default=5000)
    parser.add_argument("--linear-threshold", type=float, default=0.90)
    parser.add_argument("--planar-threshold", type=float, default=0.55)
    parser.add_argument("--equipment-demote-large-span", type=float, default=3.0)
    parser.add_argument("--equipment-demote-long-xy", type=float, default=1.8)
    parser.add_argument("--equipment-demote-planar-xy", type=float, default=1.6)
    parser.add_argument("--equipment-demote-aspect-xy", type=float, default=4.0)
    parser.add_argument("--equipment-linear-demote", type=float, default=0.82)
    parser.add_argument("--equipment-planar-demote", type=float, default=0.30)
    parser.add_argument("--equipment-review-large-points", type=int, default=350)
    parser.add_argument("--equipment-review-xy", type=float, default=1.15)
    parser.add_argument("--equipment-review-z", type=float, default=1.5)
    parser.add_argument("--linear-object-min-extent", type=float, default=0.45)
    parser.add_argument("--linear-object-min-linearity", type=float, default=0.78)
    parser.add_argument("--linear-object-max-mid-extent", type=float, default=0.45)
    parser.add_argument("--linear-object-max-thickness", type=float, default=0.35)
    parser.add_argument("--linear-object-surface-planarity", type=float, default=0.18)
    args = parser.parse_args()

    accepted = json.loads(args.accepted_report.read_text(encoding="utf-8"))
    rows = []
    action_counts = Counter()
    point_counts = Counter()
    candidate_status = {}
    kept_candidates: list[dict] = []
    for row in accepted.get("top_candidates", []):
        action, status, reasons, focus = classify(row, args)
        candidate_id = int(row["candidate_id"])
        candidate_status[candidate_id] = status
        if status == STATUS_KEEP:
            kept_candidates.append(dict(row))
        sx, sy, sz = bbox_span(row)
        out = {
            "candidate_id": candidate_id,
            "focus": focus,
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
    guarded_report = build_guarded_report(
        accepted,
        rows,
        kept_candidates,
        kept_points,
        args.output_filtered_ply,
        args.accepted_report,
    )

    summary = {
        "accepted_report": str(args.accepted_report),
        "accepted_ply": str(args.accepted_ply),
        "output_status_ply": str(args.output_status_ply),
        "output_filtered_ply": str(args.output_filtered_ply),
        "output_accepted_report": str(args.output_accepted_report),
        "params": {
            "long_xy_span": args.long_xy_span,
            "large_xy_span": args.large_xy_span,
            "large_span": args.large_span,
            "tall_z_span": args.tall_z_span,
            "large_points": args.large_points,
            "linear_threshold": args.linear_threshold,
            "planar_threshold": args.planar_threshold,
            "equipment_demote_large_span": args.equipment_demote_large_span,
            "equipment_demote_long_xy": args.equipment_demote_long_xy,
            "equipment_demote_planar_xy": args.equipment_demote_planar_xy,
            "equipment_demote_aspect_xy": args.equipment_demote_aspect_xy,
            "equipment_linear_demote": args.equipment_linear_demote,
            "equipment_planar_demote": args.equipment_planar_demote,
            "equipment_review_large_points": args.equipment_review_large_points,
            "equipment_review_xy": args.equipment_review_xy,
            "equipment_review_z": args.equipment_review_z,
            "linear_object_min_extent": args.linear_object_min_extent,
            "linear_object_min_linearity": args.linear_object_min_linearity,
            "linear_object_max_mid_extent": args.linear_object_max_mid_extent,
            "linear_object_max_thickness": args.linear_object_max_thickness,
            "linear_object_surface_planarity": args.linear_object_surface_planarity,
        },
        "candidate_count": len(rows),
        "input_points": int(len(data)),
        "kept_points": int(kept_points),
        "kept_candidate_count": int(len(kept_candidates)),
        "action_counts": dict(action_counts),
        "point_counts": dict(point_counts),
        "candidates": rows,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_accepted_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_accepted_report.write_text(json.dumps(guarded_report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_id",
        "focus",
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
    print(
        json.dumps(
            {
                k: summary[k]
                for k in [
                    "candidate_count",
                    "kept_candidate_count",
                    "input_points",
                    "kept_points",
                    "action_counts",
                    "point_counts",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
