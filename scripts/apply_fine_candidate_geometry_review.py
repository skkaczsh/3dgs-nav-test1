#!/usr/bin/env python3
"""Apply conservative geometry/evidence review to split fine candidates.

This pass converts local split candidates into a safer review dataset:

- promote railings only when local geometry is linear/thin and image evidence exists
- keep plausible cars when compact/irregular 3D shape and image evidence agree
- demote planar fragments and weak/no-evidence splits to review labels

It does not modify the source full-scene bundle.  It writes a split-candidate
PLY/JSONL that can be loaded in the viewer as an intermediate QA layer.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SEMANTIC_IDS = {
    "unknown": 0,
    "wall": 2,
    "floor": 3,
    "fine_candidate": 7,
    "car": 8,
    "railing": 9,
}

DISPLAY_COLORS = {
    "car": (235, 90, 80),
    "railing": (245, 200, 35),
    "fine_candidate": (230, 55, 220),
    "surface_fragment": (170, 170, 170),
    "weak_candidate": (120, 120, 120),
    "unknown": (90, 90, 90),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def parse_ply_header(path: Path) -> tuple[list[str], int, int]:
    props: list[str] = []
    vertex_count = 0
    header_lines = 0
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header_lines += 1
            s = line.strip()
            parts = s.split()
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                vertex_count = int(parts[2])
                in_vertex = True
            elif len(parts) >= 2 and parts[0] == "element":
                in_vertex = False
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                props.append(parts[-1])
            elif s == "end_header":
                break
    return props, vertex_count, header_lines


def evidence_by_object(evidence_rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence_rows:
        out[int(row["object_id"])].append(row)
    for rows in out.values():
        rows.sort(key=lambda r: int(r.get("rank", 999)))
    return out


def has_evidence(evidence: list[dict[str, Any]]) -> bool:
    return bool(evidence)


def best_evidence(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    return evidence[0] if evidence else {}


def review_object(obj: dict[str, Any], evidence: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(obj)
    candidate = str(obj.get("candidate_label") or obj.get("semantic_label") or "fine_candidate")
    geometry = str(obj.get("geometry_class") or "")
    point_count = int(obj.get("point_count") or 0)
    extent_max = float(obj.get("extent_max") or obj.get("max_extent") or 0.0)
    extent_mid = float(obj.get("extent_mid") or 0.0)
    extent_min = float(obj.get("extent_min") or 0.0)
    linearity = float(obj.get("linearity") or 0.0)
    thickness = float(obj.get("thickness_rms") or 0.0)
    ev = best_evidence(evidence)
    evidence_ok = has_evidence(evidence)

    review_label = "fine_candidate"
    review_status = "needs_review"
    reasons: list[str] = []

    if geometry == "planar_surface_fragment":
        review_label = "surface_fragment"
        review_status = "demoted_planar_surface_fragment"
        reasons.append("local geometry is planar surface fragment")
    elif not evidence_ok:
        review_label = "weak_candidate"
        review_status = "hold_no_image_evidence"
        reasons.append("no image evidence")
    elif candidate == "railing":
        if (
            geometry == "linear_candidate"
            and linearity >= args.railing_min_linearity
            and extent_max >= args.railing_min_extent
            and extent_min <= args.railing_max_thickness
            and point_count >= args.railing_min_points
        ):
            review_label = "railing"
            review_status = "promoted_railing_geometry_evidence"
        else:
            review_label = "fine_candidate"
            review_status = "hold_railing_geometry_mismatch"
            reasons.append("railing candidate lacks thin linear local geometry")
    elif candidate == "car":
        if (
            geometry in {"compact_candidate", "irregular_candidate"}
            and args.car_min_extent <= extent_max <= args.car_max_extent
            and extent_mid >= args.car_min_mid_extent
            and point_count >= args.car_min_points
        ):
            review_label = "car"
            review_status = "promoted_car_geometry_evidence"
        else:
            review_label = "fine_candidate"
            review_status = "hold_car_geometry_mismatch"
            reasons.append("car candidate geometry is not plausible compact/volumetric object")
    else:
        review_label = "fine_candidate"
        review_status = "hold_generic_fine_candidate"

    out["review_label"] = review_label
    out["semantic_label"] = review_label if review_label in {"car", "railing"} else "fine_candidate"
    out["scene_description"] = {
        "car": "geometry and image-evidence promoted vehicle candidate",
        "railing": "geometry and image-evidence promoted railing / fence candidate",
        "surface_fragment": "demoted planar surface fragment from a fine-object candidate",
        "weak_candidate": "weak fine-object candidate without usable image evidence",
        "fine_candidate": "fine-object candidate held for further review",
    }.get(review_label, "fine-object candidate")
    out["review_status"] = review_status
    out["review_reasons"] = reasons
    out["image_evidence_count"] = len(evidence)
    out["best_evidence"] = {
        "frame_id": ev.get("frame_id"),
        "cam_id": ev.get("cam_id"),
        "rank": ev.get("rank"),
        "projected_points": ev.get("projected_points"),
        "bbox_area": ev.get("bbox_area"),
        "bbox_area_ratio": ev.get("bbox_area_ratio"),
        "crop_path": ev.get("crop_path"),
        "overlay_path": ev.get("overlay_path"),
    } if ev else {}
    out["downstream_stage"] = "accepted_fine_candidate" if review_label in {"car", "railing"} else "fine_geometry_review"
    return out


def rewrite_ply(input_ply: Path, output_ply: Path, objects_by_id: dict[int, dict[str, Any]]) -> dict[str, Any]:
    props, vertex_count, header_lines = parse_ply_header(input_ply)
    idx = {name: i for i, name in enumerate(props)}
    object_col = idx.get("object", idx.get("object_id"))
    if object_col is None:
        raise ValueError(f"PLY missing object/object_id: {input_ply}")
    xyz_cols = [idx["x"], idx["y"], idx["z"]]

    changed_points = 0
    label_counts = Counter()
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    with input_ply.open("r", encoding="utf-8", errors="replace") as src, output_ply.open("w", encoding="utf-8") as dst:
        for _ in range(header_lines):
            dst.write(src.readline())
        for line in src:
            parts = line.strip().split()
            if len(parts) <= object_col:
                continue
            oid = int(round(float(parts[object_col])))
            obj = objects_by_id.get(oid)
            if obj is not None:
                review_label = str(obj.get("review_label") or obj.get("semantic_label") or "fine_candidate")
                color = DISPLAY_COLORS.get(review_label, DISPLAY_COLORS["fine_candidate"])
                sem = SEMANTIC_IDS.get(str(obj.get("semantic_label") or "fine_candidate"), 7)
                if {"red", "green", "blue"}.issubset(idx):
                    parts[idx["red"]] = str(color[0])
                    parts[idx["green"]] = str(color[1])
                    parts[idx["blue"]] = str(color[2])
                if "semantic" in idx:
                    parts[idx["semantic"]] = str(sem)
                changed_points += 1
                label_counts[review_label] += 1
            xyz = [float(parts[c]) for c in xyz_cols]
            dst.write(" ".join([f"{xyz[0]:.6f}", f"{xyz[1]:.6f}", f"{xyz[2]:.6f}"] + parts[3:]) + "\n")

    return {
        "input_vertices": vertex_count,
        "changed_points": changed_points,
        "point_review_label_counts": dict(label_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-jsonl", type=Path, required=True)
    parser.add_argument("--split-ply", type=Path, required=True)
    parser.add_argument("--evidence-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--railing-min-linearity", type=float, default=0.62)
    parser.add_argument("--railing-min-extent", type=float, default=0.8)
    parser.add_argument("--railing-max-thickness", type=float, default=0.45)
    parser.add_argument("--railing-min-points", type=int, default=40)
    parser.add_argument("--car-min-extent", type=float, default=1.0)
    parser.add_argument("--car-max-extent", type=float, default=8.5)
    parser.add_argument("--car-min-mid-extent", type=float, default=0.8)
    parser.add_argument("--car-min-points", type=int, default=80)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    objects = read_jsonl(args.split_jsonl)
    evidence = evidence_by_object(read_jsonl(args.evidence_jsonl))
    reviewed = [review_object(obj, evidence.get(int(obj["object_id"]), []), args) for obj in objects]
    reviewed.sort(key=lambda row: int(row["object_id"]))

    out_jsonl = args.output_dir / "fine_candidate_reviewed.jsonl"
    out_ply = args.output_dir / "fine_candidate_reviewed.ply"
    write_jsonl(out_jsonl, reviewed)
    ply_report = rewrite_ply(args.split_ply, out_ply, {int(row["object_id"]): row for row in reviewed})

    report = {
        "split_jsonl": str(args.split_jsonl),
        "split_ply": str(args.split_ply),
        "evidence_jsonl": str(args.evidence_jsonl),
        "output_jsonl": str(out_jsonl),
        "output_ply": str(out_ply),
        "object_count": len(reviewed),
        "review_label_counts": dict(Counter(str(row.get("review_label")) for row in reviewed)),
        "review_status_counts": dict(Counter(str(row.get("review_status")) for row in reviewed)),
        "candidate_label_counts": dict(Counter(str(row.get("candidate_label")) for row in reviewed)),
        "geometry_class_counts": dict(Counter(str(row.get("geometry_class")) for row in reviewed)),
        "objects_with_evidence": sum(1 for row in reviewed if int(row.get("image_evidence_count") or 0) > 0),
        "ply_report": ply_report,
    }
    (args.output_dir / "fine_candidate_review_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
