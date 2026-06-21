#!/usr/bin/env python3
"""Build a conservative V2/V8 hybrid semantic viewer preview.

V8 is used as the geometry/object-split base because it preserves more local
surface structure.  V2 is used only as a conservative teacher for labels that
were empirically more stable: outdoor vegetation, broad floor/wall consistency,
and car false-positive vetoes.

The merge is point/voxel based because V2 and V8 object ids are not compatible.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABEL_TO_SEMANTIC = {
    "unknown": 0,
    "other": 1,
    "wall": 2,
    "floor": 3,
    "ground": 3,
    "ceiling": 4,
    "grass": 5,
    "tree": 6,
    "person": 7,
    "car": 8,
    "railing": 9,
    "building": 10,
    "sky": 11,
    "road": 12,
    "water": 13,
    "furniture": 14,
    "pipe": 15,
    "equipment": 16,
    "fine_candidate": 17,
    "ignore": 255,
}

SEMANTIC_TO_LABEL = {
    0: "unknown",
    1: "other",
    2: "wall",
    3: "floor",
    4: "ceiling",
    5: "grass",
    6: "tree",
    7: "person",
    8: "car",
    9: "railing",
    10: "building",
    11: "sky",
    12: "road",
    13: "water",
    14: "furniture",
    15: "pipe",
    16: "equipment",
    17: "fine_candidate",
    255: "ignore",
}

SEMANTIC_COLORS = {
    0: (128, 128, 128),
    1: (120, 120, 120),
    2: (185, 185, 185),
    3: (192, 166, 120),
    4: (170, 170, 210),
    5: (70, 170, 80),
    6: (35, 125, 55),
    7: (255, 80, 80),
    8: (245, 200, 35),
    9: (230, 55, 220),
    10: (160, 160, 170),
    15: (80, 200, 220),
    16: (255, 120, 40),
    17: (90, 140, 255),
    255: (20, 20, 20),
}

SURFACE_TEACHER_LABELS = {"floor", "wall", "grass", "tree"}
FINE_LABELS = {"car", "railing"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def parse_header(path: Path) -> tuple[list[str], list[str], int, int]:
    header: list[str] = []
    props: list[str] = []
    vertex_count = 0
    header_lines = 0
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header_lines += 1
            header.append(line)
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                vertex_count = int(parts[2])
                in_vertex = True
            elif len(parts) >= 2 and parts[0] == "element":
                in_vertex = False
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                props.append(parts[-1])
            if line.strip() == "end_header":
                break
    return header, props, vertex_count, header_lines


def voxel_key(parts: list[str], idx: dict[str, int], voxel_size: float) -> tuple[int, int, int]:
    return (
        math.floor(float(parts[idx["x"]]) / voxel_size),
        math.floor(float(parts[idx["y"]]) / voxel_size),
        math.floor(float(parts[idx["z"]]) / voxel_size),
    )


def semantic_label(parts: list[str], idx: dict[str, int]) -> str:
    value = int(round(float(parts[idx["semantic"]])))
    return SEMANTIC_TO_LABEL.get(value, "unknown")


def load_voxel_teacher(source_ply: Path, voxel_size: float) -> dict[tuple[int, int, int], str]:
    _header, props, _vertex_count, header_lines = parse_header(source_ply)
    idx = {name: i for i, name in enumerate(props)}
    for required in ("x", "y", "z", "semantic"):
        if required not in idx:
            raise ValueError(f"PLY missing {required}: {source_ply}")
    votes: dict[tuple[int, int, int], Counter[str]] = defaultdict(Counter)
    with source_ply.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(header_lines):
            next(f)
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) <= idx["semantic"]:
                continue
            votes[voxel_key(parts, idx, voxel_size)][semantic_label(parts, idx)] += 1
    return {key: counter.most_common(1)[0][0] for key, counter in votes.items()}


def hybrid_label(v8_label: str, v2_label: str | None) -> tuple[str, str]:
    if not v2_label:
        return v8_label, "keep_no_v2_voxel"
    if v8_label == "ceiling":
        return v8_label, "keep_v8_ceiling"
    if v8_label in FINE_LABELS:
        if v2_label == v8_label:
            return v8_label, f"keep_v8_{v8_label}_confirmed_by_v2"
        if v2_label in SURFACE_TEACHER_LABELS:
            return v2_label, f"v2_surface_overrides_v8_{v8_label}"
        return "fine_candidate", f"v8_{v8_label}_unconfirmed"
    if v2_label in {"grass", "tree"} and v8_label in {"wall", "floor", "unknown", "grass"}:
        return v2_label, "v2_vegetation_teacher"
    if v8_label in {"unknown", "other"} and v2_label in SURFACE_TEACHER_LABELS:
        return v2_label, "v2_surface_fills_unknown"
    if v8_label == "grass" and v2_label in {"floor", "wall"}:
        return v2_label, "v2_surface_vetoes_v8_grass"
    return v8_label, "keep_v8"


def rewrite_hybrid_ply(
    v8_ply: Path,
    v2_teacher: dict[tuple[int, int, int], str],
    output_ply: Path,
    voxel_size: float,
) -> tuple[dict[str, Any], dict[int, Counter[str]]]:
    header, props, vertex_count, header_lines = parse_header(v8_ply)
    idx = {name: i for i, name in enumerate(props)}
    for required in ("x", "y", "z", "red", "green", "blue", "object", "semantic"):
        if required not in idx:
            raise ValueError(f"PLY missing {required}: {v8_ply}")

    label_counts = Counter()
    reason_counts = Counter()
    transition_counts = Counter()
    object_votes: dict[int, Counter[str]] = defaultdict(Counter)
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with v8_ply.open("r", encoding="utf-8", errors="replace") as src, output_ply.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in header:
            dst.write(line)
        for _ in range(header_lines):
            next(src)
        for line in src:
            if not line.strip():
                continue
            parts = line.strip().split()
            key = voxel_key(parts, idx, voxel_size)
            v8_label = semantic_label(parts, idx)
            v2_label = v2_teacher.get(key)
            new_label, reason = hybrid_label(v8_label, v2_label)
            semantic = LABEL_TO_SEMANTIC.get(new_label, 0)
            color = SEMANTIC_COLORS.get(semantic, SEMANTIC_COLORS[0])
            parts[idx["red"]] = str(color[0])
            parts[idx["green"]] = str(color[1])
            parts[idx["blue"]] = str(color[2])
            parts[idx["semantic"]] = str(semantic)
            dst.write(" ".join(parts) + "\n")
            label_counts[new_label] += 1
            reason_counts[reason] += 1
            if new_label != v8_label:
                transition_counts[f"{v8_label}->{new_label}"] += 1
            object_votes[int(round(float(parts[idx["object"]])))][new_label] += 1
            rows += 1
    report = {
        "source_v8_ply": str(v8_ply),
        "output_ply": str(output_ply),
        "voxel_size": voxel_size,
        "vertex_count_header": vertex_count,
        "rows": rows,
        "label_counts": dict(label_counts),
        "reason_counts": dict(reason_counts),
        "transition_counts": dict(transition_counts),
    }
    return report, object_votes


def update_objects(
    v8_objects_jsonl: Path,
    object_votes: dict[int, Counter[str]],
    output_jsonl: Path,
) -> dict[str, Any]:
    rows = read_jsonl(v8_objects_jsonl)
    label_counts = Counter()
    changed = 0
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        oid_raw = out.get("viewer_object_id", out.get("object_id"))
        try:
            oid = int(oid_raw)
        except (TypeError, ValueError):
            oid = None
        old_label = str(out.get("semantic_label") or "unknown")
        votes = object_votes.get(oid, Counter()) if oid is not None else Counter()
        if votes:
            new_label, new_count = votes.most_common(1)[0]
            ratio = float(new_count / max(sum(votes.values()), 1))
            out["semantic_label_original"] = out.get("semantic_label_original") or old_label
            out["semantic_label"] = new_label
            out["hybrid_v2_v8_votes"] = dict(votes)
            out["hybrid_v2_v8_vote_ratio"] = ratio
            out["hybrid_v2_v8_status"] = "changed" if new_label != old_label else "unchanged"
            if new_label != old_label:
                changed += 1
                out["status"] = f"hybrid_v2_teacher_{old_label}_to_{new_label}"
        label_counts[str(out.get("semantic_label") or "unknown")] += 1
        out_rows.append(out)
    write_jsonl(output_jsonl, out_rows)
    return {
        "source_objects_jsonl": str(v8_objects_jsonl),
        "output_objects_jsonl": str(output_jsonl),
        "object_count": len(out_rows),
        "changed_object_count": changed,
        "object_label_counts": dict(label_counts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-ply", type=Path, required=True)
    parser.add_argument("--v8-ply", type=Path, required=True)
    parser.add_argument("--v8-objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    teacher = load_voxel_teacher(args.v2_ply, args.voxel_size)
    output_ply = args.output_dir / "v2_v8_hybrid_semantic.ply"
    output_jsonl = args.output_dir / "v2_v8_hybrid_objects.jsonl"
    ply_report, object_votes = rewrite_hybrid_ply(args.v8_ply, teacher, output_ply, args.voxel_size)
    object_report = update_objects(args.v8_objects_jsonl, object_votes, output_jsonl)
    report = {
        "v2_ply": str(args.v2_ply),
        "v8_ply": str(args.v8_ply),
        "v8_objects_jsonl": str(args.v8_objects_jsonl),
        "output_dir": str(args.output_dir),
        "teacher_voxels": len(teacher),
        **ply_report,
        **object_report,
    }
    report_path = args.output_dir / "v2_v8_hybrid_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
