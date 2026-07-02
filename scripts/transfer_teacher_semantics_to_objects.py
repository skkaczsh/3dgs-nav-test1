#!/usr/bin/env python3
"""Transfer a trusted semantic viewer PLY onto a newer object/Patch PLY.

This is meant for regression recovery: when a new geometry/Patch route has
better boundaries but weaker semantic classification, use an older validated
semantic route as a teacher field.  The teacher votes are spatial nearest-neighbor
votes over the viewer points, then aggregated per object and gated by the new
object geometry type.  Point ownership is never changed here.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.spatial import cKDTree
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit("scipy is required: python -m pip install scipy") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.geometry_input_contract import is_geometry_only_row
from scripts.semantic_label_contract import LABEL_TO_SEMANTIC, SEMANTIC_COLORS, SEMANTIC_TO_LABEL

GEOMETRY_LABELS = {"horizontal", "vertical", "thin_linear", "rough_mixed", "mixed", "unknown"}
SKIP_LABELS = {"unknown", "sky", "ignore", "water"}

HORIZONTAL_LABELS = {"floor", "ground", "road", "grass", "stair", "indoor_floor", "roof"}
VERTICAL_LABELS = {"wall", "building", "railing", "pipe", "equipment"}
THIN_LABELS = {"railing", "pipe", "equipment", "tree"}
ROUGH_LABELS = {
    "car",
    "tree",
    "grass",
    "railing",
    "pipe",
    "equipment",
    "wall",
    "building",
    "floor",
    "road",
    "stair",
    "roof",
    "indoor_floor",
}


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


def read_ply(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    header: list[str] = []
    props: list[str] = []
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header.append(line)
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                in_vertex = True
            elif len(parts) >= 2 and parts[0] == "element":
                in_vertex = False
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                props.append(parts[-1])
            elif line.strip() == "end_header":
                break
    data = np.loadtxt(path, skiprows=len(header), dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return header, props, data


def object_key(row: dict[str, Any]) -> int | None:
    for key in ("viewer_object_id", "object_id"):
        value = row.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def geometry_type(row: dict[str, Any]) -> str:
    return str(row.get("geometry_type") or row.get("object_type_geometry") or "unknown")


def normalized_original_label(row: dict[str, Any]) -> str:
    label = str(row.get("semantic_label") or "unknown")
    if is_geometry_only_row(row):
        return "unknown"
    if label not in GEOMETRY_LABELS:
        return label
    geom = geometry_type(row)
    if geom == "horizontal":
        return "floor"
    if geom == "vertical":
        return "wall"
    if geom == "thin_linear":
        return "railing"
    return "unknown"


def label_allowed(label: str, geom: str, *, allow_surface_teacher_on_unknown: bool) -> bool:
    if label in SKIP_LABELS:
        return False
    if geom == "horizontal":
        return label in HORIZONTAL_LABELS
    if geom == "vertical":
        return label in VERTICAL_LABELS
    if geom == "thin_linear":
        return label in THIN_LABELS
    if geom in {"rough_mixed", "mixed"}:
        return label in ROUGH_LABELS
    if geom == "unknown":
        return allow_surface_teacher_on_unknown or label in ROUGH_LABELS
    return True


def semantic_id(label: str) -> int:
    return int(LABEL_TO_SEMANTIC.get(label, 0))


def choose_label(
    row: dict[str, Any],
    votes: Counter[str],
    args: argparse.Namespace,
) -> tuple[str, str, float, Counter[str], Counter[str]]:
    geom = geometry_type(row)
    original = normalized_original_label(row)
    allowed = Counter({k: v for k, v in votes.items() if label_allowed(k, geom, allow_surface_teacher_on_unknown=args.allow_surface_teacher_on_unknown)})
    vetoed = Counter({k: v for k, v in votes.items() if k not in allowed and k not in SKIP_LABELS})
    total = sum(votes.values())
    allowed_total = sum(allowed.values())
    if total < args.min_teacher_votes:
        return original, "kept_original_insufficient_teacher_votes", 0.0, allowed, vetoed
    if allowed_total / max(total, 1) < args.min_allowed_ratio:
        return original, "kept_original_teacher_geometry_veto", allowed_total / max(total, 1), allowed, vetoed
    label, count = allowed.most_common(1)[0]
    ratio = count / max(allowed_total, 1)
    global_ratio = count / max(total, 1)
    if ratio < args.min_winner_ratio or global_ratio < args.min_global_winner_ratio:
        return original, "kept_original_low_teacher_consensus", global_ratio, allowed, vetoed
    return label, "teacher_semantic_transfer", global_ratio, allowed, vetoed


def aggregate_teacher_votes(
    source: np.ndarray,
    source_props: list[str],
    teacher: np.ndarray,
    teacher_props: list[str],
    args: argparse.Namespace,
) -> tuple[dict[int, Counter[str]], dict[str, Any]]:
    sidx = {name: i for i, name in enumerate(source_props)}
    tidx = {name: i for i, name in enumerate(teacher_props)}
    for name in ("x", "y", "z", "object"):
        if name not in sidx:
            raise ValueError(f"source PLY missing property {name}")
    for name in ("x", "y", "z", "semantic"):
        if name not in tidx:
            raise ValueError(f"teacher PLY missing property {name}")

    teacher_xyz = teacher[:, [tidx["x"], tidx["y"], tidx["z"]]].astype(np.float32, copy=False)
    source_xyz = source[:, [sidx["x"], sidx["y"], sidx["z"]]].astype(np.float32, copy=False)
    teacher_sem = teacher[:, tidx["semantic"]].astype(np.int32, copy=False)
    source_obj = source[:, sidx["object"]].astype(np.int64, copy=False)

    tree = cKDTree(teacher_xyz)
    dist, nn = tree.query(source_xyz, k=1, workers=args.workers, distance_upper_bound=args.max_distance)
    valid = np.isfinite(dist) & (nn < len(teacher_sem))

    votes: dict[int, Counter[str]] = defaultdict(Counter)
    distance_bins = Counter()
    matched_rows = int(valid.sum())
    for object_id, sem_id, d in zip(source_obj[valid], teacher_sem[nn[valid]], dist[valid], strict=False):
        label = SEMANTIC_TO_LABEL.get(int(sem_id), "unknown")
        if label in SKIP_LABELS:
            continue
        votes[int(object_id)][label] += 1
        bucket = int(min(args.max_distance, float(d)) / max(args.distance_bin, 1e-6))
        distance_bins[f"{bucket * args.distance_bin:.2f}-{(bucket + 1) * args.distance_bin:.2f}"] += 1

    return votes, {
        "source_points": int(len(source)),
        "teacher_points": int(len(teacher)),
        "matched_points": matched_rows,
        "matched_ratio": matched_rows / max(int(len(source)), 1),
        "max_distance": args.max_distance,
        "distance_bins": dict(distance_bins),
    }


def rewrite_ply(
    source_header: list[str],
    source_props: list[str],
    source: np.ndarray,
    output_ply: Path,
    object_labels: dict[int, str],
) -> dict[str, Any]:
    idx = {name: i for i, name in enumerate(source_props)}
    required = {"red", "green", "blue", "object", "semantic"}
    missing = required - set(idx)
    if missing:
        raise ValueError(f"source PLY missing required fields: {sorted(missing)}")

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    label_point_counts = Counter()
    rows = 0
    with output_ply.open("w", encoding="utf-8") as f:
        for line in source_header:
            f.write(line)
        for row in source:
            parts: list[str] = []
            object_id = int(row[idx["object"]])
            label = object_labels.get(object_id, "unknown")
            sem = semantic_id(label)
            color = SEMANTIC_COLORS.get(sem, SEMANTIC_COLORS[0])
            for i, value in enumerate(row):
                name = source_props[i]
                if name == "red":
                    parts.append(str(color[0]))
                elif name == "green":
                    parts.append(str(color[1]))
                elif name == "blue":
                    parts.append(str(color[2]))
                elif name == "semantic":
                    parts.append(str(sem))
                elif name == "object":
                    parts.append(str(object_id))
                elif name in {"x", "y", "z"}:
                    parts.append(f"{float(value):.6f}")
                else:
                    if abs(float(value) - round(float(value))) < 1e-6:
                        parts.append(str(int(round(float(value)))))
                    else:
                        parts.append(f"{float(value):.6f}")
            f.write(" ".join(parts) + "\n")
            label_point_counts[label] += 1
            rows += 1
    return {"rows": rows, "label_point_counts": dict(label_point_counts)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_header, source_props, source = read_ply(args.source_ply)
    teacher_header, teacher_props, teacher = read_ply(args.teacher_ply)
    del teacher_header

    source_objects = read_jsonl(args.source_objects_jsonl)
    teacher_votes, match_report = aggregate_teacher_votes(source, source_props, teacher, teacher_props, args)

    out_objects: list[dict[str, Any]] = []
    object_labels: dict[int, str] = {}
    status_counts = Counter()
    label_counts = Counter()
    changed = 0
    for row in source_objects:
        oid = object_key(row)
        if oid is None:
            continue
        old_label = normalized_original_label(row)
        label, status, confidence, allowed, vetoed = choose_label(row, teacher_votes.get(oid, Counter()), args)
        out = dict(row)
        out["semantic_label_original"] = old_label
        out["semantic_label"] = label
        out["semantic_id"] = semantic_id(label)
        out["semantic_transfer_status"] = status
        out["teacher_semantic_confidence"] = confidence
        out["teacher_semantic_votes"] = dict(teacher_votes.get(oid, Counter()))
        out["teacher_allowed_votes"] = dict(allowed)
        out["teacher_vetoed_votes"] = dict(vetoed)
        if label != old_label:
            changed += 1
        status_counts[status] += 1
        label_counts[label] += 1
        object_labels[int(oid)] = label
        out_objects.append(out)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.output_dir / f"{args.output_prefix}.jsonl"
    out_ply = args.output_dir / f"{args.output_prefix}.ply"
    report_json = args.output_dir / f"{args.output_prefix}_report.json"
    write_jsonl(out_jsonl, out_objects)
    ply_report = rewrite_ply(source_header, source_props, source, out_ply, object_labels)

    report = {
        "source_ply": str(args.source_ply),
        "source_objects_jsonl": str(args.source_objects_jsonl),
        "teacher_ply": str(args.teacher_ply),
        "output_jsonl": str(out_jsonl),
        "output_ply": str(out_ply),
        "object_count": len(out_objects),
        "changed_object_count": changed,
        "status_counts": dict(status_counts),
        "label_object_counts": dict(label_counts),
        **match_report,
        **ply_report,
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ply", type=Path, required=True)
    parser.add_argument("--source-objects-jsonl", type=Path, required=True)
    parser.add_argument("--teacher-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="objects_teacher_semantic")
    parser.add_argument("--max-distance", type=float, default=0.12)
    parser.add_argument("--distance-bin", type=float, default=0.03)
    parser.add_argument("--min-teacher-votes", type=int, default=3)
    parser.add_argument("--min-winner-ratio", type=float, default=0.55)
    parser.add_argument("--min-global-winner-ratio", type=float, default=0.35)
    parser.add_argument("--min-allowed-ratio", type=float, default=0.35)
    parser.add_argument("--allow-surface-teacher-on-unknown", action="store_true")
    parser.add_argument("--workers", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
