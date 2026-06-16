#!/usr/bin/env python3
"""Apply geometry sanity checks to Target labels before Object fusion.

This is a post-process over existing target JSONL files. It does not rerun
SAM/Qwen. The goal is to prevent large floor/wall/building masks from being
accepted when the 3D component geometry or identity text strongly contradicts
the coarse VLM label.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


SURFACE_LABELS = {"floor", "wall", "building", "ceiling", "road"}
FINE_LABELS = {"equipment", "railing", "pipe", "other"}
FLOOR_TEXT = {"floor", "ground", "landing", "stair", "stairs", "step", "steps", "tread", "tiled floor", "roof surface", "rooftop floor"}
WALL_TEXT = {"wall", "vertical", "facade", "panel", "paneling", "parapet"}
CEILING_TEXT = {"ceiling", "overhead", "underside", "roof underside"}
RAILING_TEXT = {"railing", "guardrail", "handrail", "fence", "metal fence", "barrier", "balustrade", "mesh"}
PIPE_TEXT = {"pipe", "conduit", "cable", "duct", "tube", "hose", "wire"}
EQUIPMENT_TEXT = {"equipment", "hvac", "outdoor unit", "air conditioning", "machine", "cabinet", "device", "fixture", "sensor", "antenna"}


def norm(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def has_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def normal_abs_z(target: dict) -> float:
    normal = ((target.get("pca") or {}).get("normal") or [0.0, 0.0, 1.0])
    try:
        return abs(float(normal[2]))
    except (IndexError, TypeError, ValueError):
        return 1.0


def planarity(target: dict) -> float:
    try:
        return float((target.get("pca") or {}).get("planarity", 0.0))
    except (TypeError, ValueError):
        return 0.0


def linearity(target: dict) -> float:
    try:
        return float((target.get("pca") or {}).get("linearity", 0.0))
    except (TypeError, ValueError):
        return 0.0


def cluster_size(target: dict) -> int:
    try:
        return int(target.get("cluster_size") or len(target.get("point_indices") or []))
    except (TypeError, ValueError):
        return 0


def bbox_extent(target: dict) -> list[float]:
    bbox = target.get("bbox_3d") or {}
    try:
        bmin = [float(x) for x in bbox.get("min", [0.0, 0.0, 0.0])]
        bmax = [float(x) for x in bbox.get("max", [0.0, 0.0, 0.0])]
        return [max(0.0, hi - lo) for lo, hi in zip(bmin, bmax)]
    except (TypeError, ValueError):
        return [0.0, 0.0, 0.0]


def text_blob(target: dict) -> str:
    chunks = [
        target.get("description", ""),
        target.get("identity_hint", ""),
        target.get("vlm_reason", ""),
    ]
    attrs = target.get("attributes") or {}
    if isinstance(attrs, dict):
        chunks.extend(str(v) for v in attrs.values())
    return norm(" ".join(str(x) for x in chunks if x))


def relabel_large_planar_fine_target(target: dict, args: argparse.Namespace) -> tuple[str, str] | None:
    label = str(target.get("label") or "unknown")
    if label not in FINE_LABELS:
        return None
    size = cluster_size(target)
    extents = sorted(bbox_extent(target), reverse=True)
    max_extent = extents[0] if extents else 0.0
    second_extent = extents[1] if len(extents) > 1 else 0.0
    z = normal_abs_z(target)
    pl = planarity(target)
    if (
        size < args.fine_surface_min_points
        or max_extent < args.fine_surface_min_extent
        or second_extent < args.fine_surface_min_second_extent
        or pl < args.fine_surface_min_planarity
    ):
        return None
    if z >= args.floor_normal_z:
        return "floor", f"{label}_large_planar_floor"
    if z <= args.wall_normal_z:
        return "wall", f"{label}_large_planar_wall"
    return "building", f"{label}_large_planar_building"


def relabel_target(target: dict, args: argparse.Namespace) -> tuple[str, str]:
    label = str(target.get("label") or "unknown")
    fine_override = relabel_large_planar_fine_target(target, args)
    if fine_override is not None:
        return fine_override
    if label not in SURFACE_LABELS:
        return label, "non_surface_passthrough"

    text = text_blob(target)
    z = normal_abs_z(target)
    pl = planarity(target)
    text_wall = has_any(text, WALL_TEXT)
    text_floor = has_any(text, FLOOR_TEXT)
    text_ceiling = has_any(text, CEILING_TEXT)
    text_railing = has_any(text, RAILING_TEXT)
    text_pipe = has_any(text, PIPE_TEXT)
    text_equipment = has_any(text, EQUIPMENT_TEXT)
    ln = linearity(target)
    extents = sorted(bbox_extent(target), reverse=True)
    max_extent = extents[0] if extents else 0.0
    second_extent = extents[1] if len(extents) > 1 else 0.0

    # Rescue thin fine structures that were swallowed into coarse surface labels.
    if label in SURFACE_LABELS:
        if text_railing and ln >= args.fine_linear_min_linearity and max_extent >= args.fine_linear_min_extent:
            return "railing", f"{label}_rescued_railing_linear"
        if text_pipe and ln >= args.fine_linear_min_linearity and max_extent >= args.fine_linear_min_extent:
            return "pipe", f"{label}_rescued_pipe_linear"
        if text_equipment and max_extent <= args.equipment_surface_rescue_max_extent and second_extent <= args.equipment_surface_rescue_max_second_extent:
            return "equipment", f"{label}_rescued_equipment_compact"

    if label == "floor":
        if text_ceiling:
            return "ceiling", "floor_text_ceiling"
        if text_wall:
            return "wall", "floor_text_wall"
        if z <= args.wall_normal_z and pl >= args.min_planarity:
            return "wall", "floor_vertical_geometry"
        if z < args.floor_normal_z and not text_floor:
            return "building", "floor_non_horizontal_geometry"
        return label, "floor_ok"

    if label == "wall":
        if text_ceiling:
            return "ceiling", "wall_text_ceiling"
        if text_floor and z >= args.floor_normal_z:
            return "floor", "wall_text_floor_horizontal"
        if z >= args.floor_normal_z and pl >= args.min_planarity:
            return "floor", "wall_horizontal_geometry"
        return label, "wall_ok"

    if label in {"building", "ceiling", "road"}:
        if text_wall and z <= args.wall_normal_z:
            return "wall", f"{label}_text_wall"
        if text_floor and z >= args.floor_normal_z:
            return "floor", f"{label}_text_floor_horizontal"
        if text_ceiling:
            return "ceiling", f"{label}_text_ceiling"
        return label, f"{label}_ok"

    return label, "surface_passthrough"


def iter_target_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.glob("targets_frame_*.jsonl") if p.name != "targets_all.jsonl")
    return [path]


def process_file(src: Path, dst: Path, args: argparse.Namespace) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            old = str(row.get("label") or "unknown")
            new, reason = relabel_target(row, args)
            counts["targets"] += 1
            counts[f"old:{old}"] += 1
            counts[f"new:{new}"] += 1
            counts[f"reason:{reason}"] += 1
            if new != old:
                counts["changed"] += 1
                counts[f"change:{old}->{new}"] += 1
                row["original_label"] = old
                row["label"] = new
                row["geometry_guard_reason"] = reason
                row["geometry_guard_abs_normal_z"] = normal_abs_z(row)
                row["geometry_guard_planarity"] = planarity(row)
                if new != "unknown":
                    # Keep semantic_id consistent with build_targets_from_masks label ids.
                    semantic_ids = {
                        "unknown": 0,
                        "other": 1,
                        "wall": 2,
                        "floor": 3,
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
                        "ignore": 255,
                    }
                    row["semantic_id"] = semantic_ids.get(new, 0)
                if old in {"floor", "wall", "ceiling", "road"} or new in {"floor", "wall", "ceiling", "road"}:
                    row["parent_class"] = "surface"
                elif new == "building":
                    row["parent_class"] = "structure"
            else:
                row["geometry_guard_reason"] = reason
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"source": str(src), "output": str(dst), "counts": dict(counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-targets", type=Path, required=True)
    parser.add_argument("--output-targets", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--floor-normal-z", type=float, default=0.72)
    parser.add_argument("--wall-normal-z", type=float, default=0.40)
    parser.add_argument("--min-planarity", type=float, default=0.12)
    parser.add_argument("--fine-surface-min-points", type=int, default=500)
    parser.add_argument("--fine-surface-min-extent", type=float, default=1.2)
    parser.add_argument("--fine-surface-min-second-extent", type=float, default=0.45)
    parser.add_argument("--fine-surface-min-planarity", type=float, default=0.10)
    parser.add_argument("--fine-linear-min-linearity", type=float, default=0.84)
    parser.add_argument("--fine-linear-min-extent", type=float, default=0.6)
    parser.add_argument("--equipment-surface-rescue-max-extent", type=float, default=1.8)
    parser.add_argument("--equipment-surface-rescue-max-second-extent", type=float, default=1.2)
    args = parser.parse_args()

    files = iter_target_files(args.input_targets)
    if not files:
        raise SystemExit(f"no target jsonl files found: {args.input_targets}")

    reports = []
    total = Counter()
    for src in files:
        dst = args.output_targets / src.name if args.input_targets.is_dir() else args.output_targets
        report = process_file(src, dst, args)
        reports.append(report)
        total.update(report["counts"])

    merged = args.output_targets / "targets_all.jsonl"
    if args.input_targets.is_dir():
        with merged.open("w", encoding="utf-8") as fout:
            for p in sorted(args.output_targets.glob("targets_frame_*.jsonl")):
                if p.name == "targets_all.jsonl":
                    continue
                fout.write(p.read_text(encoding="utf-8"))

    summary = {
        "input_targets": str(args.input_targets),
        "output_targets": str(args.output_targets),
        "files": len(files),
        "summary": dict(total),
        "changed_ratio": float(total.get("changed", 0) / max(total.get("targets", 0), 1)),
        "file_reports": reports[:20],
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["files", "summary", "changed_ratio"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
