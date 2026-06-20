#!/usr/bin/env python3
"""Summarize and sanity-check a semantic PLY/Object viewer candidate.

This QA stage intentionally depends only on the exported viewer artifacts:
an ASCII PLY and the companion object JSONL.  It is meant to be a cheap gate
after object-fusion/post-processing variants, before visual review.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = {
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
LABEL_IDS = {name: idx for idx, name in LABELS.items()}
LABEL_ALIASES = {"ground": "floor", "ambiguous": "unknown"}
LABEL_ZH = {
    "unknown": "未知",
    "other": "其他",
    "wall": "墙面",
    "floor": "地面",
    "ground": "地面",
    "ceiling": "天花板",
    "grass": "草地",
    "tree": "树木",
    "person": "人",
    "car": "汽车",
    "railing": "栏杆/护栏",
    "building": "建筑",
    "sky": "天空",
    "road": "路面",
    "water": "水体",
    "furniture": "家具",
    "pipe": "管线",
    "equipment": "设备",
    "fine_candidate": "细目标候选",
    "ignore": "忽略",
    "ambiguous": "模糊",
}
LARGE_FINE_OBJECT_THRESHOLDS = {
    "railing": 10000,
    "car": 25000,
}


def canonical_label(label: Any) -> str:
    value = str(label or "unknown")
    return LABEL_ALIASES.get(value, value)


def zh(label: str) -> str:
    return LABEL_ZH.get(label, label)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_ply(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        first = f.readline().strip()
        if first != "ply":
            raise ValueError(f"not a PLY file: {path}")
        vertex_count = None
        props: list[str] = []
        in_vertex = False
        for line in f:
            line = line.strip()
            if line.startswith("element "):
                parts = line.split()
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
                continue
            if line.startswith("property ") and in_vertex:
                props.append(line.split()[-1])
                continue
            if line == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"missing vertex count in {path}")
        prop_index = {name: i for i, name in enumerate(props)}
        required = {"object", "semantic"}
        missing = required - set(prop_index)
        if missing:
            raise ValueError(f"missing PLY properties {sorted(missing)} in {path}")

        semantic_counts = Counter()
        object_counts = Counter()
        object_semantic_counts: dict[int, Counter] = defaultdict(Counter)
        frame_counts = Counter()
        camera_counts = Counter()
        priority_counts = Counter()
        data_rows = 0
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            data_rows += 1
            object_id = int(float(parts[prop_index["object"]]))
            semantic_id = int(float(parts[prop_index["semantic"]]))
            label = LABELS.get(semantic_id, f"id_{semantic_id}")
            semantic_counts[label] += 1
            object_counts[object_id] += 1
            object_semantic_counts[object_id][label] += 1
            if "frame" in prop_index:
                frame_counts[int(float(parts[prop_index["frame"]]))] += 1
            if "camera" in prop_index:
                camera_counts[int(float(parts[prop_index["camera"]]))] += 1
            if "priority" in prop_index:
                priority_counts[int(float(parts[prop_index["priority"]]))] += 1

    header = {"vertex_count": vertex_count, "properties": props}
    stats = {
        "data_rows": data_rows,
        "semantic_point_counts": dict(semantic_counts),
        "object_point_counts": dict(object_counts),
        "object_semantic_counts": {k: dict(v) for k, v in object_semantic_counts.items()},
        "frame_count": len(frame_counts),
        "camera_counts": dict(camera_counts),
        "priority_counts": dict(priority_counts),
    }
    return header, stats


def object_id(row: dict[str, Any]) -> int | None:
    value = row.get("viewer_object_id", row.get("object_id"))
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def object_point_count(row: dict[str, Any]) -> int:
    try:
        return int(row.get("point_count") or 0)
    except (TypeError, ValueError):
        return 0


def summarize_objects(objects: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter()
    status_counts = Counter()
    label_point_counts = Counter()
    status_point_counts = Counter()
    top_by_points: list[dict[str, Any]] = []
    remaining_ambiguous: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []
    fine_large: list[dict[str, Any]] = []

    for row in objects:
        label = str(row.get("semantic_label") or "unknown")
        status = str(row.get("status") or "unknown")
        points = object_point_count(row)
        oid = object_id(row)
        base = {
            "object_id": oid,
            "semantic_label": label,
            "semantic_label_zh": zh(label),
            "status": status,
            "point_count": points,
            "target_count": int(row.get("target_count") or 0),
            "frames": row.get("frames", []),
            "bbox_3d": row.get("bbox_3d", {}),
            "centroid": row.get("centroid", []),
            "normal": row.get("normal", []),
            "label_votes": row.get("label_votes", {}),
        }
        label_counts[label] += 1
        status_counts[status] += 1
        label_point_counts[label] += points
        status_point_counts[status] += points
        top_by_points.append(base)
        if label == "ambiguous" or status == "ambiguous_object":
            remaining_ambiguous.append(base)
        if status == "surface_ambiguous_resolved":
            resolved.append({**base, "reason": row.get("ambiguous_surface_resolve_reason")})
        threshold = LARGE_FINE_OBJECT_THRESHOLDS.get(label)
        if threshold is not None and points >= threshold:
            fine_large.append({**base, "large_fine_threshold": threshold})

    sort_key = lambda r: int(r.get("point_count") or 0)
    return {
        "object_count": len(objects),
        "label_object_counts": dict(label_counts),
        "status_counts": dict(status_counts),
        "label_point_counts": dict(label_point_counts),
        "status_point_counts": dict(status_point_counts),
        "top_objects_by_points": sorted(top_by_points, key=sort_key, reverse=True),
        "top_remaining_ambiguous_by_points": sorted(remaining_ambiguous, key=sort_key, reverse=True),
        "top_resolved_by_points": sorted(resolved, key=sort_key, reverse=True),
        "large_fine_objects": sorted(fine_large, key=sort_key, reverse=True),
    }


def compare_ply_objects(objects: list[dict[str, Any]], ply_stats: dict[str, Any], top_n: int) -> dict[str, Any]:
    json_by_id = {oid: row for row in objects if (oid := object_id(row)) is not None}
    ply_ids = {int(k) for k in ply_stats["object_point_counts"].keys()}
    json_ids = set(json_by_id.keys())
    missing_in_json = sorted(ply_ids - json_ids)
    missing_in_ply = sorted(json_ids - ply_ids)
    semantic_mismatch = []

    for oid in sorted(ply_ids & json_ids):
        row = json_by_id[oid]
        expected = canonical_label(row.get("semantic_label"))
        counts = Counter(ply_stats["object_semantic_counts"].get(oid, {}))
        if not counts:
            continue
        observed, observed_points = counts.most_common(1)[0]
        observed = canonical_label(observed)
        total = sum(counts.values())
        if expected != observed:
            semantic_mismatch.append(
                {
                    "object_id": oid,
                    "expected_object_label": row.get("semantic_label"),
                    "observed_ply_label": observed,
                    "point_count": total,
                    "observed_ratio": round(observed_points / max(total, 1), 4),
                    "semantic_counts": dict(counts),
                }
            )

    return {
        "ply_object_count": len(ply_ids),
        "json_object_count": len(json_ids),
        "missing_in_json_count": len(missing_in_json),
        "missing_in_json_examples": missing_in_json[:top_n],
        "missing_in_ply_count": len(missing_in_ply),
        "missing_in_ply_examples": missing_in_ply[:top_n],
        "semantic_mismatch_count": len(semantic_mismatch),
        "top_semantic_mismatches": sorted(
            semantic_mismatch, key=lambda r: int(r["point_count"]), reverse=True
        )[:top_n],
    }


def limit_lists(value: Any, top_n: int) -> Any:
    if isinstance(value, dict):
        return {k: limit_lists(v, top_n) for k, v in value.items()}
    if isinstance(value, list):
        return [limit_lists(v, top_n) for v in value[:top_n]]
    return value


def load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    if not path.exists():
        return {"missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    objects = read_jsonl(args.objects_jsonl)
    ply_header, ply_stats = parse_ply(args.ply)
    object_summary = summarize_objects(objects)
    consistency = compare_ply_objects(objects, ply_stats, args.top_n)
    errors = []
    warnings = []
    if ply_header["vertex_count"] != ply_stats["data_rows"]:
        errors.append(
            f"PLY vertex header/data mismatch: {ply_header['vertex_count']} vs {ply_stats['data_rows']}"
        )
    if consistency["missing_in_json_count"]:
        errors.append(f"PLY objects missing in JSONL: {consistency['missing_in_json_count']}")
    if consistency["missing_in_ply_count"]:
        warnings.append(f"JSONL objects missing in PLY: {consistency['missing_in_ply_count']}")
    if consistency["semantic_mismatch_count"]:
        errors.append(f"object label and PLY semantic mismatch: {consistency['semantic_mismatch_count']}")
    if object_summary["status_counts"].get("ambiguous_object", 0):
        warnings.append(
            f"remaining ambiguous objects: {object_summary['status_counts'].get('ambiguous_object', 0)}"
        )
    if object_summary["large_fine_objects"]:
        warnings.append("large fine objects exceed class-aware thresholds; inspect for surface swallowing")

    report = {
        "status": "failed" if errors else "ok",
        "errors": errors,
        "warnings": warnings,
        "inputs": {"ply": str(args.ply), "objects_jsonl": str(args.objects_jsonl)},
        "ply": {**ply_header, **{k: v for k, v in ply_stats.items() if k != "object_semantic_counts"}},
        "objects": object_summary,
        "consistency": consistency,
        "ambiguous_surface_resolve_report": load_optional_json(args.ambiguous_report),
        "consolidation_report": load_optional_json(args.consolidation_report),
    }
    return limit_lists(report, args.top_n)


def format_counter(counter: dict[str, int], limit: int = 12) -> str:
    rows = sorted(counter.items(), key=lambda item: int(item[1]), reverse=True)[:limit]
    return ", ".join(f"{zh(k)}({k})={v:,}" for k, v in rows) if rows else "-"


def write_markdown(path: Path, report: dict[str, Any], top_n: int) -> None:
    obj = report["objects"]
    ply = report["ply"]
    consistency = report["consistency"]
    lines = [
        "# Viewer Candidate QA",
        "",
        f"- Status: `{report['status']}`",
        f"- PLY vertices: `{ply['data_rows']:,}` / header `{ply['vertex_count']:,}`",
        f"- PLY objects: `{consistency['ply_object_count']:,}`; JSON objects: `{consistency['json_object_count']:,}`",
        f"- Object labels: {format_counter(obj['label_object_counts'])}",
        f"- Point labels: {format_counter(obj['label_point_counts'])}",
        f"- Status counts: {format_counter(obj['status_counts'])}",
        "",
        "## Warnings",
        "",
    ]
    if report["warnings"]:
        lines.extend(f"- {w}" for w in report["warnings"])
    else:
        lines.append("- none")
    lines.extend(["", "## Errors", ""])
    if report["errors"]:
        lines.extend(f"- {e}" for e in report["errors"])
    else:
        lines.append("- none")

    lines.extend(["", "## Top Ambiguous Objects", ""])
    for row in obj["top_remaining_ambiguous_by_points"][:top_n]:
        lines.append(
            f"- object `{row['object_id']}`: {row['point_count']:,} pts, "
            f"label `{row['semantic_label']}`/{row['semantic_label_zh']}, "
            f"targets `{row['target_count']}`, frames `{row.get('frames', [])[:3]}`"
        )
    if not obj["top_remaining_ambiguous_by_points"]:
        lines.append("- none")

    lines.extend(["", "## Top Resolved Surface Objects", ""])
    for row in obj["top_resolved_by_points"][:top_n]:
        lines.append(
            f"- object `{row['object_id']}`: {row['point_count']:,} pts -> "
            f"`{row['semantic_label']}`/{row['semantic_label_zh']}, reason `{row.get('reason')}`"
        )
    if not obj["top_resolved_by_points"]:
        lines.append("- none")

    lines.extend(["", "## Large Fine Objects", ""])
    for row in obj["large_fine_objects"][:top_n]:
        lines.append(
            f"- object `{row['object_id']}`: {row['point_count']:,} pts, "
            f"label `{row['semantic_label']}`/{row['semantic_label_zh']}, status `{row['status']}`"
        )
    if not obj["large_fine_objects"]:
        lines.append("- none")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--ambiguous-report", type=Path)
    parser.add_argument("--consolidation-report", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    report = build_report(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        write_markdown(args.output_md, report, args.top_n)
    print(
        json.dumps(
            {
                "status": report["status"],
                "vertices": report["ply"]["data_rows"],
                "objects": report["objects"]["object_count"],
                "warnings": report["warnings"],
                "errors": report["errors"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
