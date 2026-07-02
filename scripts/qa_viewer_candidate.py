#!/usr/bin/env python3
"""Summarize and sanity-check a semantic PLY/Object viewer candidate.

This QA stage intentionally depends only on the exported viewer artifacts:
an ASCII PLY and the companion object JSONL.  It is meant to be a cheap gate
after object-fusion/post-processing variants, before visual review.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.semantic_label_contract import LABEL_TO_SEMANTIC as LABEL_IDS
from scripts.semantic_label_contract import SEMANTIC_TO_LABEL as LABELS

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
    "stair": "楼梯",
    "indoor_floor": "室内地面",
    "roof": "屋顶/平台",
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


def source_support_kind(row: dict[str, Any]) -> str:
    label = str(row.get("semantic_label") or "unknown")
    source_scores = row.get("semantic_evidence_source_scores")
    if not isinstance(source_scores, dict):
        return "missing_source_scores"
    active_sources: list[str] = []
    for source in ("sam", "teacher", "scene"):
        scores = source_scores.get(source)
        if not isinstance(scores, dict):
            continue
        try:
            score = float(scores.get(label, 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score > 0:
            active_sources.append(source)
    return "+".join(active_sources) if active_sources else "no_label_source_support"


def count_total(counts: dict[str, int]) -> int:
    return sum(int(value) for value in counts.values())


def ratio_for_keys(counts: dict[str, int], keys: tuple[str, ...]) -> float:
    total = count_total(counts)
    if total <= 0:
        return 0.0
    matched = sum(int(counts.get(key, 0)) for key in keys)
    return matched / total


def evidence_risk_warnings(evidence_summary: dict[str, Any]) -> list[str]:
    point_sources = evidence_summary.get("point_source_support_counts")
    if not isinstance(point_sources, dict):
        point_sources = {}
    object_sources = evidence_summary.get("object_source_support_counts")
    if not isinstance(object_sources, dict):
        object_sources = {}
    conflict_flags = evidence_summary.get("conflict_flag_counts")
    if not isinstance(conflict_flags, dict):
        conflict_flags = {}

    warnings: list[str] = []
    unsupported_keys = ("missing_object", "missing_source_scores", "no_label_source_support")
    missing_point_ratio = ratio_for_keys(point_sources, unsupported_keys)
    if missing_point_ratio >= 0.01:
        warnings.append(f"evidence provenance missing/unsupported for {missing_point_ratio:.1%} of visible points")
    missing_object_ratio = ratio_for_keys(object_sources, unsupported_keys)
    if missing_object_ratio >= 0.01:
        warnings.append(f"evidence provenance missing/unsupported for {missing_object_ratio:.1%} of visible objects")
    scene_point_ratio = ratio_for_keys(point_sources, ("scene",))
    if scene_point_ratio >= 0.05:
        warnings.append(f"scene-only support covers {scene_point_ratio:.1%} of visible points")
    scene_object_ratio = ratio_for_keys(object_sources, ("scene",))
    if scene_object_ratio >= 0.05:
        warnings.append(f"scene-only support covers {scene_object_ratio:.1%} of visible objects")
    geometry_veto_count = int(conflict_flags.get("geometry_vetoed_some_evidence", 0))
    object_total = count_total(object_sources)
    if object_total > 0 and geometry_veto_count / object_total >= 0.10:
        warnings.append(f"geometry veto evidence is dense: {geometry_veto_count} flags over {object_total} visible objects")
    return warnings


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


def summarize_evidence(objects: list[dict[str, Any]], ply_stats: dict[str, Any]) -> dict[str, Any]:
    json_by_id = {oid: row for row in objects if (oid := object_id(row)) is not None}
    point_source_counts: Counter[str] = Counter()
    object_source_counts: Counter[str] = Counter()
    fusion_status_counts: Counter[str] = Counter()
    conflict_flag_counts: Counter[str] = Counter()
    for oid, point_count in ply_stats["object_point_counts"].items():
        obj = json_by_id.get(int(oid))
        if obj is None:
            point_source_counts["missing_object"] += int(point_count)
            continue
        point_source_counts[source_support_kind(obj)] += int(point_count)
    for oid in set(int(value) for value in ply_stats["object_point_counts"].keys()):
        obj = json_by_id.get(oid)
        if obj is None:
            object_source_counts["missing_object"] += 1
            continue
        object_source_counts[source_support_kind(obj)] += 1
        fusion_status_counts[str(obj.get("semantic_fusion_status") or "missing")] += 1
        for flag in obj.get("conflict_flags") or []:
            conflict_flag_counts[str(flag)] += 1
    summary = {
        "point_source_support_counts": dict(point_source_counts),
        "object_source_support_counts": dict(object_source_counts),
        "fusion_status_counts": dict(fusion_status_counts),
        "conflict_flag_counts": dict(conflict_flag_counts),
    }
    summary["warnings"] = evidence_risk_warnings(summary)
    return summary


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
    evidence_summary = summarize_evidence(objects, ply_stats)
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
    warnings.extend(evidence_summary["warnings"])

    report = {
        "status": "failed" if errors else "ok",
        "errors": errors,
        "warnings": warnings,
        "inputs": {"ply": str(args.ply), "objects_jsonl": str(args.objects_jsonl)},
        "ply": {**ply_header, **{k: v for k, v in ply_stats.items() if k != "object_semantic_counts"}},
        "objects": object_summary,
        "evidence": evidence_summary,
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
    evidence = report.get("evidence", {})
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
        f"- Evidence source points: {format_counter(evidence.get('point_source_support_counts', {}))}",
        f"- Evidence source objects: {format_counter(evidence.get('object_source_support_counts', {}))}",
        f"- Fusion statuses: {format_counter(evidence.get('fusion_status_counts', {}))}",
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
