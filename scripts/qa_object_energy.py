#!/usr/bin/env python3
"""Diagnose object-level geometry/semantic energy without relabeling.

This QA script is deliberately read-only.  It scores existing object JSONL
rows, optionally augments them with coarse voxel overlap evidence from a viewer
PLY, and reports the objects that most likely cap the current pipeline quality.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.qa_object_voxel_overlap import measure_overlap


SURFACE_LABELS = {"floor", "ground", "indoor_floor", "roof", "wall", "ceiling", "grass"}
HORIZONTAL_LABELS = {"floor", "ground", "indoor_floor", "roof", "ceiling", "grass", "stair"}
VERTICAL_LABELS = {"wall", "building"}
FINE_LABELS = {"car", "railing", "equipment", "person"}
BUCKET_ALIASES = {
    "0": "unknown",
    "1": "horizontal",
    "2": "vertical",
    "3": "thin_linear",
    "4": "rough_mixed",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bucket_name(key: Any) -> str:
    text = str(key)
    return BUCKET_ALIASES.get(text, text)


def normalized_bucket_counts(row: dict[str, Any]) -> Counter[str]:
    out: Counter[str] = Counter()
    for key, value in (row.get("bucket_counts") or {}).items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            out[bucket_name(key)] += count
    return out


def entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        p = count / total
        value -= p * math.log2(max(p, 1e-12))
    return value


def ratio(count: int, total: int) -> float:
    return float(count / max(total, 1))


def bbox_extent(row: dict[str, Any]) -> list[float]:
    bbox = row.get("bbox_3d") or {}
    bmin = bbox.get("min") or [0.0, 0.0, 0.0]
    bmax = bbox.get("max") or [0.0, 0.0, 0.0]
    return [max(0.0, float(bmax[i]) - float(bmin[i])) for i in range(3)]


def object_id(row: dict[str, Any]) -> int:
    return int(row.get("object_id", row.get("viewer_object_id", 0)))


def label(row: dict[str, Any]) -> str:
    return str(row.get("semantic_label") or row.get("label") or "unknown")


def geometry_type(row: dict[str, Any]) -> str:
    return str(row.get("geometry_type") or "unknown")


def semantic_geometry_conflicts(row: dict[str, Any], bucket_ratios: dict[str, float]) -> list[str]:
    lab = label(row)
    geom = geometry_type(row)
    flags: list[str] = []
    horizontal = bucket_ratios.get("horizontal", 0.0)
    vertical = bucket_ratios.get("vertical", 0.0)
    linear = bucket_ratios.get("thin_linear", 0.0)
    rough = bucket_ratios.get("rough_mixed", 0.0)

    if lab in HORIZONTAL_LABELS and vertical >= 0.35 and horizontal < 0.50:
        flags.append("horizontal_label_on_vertical_dominant_object")
    if lab in VERTICAL_LABELS and horizontal >= 0.35 and vertical < 0.50:
        flags.append("vertical_label_on_horizontal_dominant_object")
    if lab == "car" and (horizontal >= 0.75 or vertical >= 0.75):
        flags.append("car_label_on_single_surface_bucket")
    if lab == "railing" and linear < 0.10 and vertical + horizontal >= 0.80:
        flags.append("railing_label_without_linear_support")
    if lab == "grass" and rough < 0.10 and horizontal < 0.40:
        flags.append("grass_label_without_rough_or_horizontal_support")
    if geom == "mixed" and lab in SURFACE_LABELS:
        flags.append("surface_label_on_mixed_geometry")
    return flags


def geometry_supports_label(lab: str, bucket_ratios: dict[str, float], purity: float) -> bool:
    horizontal = bucket_ratios.get("horizontal", 0.0)
    vertical = bucket_ratios.get("vertical", 0.0)
    linear = bucket_ratios.get("thin_linear", 0.0)
    rough = bucket_ratios.get("rough_mixed", 0.0)
    if lab in HORIZONTAL_LABELS:
        return horizontal >= 0.70 and purity >= 0.70
    if lab in VERTICAL_LABELS:
        return vertical >= 0.70 and purity >= 0.70
    if lab == "railing":
        return linear >= 0.10 or vertical >= 0.50
    if lab == "grass":
        return rough >= 0.20 or horizontal >= 0.50
    return False


def teacher_conflicts(row: dict[str, Any], geometry_supported: bool) -> list[str]:
    flags: list[str] = []
    votes = Counter({str(k): int(v) for k, v in (row.get("teacher_semantic_votes") or {}).items()})
    allowed = Counter({str(k): int(v) for k, v in (row.get("teacher_allowed_votes") or {}).items()})
    vetoed = Counter({str(k): int(v) for k, v in (row.get("teacher_vetoed_votes") or {}).items()})
    if votes:
        top_label, top_count = votes.most_common(1)[0]
        if top_label != label(row) and top_count >= max(50, 2 * votes.get(label(row), 0)):
            flags.append(
                "teacher_top_vote_differs_but_geometry_supports_label"
                if geometry_supported
                else "teacher_top_vote_differs_from_label"
            )
    if sum(vetoed.values()) > max(sum(allowed.values()), 0) * 2 and sum(vetoed.values()) >= 100:
        flags.append(
            "teacher_vetoed_votes_resolved_by_geometry"
            if geometry_supported
            else "teacher_vetoed_votes_dominate_allowed_votes"
        )
    confidence = row.get("teacher_semantic_confidence")
    if confidence is not None:
        try:
            if float(confidence) < 0.20 and label(row) != "unknown":
                flags.append(
                    "low_visual_support_for_geometry_supported_label"
                    if geometry_supported
                    else "low_teacher_confidence_for_hard_label"
                )
        except (TypeError, ValueError):
            pass
    return flags


def overlap_object_stats(overlap_report: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    if not overlap_report:
        return {}
    stats: dict[int, dict[str, Any]] = defaultdict(lambda: {"max_intersection_over_min": 0.0, "pair_count": 0})
    for pair in overlap_report.get("top_object_overlaps") or []:
        for key in ("a", "b"):
            try:
                oid = int(pair[key])
            except (TypeError, ValueError):
                continue
            item = stats[oid]
            item["pair_count"] += 1
            item["max_intersection_over_min"] = max(
                float(item["max_intersection_over_min"]),
                float(pair.get("intersection_over_min") or 0.0),
            )
    return dict(stats)


def score_object(
    row: dict[str, Any],
    overlap_stats: dict[int, dict[str, Any]],
    min_mixed_bucket_ratio: float,
) -> dict[str, Any]:
    counts = normalized_bucket_counts(row)
    total = sum(counts.values()) or int(row.get("voxel_count") or 0)
    bucket_ratios = {key: ratio(value, total) for key, value in counts.items()}
    dominant_bucket, dominant_count = ("unknown", 0)
    if counts:
        dominant_bucket, dominant_count = counts.most_common(1)[0]
    purity = ratio(dominant_count, total)
    ent = entropy(counts)
    horizontal = bucket_ratios.get("horizontal", 0.0)
    vertical = bucket_ratios.get("vertical", 0.0)

    flags: list[str] = []
    if horizontal >= min_mixed_bucket_ratio and vertical >= min_mixed_bucket_ratio:
        flags.append("horizontal_vertical_same_object")
    if purity < 0.70 and total >= 1000:
        flags.append("low_bucket_purity_large_object")
    if ent >= 1.10 and total >= 1000:
        flags.append("high_bucket_entropy_large_object")
    flags.extend(semantic_geometry_conflicts(row, bucket_ratios))
    geometry_supported = geometry_supports_label(label(row), bucket_ratios, purity)
    flags.extend(teacher_conflicts(row, geometry_supported))

    ext = bbox_extent(row)
    max_extent = max(ext) if ext else 0.0
    min_extent = max(min(ext), 1e-6) if ext else 1e-6
    if label(row) in FINE_LABELS and max_extent >= 8.0:
        flags.append("fine_label_has_large_scene_extent")
    if label(row) in {"floor", "wall", "ceiling"} and max_extent / min_extent >= 80.0:
        flags.append("extreme_aabb_aspect_surface")

    ov = overlap_stats.get(object_id(row), {})
    overlap_max = float(ov.get("max_intersection_over_min") or 0.0)
    if overlap_max >= 0.35:
        flags.append("coarse_voxel_overlap_with_other_object")

    score = 0.0
    score += (1.0 - purity) * 2.0
    score += min(ent / 2.0, 1.0)
    score += 1.5 if "horizontal_vertical_same_object" in flags else 0.0
    unresolved_semantic_flags = [
        f
        for f in flags
        if (
            "label" in f
            or f
            in {
                "teacher_top_vote_differs_from_label",
                "teacher_vetoed_votes_dominate_allowed_votes",
                "low_teacher_confidence_for_hard_label",
            }
        )
        and "geometry_supports" not in f
        and "resolved_by_geometry" not in f
        and "low_visual_support_for_geometry_supported_label" not in f
    ]
    resolved_evidence_flags = [
        f
        for f in flags
        if "resolved_by_geometry" in f
        or "geometry_supports_label" in f
        or f == "low_visual_support_for_geometry_supported_label"
    ]
    score += 1.2 * len(unresolved_semantic_flags)
    score += 0.12 * len(resolved_evidence_flags)
    score += 0.8 if overlap_max >= 0.35 else 0.0
    score += min(math.log10(max(total, 1)) / 8.0, 1.0)

    return {
        "object_id": object_id(row),
        "semantic_label": label(row),
        "geometry_type": geometry_type(row),
        "voxel_count": int(row.get("voxel_count") or total),
        "patch_count": int(row.get("patch_count") or 0),
        "bbox_extent": ext,
        "dominant_bucket": dominant_bucket,
        "bucket_purity": purity,
        "bucket_entropy": ent,
        "bucket_ratios": bucket_ratios,
        "teacher_semantic_confidence": row.get("teacher_semantic_confidence"),
        "overlap_max_intersection_over_min": overlap_max,
        "overlap_pair_count": int(ov.get("pair_count") or 0),
        "flags": sorted(set(flags)),
        "energy_score": score,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Object Energy QA",
        "",
        f"Objects: `{report['object_count']}`",
        f"Flagged objects: `{report['flagged_object_count']}`",
        f"High-risk objects: `{report['high_risk_object_count']}`",
        f"Total voxel count from objects: `{report['total_voxel_count']}`",
        "",
        "## High-Risk Flags",
        "",
    ]
    for key, value in report["risk_flag_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Evidence-Only Flags", ""])
    for key, value in report["evidence_flag_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Top Problem Objects", ""])
    lines.append("| object | label | geom | voxels | purity | entropy | score | flags |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---|")
    for row in report["top_problem_objects"][:30]:
        flags = ", ".join(row["flags"][:5])
        lines.append(
            f"| {row['object_id']} | {row['semantic_label']} | {row['geometry_type']} | "
            f"{row['voxel_count']} | {row['bucket_purity']:.3f} | {row['bucket_entropy']:.3f} | "
            f"{row['energy_score']:.3f} | {flags} |"
        )
    lines.append("")
    return "\n".join(lines)


def is_evidence_only_flag(flag: str) -> bool:
    return (
        "resolved_by_geometry" in flag
        or "geometry_supports_label" in flag
        or flag == "low_visual_support_for_geometry_supported_label"
    )


def analyze(
    objects_jsonl: Path,
    ply: Path | None,
    output_dir: Path,
    voxel_size: float,
    max_overlap_pairs: int,
    top_n: int,
    min_mixed_bucket_ratio: float,
) -> dict[str, Any]:
    objects = read_jsonl(objects_jsonl)
    overlap_report = None
    if ply is not None:
        overlap_report = measure_overlap(ply, voxel_size=voxel_size, max_pairs=max_overlap_pairs)
    overlap_stats = overlap_object_stats(overlap_report)
    scored = [score_object(row, overlap_stats, min_mixed_bucket_ratio) for row in objects]
    scored.sort(key=lambda row: (row["energy_score"], row["voxel_count"]), reverse=True)

    flag_counts: Counter[str] = Counter()
    risk_flag_counts: Counter[str] = Counter()
    evidence_flag_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    geometry_counts: Counter[str] = Counter()
    total_voxels = 0
    high_risk_object_count = 0
    for row in scored:
        label_counts[row["semantic_label"]] += 1
        geometry_counts[row["geometry_type"]] += 1
        total_voxels += int(row["voxel_count"])
        flag_counts.update(row["flags"])
        risk_flags = [flag for flag in row["flags"] if not is_evidence_only_flag(flag)]
        evidence_flags = [flag for flag in row["flags"] if is_evidence_only_flag(flag)]
        risk_flag_counts.update(risk_flags)
        evidence_flag_counts.update(evidence_flags)
        if risk_flags:
            high_risk_object_count += 1

    report = {
        "schema": "object-energy-qa/v1",
        "objects_jsonl": str(objects_jsonl),
        "ply": str(ply) if ply else None,
        "voxel_size": float(voxel_size),
        "object_count": len(scored),
        "flagged_object_count": sum(1 for row in scored if row["flags"]),
        "high_risk_object_count": high_risk_object_count,
        "total_voxel_count": int(total_voxels),
        "label_object_counts": dict(label_counts),
        "geometry_object_counts": dict(geometry_counts),
        "flag_counts": dict(flag_counts.most_common()),
        "risk_flag_counts": dict(risk_flag_counts.most_common()),
        "evidence_flag_counts": dict(evidence_flag_counts.most_common()),
        "overlap_summary": None
        if overlap_report is None
        else {
            "mixed_object_voxel_ratio": overlap_report["mixed_object_voxel_ratio"],
            "mixed_semantic_voxel_ratio": overlap_report["mixed_semantic_voxel_ratio"],
            "top_object_overlaps": overlap_report["top_object_overlaps"][:10],
            "top_semantic_overlaps": overlap_report["top_semantic_overlaps"][:10],
        },
        "top_problem_objects": scored[:top_n],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "object_energy_qa_report.json", report)
    (output_dir / "object_energy_qa.md").write_text(build_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--ply", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.20)
    parser.add_argument("--max-overlap-pairs", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--min-mixed-bucket-ratio", type=float, default=0.20)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    report = analyze(
        objects_jsonl=args.objects_jsonl,
        ply=args.ply,
        output_dir=args.output_dir,
        voxel_size=args.voxel_size,
        max_overlap_pairs=args.max_overlap_pairs,
        top_n=args.top_n,
        min_mixed_bucket_ratio=args.min_mixed_bucket_ratio,
    )
    if args.summary_only:
        print(
            json.dumps(
                {
                    "schema": report["schema"],
                    "object_count": report["object_count"],
                    "flagged_object_count": report["flagged_object_count"],
                    "high_risk_object_count": report["high_risk_object_count"],
                    "flag_counts": report["flag_counts"],
                    "risk_flag_counts": report["risk_flag_counts"],
                    "evidence_flag_counts": report["evidence_flag_counts"],
                    "top_problem_objects": report["top_problem_objects"][:10],
                    "overlap_summary": report["overlap_summary"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
