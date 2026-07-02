#!/usr/bin/env python3
"""Map high-risk mixed objects back to patch-level split work.

Object-energy actions tell us which objects are suspicious.  This planner adds
the missing ownership detail: whether the fix belongs inside one dirty patch or
at object assembly time across multiple patches.  It is read-only and produces a
queue for the next dense patch optimizer run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


BUCKET_NAMES = {
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def object_id(row: dict[str, Any]) -> int:
    return int(row.get("object_id", row.get("viewer_object_id", 0)))


def bucket_counts(row: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for key, value in (row.get("bucket_counts") or {}).items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            counts[BUCKET_NAMES.get(str(key), str(key))] += count
    return counts


def bucket_ratios(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values())
    return {key: float(value / max(total, 1)) for key, value in counts.items()}


def split_groups(counts: Counter[str], min_ratio: float) -> list[dict[str, Any]]:
    total = sum(counts.values())
    groups = []
    for bucket, count in counts.most_common():
        r = float(count / max(total, 1))
        if r >= min_ratio:
            groups.append({"bucket": bucket, "voxel_count": int(count), "ratio": r})
    return groups


def classify_scope(obj: dict[str, Any], groups: list[dict[str, Any]]) -> tuple[str, str]:
    patch_count = int(obj.get("patch_count") or 0)
    if len(groups) < 2:
        return "monitor", "only one material bucket above split threshold"
    if patch_count <= 1:
        return "split_single_patch_by_bucket_connectivity", "object is one mixed patch; fix must happen inside patch generation"
    return "split_or_regroup_multi_patch_object", "object spans multiple patches; inspect whether mixed member patches should split before regroup"


def build_candidate(
    action: dict[str, Any],
    obj: dict[str, Any],
    *,
    min_bucket_ratio: float,
) -> dict[str, Any]:
    counts = bucket_counts(obj)
    groups = split_groups(counts, min_bucket_ratio)
    scope, reason = classify_scope(obj, groups)
    patch_ids = [int(v) for v in (obj.get("patch_ids") or [])]
    return {
        "object_id": object_id(obj),
        "source_action": action.get("action"),
        "split_scope": scope,
        "reason": reason,
        "semantic_label": obj.get("semantic_label"),
        "geometry_type": obj.get("geometry_type"),
        "voxel_count": int(obj.get("voxel_count") or 0),
        "patch_count": int(obj.get("patch_count") or 0),
        "patch_ids": patch_ids,
        "patch_ids_truncated": bool(obj.get("patch_ids_truncated")),
        "bucket_counts": dict(counts),
        "bucket_ratios": bucket_ratios(counts),
        "split_groups": groups,
        "priority_score": float(action.get("priority_score") or 0.0),
        "action_flags": action.get("flags") or [],
        "recommended_stage": "patch_generation" if scope == "split_single_patch_by_bucket_connectivity" else "patch_then_object_assembly",
    }


def plan_splits(
    actions: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    *,
    limit: int,
    min_bucket_ratio: float,
    include_monitor: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    objects_by_id = {object_id(row): row for row in objects}
    candidates: list[dict[str, Any]] = []
    missing = 0
    for action in actions:
        if not str(action.get("action") or "").startswith("split"):
            continue
        obj = objects_by_id.get(int(action["object_id"]))
        if obj is None:
            missing += 1
            continue
        candidate = build_candidate(action, obj, min_bucket_ratio=min_bucket_ratio)
        if candidate["split_scope"] == "monitor" and not include_monitor:
            continue
        candidates.append(candidate)
    candidates.sort(key=lambda row: (-float(row["priority_score"]), -int(row["voxel_count"]), int(row["object_id"])))
    if limit > 0:
        candidates = candidates[:limit]

    report = {
        "schema": "mixed-object-split-plan/v1",
        "input_action_count": len(actions),
        "input_object_count": len(objects),
        "missing_object_count": missing,
        "candidate_count": len(candidates),
        "scope_counts": dict(Counter(row["split_scope"] for row in candidates)),
        "recommended_stage_counts": dict(Counter(row["recommended_stage"] for row in candidates)),
        "semantic_label_counts": dict(Counter(str(row.get("semantic_label") or "unknown") for row in candidates)),
        "top_candidates": candidates[:30],
    }
    return candidates, report


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Mixed Object Split Plan",
        "",
        f"Candidates: `{report['candidate_count']}`",
        f"Missing objects: `{report['missing_object_count']}`",
        "",
        "## Scope Counts",
        "",
    ]
    for key, value in report["scope_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Top Candidates", ""])
    lines.append("| object | scope | label | geom | voxels | patches | groups |")
    lines.append("|---:|---|---|---|---:|---:|---|")
    for row in report["top_candidates"]:
        groups = ", ".join(f"{g['bucket']}={g['ratio']:.2f}" for g in row["split_groups"])
        lines.append(
            f"| {row['object_id']} | {row['split_scope']} | {row['semantic_label']} | "
            f"{row['geometry_type']} | {row['voxel_count']} | {row['patch_count']} | {groups} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions-jsonl", type=Path, required=True)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--min-bucket-ratio", type=float, default=0.15)
    parser.add_argument("--include-monitor", action="store_true")
    args = parser.parse_args()

    candidates, report = plan_splits(
        read_jsonl(args.actions_jsonl),
        read_jsonl(args.objects_jsonl),
        limit=args.limit,
        min_bucket_ratio=args.min_bucket_ratio,
        include_monitor=args.include_monitor,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "mixed_object_split_candidates.jsonl", candidates)
    (args.output_dir / "mixed_object_split_plan.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "mixed_object_split_plan.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
