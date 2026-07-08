#!/usr/bin/env python3
"""List candidate SPG objects that newly merge baseline objects."""

from __future__ import annotations

import argparse
import struct
from array import array
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_labels(path: Path) -> array:
    with path.open("rb") as f:
        if f.read(len(b"GPRGlabels1\n")) != b"GPRGlabels1\n":
            raise ValueError(f"invalid labels magic: {path}")
        n = struct.unpack("<q", f.read(8))[0]
        labels = array("i")
        labels.fromfile(f, n)
    if len(labels) != n:
        raise ValueError(f"label file ended early: expected={n} got={len(labels)}")
    return labels


def object_id(row: dict[str, Any]) -> int:
    return int(row.get("object", row.get("object_id", row.get("patch_id"))))


def source_ids(row: dict[str, Any]) -> list[int]:
    values = row.get("source_patch_ids") or [row.get("patch_id")]
    return [int(v) for v in values if v is not None]


def build_source_to_object(rows: list[dict[str, Any]]) -> dict[int, int]:
    out: dict[int, int] = {}
    for row in rows:
        oid = object_id(row)
        for source_id in source_ids(row):
            out[source_id] = oid
    return out


def row_by_object(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {object_id(row): row for row in rows}


def compare_from_jsonl(baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    baseline_owner = build_source_to_object(baseline_rows)
    changed = []
    truncated = 0
    for row in candidate_rows:
        ids = source_ids(row)
        owners = sorted({baseline_owner.get(source_id, source_id) for source_id in ids})
        if len(owners) <= 1:
            continue
        if row.get("source_patch_ids_truncated"):
            truncated += 1
        changed.append(
            {
                "candidate_object": object_id(row),
                "voxel_count": int(row.get("voxel_count", 0)),
                "geometry_type": row.get("geometry_type"),
                "bucket_entropy": float(row.get("bucket_entropy", 0.0)),
                "source_patch_count": int(row.get("source_patch_count", len(ids))),
                "source_patch_ids": ids,
                "baseline_objects_merged": owners,
                "baseline_object_count": len(owners),
                "source_patch_ids_truncated": bool(row.get("source_patch_ids_truncated")),
            }
        )
    changed.sort(key=lambda item: (item["voxel_count"], item["baseline_object_count"]), reverse=True)
    return changed, truncated


def compare_from_labels(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    baseline_labels: Path,
    candidate_labels: Path,
) -> tuple[list[dict[str, Any]], int]:
    base = read_labels(baseline_labels)
    cand = read_labels(candidate_labels)
    if len(base) != len(cand):
        raise ValueError(f"label length mismatch: {baseline_labels}={len(base)} {candidate_labels}={len(cand)}")
    candidate_meta = row_by_object(candidate_rows)
    by_candidate: dict[int, list[tuple[int, int]]] = {}
    pair_counts: dict[tuple[int, int], int] = {}
    for candidate_id, baseline_id in zip(cand, base, strict=True):
        pair = (int(candidate_id), int(baseline_id))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    for (candidate_id, baseline_id), count in pair_counts.items():
        by_candidate.setdefault(candidate_id, []).append((baseline_id, count))
    changed = []
    for candidate_id, owners in by_candidate.items():
        if len(owners) <= 1:
            continue
        owners.sort(key=lambda item: item[1], reverse=True)
        meta = candidate_meta.get(candidate_id, {})
        changed.append(
            {
                "candidate_object": candidate_id,
                "voxel_count": int(sum(count for _, count in owners)),
                "geometry_type": meta.get("geometry_type"),
                "bucket_entropy": float(meta.get("bucket_entropy", 0.0)),
                "source_patch_count": int(meta.get("source_patch_count", len(owners))),
                "source_patch_ids": [int(v) for v, _ in owners[:24]],
                "baseline_objects_merged": [int(v) for v, _ in owners],
                "baseline_object_count": len(owners),
                "baseline_object_voxel_counts": [{"object": int(v), "voxels": int(c)} for v, c in owners[:24]],
                "source_patch_ids_truncated": len(owners) > 24,
            }
        )
    changed.sort(key=lambda item: (item["voxel_count"], item["baseline_object_count"]), reverse=True)
    return changed, sum(1 for row in changed if row["source_patch_ids_truncated"])


def compare(
    baseline_jsonl: Path,
    candidate_jsonl: Path,
    *,
    baseline_labels: Path | None = None,
    candidate_labels: Path | None = None,
    top_n: int = 40,
) -> dict[str, Any]:
    baseline_rows = read_jsonl(baseline_jsonl)
    candidate_rows = read_jsonl(candidate_jsonl)
    if baseline_labels and candidate_labels:
        changed, truncated = compare_from_labels(baseline_rows, candidate_rows, baseline_labels, candidate_labels)
        method = "labels"
    else:
        changed, truncated = compare_from_jsonl(baseline_rows, candidate_rows)
        method = "jsonl_source_patch_ids"
    return {
        "schema": "spg-object-delta/v1",
        "method": method,
        "baseline_jsonl": str(baseline_jsonl),
        "candidate_jsonl": str(candidate_jsonl),
        "baseline_labels": str(baseline_labels) if baseline_labels else None,
        "candidate_labels": str(candidate_labels) if candidate_labels else None,
        "baseline_object_count": len(baseline_rows),
        "candidate_object_count": len(candidate_rows),
        "new_merge_object_count": len(changed),
        "truncated_source_list_count": truncated,
        "top_new_merges": changed[:top_n],
    }


def markdown(report: dict[str, Any], *, viewer_base_url: str = "") -> str:
    lines = [
        "# SPG Object Delta",
        "",
        f"- baseline objects: `{report['baseline_object_count']}`",
        f"- candidate objects: `{report['candidate_object_count']}`",
        f"- new merge objects: `{report['new_merge_object_count']}`",
        f"- truncated source lists: `{report['truncated_source_list_count']}`",
        "",
        "| candidate object | voxels | geometry | entropy | baseline objects merged | source patches | viewer |",
        "| ---: | ---: | --- | ---: | --- | --- | --- |",
    ]
    for row in report["top_new_merges"]:
        url = "-"
        if viewer_base_url:
            url = f"[open]({viewer_base_url}&object={row['candidate_object']})"
        owners = ", ".join(str(v) for v in row["baseline_objects_merged"][:12])
        if len(row["baseline_objects_merged"]) > 12:
            owners += ", ..."
        sources = ", ".join(str(v) for v in row["source_patch_ids"][:12])
        if row["source_patch_ids_truncated"] or len(row["source_patch_ids"]) > 12:
            sources += ", ..."
        lines.append(
            f"| {row['candidate_object']} | {row['voxel_count']} | {row['geometry_type']} | "
            f"{row['bucket_entropy']:.3f} | {owners} | {sources} | {url} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-jsonl", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--baseline-labels", type=Path)
    parser.add_argument("--candidate-labels", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--viewer-base-url", default="")
    args = parser.parse_args()
    report = compare(
        args.baseline_jsonl,
        args.candidate_jsonl,
        baseline_labels=args.baseline_labels,
        candidate_labels=args.candidate_labels,
        top_n=args.top_n,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown(report, viewer_base_url=args.viewer_base_url), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md), "new_merge_object_count": report["new_merge_object_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
