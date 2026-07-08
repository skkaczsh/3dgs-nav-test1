#!/usr/bin/env python3
"""Compare GeoPatch optimizer runs from JSONL/report artifacts."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def bucket_purity(bucket_counts: dict[str, int]) -> float:
    total = sum(int(v) for v in bucket_counts.values())
    if total <= 0:
        return 0.0
    return max(int(v) for v in bucket_counts.values()) / float(total)


def bucket_entropy(bucket_counts: dict[str, int]) -> float:
    total = sum(int(v) for v in bucket_counts.values())
    if total <= 0:
        return 0.0
    out = 0.0
    for count in bucket_counts.values():
        p = int(count) / float(total)
        if p > 0:
            out -= p * math.log2(p)
    return out


def extent_ratio(extent: list[float]) -> float:
    if not extent:
        return 0.0
    values = [max(float(v), 1e-6) for v in extent[:3]]
    return max(values) / min(values)


def summarize_jsonl(path: Path, large_voxels: int, entropy_threshold: float) -> dict[str, Any]:
    counts = Counter()
    geom = Counter()
    conflict = Counter()
    voxel_counts: list[int] = []
    entropy_values: list[float] = []
    large_high_entropy = 0
    large_low_purity = 0
    large_extreme_aspect = 0
    total_voxels = 0
    top_rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            count = int(row.get("voxel_count") or 0)
            bucket_counts = row.get("bucket_counts") or {}
            ent = float(row.get("bucket_entropy") if row.get("bucket_entropy") is not None else bucket_entropy(bucket_counts))
            purity = bucket_purity(bucket_counts)
            ext = [float(v) for v in (row.get("extent") or [])[:3]]
            ratio = extent_ratio(ext)

            counts["patches"] += 1
            total_voxels += count
            voxel_counts.append(count)
            entropy_values.append(ent)
            geom[str(row.get("geometry_type") or "unknown")] += 1
            for flag in row.get("conflict_flags") or []:
                conflict[str(flag)] += 1
            if ent >= entropy_threshold:
                counts["high_entropy"] += 1
            if count >= large_voxels:
                counts["large"] += 1
                if ent >= entropy_threshold:
                    large_high_entropy += 1
                if purity < 0.75:
                    large_low_purity += 1
                if ratio >= 80.0:
                    large_extreme_aspect += 1
            if len(top_rows) < 20 or count > top_rows[-1]["voxel_count"]:
                top_rows.append(
                    {
                        "patch_id": int(row.get("patch_id") or row.get("object") or -1),
                        "voxel_count": count,
                        "geometry_type": row.get("geometry_type"),
                        "bucket_entropy": ent,
                        "bucket_purity": purity,
                        "extent": ext,
                        "extent_ratio": ratio,
                    }
                )
                top_rows.sort(key=lambda item: item["voxel_count"], reverse=True)
                top_rows = top_rows[:20]

    voxel_counts.sort()
    entropy_values.sort()

    def percentile(values: list[float] | list[int], q: float) -> float:
        if not values:
            return 0.0
        idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
        return float(values[idx])

    return {
        "jsonl": str(path),
        "patch_count": int(counts["patches"]),
        "total_voxels": int(total_voxels),
        "high_entropy_count": int(counts["high_entropy"]),
        "large_patch_count": int(counts["large"]),
        "large_high_entropy_count": int(large_high_entropy),
        "large_low_purity_count": int(large_low_purity),
        "large_extreme_aspect_count": int(large_extreme_aspect),
        "voxel_count_p50": percentile(voxel_counts, 0.50),
        "voxel_count_p90": percentile(voxel_counts, 0.90),
        "voxel_count_p99": percentile(voxel_counts, 0.99),
        "voxel_count_max": float(voxel_counts[-1]) if voxel_counts else 0.0,
        "bucket_entropy_p50": percentile(entropy_values, 0.50),
        "bucket_entropy_p90": percentile(entropy_values, 0.90),
        "bucket_entropy_p99": percentile(entropy_values, 0.99),
        "geometry_counts": dict(geom),
        "conflict_counts": dict(conflict),
        "top_patches": top_rows,
    }


def summarize_merge_log(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    status = Counter()
    reasons = Counter()
    accepted_profiles = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            status[str(row.get("status"))] += 1
            reason = str(row.get("reason"))
            reasons[reason] += 1
            if row.get("status") == "accept":
                accepted_profiles[reason] += 1
    return {
        "status_counts": dict(status),
        "reason_counts": dict(reasons),
        "accepted_profiles": dict(accepted_profiles),
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = ["# GeoPatch Run Comparison", ""]
    lines.append("| run | patches | high entropy | large high entropy | large low purity | merge accepts | accepted profiles |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for run in report["runs"]:
        merge = run.get("merge_log_summary") or {}
        status = merge.get("status_counts") or {}
        accepts = status.get("accept")
        if accepts is None:
            accepts = (run.get("source_report") or {}).get("accepted_edges", 0)
        profiles = merge.get("accepted_profiles") or {}
        profile_text = ", ".join(f"{k}:{v}" for k, v in sorted(profiles.items()) if k != "None") or "-"
        lines.append(
            f"| {run['name']} | {run['patch_count']} | {run['high_entropy_count']} | "
            f"{run['large_high_entropy_count']} | {run['large_low_purity_count']} | "
            f"{accepts} | {profile_text} |"
        )
    lines.append("")
    for run in report["runs"]:
        lines.append(f"## {run['name']}")
        lines.append("")
        lines.append(
            f"- voxel p50/p90/p99/max: `{run['voxel_count_p50']:.0f}` / "
            f"`{run['voxel_count_p90']:.0f}` / `{run['voxel_count_p99']:.0f}` / `{run['voxel_count_max']:.0f}`"
        )
        lines.append(
            f"- entropy p50/p90/p99: `{run['bucket_entropy_p50']:.3f}` / "
            f"`{run['bucket_entropy_p90']:.3f}` / `{run['bucket_entropy_p99']:.3f}`"
        )
        lines.append("- top patches:")
        for row in run["top_patches"][:5]:
            lines.append(
                f"  - `{row['patch_id']}` voxels={row['voxel_count']} "
                f"geom={row['geometry_type']} entropy={row['bucket_entropy']:.3f} "
                f"purity={row['bucket_purity']:.3f} extent_ratio={row['extent_ratio']:.1f}"
            )
        lines.append("")
    return "\n".join(lines)


def parse_run(value: str) -> tuple[str, Path, Path | None, Path | None]:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--run must be name=/path/to/run.jsonl")
    name, rest = parts
    paths = rest.split(",")
    jsonl = Path(paths[0])
    report = Path(paths[1]) if len(paths) > 1 and paths[1] else None
    merge = Path(paths[2]) if len(paths) > 2 and paths[2] else None
    return name, jsonl, report, merge


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True, help="name=jsonl[,report,merge_log]")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--large-voxels", type=int, default=10000)
    parser.add_argument("--entropy-threshold", type=float, default=1.1)
    args = parser.parse_args()

    runs = []
    for name, jsonl, report_path, merge_path in args.run:
        if not jsonl.exists():
            raise FileNotFoundError(jsonl)
        run = summarize_jsonl(jsonl, large_voxels=args.large_voxels, entropy_threshold=args.entropy_threshold)
        run["name"] = name
        run["source_report"] = load_json(report_path)
        run["merge_log_summary"] = summarize_merge_log(merge_path)
        runs.append(run)

    report = {
        "schema": "geo-patch-run-comparison/v1",
        "large_voxels": args.large_voxels,
        "entropy_threshold": args.entropy_threshold,
        "runs": runs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md), "runs": len(runs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
