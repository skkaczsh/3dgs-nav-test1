#!/usr/bin/env python3
"""Apply oversized-mask hygiene decisions to fine residual cluster QA PLYs.

This creates non-destructive QA artifacts:

- a status PLY with every point and a cluster_status scalar
- a filtered PLY with demoted clusters removed

It does not rewrite the source fine residual cluster output.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


STATUS_IDS = {
    "other": 0,
    "fine_object_candidate": 1,
    "manual_review": 2,
    "pre_fusion_split_or_demote": 3,
}


def read_header(path: Path) -> tuple[list[str], list[str], int]:
    props: list[str] = []
    header: list[str] = []
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header.append(line)
            s = line.strip()
            if s.startswith("element vertex"):
                in_vertex = True
            elif s.startswith("element "):
                in_vertex = False
            elif in_vertex and s.startswith("property"):
                props.append(s.split()[-1])
            elif s == "end_header":
                break
    return header, props, len(header)


def load_actions(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    actions = {}
    for row in raw.get("clusters", []):
        actions[int(row["cluster_id"])] = str(row.get("recommended_action", "other"))
    return actions


def write_status_header(out, input_header: list[str]) -> None:
    for line in input_header:
        if line.strip() == "end_header":
            out.write("property uchar cluster_status\n")
            out.write("end_header\n")
        else:
            out.write(line)


def write_filtered_header(out, input_header: list[str], vertex_count: int) -> None:
    for line in input_header:
        if line.startswith("element vertex"):
            out.write(f"element vertex {vertex_count}\n")
        else:
            out.write(line)


def process(args: argparse.Namespace) -> dict:
    header, props, header_lines = read_header(args.cluster_ply)
    idx = {name: i for i, name in enumerate(props)}
    if "cluster" not in idx:
        raise ValueError(f"missing cluster field in {args.cluster_ply}: {props}")
    actions = load_actions(args.hygiene_eval_json)
    demoted = {cluster_id for cluster_id, action in actions.items() if action == "pre_fusion_split_or_demote"}

    status_counts = Counter()
    cluster_counts = Counter()
    kept_lines: list[str] = []
    total = 0
    kept = 0
    demoted_points = 0
    args.output_status_ply.parent.mkdir(parents=True, exist_ok=True)
    with args.cluster_ply.open("r", encoding="utf-8", errors="replace") as src, args.output_status_ply.open(
        "w", encoding="utf-8"
    ) as status_out:
        for _ in range(header_lines):
            next(src)
        write_status_header(status_out, header)
        for line in src:
            parts = line.split()
            if not parts:
                continue
            total += 1
            cluster_id = int(float(parts[idx["cluster"]]))
            action = actions.get(cluster_id, "other")
            status_id = STATUS_IDS.get(action, 0)
            status_counts[action] += 1
            cluster_counts[cluster_id] += 1
            status_out.write(line.rstrip("\n") + f" {status_id}\n")
            if cluster_id in demoted:
                demoted_points += 1
            else:
                kept += 1
                kept_lines.append(line)

    args.output_filtered_ply.parent.mkdir(parents=True, exist_ok=True)
    with args.output_filtered_ply.open("w", encoding="utf-8") as f:
        write_filtered_header(f, header, len(kept_lines))
        for line in kept_lines:
            f.write(line)

    report = {
        "cluster_ply": str(args.cluster_ply),
        "hygiene_eval_json": str(args.hygiene_eval_json),
        "output_status_ply": str(args.output_status_ply),
        "output_filtered_ply": str(args.output_filtered_ply),
        "total_points": int(total),
        "kept_points": int(kept),
        "demoted_points": int(demoted_points),
        "demoted_ratio": float(demoted_points / max(total, 1)),
        "status_counts": dict(status_counts),
        "demoted_clusters": sorted(int(x) for x in demoted),
        "cluster_counts": {str(k): int(v) for k, v in sorted(cluster_counts.items())},
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-ply", type=Path, required=True)
    parser.add_argument("--hygiene-eval-json", type=Path, required=True)
    parser.add_argument("--output-status-ply", type=Path, required=True)
    parser.add_argument("--output-filtered-ply", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    report = process(args)
    print(
        json.dumps(
            {
                "total_points": report["total_points"],
                "kept_points": report["kept_points"],
                "demoted_points": report["demoted_points"],
                "demoted_ratio": report["demoted_ratio"],
                "status_counts": report["status_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
