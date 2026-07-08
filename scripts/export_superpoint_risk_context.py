#!/usr/bin/env python3
"""Export a local context PLY for two superpoint graph objects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COLORS = ([255, 40, 40], [0, 220, 255])
CONTEXT = [150, 150, 150]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bbox_of(obj: dict[str, Any]) -> tuple[list[float], list[float]]:
    bbox = obj.get("bbox_3d") or {}
    return [float(x) for x in bbox["min"]], [float(x) for x in bbox["max"]]


def parse_header(path: Path) -> tuple[list[str], list[str], int]:
    header: list[str] = []
    props: list[str] = []
    vertex_count = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            header.append(line)
            parts = line.strip().split()
            if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
            elif len(parts) == 3 and parts[0] == "property":
                props.append(parts[2])
            elif line.strip() == "end_header":
                break
    if not vertex_count or "object" not in props:
        raise ValueError(f"expected ASCII PLY with object field: {path}")
    return header, props, vertex_count


def in_bbox(xyz: list[float], lo: list[float], hi: list[float]) -> bool:
    return all(lo[i] <= xyz[i] <= hi[i] for i in range(3))


def fmt(value: str, prop: str, color: list[int] | None) -> str:
    if color is None:
        return value
    if prop == "red":
        return str(color[0])
    if prop == "green":
        return str(color[1])
    if prop == "blue":
        return str(color[2])
    return value


def export_context(args: argparse.Namespace) -> dict[str, Any]:
    objects = {int(o.get("object_id", o.get("object", o.get("patch_id")))): o for o in read_jsonl(args.objects)}
    missing = [pid for pid in args.patch_ids if pid not in objects]
    if missing:
        raise ValueError(f"missing patch ids in objects JSONL: {missing}")

    lows, highs = zip(*(bbox_of(objects[pid]) for pid in args.patch_ids))
    lo = [min(v[i] for v in lows) - args.padding for i in range(3)]
    hi = [max(v[i] for v in highs) + args.padding for i in range(3)]
    header, props, _vertex_count = parse_header(args.ply)
    idx = {name: i for i, name in enumerate(props)}
    target_colors = {pid: COLORS[i % len(COLORS)] for i, pid in enumerate(args.patch_ids)}
    target_counts = {str(pid): 0 for pid in args.patch_ids}
    context_count = 0
    rows: list[str] = []

    with args.ply.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip() == "end_header":
                break
        for line in fh:
            parts = line.strip().split()
            if not parts:
                continue
            xyz = [float(parts[idx[name]]) for name in ("x", "y", "z")]
            obj_id = int(float(parts[idx["object"]]))
            color = target_colors.get(obj_id)
            if color is None:
                if not in_bbox(xyz, lo, hi):
                    continue
                color = CONTEXT
                context_count += 1
            else:
                target_counts[str(obj_id)] += 1
            rows.append(" ".join(fmt(value, prop, color) for value, prop in zip(parts, props)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_ply = args.output_dir / f"{args.output_stem}.ply"
    out_jsonl = args.output_dir / f"{args.output_stem}.jsonl"
    out_report = args.output_dir / f"{args.output_stem}_report.json"

    with out_ply.open("w", encoding="utf-8") as fh:
        for line in header:
            if line.startswith("element vertex "):
                fh.write(f"element vertex {len(rows)}\n")
            else:
                fh.write(line)
        fh.write("\n".join(rows))
        fh.write("\n")

    qa_rows = []
    for i, pid in enumerate(args.patch_ids):
        row = dict(objects[pid])
        row["qa_color"] = "red" if i == 0 else "cyan"
        row["qa_stride_points"] = target_counts[str(pid)]
        qa_rows.append(row)
    qa_rows.append({
        "object_id": 0,
        "object": 0,
        "patch_id": 0,
        "status": "qa_context",
        "semantic_label": "context",
        "description": "local bbox context points recolored gray",
        "voxel_count": context_count,
    })
    out_jsonl.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in qa_rows) + "\n", encoding="utf-8")

    report = {
        "schema": "superpoint-risk-context/v1",
        "source_ply": str(args.ply),
        "source_jsonl": str(args.objects),
        "patch_ids": args.patch_ids,
        "bbox_min": lo,
        "bbox_max": hi,
        "output_ply": str(out_ply),
        "output_jsonl": str(out_jsonl),
        "output_points": len(rows),
        "target_stride_points": target_counts,
        "context_points": context_count,
    }
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--patch-ids", type=int, nargs=2, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default="risk_context")
    parser.add_argument("--padding", type=float, default=0.6)
    print(json.dumps(export_context(parser.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
