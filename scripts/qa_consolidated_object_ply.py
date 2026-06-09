#!/usr/bin/env python3
"""Validate consolidated object QA outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def ply_vertex_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("element vertex"):
                return int(s.split()[-1])
            if s == "end_header":
                break
    raise ValueError(f"missing element vertex in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--preview", type=Path, default=None)
    parser.add_argument("--min-total-vertices", type=int, default=1)
    parser.add_argument("--min-absorbed-residual", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    errors = []
    if not args.ply.exists():
        errors.append(f"missing ply: {args.ply}")
        vertex_count = 0
    else:
        vertex_count = ply_vertex_count(args.ply)

    expected = int(report.get("total_vertices", 0))
    if vertex_count != expected:
        errors.append(f"ply vertex count mismatch: header={vertex_count} report={expected}")
    if expected < args.min_total_vertices:
        errors.append(f"total vertices below threshold: {expected} < {args.min_total_vertices}")
    point_counts = report.get("point_counts", {})
    absorbed = int(point_counts.get("absorbed_residual", 0))
    if absorbed < args.min_absorbed_residual:
        errors.append(f"absorbed residual below threshold: {absorbed} < {args.min_absorbed_residual}")
    if "object_status_counts" not in report:
        errors.append("missing object_status_counts")
    if args.preview and (not args.preview.exists() or args.preview.stat().st_size == 0):
        errors.append(f"missing or empty preview: {args.preview}")

    qa = {
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "ply": str(args.ply),
        "report": str(args.report),
        "preview": str(args.preview) if args.preview else "",
        "vertex_count": int(vertex_count),
        "absorbed_residual": absorbed,
        "object_status_counts": report.get("object_status_counts", {}),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
