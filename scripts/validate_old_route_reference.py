#!/usr/bin/env python3
"""Validate the old-route visual/color reference package.

The old route is intentionally kept as a side-track visual reference. This
validator checks that the smoke artifact is usable for comparison without
claiming semantic correctness or reviving the deprecated 3DGS/transforms route.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_ply_header(path: Path) -> dict[str, Any]:
    header_lines: list[str] = []
    with path.open("rb") as fh:
        for raw in fh:
            line = raw.decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break
            if len(header_lines) > 200:
                raise ValueError(f"PLY header too long or missing end_header: {path}")
    if not header_lines or header_lines[0] != "ply":
        raise ValueError(f"not a PLY file: {path}")
    vertex_count = None
    properties: list[str] = []
    for line in header_lines:
        parts = line.split()
        if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
        if len(parts) >= 3 and parts[0] == "property":
            properties.append(parts[-1])
    return {
        "vertex_count": vertex_count,
        "properties": properties,
        "has_rgb": all(name in properties for name in ("red", "green", "blue")),
        "header_lines": header_lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--debug-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-colored-ratio", type=float, default=0.85)
    args = parser.parse_args()

    errors: list[str] = []
    summary = read_json(args.summary)
    ply_header = read_ply_header(args.ply)
    debug_images = sorted(str(path) for path in args.debug_dir.glob("*_world_visible_non_sky.png"))

    if not args.preview.exists():
        errors.append(f"missing preview: {args.preview}")
    if not args.ply.exists():
        errors.append(f"missing ply: {args.ply}")
    if not args.summary.exists():
        errors.append(f"missing summary: {args.summary}")
    if not debug_images:
        errors.append(f"missing debug visible images under {args.debug_dir}")
    if not ply_header["has_rgb"]:
        errors.append("PLY does not expose red/green/blue properties")
    if ply_header["vertex_count"] != summary.get("fuse", {}).get("fused_points"):
        errors.append(
            f"PLY vertex_count {ply_header['vertex_count']} != summary fused_points {summary.get('fuse', {}).get('fused_points')}"
        )
    colored_ratio = float(summary.get("colored_ratio", 0.0))
    if colored_ratio < args.min_colored_ratio:
        errors.append(f"colored_ratio {colored_ratio:.4f} < {args.min_colored_ratio:.4f}")

    report = {
        "passed": not errors,
        "summary": str(args.summary),
        "ply": str(args.ply),
        "preview": str(args.preview),
        "debug_dir": str(args.debug_dir),
        "debug_visible_images": len(debug_images),
        "ply_vertex_count": ply_header["vertex_count"],
        "ply_has_rgb": ply_header["has_rgb"],
        "colored_ratio": colored_ratio,
        "sample_mode": summary.get("sample_mode"),
        "sample_radius": summary.get("sample_radius"),
        "fusion_mode": summary.get("fusion_mode"),
        "sections": summary.get("fuse", {}).get("sections"),
        "source_points": summary.get("fuse", {}).get("source_points"),
        "fused_points": summary.get("fuse", {}).get("fused_points"),
        "color_frames": summary.get("color_frames"),
        "policy": {
            "role": "visual_color_reference_only",
            "not_semantic_source": True,
            "deprecated_route_not_promoted": "transforms.json + project_world_points",
        },
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
