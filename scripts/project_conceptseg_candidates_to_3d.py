#!/usr/bin/env python3
"""Project accepted ConceptSeg candidates to 3D components.

ConceptSeg outputs a 3-panel visualization. The right panel contains the red
candidate overlay at a resized image scale. This script uses the same accepted
candidate rows produced by build_conceptseg_integration_plan.py, intersects the
right-panel red mask with the source SAM2/Qwen instance mask, projects visible
frame points through the validated scanner route, and writes 3D connected
components for review.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from build_targets_from_masks import (
    connected_components,
    read_colored_ply,
    summarize_points,
    transform_project,
)
from project_color import load_ply_xyz
from project_semantic import LABEL_COLORS, LABEL_NAMES, zbuffer_visible_indices


LABEL_IDS = {v: k for k, v in LABEL_NAMES.items()}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def conceptseg_red_panel(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.int16)
    h, w, _ = arr.shape
    panel = arr[:, (2 * w) // 3 :, :]
    red = panel[:, :, 0]
    green = panel[:, :, 1]
    blue = panel[:, :, 2]
    return (red > 110) & (red > green + 20) & (red > blue + 20)


def load_instance(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def resolve_conceptseg_output(row: dict[str, Any], output_dirs: list[Path]) -> Path:
    name = Path(str(row.get("output_path", ""))).name
    for output_dir in output_dirs:
        candidate = output_dir / name
        if candidate.exists():
            return candidate
    direct = Path(str(row.get("output_path", "")))
    if direct.exists():
        return direct
    return output_dirs[0] / name if output_dirs else direct


def resolve_asset_path(row: dict[str, Any], key: str) -> Path:
    for asset_group in ("local_assets", "remote_assets"):
        path = Path(str((row.get(asset_group) or {}).get(key, "")))
        if path.exists():
            return path
    return Path(str((row.get("local_assets") or row.get("remote_assets") or {}).get(key, "")))


def load_points_for_frame(frame_id: int, args: argparse.Namespace, config) -> tuple[np.ndarray, np.ndarray, str]:
    color_path = args.color_dir / f"frame_{frame_id:04d}.ply" if args.color_dir else Path()
    if args.color_dir and color_path.exists():
        points, colors = read_colored_ply(color_path)
        return points, colors, str(color_path)
    raw_path = Path(config.EXTRACTED_DIR) / f"section_{frame_id:04d}.ply"
    if not raw_path.exists():
        raise FileNotFoundError(str(raw_path))
    points = load_ply_xyz(str(raw_path)).astype(np.float32)
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    return points, colors, str(raw_path)


def stable_color(class_name: str) -> tuple[int, int, int]:
    semantic_id = LABEL_IDS.get(class_name)
    if semantic_id in LABEL_COLORS:
        return LABEL_COLORS[int(semantic_id)]
    fallback = {
        "pipe": (255, 165, 0),
        "railing": (255, 210, 40),
        "equipment": (255, 0, 255),
    }
    return fallback.get(class_name, (255, 80, 80))


def write_components_ply(path: Path, components: list[dict[str, Any]]) -> None:
    total = sum(len(row["points"]) for row in components)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {total}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uchar visual_red\nproperty uchar visual_green\nproperty uchar visual_blue\n")
        f.write("property int proposal\nproperty uchar semantic\n")
        f.write("property int frame\nproperty int camera\nproperty int mask\n")
        f.write("end_header\n")
        for row in components:
            color = stable_color(str(row["concept_class"]))
            semantic_id = int(LABEL_IDS.get(str(row["concept_class"]), LABEL_IDS.get(str(row["source_label"]), 0)))
            for point, visual in zip(row["points"], row["visual_colors"]):
                f.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{color[0]} {color[1]} {color[2]} "
                    f"{int(visual[0])} {int(visual[1])} {int(visual[2])} "
                    f"{row['proposal_index']} {semantic_id} {row['frame']} {row['cam']} {row['mask']}\n"
                )


def process_candidate(
    row: dict[str, Any],
    args: argparse.Namespace,
    config,
    output_dirs: list[Path],
    frame_cache: dict[int, tuple[np.ndarray, np.ndarray, str]],
    proposal_index_base: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame_id = int(row["frame"])
    cam_id = int(row["cam"])
    mask_id = int(row["mask"])
    if frame_id not in frame_cache:
        try:
            frame_cache[frame_id] = load_points_for_frame(frame_id, args, config)
        except FileNotFoundError as exc:
            return [], {**row, "status": "missing_points", "errors": [str(exc)]}
    points, colors, point_source = frame_cache[frame_id]
    concept_path = resolve_conceptseg_output(row, output_dirs)
    instance_path = resolve_asset_path(row, "instance")
    errors: list[str] = []
    if not concept_path.exists():
        errors.append(f"missing_conceptseg_output:{concept_path}")
    if not instance_path.exists():
        errors.append(f"missing_instance:{instance_path}")
    if errors:
        return [], {**row, "status": "error", "errors": errors}

    candidate = conceptseg_red_panel(concept_path)
    panel_h, panel_w = candidate.shape
    instance = load_instance(instance_path)
    img_h, img_w = instance.shape[:2]

    projected = transform_project(points, frame_id, cam_id, config, args.min_depth)
    if projected is None:
        return [], {**row, "status": "no_projection", "errors": []}
    point_idx, u, v, depths = projected
    in_img = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    if not np.any(in_img):
        return [], {**row, "status": "no_points_in_image", "errors": []}
    point_idx = point_idx[in_img]
    u = u[in_img]
    v = v[in_img]
    depths = depths[in_img]
    uu = np.clip(np.rint(u).astype(np.int32), 0, img_w - 1)
    vv = np.clip(np.rint(v).astype(np.int32), 0, img_h - 1)
    if args.zbuffer_visible:
        visible = zbuffer_visible_indices(point_idx, np.column_stack([uu, vv]), depths, img_w)
        point_idx, uu, vv, u, v = point_idx[visible], uu[visible], vv[visible], u[visible], v[visible]
    if len(point_idx) == 0:
        return [], {**row, "status": "no_visible_points", "errors": []}

    panel_u = np.clip(np.rint(u * panel_w / img_w).astype(np.int32), 0, panel_w - 1)
    panel_v = np.clip(np.rint(v * panel_h / img_h).astype(np.int32), 0, panel_h - 1)
    in_instance = instance[vv, uu].astype(np.int64) == mask_id
    in_candidate = candidate[panel_v, panel_u]
    selected = point_idx[in_instance & in_candidate]
    if len(selected) == 0:
        return [], {
            **row,
            "status": "no_candidate_points",
            "errors": [],
            "visible_points": int(len(point_idx)),
            "candidate_pixels": int(candidate.sum()),
        }

    comps, residual = connected_components(points[selected], args.voxel_size, args.min_component_points)
    components: list[dict[str, Any]] = []
    report_components = []
    for comp_id, comp_local in enumerate(comps):
        global_idx = selected[comp_local]
        pts = points[global_idx]
        vis = colors[global_idx]
        summary = summarize_points(pts, vis)
        proposal_index = proposal_index_base + len(components) + 1
        proposal_id = f"cs_{frame_id:04d}_cam{cam_id}_m{mask_id:04d}_{row.get('concept_class')}_c{comp_id:02d}"
        component = {
            "proposal_id": proposal_id,
            "proposal_index": int(proposal_index),
            "component_id": int(comp_id),
            "target_id": row.get("target_id"),
            "object_id": row.get("object_id"),
            "tracklet_id": row.get("tracklet_id"),
            "frame": frame_id,
            "cam": cam_id,
            "mask": mask_id,
            "source_label": row.get("source_label"),
            "concept": row.get("concept"),
            "concept_class": row.get("concept_class"),
            "answer": row.get("answer"),
            "answer_class": row.get("answer_class"),
            "point_source": point_source,
            "conceptseg_output": str(concept_path),
            "instance_path": str(instance_path),
            "point_indices": [int(x) for x in global_idx.tolist()],
            **summary,
        }
        components.append({**component, "points": pts, "visual_colors": vis})
        report_components.append(component)

    return components, {
        **row,
        "status": "ok" if components else "small_component_only",
        "errors": [],
        "point_source": point_source,
        "conceptseg_output": str(concept_path),
        "instance_path": str(instance_path),
        "visible_points": int(len(point_idx)),
        "selected_points": int(len(selected)),
        "component_count": int(len(components)),
        "component_points": int(sum(len(c["points"]) for c in components)),
        "small_component_residual_points": int(residual.sum()),
        "components": report_components,
    }


def strip_arrays(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in {"points", "visual_colors"}}


def build_refinement(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config

    rows = read_jsonl(args.accepted_candidates)
    output_dirs = [Path(p) for p in args.conceptseg_output_dirs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    components: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    frame_cache: dict[int, tuple[np.ndarray, np.ndarray, str]] = {}
    for i, row in enumerate(rows):
        comps, report = process_candidate(
            row,
            args,
            config,
            output_dirs,
            frame_cache,
            args.proposal_index_base + i * args.proposal_index_stride,
        )
        components.extend(comps)
        reports.append(report)
        print(
            f"candidate={i} frame={row.get('frame')} cam={row.get('cam')} "
            f"concept={row.get('concept_class')} status={report.get('status')} "
            f"selected={report.get('selected_points', 0)} comps={report.get('component_count', 0)}"
        )

    component_rows = [strip_arrays(row) for row in components]
    component_jsonl = args.output_dir / "conceptseg_3d_components.jsonl"
    component_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in component_rows),
        encoding="utf-8",
    )
    if args.write_ply:
        write_components_ply(args.output_dir / "conceptseg_3d_components.ply", components)

    status_counts = Counter(str(row.get("status")) for row in reports)
    concept_counts = Counter(str(row.get("concept_class")) for row in component_rows)
    report = {
        "accepted_candidates": len(rows),
        "candidate_status_counts": dict(sorted(status_counts.items())),
        "component_count": len(component_rows),
        "component_points": int(sum(row.get("cluster_size", 0) for row in component_rows)),
        "component_concept_counts": dict(sorted(concept_counts.items())),
        "output_components_jsonl": str(component_jsonl),
        "output_components_ply": str(args.output_dir / "conceptseg_3d_components.ply") if args.write_ply else "",
        "voxel_size": args.voxel_size,
        "min_component_points": args.min_component_points,
        "candidates": reports,
        "interpretation": {
            "use": "3D review proposals for fine-object split/refine only.",
            "limitation": "Components are not applied to object labels automatically.",
        },
    }
    (args.output_dir / "conceptseg_3d_refinement_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("/Users/skkac/Work/SCAN")
    parser.add_argument("--accepted-candidates", type=Path, default=root / "route_status_20260610/conceptseg_accepted_integration_candidates_20260611.jsonl")
    parser.add_argument("--conceptseg-output-dirs", nargs="+", default=[
        str(root / "server_conceptseg_fine_object_runlist_v008_outputs"),
        str(root / "server_conceptseg_fine_object_runlist_v008_outputs_full"),
    ])
    parser.add_argument("--color-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=root / "server_conceptseg_3d_refinement_v008")
    parser.add_argument("--voxel-size", type=float, default=0.08)
    parser.add_argument("--min-component-points", type=int, default=10)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--zbuffer-visible", action="store_true", default=True)
    parser.add_argument("--no-zbuffer-visible", dest="zbuffer_visible", action="store_false")
    parser.add_argument("--proposal-index-base", type=int, default=2000000)
    parser.add_argument("--proposal-index-stride", type=int, default=1000)
    parser.add_argument("--write-ply", action="store_true")
    args = parser.parse_args()

    report = build_refinement(args)
    print(json.dumps({k: report[k] for k in ["accepted_candidates", "candidate_status_counts", "component_count", "component_points", "component_concept_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
