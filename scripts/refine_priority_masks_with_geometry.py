#!/usr/bin/env python3
"""Refine priority masks with projected depth and trusted semantic priors.

This is a conservative prototype for geometry-guided 2D segmentation:

- keep the original segmentation model as a candidate generator
- use trusted projected 3D surface labels to fill residual surface holes
- report fine-object / surface-prior conflicts without deleting fine targets by default
- optionally cut car/railing masks at strong depth edges for diagnostics only

It writes refined priority PNGs plus review contact sheets so the effect can be
judged visually before any full-scene production run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PRIORITY_NAMES = {
    0: "residual",
    1: "ground",
    2: "wall",
    3: "grass",
    4: "car",
    5: "railing",
    6: "sky",
}

PRIORITY_COLORS = {
    0: (30, 30, 30),
    1: (196, 168, 112),
    2: (120, 150, 180),
    3: (80, 160, 80),
    4: (235, 90, 80),
    5: (240, 210, 60),
    6: (90, 170, 235),
}

SEMANTIC_TO_PRIORITY = {
    2: 2,   # wall
    3: 1,   # floor
    4: 2,   # ceiling/overhead surface uses wall-like surface guard in image space
    5: 3,   # grass
    8: 4,   # car
    9: 5,   # railing
}

FINE_PRIORITY = {4, 5}
SURFACE_PRIORITY = {1, 2, 3}


def priority_path(base: Path, cam_id: int, frame_id: int, suffix: str = "_priority") -> Path:
    return base / "priority" / f"cam{cam_id}_{frame_id:06d}{suffix}.png"


def frame_path(base: Path, cam_id: int, frame_id: int) -> Path:
    return base / f"cam{cam_id}" / f"frame_{frame_id:06d}.jpg"


def geometry_path(base: Path, cam_id: int, frame_id: int) -> Path:
    return base / "maps" / f"cam{cam_id}_{frame_id:06d}_geometry.npz"


def colorize_priority(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, color in PRIORITY_COLORS.items():
        rgb[mask == label] = color
    return rgb


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    rgb = colorize_priority(mask)[:, :, ::-1]
    return cv2.addWeighted(image, 1.0 - alpha, rgb, alpha, 0.0)


def semantic_to_priority_map(semantic: np.ndarray) -> np.ndarray:
    out = np.zeros(semantic.shape, dtype=np.uint8)
    for sem, pri in SEMANTIC_TO_PRIORITY.items():
        out[semantic == sem] = pri
    return out


def component_count(mask: np.ndarray, min_area: int) -> int:
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    count = 0
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            count += 1
    return count


def count_priority_pixels(mask: np.ndarray) -> dict[str, int]:
    counts = np.bincount(mask.reshape(-1), minlength=max(PRIORITY_NAMES) + 1)
    return {
        PRIORITY_NAMES.get(int(label), str(int(label))): int(count)
        for label, count in enumerate(counts)
        if count
    }


def guarded_fine_surface_override(
    refined: np.ndarray,
    prior_priority: np.ndarray,
    surface_prior: np.ndarray,
    valid_bool: np.ndarray,
    args: argparse.Namespace,
) -> tuple[int, Counter[str], list[dict[str, Any]]]:
    """Recover obvious surface pixels from oversized fine-label components.

    The model sometimes labels large wall/ground regions as railing/car.  A
    blanket overwrite is too aggressive, so this only acts on connected fine
    components whose projected surface-prior overlap is both large enough and
    internally dominated by one surface class.
    """
    if not args.guarded_fine_surface_override:
        return 0, Counter(), []

    support_priority = prior_priority.copy()
    support_mask = surface_prior.copy()
    if args.fine_surface_neighbor_radius > 0:
        kernel_size = args.fine_surface_neighbor_radius * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        class_counts = []
        for surface_label in sorted(SURFACE_PRIORITY):
            class_map = ((prior_priority == surface_label) & surface_prior).astype(np.uint8)
            class_counts.append(cv2.filter2D(class_map, -1, kernel, borderType=cv2.BORDER_CONSTANT))
        counts = np.stack(class_counts, axis=0)
        best_index = np.argmax(counts, axis=0)
        best_count = np.max(counts, axis=0)
        labels = np.array(sorted(SURFACE_PRIORITY), dtype=np.uint8)
        neighbor_support = best_count >= args.fine_surface_neighbor_min_support
        support_priority = np.where(neighbor_support, labels[best_index], support_priority).astype(np.uint8)
        support_mask = support_mask | neighbor_support

    total = 0
    pairs: Counter[str] = Counter()
    components: list[dict[str, Any]] = []
    for label in sorted(FINE_PRIORITY):
        component_mask = refined == label
        n, labels, stats, _centroids = cv2.connectedComponentsWithStats(component_mask.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            component = labels == i
            projected = component & valid_bool
            projected_area = int(projected.sum())
            if projected_area <= 0:
                continue
            overlap = projected & support_mask
            overlap_area = int(overlap.sum())
            overlap_ratio = overlap_area / float(projected_area)
            if overlap_area < args.fine_surface_min_pixels or overlap_ratio < args.fine_surface_min_ratio:
                continue

            prior_vals = support_priority[overlap]
            prior_counts = Counter(int(x) for x in prior_vals.tolist())
            dominant_label, dominant_count = prior_counts.most_common(1)[0]
            dominant_ratio = dominant_count / float(overlap_area)
            if dominant_ratio < args.fine_surface_dominant_ratio:
                continue

            old_vals = refined[overlap]
            new_vals = support_priority[overlap]
            pairs.update(
                f"{PRIORITY_NAMES.get(int(o), o)}->{PRIORITY_NAMES.get(int(n), n)}"
                for o, n in zip(old_vals.tolist(), new_vals.tolist())
            )
            refined[overlap] = new_vals
            total += overlap_area
            components.append({
                "source_label": PRIORITY_NAMES.get(label, str(label)),
                "surface_label": PRIORITY_NAMES.get(dominant_label, str(dominant_label)),
                "component_area": area,
                "projected_component_area": projected_area,
                "overlap_area": overlap_area,
                "overlap_ratio": overlap_ratio,
                "surface_dominant_ratio": dominant_ratio,
            })
    return total, pairs, components


def refine(priority: np.ndarray, semantic: np.ndarray, valid: np.ndarray, edge: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    if semantic.shape != priority.shape:
        semantic = cv2.resize(semantic, (priority.shape[1], priority.shape[0]), interpolation=cv2.INTER_NEAREST)
    if valid.shape != priority.shape:
        valid = cv2.resize(valid, (priority.shape[1], priority.shape[0]), interpolation=cv2.INTER_NEAREST)
    if edge.shape != priority.shape:
        edge = cv2.resize(edge, (priority.shape[1], priority.shape[0]), interpolation=cv2.INTER_NEAREST)

    refined = priority.copy()
    prior_priority = semantic_to_priority_map(semantic)
    valid_bool = valid > 0
    surface_prior = valid_bool & np.isin(prior_priority, list(SURFACE_PRIORITY))
    eligible = surface_prior & np.isin(priority, list(args.surface_override_from))
    old_vals = priority[eligible]
    new_vals = prior_priority[eligible]
    override_pairs = Counter(f"{PRIORITY_NAMES.get(int(o), o)}->{PRIORITY_NAMES.get(int(n), n)}" for o, n in zip(old_vals.tolist(), new_vals.tolist()))
    refined[eligible] = prior_priority[eligible]

    guarded_pixels, guarded_pairs, guarded_components = guarded_fine_surface_override(
        refined,
        prior_priority,
        surface_prior,
        valid_bool,
        args,
    )
    override_pairs.update(guarded_pairs)

    cut_pixels = 0
    if args.cut_fine_at_depth_edge:
        fine_edge = (edge > 0) & np.isin(refined, list(FINE_PRIORITY))
        cut_pixels = int(fine_edge.sum())
        refined[fine_edge] = 0

    small_removed = 0
    if args.min_fine_component_area > 0:
        for label in FINE_PRIORITY:
            mask = refined == label
            n, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            for i in range(1, n):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < args.min_fine_component_area:
                    refined[labels == i] = 0
                    small_removed += area

    before_counts = count_priority_pixels(priority)
    after_counts = count_priority_pixels(refined)
    before_components = {
        PRIORITY_NAMES[label]: component_count(priority == label, args.component_min_area)
        for label in sorted(FINE_PRIORITY)
    }
    after_components = {
        PRIORITY_NAMES[label]: component_count(refined == label, args.component_min_area)
        for label in sorted(FINE_PRIORITY)
    }
    fine_surface_overlap = int((np.isin(priority, list(FINE_PRIORITY)) & surface_prior).sum())
    return refined, {
        "override_pixels": int(eligible.sum()),
        "override_pairs": dict(override_pairs),
        "guarded_fine_surface_override_pixels": guarded_pixels,
        "guarded_fine_surface_components": guarded_components,
        "depth_edge_cut_pixels": cut_pixels,
        "small_fine_component_removed_pixels": small_removed,
        "fine_surface_overlap_before": fine_surface_overlap,
        "priority_counts_before": before_counts,
        "priority_counts_after": after_counts,
        "fine_component_count_before": before_components,
        "fine_component_count_after": after_components,
    }


def make_review_panel(image: np.ndarray, priority: np.ndarray, refined: np.ndarray, depth_viz: np.ndarray, edge: np.ndarray, semantic: np.ndarray, alpha: float) -> np.ndarray:
    h, w = image.shape[:2]
    def resize(img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    before = overlay(image, priority, alpha)
    after = overlay(image, refined, alpha)
    edge_rgb = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    semantic_rgb = resize(colorize_priority(semantic_to_priority_map(semantic))[:, :, ::-1])
    depth_viz = resize(depth_viz)
    return np.hstack([image, before, depth_viz, edge_rgb, semantic_rgb, after])


def write_contact_sheet(panels: list[np.ndarray], output: Path, max_width: int = 1800) -> None:
    if not panels:
        return
    resized = []
    for panel in panels:
        scale = min(1.0, max_width / panel.shape[1])
        resized.append(cv2.resize(panel, (int(panel.shape[1] * scale), int(panel.shape[0] * scale))))
    width = max(p.shape[1] for p in resized)
    padded = []
    for panel in resized:
        if panel.shape[1] < width:
            pad = np.zeros((panel.shape[0], width - panel.shape[1], 3), dtype=np.uint8)
            panel = np.hstack([panel, pad])
        padded.append(panel)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.vstack(padded))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--priority-dir", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--priority-suffix", default="_priority", help="Mask filename suffix before .png")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--surface-override-from",
        type=int,
        nargs="+",
        default=[0],
        help="Priority ids that trusted projected surfaces may overwrite. Default only fills residual holes.",
    )
    parser.add_argument(
        "--cut-fine-at-depth-edge",
        action="store_true",
        default=False,
        help="Diagnostic/aggressive mode: demote fine labels on projected depth edges.",
    )
    parser.add_argument(
        "--guarded-fine-surface-override",
        action="store_true",
        default=False,
        help="Overwrite fine-label pixels only inside components strongly supported by trusted surface priors.",
    )
    parser.add_argument("--fine-surface-min-pixels", type=int, default=240)
    parser.add_argument("--fine-surface-min-ratio", type=float, default=0.35)
    parser.add_argument("--fine-surface-dominant-ratio", type=float, default=0.70)
    parser.add_argument("--fine-surface-neighbor-radius", type=int, default=0)
    parser.add_argument("--fine-surface-neighbor-min-support", type=int, default=1)
    parser.add_argument("--min-fine-component-area", type=int, default=24)
    parser.add_argument("--component-min-area", type=int, default=80)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--skip-overlays", action="store_true", help="Do not write per-image overlay JPGs; contact sheet is still written.")
    parser.add_argument("--max-review-panels", type=int, default=24)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "priority").mkdir(exist_ok=True)
    (args.output_dir / "overlay").mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    panels: list[np.ndarray] = []

    for frame_id in range(args.start, args.end + 1, max(args.stride, 1)):
        for cam_id in args.cams:
            pri_path = priority_path(args.priority_dir, cam_id, frame_id, args.priority_suffix)
            geom_path = geometry_path(args.geometry_dir, cam_id, frame_id)
            img_path = frame_path(args.frame_root, cam_id, frame_id)
            image_id = f"cam{cam_id}_{frame_id:06d}"
            if not pri_path.exists() or not geom_path.exists() or not img_path.exists():
                rows.append({
                    "image_id": image_id,
                    "frame_id": frame_id,
                    "cam_id": cam_id,
                    "status": "missing_input",
                    "priority_path": str(pri_path),
                    "geometry_path": str(geom_path),
                    "image_path": str(img_path),
                })
                continue
            priority = cv2.imread(str(pri_path), cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(str(img_path))
            geom = np.load(str(geom_path))
            semantic = geom["semantic"].astype(np.uint8)
            valid = geom["valid"].astype(np.uint8)
            edge = geom["edge"].astype(np.uint8)
            refined, stats = refine(priority, semantic, valid, edge, args)
            out_mask = args.output_dir / "priority" / f"{image_id}_priority_refined.png"
            out_overlay = args.output_dir / "overlay" / f"{image_id}_overlay_refined.jpg"
            cv2.imwrite(str(out_mask), refined)
            if not args.skip_overlays:
                cv2.imwrite(str(out_overlay), overlay(image, refined, args.overlay_alpha))
            depth_viz_path = args.geometry_dir / "depth_viz" / f"{image_id}_depth.jpg"
            depth_viz = cv2.imread(str(depth_viz_path))
            if depth_viz is None:
                depth_viz = np.zeros_like(image)
            if len(panels) < args.max_review_panels:
                panels.append(make_review_panel(image, priority, refined, depth_viz, edge, semantic, args.overlay_alpha))
            rows.append({
                "image_id": image_id,
                "frame_id": frame_id,
                "cam_id": cam_id,
                "status": "ok",
                "priority_path": str(pri_path),
                "geometry_path": str(geom_path),
                "refined_priority_path": str(out_mask),
                "refined_overlay_path": "" if args.skip_overlays else str(out_overlay),
                **stats,
            })

    status_counts = Counter(row["status"] for row in rows)
    aggregate_override = Counter()
    total_override = 0
    total_guarded = 0
    total_cut = 0
    total_overlap = 0
    for row in rows:
        if row.get("status") != "ok":
            continue
        total_override += int(row.get("override_pixels") or 0)
        total_guarded += int(row.get("guarded_fine_surface_override_pixels") or 0)
        total_cut += int(row.get("depth_edge_cut_pixels") or 0)
        total_overlap += int(row.get("fine_surface_overlap_before") or 0)
        aggregate_override.update(row.get("override_pairs", {}))
    report = {
        "frame_root": str(args.frame_root),
        "priority_dir": str(args.priority_dir),
        "geometry_dir": str(args.geometry_dir),
        "output_dir": str(args.output_dir),
        "start": args.start,
        "end": args.end,
        "stride": args.stride,
        "cams": args.cams,
        "status_counts": dict(status_counts),
        "image_count": len(rows),
        "total_surface_override_pixels": total_override,
        "total_guarded_fine_surface_override_pixels": total_guarded,
        "total_depth_edge_cut_pixels": total_cut,
        "total_fine_surface_overlap_before": total_overlap,
        "override_pairs": dict(aggregate_override),
        "items": rows,
    }
    (args.output_dir / "geometry_refine_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_contact_sheet(panels, args.output_dir / "geometry_refine_contact.jpg")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "image_count": len(rows),
        "status_counts": dict(status_counts),
        "total_surface_override_pixels": total_override,
        "total_guarded_fine_surface_override_pixels": total_guarded,
        "total_depth_edge_cut_pixels": total_cut,
        "override_pairs": dict(aggregate_override),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
