#!/usr/bin/env python3
"""Trace fine residual clusters back to source 2D masks.

Fine residual cluster PLYs intentionally stay small and only contain geometry,
semantic id, cluster id, and visual RGB. This script joins those clustered
points back to per-frame residual PLYs, which still contain frame/camera/mask
metadata, then writes a review table plus optional mask overlays.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def read_ply(path: Path) -> tuple[list[str], np.ndarray]:
    props: list[str] = []
    vertex_count = 0
    header_lines = 0
    in_vertex = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header_lines += 1
            s = line.strip()
            if s.startswith("element vertex"):
                vertex_count = int(s.split()[-1])
                in_vertex = True
            elif s.startswith("element "):
                in_vertex = False
            elif in_vertex and s.startswith("property"):
                props.append(s.split()[-1])
            elif s == "end_header":
                break
    if vertex_count == 0:
        return props, np.empty((0, len(props)), dtype=np.float64)
    data = np.loadtxt(path, skiprows=header_lines, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return props, data


def load_labels(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "labels" in raw:
        raw = raw["labels"]
    labels: dict[int, str] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                mask_id = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                labels[mask_id] = str(value.get("label") or value.get("name") or "unknown")
            else:
                labels[mask_id] = str(value)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                mask_id = int(item.get("id", item.get("mask_id", 0)))
                labels[mask_id] = str(item.get("label") or item.get("name") or "unknown")
    return labels


def point_key(row: np.ndarray, idx: dict[str, int], scale: int) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(round(float(row[idx["x"]]) * scale)),
        int(round(float(row[idx["y"]]) * scale)),
        int(round(float(row[idx["z"]]) * scale)),
        int(round(float(row[idx["semantic"]]))),
        int(round(float(row[idx["visual_red"]]))),
        int(round(float(row[idx["visual_green"]]))),
        int(round(float(row[idx["visual_blue"]]))),
    )


def artifact_dir(base: Path, combo: str, cam_id: int, frame_id: int) -> Path:
    return base / "images" / f"cam{cam_id}_{frame_id:06d}" / combo


def make_overlay(image_path: Path, instance_path: Path, mask_id: int, label: str, out_path: Path) -> bool:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    instance = cv2.imread(str(instance_path), cv2.IMREAD_UNCHANGED)
    if image is None or instance is None:
        return False
    if instance.ndim == 3:
        instance = instance[:, :, 0]
    mask = instance.astype(np.int64) == int(mask_id)
    if not np.any(mask):
        return False
    overlay = image.copy()
    color = np.array([0, 255, 255], dtype=np.uint8)
    overlay[mask] = (0.45 * overlay[mask] + 0.55 * color).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    text = f"mask {mask_id} {label}"
    cv2.rectangle(overlay, (8, 8), (min(8 + 18 * len(text), overlay.shape[1] - 8), 42), (0, 0, 0), -1)
    cv2.putText(overlay, text, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), overlay))


def make_contact_sheet(paths: list[Path], output_path: Path, thumb_width: int = 420) -> None:
    images = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        scale = thumb_width / max(img.shape[1], 1)
        thumb = cv2.resize(img, (thumb_width, max(1, int(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        images.append(thumb)
    if not images:
        return
    cols = min(3, len(images))
    rows = int(np.ceil(len(images) / cols))
    h = max(img.shape[0] for img in images)
    sheet = np.zeros((rows * h, cols * thumb_width, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        sheet[r * h : r * h + img.shape[0], c * thumb_width : c * thumb_width + img.shape[1]] = img
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-ply", type=Path, required=True)
    parser.add_argument("--review-json", type=Path, required=True)
    parser.add_argument("--residual-dir", type=Path, required=True)
    parser.add_argument("--semantic-eval-dir", type=Path, required=True)
    parser.add_argument("--combo", default="sam2_prompt_v3_sky_label_merge_completion")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--statuses", nargs="+", default=["review_suspicious"])
    parser.add_argument("--top-sources-per-cluster", type=int, default=8)
    parser.add_argument("--overlay-limit", type=int, default=36)
    parser.add_argument("--coord-scale", type=int, default=1000000)
    args = parser.parse_args()

    review = json.loads(args.review_json.read_text(encoding="utf-8"))
    wanted_clusters = {
        int(row["cluster_id"])
        for row in review.get("review_rows", [])
        if row.get("status") in set(args.statuses)
    }
    props, data = read_ply(args.cluster_ply)
    idx = {name: i for i, name in enumerate(props)}
    if not {"x", "y", "z", "semantic", "cluster", "visual_red", "visual_green", "visual_blue"}.issubset(idx):
        raise ValueError(f"missing required cluster PLY properties in {args.cluster_ply}: {props}")

    key_to_clusters: dict[tuple[int, int, int, int, int, int, int], list[int]] = defaultdict(list)
    cluster_point_counts = Counter()
    for row in data:
        cluster_id = int(round(float(row[idx["cluster"]])))
        if cluster_id not in wanted_clusters:
            continue
        key_to_clusters[point_key(row, idx, args.coord_scale)].append(cluster_id)
        cluster_point_counts[cluster_id] += 1

    source_counts: dict[int, Counter] = {cluster_id: Counter() for cluster_id in wanted_clusters}
    matched_points = Counter()
    residual_files = sorted(args.residual_dir.glob("residuals_frame_*.ply"))
    for residual_path in residual_files:
        rprops, rdata = read_ply(residual_path)
        ridx = {name: i for i, name in enumerate(rprops)}
        required = {"x", "y", "z", "semantic", "frame", "camera", "mask", "visual_red", "visual_green", "visual_blue"}
        if not required.issubset(ridx):
            continue
        for row in rdata:
            clusters = key_to_clusters.get(point_key(row, ridx, args.coord_scale))
            if not clusters:
                continue
            source = (
                int(round(float(row[ridx["frame"]]))),
                int(round(float(row[ridx["camera"]]))),
                int(round(float(row[ridx["mask"]]))),
            )
            for cluster_id in clusters:
                source_counts[cluster_id][source] += 1
                matched_points[cluster_id] += 1

    rows = []
    overlays: list[Path] = []
    overlay_count = 0
    for cluster_id in sorted(wanted_clusters):
        total = int(cluster_point_counts[cluster_id])
        matched = min(int(matched_points[cluster_id]), total)
        sources = []
        for (frame_id, cam_id, mask_id), count in source_counts[cluster_id].most_common(args.top_sources_per_cluster):
            combo_dir = artifact_dir(args.semantic_eval_dir, args.combo, cam_id, frame_id)
            labels = load_labels(combo_dir / "labels.json")
            label = labels.get(mask_id, "unknown")
            image_path = combo_dir / "image.png"
            instance_path = combo_dir / "instance.png"
            overlay_path = args.output_dir / "overlays" / f"cluster_{cluster_id:04d}_f{frame_id:04d}_cam{cam_id}_m{mask_id:04d}.png"
            wrote_overlay = False
            if overlay_count < args.overlay_limit:
                wrote_overlay = make_overlay(image_path, instance_path, mask_id, label, overlay_path)
                if wrote_overlay:
                    overlays.append(overlay_path)
                    overlay_count += 1
            source_row = {
                "frame_id": frame_id,
                "cam_id": cam_id,
                "mask_id": mask_id,
                "label": label,
                "points": int(count),
                "share_of_cluster": float(count / max(total, 1)),
                "image_path": str(image_path),
                "instance_path": str(instance_path),
                "labels_path": str(combo_dir / "labels.json"),
                "overlay_path": str(overlay_path) if wrote_overlay else "",
            }
            sources.append(source_row)
            rows.append({"cluster_id": cluster_id, "cluster_points": total, "matched_points": matched, **source_row})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_matched_points = int(sum(matched_points.values()))
    cluster_points = int(sum(cluster_point_counts.values()))
    capped_matched_points = int(
        sum(min(int(matched_points[cluster_id]), int(cluster_point_counts[cluster_id])) for cluster_id in wanted_clusters)
    )
    summary = {
        "cluster_ply": str(args.cluster_ply),
        "review_json": str(args.review_json),
        "residual_dir": str(args.residual_dir),
        "semantic_eval_dir": str(args.semantic_eval_dir),
        "combo": args.combo,
        "statuses": args.statuses,
        "cluster_count": len(wanted_clusters),
        "cluster_points": cluster_points,
        "matched_points": capped_matched_points,
        "raw_matched_points": raw_matched_points,
        "matched_ratio": float(capped_matched_points / max(cluster_points, 1)),
        "duplicate_match_points": int(max(0, raw_matched_points - capped_matched_points)),
        "clusters": [
            {
                "cluster_id": int(cluster_id),
                "cluster_points": int(cluster_point_counts[cluster_id]),
                "matched_points": int(min(matched_points[cluster_id], cluster_point_counts[cluster_id])),
                "raw_matched_points": int(matched_points[cluster_id]),
                "matched_ratio": float(
                    min(matched_points[cluster_id], cluster_point_counts[cluster_id])
                    / max(cluster_point_counts[cluster_id], 1)
                ),
                "top_sources": [
                    {
                        "frame_id": int(frame_id),
                        "cam_id": int(cam_id),
                        "mask_id": int(mask_id),
                        "points": int(count),
                    }
                    for (frame_id, cam_id, mask_id), count in source_counts[cluster_id].most_common(args.top_sources_per_cluster)
                ],
            }
            for cluster_id in sorted(wanted_clusters)
        ],
    }
    (args.output_dir / "fine_cluster_mask_trace.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "fine_cluster_mask_trace.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "cluster_id",
            "cluster_points",
            "matched_points",
            "frame_id",
            "cam_id",
            "mask_id",
            "label",
            "points",
            "share_of_cluster",
            "image_path",
            "instance_path",
            "labels_path",
            "overlay_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    make_contact_sheet(overlays, args.output_dir / "fine_cluster_mask_trace_contact_sheet.png")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in [
                    "cluster_count",
                    "cluster_points",
                    "matched_points",
                    "raw_matched_points",
                    "matched_ratio",
                    "duplicate_match_points",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
