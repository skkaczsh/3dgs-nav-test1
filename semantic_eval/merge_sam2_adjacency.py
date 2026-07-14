#!/usr/bin/env python3
"""Merge SAM2 masks by sky exclusion and semantic adjacency.

This is a fast experiment for the high-coverage route:
SAM2 dense masks -> Qwen labels -> SkyMask exclusion -> connected components
within the same semantic label.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from run_eval import LABEL_TO_ID, LABEL_ALIASES, write_combo_artifacts, Mask


ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
MERGE_LABELS = {
    "other",
    "wall",
    "floor",
    "ceiling",
    "grass",
    "tree",
    "person",
    "car",
    "railing",
    "building",
    "road",
    "water",
    "furniture",
    "pipe",
    "equipment",
}


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    import cv2
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return [labels == i for i in range(1, n)]


def load_summary(combo_dir: Path) -> dict[str, Any]:
    return json.loads((combo_dir / "summary.json").read_text())


def normalize_label(raw: str) -> str:
    value = str(raw or "").strip().lower()
    if value in LABEL_ALIASES:
        return LABEL_ALIASES[value]
    for key, label in LABEL_ALIASES.items():
        if key.lower() in value:
            return label
    return "other"


def merge_one(image_dir: Path, source_combo: str, output_combo: str, min_area: int) -> dict[str, Any]:
    src = image_dir / source_combo
    out = image_dir / output_combo
    image = np.array(Image.open(src / "image.png").convert("RGB"))
    sky = np.array(Image.open(src / "sky_mask.png").convert("L")) > 128
    inst = np.array(Image.open(src / "instance.png"))
    src_labels = json.loads((src / "labels.json").read_text())
    source_summary = load_summary(src)
    h, w = inst.shape

    masks: list[Mask] = []
    labels: dict[str, str] = {}
    label_components = Counter()

    label_unions: dict[str, np.ndarray] = {}
    for mask_id in sorted(int(x) for x in np.unique(inst) if int(x) > 0):
        label = normalize_label(src_labels.get(str(mask_id), "other"))
        if label not in MERGE_LABELS:
            continue
        label_unions.setdefault(label, np.zeros((h, w), dtype=bool))
        label_unions[label] |= (inst == mask_id) & (~sky)

    for label, base in sorted(label_unions.items()):
        for comp in connected_components(base):
            area = int(comp.sum())
            if area < min_area:
                continue
            ys, xs = np.where(comp)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            masks.append(Mask(comp, area, 1.0, bbox, "sam2_sky_label_merge"))
            labels[str(len(masks))] = label
            label_components[label] += 1

    masks.sort(key=lambda m: m.area, reverse=True)
    # Re-number labels after sorting by looking up the dominant source label.
    labels = {}
    for i, mask in enumerate(masks, start=1):
        votes = Counter()
        for label, union in label_unions.items():
            inter = int((mask.segmentation & union).sum())
            if inter:
                votes[label] += inter
        labels[str(i)] = votes.most_common(1)[0][0] if votes else "other"

    coverage = float(np.logical_or.reduce([m.segmentation for m in masks]).sum() / (h * w)) if masks else 0.0
    non_sky = int((~sky).sum())
    non_sky_coverage = float(np.logical_or.reduce([m.segmentation for m in masks]).sum() / max(non_sky, 1)) if masks else 0.0
    ground_non_sky = float(((build_semantic_from_masks(masks, labels, (h, w)) == LABEL_TO_ID["floor"]) & (~sky)).sum() / max(non_sky, 1))

    summary = {
        "image_id": image_dir.name,
        "combo": output_combo,
        "source_combo": source_combo,
        "blocked": False,
        "blocker": "",
        "mask_count": len(masks),
        "coverage": coverage,
        "non_sky_coverage": non_sky_coverage,
        "ground_non_sky_ratio": ground_non_sky,
        "sky_source": source_summary.get("sky_source", ""),
        "sky_mask_ratio": float(sky.sum() / (h * w)),
        "sky_labeled_ratio": float(sky.sum() / (h * w)),
        "label_components": dict(label_components),
        "vlm": {
            "parse_ok": bool(source_summary.get("vlm", {}).get("parse_ok")),
            "source": source_combo,
            "review_mode": "reuse_qwen_labels_then_merge_same_semantic_connected_components",
        },
    }
    write_combo_artifacts(out, image, masks, labels, sky, summary, mark_sky_semantic=True)
    return summary


def build_semantic_from_masks(masks: list[Mask], labels: dict[str, str], shape: tuple[int, int]) -> np.ndarray:
    sem = np.zeros(shape, dtype=np.uint8)
    for i, mask in enumerate(masks, start=1):
        sem[mask.segmentation] = LABEL_TO_ID.get(labels.get(str(i), "unknown"), 0)
    return sem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/root/epfs/manifold_3dgs_project/processed/semantic_eval_20260605"))
    parser.add_argument("--source-combo", default="sam2_qwen")
    parser.add_argument("--output-combo", default="sam2_sky_label_merge_qwen_review")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Optional manifest; when set, process only image_ids listed in it.")
    parser.add_argument("--min-area", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    image_dirs = sorted(p for p in (args.output_dir / "images").iterdir() if (p / args.source_combo).exists())
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        wanted = {item["image_id"] for item in manifest.get("items", [])}
        image_dirs = [p for p in image_dirs if p.name in wanted]
    if args.limit:
        image_dirs = image_dirs[:args.limit]

    rows = []
    for image_dir in image_dirs:
        row = merge_one(image_dir, args.source_combo, args.output_combo, args.min_area)
        rows.append(row)
        print(f"{row['image_id']} {args.output_combo}: masks={row['mask_count']} cov={row['coverage']:.3f} non_sky={row['non_sky_coverage']:.3f} ground={row['ground_non_sky_ratio']:.3f}")
    report = {
        "combo": args.output_combo,
        "source_combo": args.source_combo,
        "images": len(rows),
        "avg_mask_count": float(np.mean([r["mask_count"] for r in rows])) if rows else 0.0,
        "avg_coverage": float(np.mean([r["coverage"] for r in rows])) if rows else 0.0,
        "avg_non_sky_coverage": float(np.mean([r["non_sky_coverage"] for r in rows])) if rows else 0.0,
        "avg_ground_non_sky_ratio": float(np.mean([r["ground_non_sky_ratio"] for r in rows])) if rows else 0.0,
        "rows": rows,
    }
    report_path = args.output_dir / f"{args.output_combo}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {report_path}")
    print(f"Wrote {len(rows)} merged samples")


if __name__ == "__main__":
    main()
