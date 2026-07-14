#!/usr/bin/env python3
"""Aggregate semantic evaluation artifacts into review metrics."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ID_TO_LABEL = {
    0: "unknown",
    1: "other",
    2: "wall",
    3: "floor",
    4: "ceiling",
    5: "grass",
    6: "tree",
    7: "person",
    8: "car",
    9: "railing",
    10: "building",
    11: "sky",
    12: "road",
    13: "water",
    14: "furniture",
    15: "pipe",
    16: "equipment",
    255: "ignore",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def summarize_combo(combo_dir: Path, small_area: int) -> dict[str, Any]:
    summary = load_json(combo_dir / "summary.json")
    inst = np.array(Image.open(combo_dir / "instance.png"))
    sem = np.array(Image.open(combo_dir / "semantic.png"))
    sky = np.array(Image.open(combo_dir / "sky_mask.png").convert("L")) > 128

    ids, counts = np.unique(inst[inst > 0], return_counts=True)
    areas = counts.astype(np.int64)
    small_fraction = float(np.mean(areas < small_area)) if len(areas) else 0.0
    median_area = float(np.median(areas)) if len(areas) else 0.0

    sky_sem = sem == 11
    sky_pixels = int(sky.sum())
    sky_labeled = int(sky_sem.sum())
    sky_recall = float((sky_sem & sky).sum() / sky_pixels) if sky_pixels else None
    sky_pollution = float((sky_sem & ~sky).sum() / sky_labeled) if sky_labeled else None

    labels = Counter()
    ids_sem, counts_sem = np.unique(sem, return_counts=True)
    for label_id, count in zip(ids_sem.tolist(), counts_sem.tolist()):
        labels[ID_TO_LABEL.get(int(label_id), f"id_{label_id}")] += int(count)

    return {
        "blocked": bool(summary.get("blocked")),
        "blocker": summary.get("blocker", ""),
        "mask_count": int(summary.get("mask_count", len(areas))),
        "coverage": float(summary.get("coverage", 0.0)),
        "vlm_parse_ok": bool(summary.get("vlm", {}).get("parse_ok")),
        "small_mask_fraction": small_fraction,
        "median_mask_area": median_area,
        "sky_mask_ratio": float(summary.get("sky_mask_ratio", sky_pixels / max(sem.size, 1))),
        "sky_label_recall": sky_recall,
        "sky_label_pollution": sky_pollution,
        "label_pixels": dict(labels),
    }


def mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return float(np.mean(clean)) if clean else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize semantic eval artifacts")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/root/epfs/manifold_3dgs_project/processed/semantic_eval_20260605"))
    parser.add_argument("--small-area", type=int, default=2000)
    args = parser.parse_args()

    images_dir = args.output_dir / "images"
    combo_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image_dir in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        for combo_dir in sorted(p for p in image_dir.iterdir() if p.is_dir()):
            row = summarize_combo(combo_dir, args.small_area)
            row["image_id"] = image_dir.name
            row["combo"] = combo_dir.name
            combo_rows[combo_dir.name].append(row)

    report: dict[str, Any] = {"combos": {}, "small_area": args.small_area}
    for combo, rows in sorted(combo_rows.items()):
        label_totals: Counter[str] = Counter()
        for row in rows:
            label_totals.update(row["label_pixels"])
        mask_positive_rows = [row for row in rows if row["mask_count"] > 0]
        report["combos"][combo] = {
            "images": len(rows),
            "blocked_images": sum(1 for row in rows if row["blocked"]),
            "zero_mask_images": sum(1 for row in rows if row["mask_count"] == 0),
            "vlm_parse_success_rate": mean([float(row["vlm_parse_ok"]) for row in rows]),
            "mask_positive_vlm_parse_success_rate": mean([float(row["vlm_parse_ok"]) for row in mask_positive_rows]),
            "avg_mask_count": mean([row["mask_count"] for row in rows]),
            "avg_coverage": mean([row["coverage"] for row in rows]),
            "avg_small_mask_fraction": mean([row["small_mask_fraction"] for row in rows]),
            "avg_median_mask_area": mean([row["median_mask_area"] for row in rows]),
            "avg_sky_mask_ratio": mean([row["sky_mask_ratio"] for row in rows]),
            "avg_sky_label_recall": mean([row["sky_label_recall"] for row in rows]),
            "avg_sky_label_pollution": mean([row["sky_label_pollution"] for row in rows]),
            "label_pixels": dict(label_totals),
        }

    out = args.output_dir / "analysis_summary.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
