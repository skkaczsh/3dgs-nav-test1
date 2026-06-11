#!/usr/bin/env python3
"""Sweep same-label surface object consolidation parameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from consolidate_same_label_surface_objects import consolidate, load_jsonl


DEFAULT_CONFIGS = [
    "name=conservative,min_points=200,bbox=0.20,centroid=0.70,normal=8,plane=0.10,color=45",
    "name=balanced,min_points=100,bbox=0.35,centroid=1.00,normal=15,plane=0.20,color=65",
    "name=wall_building_friendly,min_points=100,bbox=0.45,centroid=1.25,normal=18,plane=0.25,color=75",
    "name=aggressive,min_points=50,bbox=0.60,centroid=1.50,normal=22,plane=0.30,color=90",
]


def parse_config(text: str) -> dict:
    out = {}
    for part in text.split(","):
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    required = {"name", "min_points", "bbox", "centroid", "normal", "plane", "color"}
    missing = required - set(out)
    if missing:
        raise ValueError(f"missing keys in config {text}: {sorted(missing)}")
    return {
        "name": out["name"],
        "min_points": int(out["min_points"]),
        "max_bbox_gap": float(out["bbox"]),
        "max_centroid_distance": float(out["centroid"]),
        "max_normal_angle": float(out["normal"]),
        "max_plane_distance": float(out["plane"]),
        "max_color_distance": float(out["color"]),
    }


def summarize_config(objects: list[dict], labels: set[str], cfg: dict) -> dict:
    args = SimpleNamespace(
        min_points=cfg["min_points"],
        max_bbox_gap=cfg["max_bbox_gap"],
        max_centroid_distance=cfg["max_centroid_distance"],
        max_normal_angle=cfg["max_normal_angle"],
        max_plane_distance=cfg["max_plane_distance"],
        max_color_distance=cfg["max_color_distance"],
    )
    _, result = consolidate(objects, labels, args)
    report = result["report"]
    label_rows = {}
    for label, row in report.get("labels", {}).items():
        source = int(row.get("source_objects", 0))
        output = int(row.get("consolidated_objects", 0))
        largest = int(row.get("largest_group_source_objects", 0))
        label_rows[label] = {
            "source_objects": source,
            "consolidated_objects": output,
            "merged_object_reduction": int(row.get("merged_object_reduction", 0)),
            "reduction_ratio": float((source - output) / max(source, 1)),
            "largest_group_source_objects": largest,
            "largest_group_points": int(row.get("largest_group_points", 0)),
        }
    return {
        "name": cfg["name"],
        "params": cfg,
        "input_objects": report["input_objects"],
        "output_objects": report["output_objects"],
        "merged_object_reduction": report["merged_object_reduction"],
        "reduction_ratio": float(report["merged_object_reduction"] / max(report["input_objects"], 1)),
        "labels": label_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=["building", "floor", "wall"])
    parser.add_argument("--config", action="append", default=DEFAULT_CONFIGS)
    args = parser.parse_args()

    objects = load_jsonl(args.objects_jsonl)
    labels = set(args.labels)
    configs = [parse_config(text) for text in args.config]
    rows = [summarize_config(objects, labels, cfg) for cfg in configs]
    report = {"objects_jsonl": str(args.objects_jsonl), "labels": sorted(labels), "configs": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
