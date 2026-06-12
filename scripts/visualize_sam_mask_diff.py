#!/usr/bin/env python3
"""Visualize SAM mask union differences between baseline and candidate dirs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from compare_sam_mask_dirs import load_masks


def union_mask(masks: list[np.ndarray]) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), dtype=bool)
    out = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        out |= mask
    return out


def make_diff_rgb(baseline: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if baseline.shape != candidate.shape:
        raise ValueError(f"shape mismatch: baseline={baseline.shape}, candidate={candidate.shape}")
    rgb = np.zeros((*baseline.shape, 3), dtype=np.uint8)
    both = baseline & candidate
    extra = candidate & ~baseline
    missing = baseline & ~candidate
    rgb[both] = (180, 180, 180)
    rgb[extra] = (255, 0, 255)
    rgb[missing] = (255, 220, 0)
    return rgb


def blend_overlay(image_path: Path, diff_rgb: np.ndarray) -> Image.Image:
    base = Image.open(image_path).convert("RGB").resize((diff_rgb.shape[1], diff_rgb.shape[0]))
    base_arr = np.asarray(base).astype(np.float32)
    mask = diff_rgb.any(axis=2)
    out = base_arr.copy()
    out[mask] = base_arr[mask] * 0.45 + diff_rgb[mask].astype(np.float32) * 0.55
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--image-id", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--min-area", type=int, default=500)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_id in args.image_id:
        baseline_path = args.baseline_dir / f"{image_id}_sam_masks.json"
        candidate_path = args.candidate_dir / f"{image_id}_sam_masks.json"
        baseline = union_mask(load_masks(baseline_path, args.min_area))
        candidate = union_mask(load_masks(candidate_path, args.min_area))
        diff_rgb = make_diff_rgb(baseline, candidate)

        extra = candidate & ~baseline
        missing = baseline & ~candidate
        both = baseline & candidate
        total = max(int(baseline.size), 1)
        row = {
            "image_id": image_id,
            "baseline_coverage": float(baseline.sum() / total),
            "candidate_coverage": float(candidate.sum() / total),
            "extra_ratio": float(extra.sum() / total),
            "missing_ratio": float(missing.sum() / total),
            "intersection_ratio": float(both.sum() / total),
        }
        rows.append(row)

        Image.fromarray(diff_rgb, mode="RGB").save(args.output_dir / f"{image_id}_union_diff.png")
        if args.image_dir:
            image_path = args.image_dir / f"{image_id}.png"
            if image_path.exists():
                blend_overlay(image_path, diff_rgb).save(args.output_dir / f"{image_id}_union_diff_overlay.png")

    report = {"rows": rows}
    args.output_dir.joinpath("sam_mask_diff_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
