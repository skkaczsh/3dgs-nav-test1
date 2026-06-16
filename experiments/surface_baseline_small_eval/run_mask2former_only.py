#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

TARGET_CLASSES = ["floor_ground", "wall", "ceiling", "building", "sky"]
PALETTE = {
    "background": (0, 0, 0),
    "floor_ground": (230, 170, 40),
    "wall": (70, 130, 180),
    "ceiling": (180, 120, 220),
    "building": (220, 80, 80),
    "sky": (100, 180, 255),
}
VISUAL_IDS = ["cam0_002010", "cam0_002020", "cam1_002010", "cam1_002220", "cam2_002050"]


def normalize_label(label: str) -> str:
    return label.lower().replace("-", " ").replace("/", " ").replace(",", " ")


def map_universal_label(label: str) -> str | None:
    n = normalize_label(label)
    if "sky" in n:
        return "sky"
    if "ceiling" in n:
        return "ceiling"
    if "wall" in n:
        return "wall"
    if any(token in n for token in ["building", "edifice", "house", "skyscraper"]):
        return "building"
    if any(
        token in n
        for token in ["floor", "ground", "earth", "road", "sidewalk", "path", "runway", "dirt track"]
    ):
        return "floor_ground"
    return None


def map_baseline_label(label: str) -> str | None:
    n = normalize_label(label)
    if n in {"sky"}:
        return "sky"
    if n in {"ceiling"}:
        return "ceiling"
    if n in {"wall"}:
        return "wall"
    if n in {"building"}:
        return "building"
    if n in {"floor", "ground", "road"}:
        return "floor_ground"
    return None


def semantic_to_rgb(mapped: np.ndarray) -> Image.Image:
    rgb = np.zeros((mapped.shape[0], mapped.shape[1], 3), dtype=np.uint8)
    for idx, name in enumerate(["background"] + TARGET_CLASSES):
        rgb[mapped == idx] = PALETTE[name]
    return Image.fromarray(rgb)


def alpha_overlay(image: Image.Image, semantic_rgb: Image.Image, alpha: float = 0.45) -> Image.Image:
    return Image.blend(image.convert("RGB"), semantic_rgb.convert("RGB"), alpha)


def count_transitions(mapped: np.ndarray) -> float:
    h = (mapped[:, 1:] != mapped[:, :-1]).sum()
    v = (mapped[1:, :] != mapped[:-1, :]).sum()
    return float(h + v) / float(mapped.size)


def encode_ratios(mapped: np.ndarray) -> dict[str, float]:
    total = float(mapped.size)
    return {name: float((mapped == (idx + 1)).sum()) / total for idx, name in enumerate(TARGET_CLASSES)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.samples_dir / "sample_manifest.json").read_text())
    processor = AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-tiny-ade-semantic", use_fast=True
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-tiny-ade-semantic", use_safetensors=True
    ).to(args.device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    summary: dict[str, object] = {
        "sample_count": len(manifest["samples"]),
        "model": "mask2former_ade20k",
        "samples": [],
        "blockers": {
            "oneformer": "blocked on remote vlm_seg: transformers requires torch>=2.6 for torch.load path; current env is torch 2.5.1"
        },
    }
    agg = defaultdict(float)

    for sample in manifest["samples"]:
        image_id = sample["image_id"]
        sample_dir = args.samples_dir / image_id
        image = Image.open(sample_dir / "image.png").convert("RGB")
        baseline_overlay = Image.open(sample_dir / "baseline_overlay.png").convert("RGB")
        baseline_labels = json.loads((sample_dir / "baseline_labels.json").read_text())
        baseline_summary = json.loads((sample_dir / "baseline_summary.json").read_text())

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        pred = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[
            0
        ].cpu().numpy()

        mapped = np.zeros_like(pred, dtype=np.uint8)
        raw_counts = Counter()
        for raw_id in np.unique(pred):
            label = id2label[int(raw_id)]
            raw_counts[label] += int((pred == raw_id).sum())
            target = map_universal_label(label)
            if target is not None:
                mapped[pred == raw_id] = TARGET_CLASSES.index(target) + 1

        ratios = encode_ratios(mapped)
        transition_rate = count_transitions(mapped)
        semantic_rgb = semantic_to_rgb(mapped)
        overlay = alpha_overlay(image, semantic_rgb)

        pred_dir = args.output_dir / "predictions" / image_id
        pred_dir.mkdir(parents=True, exist_ok=True)
        semantic_rgb.save(pred_dir / "semantic_rgb.png")
        overlay.save(pred_dir / "overlay.png")
        np.save(pred_dir / "mapped_semantic.npy", mapped)

        baseline_counts = Counter()
        for label in baseline_labels.values():
            mapped_label = map_baseline_label(label)
            if mapped_label:
                baseline_counts[mapped_label] += 1

        row = {
            "image_id": image_id,
            "rationale": sample["rationale"],
            "baseline": {
                "combo": baseline_summary.get("combo"),
                "sky_mask_ratio": baseline_summary.get("sky_mask_ratio"),
                "coverage": baseline_summary.get("coverage"),
                "non_sky_coverage": baseline_summary.get("non_sky_coverage"),
                "ground_non_sky_ratio": baseline_summary.get("ground_non_sky_ratio"),
                "mask_label_counts": dict(baseline_counts),
            },
            "mask2former": {
                "target_ratios": ratios,
                "transition_rate": transition_rate,
                "raw_top_labels": dict(
                    sorted(
                        ((k, v / float(pred.size)) for k, v in raw_counts.items()),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )[:10]
                ),
            },
        }
        summary["samples"].append(row)

        agg["transition_rate"] += transition_rate
        for key, value in ratios.items():
            agg[key] += value

        if image_id in VISUAL_IDS:
            width, height = image.size
            header = 40
            panel = Image.new("RGB", (width * 3, height + header), color=(18, 18, 18))
            draw = ImageDraw.Draw(panel)
            panel.paste(image, (0, header))
            panel.paste(baseline_overlay, (width, header))
            panel.paste(overlay, (width * 2, header))
            draw.text((8, 10), "original", fill=(255, 255, 255))
            draw.text((width + 8, 10), "sam2_mimo_baseline", fill=(255, 255, 255))
            draw.text((width * 2 + 8, 10), "mask2former_ade20k", fill=(255, 255, 255))
            panel.save(args.output_dir / "visualizations" / f"{image_id}_compare.png")

    count = float(summary["sample_count"])
    summary["aggregate"] = {
        "avg_transition_rate": agg["transition_rate"] / count,
        "avg_target_ratios": {key: agg[key] / count for key in TARGET_CLASSES},
    }
    (args.output_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    lines = [
        "# Surface Baseline Small Eval",
        "",
        "- model: `facebook/mask2former-swin-tiny-ade-semantic`",
        "- dataset: `ADE20K`",
        "- blocker: `OneFormer on remote vlm_seg is blocked by transformers x torch 2.5.1 loading policy`",
        "",
        "## Aggregate",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| avg transition rate | {summary['aggregate']['avg_transition_rate']:.4f} |",
    ]
    for key, value in summary["aggregate"]["avg_target_ratios"].items():
        lines.append(f"| avg {key} ratio | {value:.4f} |")

    lines.extend(
        [
            "",
            "## Per Sample",
            "",
            "| image_id | rationale | baseline labels | mask2former ratios | raw top labels |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["samples"]:
        ratios = row["mask2former"]["target_ratios"]
        raw_top = "<br>".join(
            f"{key}:{value:.3f}" for key, value in list(row["mask2former"]["raw_top_labels"].items())[:5]
        )
        lines.append(
            "| {} | {} | {} | floor {:.3f}, wall {:.3f}, ceiling {:.3f}, building {:.3f}, sky {:.3f} | {} |".format(
                row["image_id"],
                row["rationale"],
                ", ".join(sorted(row["baseline"]["mask_label_counts"].keys())) or "-",
                ratios["floor_ground"],
                ratios["wall"],
                ratios["ceiling"],
                ratios["building"],
                ratios["sky"],
                raw_top,
            )
        )
    (args.output_dir / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
