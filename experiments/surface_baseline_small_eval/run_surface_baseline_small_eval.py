#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
    OneFormerForUniversalSegmentation,
    OneFormerProcessor,
)


TARGET_CLASSES = ["floor_ground", "wall", "ceiling", "building", "sky"]
PALETTE = {
    "background": (0, 0, 0),
    "floor_ground": (230, 170, 40),
    "wall": (70, 130, 180),
    "ceiling": (180, 120, 220),
    "building": (220, 80, 80),
    "sky": (100, 180, 255),
}

MODEL_SPECS = {
    "oneformer_ade20k": {
        "hf_id": "shi-labs/oneformer_ade20k_swin_tiny",
        "kind": "oneformer",
        "dataset": "ADE20K",
    },
    "mask2former_ade20k": {
        "hf_id": "facebook/mask2former-swin-tiny-ade-semantic",
        "kind": "mask2former",
        "dataset": "ADE20K",
    },
}


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
        for token in [
            "floor",
            "ground",
            "earth",
            "road",
            "sidewalk",
            "path",
            "runway",
            "dirt track",
        ]
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


def alpha_overlay(image: Image.Image, semantic_rgb: Image.Image, alpha: float = 0.45) -> Image.Image:
    return Image.blend(image.convert("RGB"), semantic_rgb.convert("RGB"), alpha)


def semantic_to_rgb(mapped: np.ndarray) -> Image.Image:
    rgb = np.zeros((mapped.shape[0], mapped.shape[1], 3), dtype=np.uint8)
    for idx, name in enumerate(["background"] + TARGET_CLASSES):
        rgb[mapped == idx] = PALETTE[name]
    return Image.fromarray(rgb)


def encode_target_mask(mapped: np.ndarray) -> Dict[str, float]:
    total = float(mapped.size)
    out: Dict[str, float] = {}
    for idx, name in enumerate(TARGET_CLASSES, start=1):
        out[name] = float((mapped == idx).sum()) / total
    return out


def count_transitions(mapped: np.ndarray) -> float:
    # Lower is smoother. Only counts transitions among target classes/background.
    h = mapped[:, 1:] != mapped[:, :-1]
    v = mapped[1:, :] != mapped[:-1, :]
    denom = mapped.size
    return float(h.sum() + v.sum()) / denom


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class LoadedModel:
    name: str
    kind: str
    hf_id: str
    processor: object
    model: torch.nn.Module
    id2label: Dict[int, str]


def load_model(name: str, device: torch.device) -> LoadedModel:
    spec = MODEL_SPECS[name]
    if spec["kind"] == "oneformer":
        processor = OneFormerProcessor.from_pretrained(spec["hf_id"])
        model = OneFormerForUniversalSegmentation.from_pretrained(
            spec["hf_id"], use_safetensors=True
        )
    else:
        processor = AutoImageProcessor.from_pretrained(spec["hf_id"])
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            spec["hf_id"], use_safetensors=True
        )
    if device.type == "mps":
        model = model.to(device=device, dtype=torch.float32)
    else:
        model = model.to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    return LoadedModel(
        name=name,
        kind=spec["kind"],
        hf_id=spec["hf_id"],
        processor=processor,
        model=model,
        id2label=id2label,
    )


def run_inference(loaded: LoadedModel, image: Image.Image, device: torch.device) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
    if loaded.kind == "oneformer":
        inputs = loaded.processor(images=image, task_inputs=["semantic"], return_tensors="pt")
        target_sizes = [image.size[::-1]]
    else:
        inputs = loaded.processor(images=image, return_tensors="pt")
        target_sizes = [image.size[::-1]]
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = loaded.model(**inputs)
    if loaded.kind == "oneformer":
        pred = loaded.processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)[0]
    else:
        pred = loaded.processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)[0]
    pred_np = pred.detach().cpu().numpy().astype(np.int32)
    mapped = np.zeros_like(pred_np, dtype=np.uint8)
    raw_counts: Counter[str] = Counter()
    for raw_id in np.unique(pred_np):
        label = loaded.id2label.get(int(raw_id), f"id_{raw_id}")
        raw_counts[label] += int((pred_np == raw_id).sum())
        target = map_universal_label(label)
        if target is None:
            continue
        mapped[pred_np == raw_id] = TARGET_CLASSES.index(target) + 1
    ratios = encode_target_mask(mapped)
    raw_ratios = {k: v / float(pred_np.size) for k, v in raw_counts.items()}
    return mapped, ratios, raw_ratios


def load_baseline_meta(sample_dir: Path) -> Dict[str, object]:
    labels = json.loads((sample_dir / "baseline_labels.json").read_text())
    summary = json.loads((sample_dir / "baseline_summary.json").read_text())
    mapped_labels = [mapped for mapped in (map_baseline_label(v) for v in labels.values()) if mapped]
    return {
        "combo": summary.get("combo"),
        "sky_mask_ratio": summary.get("sky_mask_ratio"),
        "coverage": summary.get("coverage"),
        "non_sky_coverage": summary.get("non_sky_coverage"),
        "ground_non_sky_ratio": summary.get("ground_non_sky_ratio"),
        "mask_label_counts": dict(Counter(mapped_labels)),
        "raw_mask_label_counts": dict(Counter(labels.values())),
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def compose_panel(
    image_id: str,
    image: Image.Image,
    baseline_overlay: Image.Image,
    overlays: Dict[str, Image.Image],
    model_names: Iterable[str],
) -> Image.Image:
    tiles = [("original", image), ("sam2_mimo_baseline", baseline_overlay)]
    for name in model_names:
        tiles.append((name, overlays[name]))
    width, height = image.size
    header = 40
    panel = Image.new("RGB", (width * len(tiles), height + header), color=(18, 18, 18))
    draw = ImageDraw.Draw(panel)
    for idx, (title, tile) in enumerate(tiles):
        x = idx * width
        panel.paste(tile.resize((width, height)), (x, header))
        draw.text((x + 8, 10), title, fill=(255, 255, 255))
    draw.text((8, height + 10), image_id, fill=(255, 255, 255))
    return panel


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--visualize-count", type=int, default=5)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_SPECS.keys()),
        choices=list(MODEL_SPECS.keys()),
    )
    args = parser.parse_args()

    device = pick_device(args.device)
    ensure_dir(args.output_dir)
    ensure_dir(args.output_dir / "predictions")
    ensure_dir(args.output_dir / "visualizations")

    manifest_path = args.samples_dir / "sample_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    selected_model_names = list(args.models)
    models = {name: load_model(name, device) for name in selected_model_names}

    aggregate: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    sample_rows: List[Dict[str, object]] = []
    visualized = 0

    for sample in manifest["samples"]:
        image_id = sample["image_id"]
        sample_dir = args.samples_dir / image_id
        image = Image.open(sample_dir / "image.png").convert("RGB")
        baseline_overlay = Image.open(sample_dir / "baseline_overlay.png").convert("RGB")
        baseline = load_baseline_meta(sample_dir)
        row: Dict[str, object] = {
            "image_id": image_id,
            "rationale": sample.get("rationale", ""),
            "baseline": baseline,
            "models": {},
        }
        overlays: Dict[str, Image.Image] = {}

        for model_name, loaded in models.items():
            model_dir = args.output_dir / "predictions" / model_name / image_id
            ensure_dir(model_dir)
            mapped, ratios, raw_ratios = run_inference(loaded, image, device)
            semantic_rgb = semantic_to_rgb(mapped)
            overlay = alpha_overlay(image, semantic_rgb)
            semantic_rgb.save(model_dir / "semantic_rgb.png")
            overlay.save(model_dir / "overlay.png")
            np.save(model_dir / "mapped_semantic.npy", mapped)
            metrics = {
                "target_ratios": ratios,
                "transition_rate": count_transitions(mapped),
                "raw_top_labels": dict(sorted(raw_ratios.items(), key=lambda kv: kv[1], reverse=True)[:10]),
            }
            write_json(model_dir / "metrics.json", metrics)
            row["models"][model_name] = metrics
            overlays[model_name] = overlay

            aggregate[model_name]["transition_rate_sum"] += metrics["transition_rate"]
            for k, v in ratios.items():
                aggregate[model_name][f"{k}_sum"] += v

        sample_rows.append(row)

        if visualized < args.visualize_count:
            panel = compose_panel(image_id, image, baseline_overlay, overlays, selected_model_names)
            panel.save(args.output_dir / "visualizations" / f"{image_id}_compare.png")
            visualized += 1

    summary: Dict[str, object] = {
        "device": str(device),
        "sample_count": len(sample_rows),
        "models": {},
        "samples": sample_rows,
    }

    for model_name in selected_model_names:
        count = float(len(sample_rows))
        summary["models"][model_name] = {
            "avg_transition_rate": aggregate[model_name]["transition_rate_sum"] / count,
            "avg_target_ratios": {
                cls: aggregate[model_name][f"{cls}_sum"] / count for cls in TARGET_CLASSES
            },
        }

    write_json(args.output_dir / "report.json", summary)

    report_lines = [
        "# Surface Baseline Small Eval",
        "",
        f"- device: `{device}`",
        f"- sample_count: `{len(sample_rows)}`",
        "- models: `shi-labs/oneformer_ade20k_swin_tiny`, `facebook/mask2former-swin-tiny-ade-semantic`",
        "- baseline reference: `sam2_prompt_v3_sky_label_merge_completion`",
        "",
        "## Aggregate",
        "",
        "| model | avg transition | avg floor/ground | avg wall | avg ceiling | avg building | avg sky |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name, model_summary in summary["models"].items():
        ratios = model_summary["avg_target_ratios"]
        report_lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                model_name,
                model_summary["avg_transition_rate"],
                ratios["floor_ground"],
                ratios["wall"],
                ratios["ceiling"],
                ratios["building"],
                ratios["sky"],
            )
        )

    report_lines.extend(
        [
            "",
            "## Per Sample Notes",
            "",
            "| image_id | rationale | baseline labels | model highlights |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in sample_rows:
        baseline_labels = ", ".join(sorted(row["baseline"]["mask_label_counts"].keys()))
        model_chunks = []
        for model_name in selected_model_names:
            ratios = row["models"][model_name]["target_ratios"]
            model_chunks.append(
                "{}: floor {:.3f}, wall {:.3f}, ceiling {:.3f}, building {:.3f}, sky {:.3f}".format(
                    model_name,
                    ratios["floor_ground"],
                    ratios["wall"],
                    ratios["ceiling"],
                    ratios["building"],
                    ratios["sky"],
                )
            )
        report_lines.append(
            "| {} | {} | {} | {} |".format(
                row["image_id"],
                row["rationale"],
                baseline_labels or "-",
                "<br>".join(model_chunks),
            )
        )

    (args.output_dir / "report.md").write_text("\n".join(report_lines))


if __name__ == "__main__":
    main()
