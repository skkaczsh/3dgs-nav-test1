#!/usr/bin/env python3
"""Run a closed-vocabulary TVP crop/text benchmark on rooftop candidates."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoProcessor


LABELS = ["railing", "pipe", "equipment", "none"]
LABEL_SYNONYMS = {
    "railing": {"railing", "guardrail", "handrail", "fence", "barrier", "balustrade", "rail"},
    "pipe": {"pipe", "conduit", "cable", "duct", "tube", "hose", "wire"},
    "equipment": {
        "equipment",
        "hvac",
        "machine",
        "device",
        "cabinet",
        "box",
        "compressor",
        "condensing unit",
        "air conditioning unit",
        "outdoor unit",
        "air conditioner",
    },
    "none": {"none", "nonexistent", "no object", "background", "wall", "floor", "sky", "empty", "unknown"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tvp-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--crop-pad", type=int, default=48)
    parser.add_argument("--crop-dir", type=Path, default=None)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--truth-fields",
        nargs="+",
        default=["answer_class", "concept_class", "source_label"],
        help="Ordered manifest fields to consult when resolving the ground-truth closed-vocabulary label.",
    )
    return parser.parse_args()


def add_repo(repo: Path) -> None:
    text = str(repo.resolve())
    if text not in sys.path:
        sys.path.insert(0, text)


def build_messages(image: Image.Image, prompt: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "You are a careful classifier for cropped rooftop objects.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def resolve_model_path(model_path: str, download_dir: Path | None) -> str:
    if "://" not in model_path and not Path(model_path).exists() and "/" in model_path:
        base_download_dir = download_dir or (
            Path(os.environ["TVP_DOWNLOAD_DIR"]) if os.environ.get("TVP_DOWNLOAD_DIR") else Path(tempfile.gettempdir())
        )
        local_dir = base_download_dir / ("tvp_" + model_path.replace("/", "__"))
        local_dir.mkdir(parents=True, exist_ok=True)
        model_path = snapshot_download(repo_id=model_path, local_dir=str(local_dir), local_dir_use_symlinks=False)
    return model_path


def truth_label(sample: dict, truth_fields: list[str]) -> tuple[str, str]:
    for field in truth_fields:
        value = str(sample.get(field, ""))
        normalized = normalize_label(value)
        if normalized != "none":
            return normalized, field
    return "none", ""


def normalize_label(text: str) -> str:
    lower = text.strip().lower()
    if not lower:
        return "none"
    for label, terms in LABEL_SYNONYMS.items():
        if lower in terms:
            return label
    for label, terms in LABEL_SYNONYMS.items():
        if any(term in lower for term in sorted(terms, key=len, reverse=True)):
            return label
    return "none"


def crop_image(image: Image.Image, bbox: list[float] | None, pad: int) -> tuple[Image.Image, list[int] | None]:
    if not bbox or len(bbox) != 4:
        return image, None
    w, h = image.size
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return image, None
    return image.crop((x1, y1, x2, y2)), [x1, y1, x2, y2]


def closed_vocab_prompt() -> str:
    return (
        "Choose one label for the cropped rooftop object: railing, pipe, equipment, none. "
        "Return one label only."
    )


def main() -> None:
    args = parse_args()
    add_repo(args.tvp_repo)
    from model import VisualPrimitiveVLM  # type: ignore

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    samples = manifest.get("samples", manifest.get("items", []))[: args.max_samples]
    device = args.device if torch.cuda.is_available() else "cpu"

    model_path = resolve_model_path(args.model_path, args.download_dir)
    model = VisualPrimitiveVLM.from_pretrained(
        model_path,
        device_map=device,
        load_in_4bit=args.load_in_4bit,
    )
    model.eval()
    tokenizer = model.tokenizer
    processor_path = getattr(model, "base_model_path", "Qwen/Qwen2-VL-2B-Instruct")
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.crop and args.crop_dir is not None:
        args.crop_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    confusion = defaultdict(Counter)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(samples):
            raw_image_path = Path(sample["image_path"])
            image = Image.open(raw_image_path).convert("RGB")
            cropped_bbox = None
            effective_image_path = raw_image_path
            if args.crop:
                image, cropped_bbox = crop_image(image, sample.get("bbox"), args.crop_pad)
                if args.crop_dir is not None:
                    crop_path = args.crop_dir / f"{idx:04d}_{Path(raw_image_path).stem}.png"
                    image.save(crop_path)
                    effective_image_path = crop_path

            prompt = closed_vocab_prompt()
            messages = build_messages(image, prompt)
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
            inputs = {k: v.to(model.vlm.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

            gen_kwargs = {
                "max_new_tokens": int(args.max_new_tokens),
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if args.do_sample:
                gen_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": 0.9})
            else:
                gen_kwargs.update({"do_sample": False})

            with torch.no_grad():
                output_ids = model.vlm.generate(**inputs, **gen_kwargs)
            new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
            response = tokenizer.batch_decode(new_tokens, skip_special_tokens=False)[0]
            response = response.replace("<|im_end|>", "").strip()

            truth, truth_field = truth_label(sample, list(args.truth_fields))
            pred = normalize_label(response)
            confusion[truth][pred] += 1
            row = {
                "sample_id": sample.get("id", sample.get("image_id", f"sample_{idx:04d}")),
                "image_path": str(raw_image_path),
                "effective_image_path": str(effective_image_path),
                "crop_bbox_xyxy": cropped_bbox,
                "prompt": prompt,
                "truth_label": truth,
                "truth_field": truth_field,
                "pred_label": pred,
                "response": response,
                "source_label": sample.get("source_label", ""),
                "answer_class": sample.get("answer_class", ""),
                "concept_class": sample.get("concept_class", ""),
                "metadata": sample.get("metadata", {}),
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({"sample_id": row["sample_id"], "truth": truth, "pred": pred}, ensure_ascii=False))

    accuracy = float(sum(row["truth_label"] == row["pred_label"] for row in rows) / max(len(rows), 1))
    report = {
        "manifest": str(args.manifest),
        "output_jsonl": str(args.output_jsonl),
        "sample_count": len(rows),
        "labels": LABELS,
        "truth_fields": list(args.truth_fields),
        "accuracy": accuracy,
        "label_counts": dict(Counter(row["truth_label"] for row in rows)),
        "confusion": {truth: dict(counter) for truth, counter in confusion.items()},
        "per_label_accuracy": {
            label: float(confusion[label][label] / max(sum(confusion[label].values()), 1))
            for label in LABELS
        },
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"sample_count": len(rows), "accuracy": accuracy}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
