#!/usr/bin/env python3
"""Run Thinking-with-Visual-Primitives on a manifest and save raw outputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoProcessor


BOX_PATTERN = re.compile(r"<\|ref\|>(.*?)<\|/ref\|><\|box\|>\[(.*?)\]<\|/box\|>")
POINT_PATTERN = re.compile(r"<\|point\|>\[(.*?)\]<\|/point\|>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tvp-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()


def add_repo(repo: Path) -> None:
    text = str(repo.resolve())
    if text not in sys.path:
        sys.path.insert(0, text)


def parse_visual_primitives(text: str) -> tuple[list[dict], list[list[int]]]:
    boxes: list[dict] = []
    for match in BOX_PATTERN.finditer(text):
        label = match.group(1).strip()
        coords_str = match.group(2)
        coord_groups = re.findall(r"\[?(\d+),(\d+),(\d+),(\d+)\]?", coords_str)
        for c in coord_groups:
            boxes.append(
                {
                    "label": label or "object",
                    "box_999": [int(c[0]), int(c[1]), int(c[2]), int(c[3])],
                }
            )
    points: list[list[int]] = []
    for match in POINT_PATTERN.finditer(text):
        coords_str = match.group(1)
        point_groups = re.findall(r"\[(\d+),(\d+)\]", coords_str)
        for p in point_groups:
            points.append([int(p[0]), int(p[1])])
    return boxes, points


def build_messages(image: Image.Image, prompt: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant that can understand images and reason with visual primitives.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def main() -> None:
    args = parse_args()
    add_repo(args.tvp_repo)
    from model import VisualPrimitiveVLM  # type: ignore

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])[: args.max_samples]
    device = args.device if torch.cuda.is_available() else "cpu"

    model_path = args.model_path
    if "://" not in model_path and not Path(model_path).exists() and "/" in model_path:
        local_dir = Path(tempfile.gettempdir()) / ("tvp_" + model_path.replace("/", "__"))
        local_dir.mkdir(parents=True, exist_ok=True)
        model_path = snapshot_download(repo_id=model_path, local_dir=str(local_dir), local_dir_use_symlinks=False)

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
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for sample in samples:
            image_path = Path(sample["image_path"])
            image = Image.open(image_path).convert("RGB")
            messages = build_messages(image, str(sample["prompt"]))
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
            boxes, points = parse_visual_primitives(response)
            row = {
                "sample_id": sample["id"],
                "target_id": sample.get("target_id", ""),
                "frame": int(sample.get("frame", -1)),
                "cam": int(sample.get("cam", -1)),
                "mask": int(sample.get("mask", -1)),
                "prompt": sample["prompt"],
                "image_path": str(image_path),
                "response": response,
                "boxes": boxes,
                "points": points,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({"sample_id": row["sample_id"], "boxes": len(boxes), "points": len(points)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
