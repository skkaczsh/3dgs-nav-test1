#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_focus_rich_manifest import FOCUS_RULES, parse_image_dir_name


def build_manifest(args: argparse.Namespace) -> dict:
    rules = FOCUS_RULES[args.focus]
    prompt_terms = rules["fallback_terms"] if args.include_fallback else rules["strict_terms"]
    samples = []
    for image_path in sorted(args.images_dir.glob("cam*_*.png")):
        parsed = parse_image_dir_name(image_path.stem)
        if parsed is None:
            continue
        cam, frame = parsed
        if args.cameras and cam not in args.cameras:
            continue
        if frame < args.frame_start or frame > args.frame_end:
            continue
        if args.frame_stride > 1 and ((frame - args.frame_start) % args.frame_stride) != 0:
            continue
        samples.append(
            {
                "id": image_path.stem,
                "rel": image_path.name,
                "focus": [args.focus],
                "image": str(image_path),
                "cam": cam,
                "frame": frame,
                "label_counts": {},
                "all_labels": [],
                "prompt_terms": prompt_terms,
                "prompt_groups": [{"focus": args.focus, "terms": prompt_terms}],
            }
        )
    samples.sort(key=lambda row: (int(row["frame"]), int(row["cam"])))
    if args.limit > 0:
        samples = samples[: args.limit]
    return {
        "experiment": f"fine_object_grounded_{args.focus}_raw_window_eval",
        "images_dir": str(args.images_dir),
        "focus": args.focus,
        "prompt_variant": "fallback" if args.include_fallback else "strict",
        "prompt_terms": prompt_terms,
        "prompt_text": " . ".join(prompt_terms),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frame_stride": int(args.frame_stride),
        "cameras": list(args.cameras),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--focus", choices=sorted(FOCUS_RULES), required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--camera", dest="cameras", type=int, action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-fallback", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(
        json.dumps(
            {
                "samples": len(manifest["samples"]),
                "focus": args.focus,
                "frame_start": args.frame_start,
                "frame_end": args.frame_end,
                "cameras": args.cameras,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
