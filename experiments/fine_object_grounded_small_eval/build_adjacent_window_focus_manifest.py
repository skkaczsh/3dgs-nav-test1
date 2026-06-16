#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_focus_rich_manifest import FOCUS_RULES, label_score, load_labels, parse_image_dir_name


def build_manifest(args: argparse.Namespace) -> dict:
    root = args.semantic_eval_dir
    rules = FOCUS_RULES[args.focus]
    prompt_terms = rules["fallback_terms"] if args.include_fallback else rules["strict_terms"]
    samples = []
    for image_dir in sorted((root / "images").glob("cam*_*")):
        parsed = parse_image_dir_name(image_dir.name)
        if parsed is None:
            continue
        cam, frame = parsed
        if frame < args.frame_start or frame > args.frame_end:
            continue
        if args.frame_stride > 1 and ((frame - args.frame_start) % args.frame_stride) != 0:
            continue
        combo_dir = image_dir / args.combo
        if not combo_dir.exists():
            continue
        labels = load_labels(combo_dir / "labels.json")
        exact, fuzzy = label_score(labels, rules["score_terms"])
        if args.require_focus_hit and exact == 0 and fuzzy == 0:
            continue
        label_counts: dict[str, int] = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        samples.append(
            {
                "id": image_dir.name,
                "rel": image_dir.name,
                "focus": [args.focus],
                "image": str(combo_dir / "image.png"),
                "overlay": str(combo_dir / "overlay.png"),
                "instance": str(combo_dir / "instance.png"),
                "semantic": str(combo_dir / "semantic.png"),
                "labels_path": str(combo_dir / "labels.json"),
                "cam": cam,
                "frame": frame,
                "label_counts": label_counts,
                "prompt_terms": prompt_terms,
                "prompt_groups": [{"focus": args.focus, "terms": prompt_terms}],
                f"{args.focus}_exact_count": exact,
                f"{args.focus}_fuzzy_count": fuzzy,
                "all_labels": labels,
            }
        )
    samples.sort(key=lambda row: (int(row["frame"]), int(row["cam"])))
    if args.limit > 0:
        samples = samples[: args.limit]
    return {
        "experiment": f"fine_object_grounded_{args.focus}_adjacent_window_eval",
        "semantic_eval_dir": str(root),
        "combo": args.combo,
        "focus": args.focus,
        "prompt_variant": "fallback" if args.include_fallback else "strict",
        "prompt_terms": prompt_terms,
        "prompt_text": " . ".join(prompt_terms),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frame_stride": int(args.frame_stride),
        "require_focus_hit": bool(args.require_focus_hit),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-eval-dir", type=Path, required=True)
    parser.add_argument("--focus", choices=sorted(FOCUS_RULES), required=True)
    parser.add_argument("--combo", default="sam2_prompt_v3_sky_label_merge_completion")
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-fallback", action="store_true")
    parser.add_argument("--require-focus-hit", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(json.dumps({"samples": len(manifest["samples"]), "focus": args.focus, "frame_start": args.frame_start, "frame_end": args.frame_end}, ensure_ascii=False))


if __name__ == "__main__":
    main()
