#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


FOCUS_RULES = {
    "railing": {
        "score_terms": {
            "railing", "guardrail", "handrail", "fence", "barrier", "balustrade",
            "栏杆", "护栏", "围栏",
        },
        "strict_terms": ["railing", "guardrail", "handrail"],
        "fallback_terms": ["railing", "guardrail", "handrail", "metal fence"],
    },
    "pipe": {
        "score_terms": {
            "pipe", "conduit", "cable", "duct", "tube", "hose", "wire",
            "管", "管道", "线缆",
        },
        "strict_terms": ["pipe", "cable", "conduit"],
        "fallback_terms": ["pipe", "cable", "conduit", "utility pipe"],
    },
    "equipment": {
        "score_terms": {
            "equipment", "hvac", "air conditioning", "outdoor unit", "machine",
            "device", "cabinet", "box", "fixture", "空调外机", "设备",
        },
        "strict_terms": ["HVAC outdoor unit", "outdoor unit", "air conditioning unit"],
        "fallback_terms": ["HVAC outdoor unit", "outdoor unit", "air conditioning unit", "rooftop equipment box"],
    },
    "hvac": {
        "score_terms": {
            "hvac", "air conditioning", "outdoor unit", "compressor", "condensing unit",
            "空调外机",
        },
        "strict_terms": ["HVAC outdoor unit", "outdoor unit", "air conditioning unit"],
        "fallback_terms": ["HVAC outdoor unit", "outdoor unit", "air conditioning unit", "condensing unit"],
    },
}


def parse_image_dir_name(name: str) -> tuple[int, int] | None:
    if not name.startswith("cam"):
        return None
    try:
        left, right = name.split("_", 1)
        cam = int(left[3:])
        frame = int(right)
    except (ValueError, IndexError):
        return None
    return cam, frame


def load_labels(path: Path) -> list[str]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    labels: list[str] = []
    if isinstance(raw, dict):
        values = raw.get("labels", raw)
        if isinstance(values, dict):
            for value in values.values():
                if isinstance(value, dict):
                    label = value.get("label") or value.get("name") or value.get("freeform_label") or ""
                else:
                    label = value
                labels.append(str(label).strip())
        elif isinstance(values, list):
            for item in values:
                if isinstance(item, dict):
                    label = item.get("label") or item.get("name") or item.get("freeform_label") or ""
                else:
                    label = item
                labels.append(str(label).strip())
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                label = item.get("label") or item.get("name") or item.get("freeform_label") or ""
            else:
                label = item
            labels.append(str(label).strip())
    return [x for x in labels if x]


def label_score(labels: list[str], terms: set[str]) -> tuple[int, int]:
    exact = 0
    fuzzy = 0
    for label in labels:
        lower = label.lower()
        if lower in terms:
            exact += 1
            fuzzy += 1
            continue
        if any(term in lower for term in terms):
            fuzzy += 1
    return exact, fuzzy


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
        combo_dir = image_dir / args.combo
        if not combo_dir.exists():
            continue
        labels = load_labels(combo_dir / "labels.json")
        if not labels:
            continue
        exact, fuzzy = label_score(labels, rules["score_terms"])
        if exact == 0 and fuzzy == 0:
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
    samples.sort(
        key=lambda row: (
            -int(row[f"{args.focus}_exact_count"]),
            -int(row[f"{args.focus}_fuzzy_count"]),
            int(row["frame"]),
            int(row["cam"]),
        )
    )
    if args.limit > 0:
        samples = samples[: args.limit]
    return {
        "experiment": f"fine_object_grounded_{args.focus}_rich_eval",
        "semantic_eval_dir": str(root),
        "combo": args.combo,
        "focus": args.focus,
        "prompt_variant": "fallback" if args.include_fallback else "strict",
        "prompt_terms": prompt_terms,
        "prompt_text": " . ".join(prompt_terms),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-eval-dir", type=Path, required=True)
    parser.add_argument("--focus", choices=sorted(FOCUS_RULES), required=True)
    parser.add_argument("--combo", default="sam2_prompt_v3_sky_label_merge_completion")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--include-fallback", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(json.dumps({"samples": len(manifest["samples"]), "focus": args.focus}, ensure_ascii=False))


if __name__ == "__main__":
    main()
