#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


RAILING_TERMS = {
    "railing",
    "guardrail",
    "handrail",
    "fence",
    "barrier",
    "balustrade",
    "栏杆",
    "护栏",
    "围栏",
}

STRICT_TERMS = ["railing", "guardrail", "handrail"]
WITH_FENCE_TERMS = ["railing", "guardrail", "handrail", "metal fence"]


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


def railing_score(labels: list[str]) -> tuple[int, int]:
    exact = 0
    fuzzy = 0
    for label in labels:
        lower = label.lower()
        if lower in RAILING_TERMS:
            exact += 1
            fuzzy += 1
            continue
        if any(term in lower for term in RAILING_TERMS):
            fuzzy += 1
    return exact, fuzzy


def build_manifest(args: argparse.Namespace) -> dict:
    root = args.semantic_eval_dir
    terms = WITH_FENCE_TERMS if args.include_metal_fence else STRICT_TERMS
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
        exact, fuzzy = railing_score(labels)
        if exact == 0 and fuzzy == 0:
            continue
        label_counts: dict[str, int] = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        samples.append(
            {
                "id": image_dir.name,
                "rel": image_dir.name,
                "focus": ["railing"],
                "image": str(combo_dir / "image.png"),
                "overlay": str(combo_dir / "overlay.png"),
                "instance": str(combo_dir / "instance.png"),
                "semantic": str(combo_dir / "semantic.png"),
                "labels_path": str(combo_dir / "labels.json"),
                "cam": cam,
                "frame": frame,
                "label_counts": label_counts,
                "prompt_terms": terms,
                "prompt_groups": [
                    {
                        "focus": "railing",
                        "terms": terms,
                    }
                ],
                "railing_exact_count": exact,
                "railing_fuzzy_count": fuzzy,
                "all_labels": labels,
            }
        )

    samples.sort(
        key=lambda row: (
            -int(row["railing_exact_count"]),
            -int(row["railing_fuzzy_count"]),
            int(row["frame"]),
            int(row["cam"]),
        )
    )
    if args.limit > 0:
        samples = samples[: args.limit]

    return {
        "experiment": "fine_object_grounded_railing_rich_eval",
        "semantic_eval_dir": str(root),
        "combo": args.combo,
        "prompt_variant": "with_fence" if args.include_metal_fence else "strict",
        "prompt_terms": terms,
        "prompt_text": " . ".join(terms),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-eval-dir", type=Path, required=True)
    parser.add_argument("--combo", default="sam2_prompt_v3_sky_label_merge_completion")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--include-metal-fence", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(json.dumps({"samples": len(manifest["samples"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
