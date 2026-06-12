#!/usr/bin/env python3
"""Patch semantic_eval/run_eval.py to read compact SAM mask RLE.

The original semantic_eval loader only accepted dense bool-list segmentation.
SAM2 TensorRT production output uses uncompressed COCO-style RLE to avoid
hundreds of gigabytes of JSON. This patch is intentionally narrow and
idempotent: it replaces only decode_sam2_masks and adds a helper decoder.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


START = "def decode_sam2_masks(path: Path) -> list[Mask]:\n"

PATCHED_BLOCK = '''def decode_sam_segmentation(segmentation: Any) -> np.ndarray:
    """Decode dense bool-list or uncompressed COCO-style RLE segmentation."""
    if isinstance(segmentation, dict) and "counts" in segmentation and "size" in segmentation:
        h, w = [int(x) for x in segmentation["size"]]
        flat = np.empty(h * w, dtype=bool)
        idx = 0
        value = False
        for raw_count in segmentation["counts"]:
            count = int(raw_count)
            next_idx = min(idx + count, flat.size)
            flat[idx:next_idx] = value
            idx = next_idx
            value = not value
        if idx < flat.size:
            flat[idx:] = False
        return flat.reshape((w, h)).T
    return np.array(segmentation, dtype=bool)


def decode_sam2_masks(path: Path) -> list[Mask]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    items = data.get("masks", data if isinstance(data, list) else [])
    masks: list[Mask] = []
    for item in items:
        seg = decode_sam_segmentation(item["segmentation"])
        ys, xs = np.where(seg)
        bbox = item.get("bbox")
        if not bbox and len(xs):
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        masks.append(Mask(
            segmentation=seg,
            area=int(item.get("area", int(seg.sum()))),
            score=float(item.get("predicted_iou", item.get("score", 0.5)))
                  * float(item.get("stability_score", 1.0)),
            bbox=[int(x) for x in (bbox or [0, 0, 0, 0])],
            source="sam2",
        ))
    return masks
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-eval", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    text = args.run_eval.read_text(encoding="utf-8")
    if "def decode_sam_segmentation(segmentation: Any) -> np.ndarray:" in text:
        print(f"already_patched={args.run_eval}")
        return
    if "from typing import Any" not in text:
        text = text.replace("from typing import", "from typing import")
        if "from typing import Any" not in text:
            text = text.replace("from typing import ", "from typing import Any, ", 1)
    start = text.find(START)
    next_top_level = re.search(
        r"\n\n(?=(?:def [A-Za-z_][A-Za-z0-9_]*\(|class [A-Za-z_][A-Za-z0-9_]*(?:\(|:)))",
        text[start + len(START):],
    )
    end = start + len(START) + next_top_level.start() if start >= 0 and next_top_level else -1
    if start < 0 or end < 0 or end <= start:
        raise SystemExit(f"could not locate decode_sam2_masks block in {args.run_eval}")
    patched = text[:start] + PATCHED_BLOCK + text[end:]
    if args.dry_run:
        print(f"would_patch={args.run_eval}")
        return
    args.run_eval.write_text(patched, encoding="utf-8")
    print(f"patched={args.run_eval}")


if __name__ == "__main__":
    main()
