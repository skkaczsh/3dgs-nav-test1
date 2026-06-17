#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path("/Users/skkac/Work/SCAN")
FINE_MANIFEST = ROOT / "new_route/experiments/fine_object_grounded_small_eval/sample_manifest.json"
OUTPUT_ROOT = ROOT / "new_route/experiments/surface_fine_combo_small_eval/samples"


def main() -> None:
    manifest = json.loads(FINE_MANIFEST.read_text(encoding="utf-8"))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sample_rows: list[dict[str, object]] = []
    for sample in manifest["samples"]:
        image_id = str(sample["id"])
        sample_dir = OUTPUT_ROOT / image_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        image_path = Path(sample["image"])
        overlay_path = Path(sample["overlay"])
        labels_path = Path(sample["rel"]).parts
        source_labels = Path(sample["image"]).parent / "labels.json"

        labels_raw = json.loads(source_labels.read_text(encoding="utf-8"))
        label_counts = dict(Counter(labels_raw.values()))
        baseline_summary = {
            "combo": "sam2_prompt_v3_sky_label_merge_completion",
            "focus": sample.get("focus", []),
            "label_counts": label_counts,
        }

        (sample_dir / "image.png").write_bytes(image_path.read_bytes())
        (sample_dir / "baseline_overlay.png").write_bytes(overlay_path.read_bytes())
        (sample_dir / "baseline_labels.json").write_text(
            json.dumps(labels_raw, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (sample_dir / "baseline_summary.json").write_text(
            json.dumps(baseline_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        sample_rows.append(
            {
                "image_id": image_id,
                "rationale": "fine-object review sample with focus={}".format(
                    ",".join(sample.get("focus", []))
                ),
            }
        )

    (OUTPUT_ROOT / "sample_manifest.json").write_text(
        json.dumps({"samples": sample_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
