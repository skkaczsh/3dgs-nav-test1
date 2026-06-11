#!/usr/bin/env python3
"""Build a constrained ConceptSeg-R1 runlist from fine-object review assets.

The previous broad ConceptSeg smoke showed that floor/wall prompts are not
stable enough for automatic dense semantics. This builder therefore creates a
small, traceable side-track runlist only for fine residual targets such as
railing, equipment, and pipes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CONCEPTS = [
    "railing or thin metal guardrail",
    "rooftop equipment box or HVAC unit",
    "pipe or thin utility conduit",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def localize_pack_asset(pack_dir: Path, remote_path: str | None) -> Path | None:
    if not remote_path:
        return None
    path = Path(remote_path)
    if path.exists():
        return path
    parts = path.parts
    if "assets" in parts:
        suffix = Path(*parts[parts.index("assets") :])
        candidate = pack_dir / suffix
        if candidate.exists():
            return candidate
    return None


def copy_asset(src: Path | None, dst: Path | None) -> str | None:
    if src is None or dst is None:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def label_for_side(row: dict[str, Any], side: str) -> str:
    obj = row.get(f"object_{side}") or {}
    return str(obj.get("label") or row.get("proposal", {}).get("label") or "unknown")


def build_items(
    *,
    pack_dir: Path,
    output_dir: Path,
    concepts: list[str],
    copy_assets: bool,
    max_reps: int | None,
    remote_asset_root: str | None,
) -> list[dict[str, Any]]:
    review_jsonl = pack_dir / "cross_candidate_review_items.jsonl"
    rows = read_jsonl(review_jsonl)
    items: list[dict[str, Any]] = []

    for row in rows:
        review_id = row.get("review_id")
        proposal = row.get("proposal", {})
        representatives = row.get("representatives", [])
        if max_reps is not None:
            representatives = representatives[:max_reps]
        for rep in representatives:
            side = str(rep.get("side") or "")
            rep_index = rep.get("rep_index")
            slot = f"{side}{rep_index}"
            source_label = label_for_side(row, side)
            source_image = localize_pack_asset(pack_dir, rep.get("copied_image"))
            if source_image is None:
                continue

            asset_dir = output_dir / "assets" / str(review_id) / slot
            local_image = asset_dir / "image.png"
            local_overlay = asset_dir / "overlay.png"
            local_instance = asset_dir / "instance.png"
            local_semantic = asset_dir / "semantic.png"
            local_labels = asset_dir / "labels.json"

            if copy_assets:
                image_path = copy_asset(source_image, local_image)
                overlay_path = copy_asset(localize_pack_asset(pack_dir, rep.get("copied_overlay")), local_overlay)
                instance_path = copy_asset(localize_pack_asset(pack_dir, rep.get("copied_instance")), local_instance)
                semantic_path = copy_asset(localize_pack_asset(pack_dir, rep.get("copied_semantic")), local_semantic)
                labels_path = copy_asset(localize_pack_asset(pack_dir, rep.get("copied_labels")), local_labels)
            else:
                image_path = str(source_image)
                overlay_path = str(localize_pack_asset(pack_dir, rep.get("copied_overlay")) or "")
                instance_path = str(localize_pack_asset(pack_dir, rep.get("copied_instance")) or "")
                semantic_path = str(localize_pack_asset(pack_dir, rep.get("copied_semantic")) or "")
                labels_path = str(localize_pack_asset(pack_dir, rep.get("copied_labels")) or "")

            if remote_asset_root:
                remote_base = remote_asset_root.rstrip("/") + f"/assets/{review_id}/{slot}"
                run_image_path = f"{remote_base}/image.png"
            else:
                run_image_path = image_path

            for concept in concepts:
                safe_concept = concept.replace(" ", "_").replace("/", "_")
                image_id = f"{review_id}_{slot}_{rep.get('target_id')}_{safe_concept}"
                items.append(
                    {
                        "image_id": image_id,
                        "concept": concept,
                        "image_path": run_image_path,
                        "local_image_path": image_path,
                        "metadata": {
                            "review_id": review_id,
                            "source_label": source_label,
                            "proposal": proposal,
                            "representative": {
                                "side": side,
                                "rep_index": rep_index,
                                "object_id": rep.get("object_id"),
                                "candidate": rep.get("candidate"),
                                "tracklet_id": rep.get("tracklet_id"),
                                "target_id": rep.get("target_id"),
                                "target_meta": rep.get("target_meta"),
                                "tracklet_point_count": rep.get("tracklet_point_count"),
                                "tracklet_target_count": rep.get("tracklet_target_count"),
                            },
                            "local_assets": {
                                "image": image_path,
                                "overlay": overlay_path,
                                "instance": instance_path,
                                "semantic": semantic_path,
                                "labels": labels_path,
                            },
                            "remote_assets": {
                                "image": run_image_path,
                                "overlay": (remote_asset_root.rstrip("/") + f"/assets/{review_id}/{slot}/overlay.png")
                                if remote_asset_root
                                else overlay_path,
                                "instance": (remote_asset_root.rstrip("/") + f"/assets/{review_id}/{slot}/instance.png")
                                if remote_asset_root
                                else instance_path,
                                "semantic": (remote_asset_root.rstrip("/") + f"/assets/{review_id}/{slot}/semantic.png")
                                if remote_asset_root
                                else semantic_path,
                                "labels": (remote_asset_root.rstrip("/") + f"/assets/{review_id}/{slot}/labels.json")
                                if remote_asset_root
                                else labels_path,
                            },
                        },
                    }
                )
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concept", action="append", dest="concepts", default=None)
    parser.add_argument("--remote-asset-root", default=None)
    parser.add_argument("--max-reps-per-review", type=int, default=None)
    parser.add_argument("--no-copy-assets", action="store_true")
    args = parser.parse_args()

    concepts = args.concepts or DEFAULT_CONCEPTS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = build_items(
        pack_dir=args.pack_dir,
        output_dir=args.output_dir,
        concepts=concepts,
        copy_assets=not args.no_copy_assets,
        max_reps=args.max_reps_per_review,
        remote_asset_root=args.remote_asset_root,
    )
    runlist = {
        "schema": "conceptseg_fine_object_runlist_v1",
        "pack_dir": str(args.pack_dir),
        "output_dir": str(args.output_dir),
        "remote_asset_root": args.remote_asset_root,
        "policy": {
            "concepts": concepts,
            "excluded_broad_surface_prompts": ["floor", "wall", "building facade"],
            "note": "Use ConceptSeg-R1 only as constrained fine-object candidate generation.",
        },
        "items": items,
    }
    (args.output_dir / "runlist.json").write_text(json.dumps(runlist, ensure_ascii=False, indent=2), encoding="utf-8")

    concept_counts = Counter(item["concept"] for item in items)
    review_counts = Counter(item["metadata"]["review_id"] for item in items)
    summary = {
        "items": len(items),
        "concept_counts": dict(concept_counts),
        "review_counts": dict(review_counts),
        "asset_root": str(args.output_dir / "assets"),
        "runlist": str(args.output_dir / "runlist.json"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
