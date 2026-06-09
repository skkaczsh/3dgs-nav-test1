#!/usr/bin/env python3
"""Build a compact review pack for cross-candidate merge proposals."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path


TARGET_RE = re.compile(r"fine_t_(?P<frame>\d+)_cam(?P<cam>\d+)_mask(?P<mask>\d+)_sem(?P<semantic>\d+)_cc(?P<cc>\d+)")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dominant_vote(row: dict, key: str) -> str:
    votes = {str(k): int(v) for k, v in row.get(key, {}).items()}
    if not votes:
        return ""
    return max(votes.items(), key=lambda kv: kv[1])[0]


def parse_target_id(target_id: str) -> dict:
    m = TARGET_RE.match(target_id)
    if not m:
        return {}
    return {k: int(v) for k, v in m.groupdict().items()}


def artifact_paths(base: Path, combo: str, target_meta: dict) -> dict:
    if not target_meta:
        return {}
    frame = int(target_meta["frame"])
    cam = int(target_meta["cam"])
    artifact_dir = base / "images" / f"cam{cam}_{frame:06d}" / combo
    return {
        "artifact_dir": str(artifact_dir),
        "image": str(artifact_dir / "image.png"),
        "overlay": str(artifact_dir / "overlay.png"),
        "semantic": str(artifact_dir / "semantic.png"),
        "instance": str(artifact_dir / "instance.png"),
        "labels": str(artifact_dir / "labels.json"),
    }


def raw_image_path(raw_image_dir: Path, target_meta: dict) -> str:
    if not target_meta:
        return ""
    frame = int(target_meta["frame"])
    cam = int(target_meta["cam"])
    return str(raw_image_dir / f"cam{cam}_{frame:06d}.png")


def choose_representative_tracklets(obj: dict, tracklet_by_id: dict[str, dict], candidate: str, limit: int) -> list[dict]:
    tracklets = [tracklet_by_id[tid] for tid in obj.get("tracklet_ids", []) if tid in tracklet_by_id]
    preferred = [t for t in tracklets if dominant_vote(t, "accepted_candidate_votes") == str(candidate)]
    pool = preferred or tracklets
    pool.sort(key=lambda t: (int(t.get("point_count", 0)), int(t.get("target_count", 0))), reverse=True)
    return pool[:limit]


def representative_target(tracklet: dict, candidate: str, artifact_base: Path | None = None, combo: str = "") -> str:
    # Prefer target ids whose semantic artifacts exist. Within that set, start
    # near the middle of the tracklet span; middle frames tend to be better
    # observed than first/last frames.
    ids = list(tracklet.get("target_ids", []))
    if not ids:
        return ""
    order = sorted(range(len(ids)), key=lambda i: abs(i - len(ids) / 2.0))
    if artifact_base is not None and combo:
        for i in order:
            meta = parse_target_id(ids[i])
            paths = artifact_paths(artifact_base, combo, meta)
            if Path(paths.get("overlay", "")).exists():
                return ids[i]
    return ids[order[0]]


def copy_artifacts(paths: dict, asset_dir: Path, prefix: str) -> dict:
    copied = {}
    asset_dir.mkdir(parents=True, exist_ok=True)
    for key in ("image", "overlay", "semantic", "instance", "labels"):
        src = Path(paths.get(key, ""))
        if not src.exists():
            continue
        suffix = src.suffix
        dst = asset_dir / f"{prefix}_{key}{suffix}"
        shutil.copy2(src, dst)
        copied[f"copied_{key}"] = str(dst)
    return copied


def copy_raw_image(raw_path: str, asset_dir: Path, prefix: str) -> dict:
    src = Path(raw_path)
    if not src.exists():
        return {}
    asset_dir.mkdir(parents=True, exist_ok=True)
    dst = asset_dir / f"{prefix}_raw_image{src.suffix}"
    shutil.copy2(src, dst)
    return {"copied_raw_image": str(dst)}


def build_review_items(
    proposals: list[dict],
    objects: list[dict],
    tracklets: list[dict],
    args: argparse.Namespace,
) -> list[dict]:
    object_by_id = {row["long_object_id"]: row for row in objects}
    tracklet_by_id = {row["tracklet_id"]: row for row in tracklets}
    items = []
    selected = [
        row
        for row in proposals
        if args.priority == "all" or row.get("review_priority") == args.priority
    ][: args.max_items]
    for idx, proposal in enumerate(selected, start=1):
        obj_a = object_by_id.get(proposal["object_a"], {})
        obj_b = object_by_id.get(proposal["object_b"], {})
        reps = []
        for side, obj, candidate in (
            ("a", obj_a, proposal.get("candidate_a", "")),
            ("b", obj_b, proposal.get("candidate_b", "")),
        ):
            for rep_idx, tracklet in enumerate(choose_representative_tracklets(obj, tracklet_by_id, candidate, args.reps_per_side)):
                target_id = representative_target(tracklet, candidate, args.artifact_base, args.combo)
                target_meta = parse_target_id(target_id)
                paths = artifact_paths(args.artifact_base, args.combo, target_meta)
                raw_path = raw_image_path(args.raw_image_dir, target_meta)
                copied = {}
                if args.copy_assets:
                    asset_dir = args.output_dir / "assets" / f"proposal_{idx:03d}"
                    copied = copy_artifacts(paths, asset_dir, f"{side}{rep_idx}")
                    if "copied_image" not in copied:
                        copied.update(copy_raw_image(raw_path, asset_dir, f"{side}{rep_idx}"))
                reps.append(
                    {
                        "side": side,
                        "rep_index": rep_idx,
                        "object_id": obj.get("long_object_id", ""),
                        "candidate": str(candidate),
                        "tracklet_id": tracklet.get("tracklet_id", ""),
                        "tracklet_point_count": int(tracklet.get("point_count", 0)),
                        "tracklet_target_count": int(tracklet.get("target_count", 0)),
                        "target_id": target_id,
                        "target_meta": target_meta,
                        "artifact_paths": paths,
                        "raw_image": raw_path,
                        **copied,
                    }
                )
        item = {
            "review_id": f"review_{idx:03d}",
            "proposal": proposal,
            "object_a": obj_a,
            "object_b": obj_b,
            "representatives": reps,
            "decision": "pending",
            "review_notes": "",
        }
        items.append(item)
    return items


def write_outputs(items: list[dict], args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "cross_candidate_review_items.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    csv_path = args.output_dir / "cross_candidate_review_items.csv"
    rows = []
    for item in items:
        p = item["proposal"]
        rows.append(
            {
                "review_id": item["review_id"],
                "object_a": p["object_a"],
                "object_b": p["object_b"],
                "candidate_a": p["candidate_a"],
                "candidate_b": p["candidate_b"],
                "source_a": p["source_a"],
                "source_b": p["source_b"],
                "same_source_cluster": p["same_source_cluster"],
                "score": p["score"],
                "review_priority": p["review_priority"],
                "centroid_distance": p["centroid_distance"],
                "bbox_distance": p["bbox_distance"],
                "bbox_overlap_ratio": p["bbox_overlap_ratio"],
                "color_distance": p["color_distance"],
                "representative_count": len(item["representatives"]),
                "decision": item["decision"],
            }
        )
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    md_path = args.output_dir / "README_review.md"
    lines = [
        "# Cross-Candidate Merge Review Pack",
        "",
        f"- items: `{len(items)}`",
        f"- priority filter: `{args.priority}`",
        f"- combo: `{args.combo}`",
        f"- copied assets: `{bool(args.copy_assets)}`",
        "",
        "Review guidance:",
        "",
        "- `merge`: two objects are the same physical structure/object.",
        "- `keep_split`: two objects are adjacent or overlapping but should remain separate.",
        "- `uncertain`: insufficient visual evidence; send to VLM/ConceptSeg review.",
        "",
    ]
    for item in items[:20]:
        p = item["proposal"]
        lines.append(
            f"- `{item['review_id']}` `{p['object_a']}` + `{p['object_b']}` "
            f"candidates `{p['candidate_a']}`/`{p['candidate_b']}` "
            f"score `{p['score']:.3f}` priority `{p['review_priority']}` reps `{len(item['representatives'])}`"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {
        "proposal_jsonl": str(args.proposals),
        "objects_jsonl": str(args.objects),
        "tracklets_jsonl": str(args.tracklets),
        "output_jsonl": str(jsonl_path),
        "output_csv": str(csv_path),
        "readme": str(md_path),
        "item_count": int(len(items)),
        "representative_count": int(sum(len(item["representatives"]) for item in items)),
        "copied_overlay_count": int(
            sum(1 for item in items for rep in item["representatives"] if "copied_overlay" in rep)
        ),
        "copied_raw_image_count": int(
            sum(1 for item in items for rep in item["representatives"] if "copied_raw_image" in rep)
        ),
        "items_with_overlay": int(
            sum(1 for item in items if any("copied_overlay" in rep for rep in item["representatives"]))
        ),
        "items_with_any_image": int(
            sum(
                1
                for item in items
                if any(("copied_overlay" in rep or "copied_image" in rep or "copied_raw_image" in rep) for rep in item["representatives"])
            )
        ),
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (args.output_dir / "cross_candidate_review_pack_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"item_count": len(items), "output_dir": str(args.output_dir)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--tracklets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-base", type=Path, default=Path("/root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_0000_0999_b"))
    parser.add_argument("--raw-image-dir", type=Path, default=Path("/root/epfs/manifold_3dgs_project/processed/images"))
    parser.add_argument("--combo", default="sam2_prompt_v3_sky_label_merge_completion")
    parser.add_argument("--priority", choices=["high", "medium", "low", "all"], default="high")
    parser.add_argument("--max-items", type=int, default=20)
    parser.add_argument("--reps-per-side", type=int, default=2)
    parser.add_argument("--copy-assets", action="store_true")
    args = parser.parse_args()

    proposals = load_jsonl(args.proposals)
    objects = load_jsonl(args.objects)
    tracklets = load_jsonl(args.tracklets)
    items = build_review_items(proposals, objects, tracklets, args)
    write_outputs(items, args)


if __name__ == "__main__":
    main()
