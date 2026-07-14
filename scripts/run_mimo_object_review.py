#!/usr/bin/env python3
"""Review object image evidence with a VLM-compatible Mimo endpoint.

This stage is classification/review only. It does not rewrite point clouds.
It asks the VLM for:

- open description
- controlled semantic label
- whether the object is a true object, a surface fragment, or ambiguous
- suggested downstream action

API credentials are read from environment variables:

- MIMO_API_KEY
- MIMO_BASE_URL, default https://token-plan-sgp.xiaomimimo.com/v1
- MIMO_MODEL, default mimo-v2.5
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


CONTROLLED_LABELS = [
    "floor",
    "wall",
    "grass",
    "car",
    "railing",
    "tree_or_shrub",
    "equipment",
    "hvac_outdoor_unit",
    "traffic_cone",
    "pipe_or_pole",
    "door_or_window",
    "sign_or_box",
    "curb_or_low_barrier",
    "roof",
    "ceiling",
    "stair",
    "building_part",
    "unknown",
]
SURFACE_ATTACHMENTS = ["none", "floor", "wall", "grass", "roof", "ceiling", "stair", "unknown"]


def geometry_context(obj: dict[str, Any]) -> dict[str, Any]:
    features = obj.get("geometry_features") or {}
    normal = features.get("normal") or obj.get("pca_normal")
    normal_z = abs(float(normal[2])) if isinstance(normal, list) and len(normal) == 3 else None
    orientation = "unavailable"
    if normal_z is not None:
        orientation = "horizontal_like" if normal_z >= 0.85 else ("vertical_like" if normal_z <= 0.30 else "oblique_or_linear")
    return {
        "geometry_type": obj.get("geometry_type") or features.get("geometry_type") or "unknown",
        "geometry_features": features,
        "world_normal_abs_z": normal_z,
        "gravity_orientation_hint": orientation,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_controlled_fields(parsed: dict[str, Any]) -> list[str]:
    """Keep open descriptions, but keep graph labels inside their contract."""
    warnings: list[str] = []
    label = str(parsed.get("controlled_label") or "unknown")
    if label not in CONTROLLED_LABELS:
        parsed["controlled_label"] = "unknown"
        warnings.append(f"unsupported_controlled_label={label}")
    attachment = str(parsed.get("surface_attachment") or "unknown")
    if attachment not in SURFACE_ATTACHMENTS:
        parsed["surface_attachment"] = "unknown"
        warnings.append(f"unsupported_surface_attachment={attachment}")
    return warnings


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def evidence_by_object(rows: list[dict[str, Any]], top_k: int) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["object_id"])].append(row)
    for vals in grouped.values():
        vals.sort(key=lambda r: (int(r.get("rank", 999)), -float(r.get("score", 0.0))))
        del vals[top_k:]
    return grouped


def prompt_for_object(obj: dict[str, Any], evidence_rows: list[dict[str, Any]], task: str) -> str:
    geometry = {
        key: obj.get(key)
        for key in [
            "object_id",
            "semantic_label",
            "semantic_label_original",
            "point_count",
            "centroid",
            "bbox_min",
            "bbox_max",
            "extent",
            "pca_normal",
            "planarity",
            "thickness_rms",
            "surface_trust_guard_status",
            "surface_trust_guard_majority_label",
            "surface_trust_guard_majority_ratio",
            "first_pass_label",
            "first_pass_confidence",
            "first_pass_description_zh",
        ]
        if key in obj
    }
    geometry.update(geometry_context(obj))
    evidence = [
        {
            "rank": row.get("rank"),
            "frame_id": row.get("frame_id"),
            "cam_id": row.get("cam_id"),
            "projected_points": row.get("projected_points"),
            "bbox_area_ratio": row.get("bbox_area_ratio"),
            "median_depth": row.get("median_depth"),
            "bbox_xyxy": row.get("bbox_xyxy"),
            "world_up_image_hint": row.get("world_up_image_hint", "unavailable"),
            "world_up_image_unit_xy": row.get("world_up_image_unit_xy"),
            "camera_pose_hint": row.get("camera_pose_hint", "unavailable"),
            "camera_center_world": row.get("camera_center_world"),
            "camera_forward_world_unit": row.get("camera_forward_world_unit"),
            "camera_image_up_world_unit": row.get("camera_image_up_world_unit"),
            "object_view_direction_world_unit": row.get("object_view_direction_world_unit"),
            "object_camera_distance_m": row.get("object_camera_distance_m"),
            "object_relative_height_m": row.get("object_relative_height_m"),
            "object_view_elevation_deg": row.get("object_view_elevation_deg"),
            "camera_forward_elevation_deg": row.get("camera_forward_elevation_deg"),
        }
        for row in evidence_rows
    ]
    labels = ", ".join(CONTROLLED_LABELS)
    attachments = ", ".join(SURFACE_ATTACHMENTS)
    common = (
        "You are reviewing one 3D point-cloud object from an outdoor parking-lot scan. "
        "Images are undistorted camera evidence. Red points / yellow boxes indicate where this 3D object projects. "
        "When shown, the cyan WORLD UP arrow is the direction of world +Z in image pixels, the direction opposite gravity. "
        "Treat it as a level reference: a world-horizontal surface extends approximately perpendicular to that arrow, while a world-vertical surface extends approximately along it. "
        "The camera may be rolled or pitched, so do not assume image-top is world-up. "
        "For 3D orientation, use world_normal_abs_z and gravity_orientation_hint as authoritative. Ignore geometry_features.verticality for world up/down; it is a local PCA feature, not a gravity direction. "
        "Each evidence record may include calibrated camera_pose facts computed from the same projection chain: "
        "camera_center_world, camera_forward_world_unit, camera_image_up_world_unit, object_view_direction_world_unit, "
        "object_relative_height_m, object_view_elevation_deg, and camera_forward_elevation_deg. "
        "They are hard geometric evidence, not quantities to infer from the image. Positive object_relative_height_m means the object centroid is above the camera; "
        "positive object_view_elevation_deg means the object lies above the camera's world-horizontal plane. "
        "Use the images plus the 3D geometry summary. Do not classify the whole image; classify only the projected object.\n\n"
        f"Allowed controlled labels: {labels}.\n"
        f"Allowed surface_attachment values: {attachments}.\n"
    )
    if task == "structure":
        return common + (
            "This is a second-pass structural review of a generic building surface. "
            "Choose only a specific structure when the visible evidence supports it: wall, roof, ceiling, stair, floor, or grass. "
            "Use building_part or unknown when the crop cannot support a specific structure.\n"
            "controlled_label names this 3D superpoint itself. surface_attachment names a larger surface it is attached to; "
            "do not use a parent surface as controlled_label for a thin line, light strip, railing, pipe, or other child feature.\n"
            "The camera can be rotated: do not infer floor or wall from image up/down. "
            "Use gravity_orientation_hint from world coordinates as advisory evidence. "
            "horizontal_like may be floor/roof/ceiling/grass; vertical_like may be wall/door_or_window; "
            "If vertical_like conflicts with a horizontal surface label, keep any parent in surface_attachment but return unknown or review_manually for controlled_label. "
            "thin_linear or rough_mixed geometry needs especially strong visual support. Geometry does not force a label.\n"
            "controlled_label must be exactly one allowed value. Put free terms such as 'light strip' only in description_zh.\n"
            "Return strict JSON only with keys:"
            "{"
            "\"controlled_label\": string, "
            "\"surface_attachment\": string, "
            "\"description_zh\": string, "
            "\"is_true_object\": false, "
            "\"is_surface_fragment\": true, "
            "\"confidence\": number, "
            "\"action\": \"keep\"|\"relabel\"|\"demote_to_unknown\"|\"review_manually\", "
            "\"reason_zh\": string"
            "}.\n\n"
            f"3D geometry summary JSON:\n{json.dumps(geometry, ensure_ascii=False)}\n"
            f"Evidence metadata JSON:\n{json.dumps(evidence, ensure_ascii=False)}"
        )
    return common + (
        "Important rules:\n"
        "- Broad flat pavement is floor, not railing/car.\n"
        "- Vertical building facade or wall panels are wall/building_part, not car/railing.\n"
        "- Railing/guardrail should be thin linear metal fence/handrail geometry.\n"
        "- Car must be an actual vehicle body, not a wall reflection, facade, or flat surface.\n"
        "- HVAC outdoor units, utility boxes, exposed machines, or installed mechanical devices are equipment/hvac_outdoor_unit, not wall.\n"
        "- Doors and windows are door_or_window unless the evidence only shows an unresolved building surface fragment.\n"
        "- If evidence mostly shows a surface fragment, say surface_fragment and choose floor/wall/grass/building_part.\n"
        "- If evidence is unclear, choose unknown and action review_manually.\n\n"
        "controlled_label names the observed 3D superpoint itself. surface_attachment separately names a broad surface "
        "that contains or supports it. A thin rail, pipe, light strip, curb, or window edge must not be labeled wall/floor/ceiling "
        "only because it is attached to one; use its intrinsic label and record the parent in surface_attachment. "
        "For a broad surface superpoint, controlled_label and surface_attachment may be the same.\n\n"
        "controlled_label must be exactly one allowed value. Put free terms such as 'light strip' only in description_zh. "
        "If vertical_like conflicts with floor/roof/ceiling/grass/stair, keep the parent in surface_attachment and use unknown or review_manually for controlled_label.\n\n"
        "Return strict JSON only with keys:\n"
        "{"
        "\"controlled_label\": string, "
        "\"surface_attachment\": string, "
        "\"description_zh\": string, "
        "\"is_true_object\": boolean, "
        "\"is_surface_fragment\": boolean, "
        "\"confidence\": number, "
        "\"action\": \"keep\"|\"relabel\"|\"demote_to_unknown\"|\"review_manually\", "
        "\"reason_zh\": string"
        "}.\n\n"
        f"3D geometry summary JSON:\n{json.dumps(geometry, ensure_ascii=False)}\n"
        f"Evidence metadata JSON:\n{json.dumps(evidence, ensure_ascii=False)}"
    )


def parse_json_response(text: str) -> tuple[dict[str, Any] | None, str]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        raw = raw.rsplit("\n", 1)[0] if raw.endswith("```") else raw
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start:end + 1]
    try:
        return json.loads(raw), ""
    except json.JSONDecodeError as exc:
        return None, str(exc)


def evidence_image_paths(row: dict[str, Any], image_mode: str) -> list[Path]:
    keys_by_mode = {
        "overlay": ("overlay_path",),
        "crop": ("crop_path",),
        "both": ("overlay_path", "crop_path"),
    }
    paths: list[Path] = []
    for key in keys_by_mode[image_mode]:
        path = Path(str(row.get(key) or ""))
        if path.exists():
            paths.append(path)
    return paths


def call_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    obj: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    timeout: float,
    max_tokens: int,
    image_mode: str,
    task: str,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_for_object(obj, evidence_rows, task)}]
    for row in evidence_rows:
        for path in evidence_image_paths(row, image_mode):
            content.append({
                "type": "image_url",
                "image_url": {"url": image_data_url(path)},
            })
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if os.environ.get("VLM_DISABLE_THINKING", "").lower() in {"1", "true", "yes", "on"}:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    text = str(body.get("choices", [{}])[0].get("message", {}).get("content", ""))
    if not text.strip():
        raise ValueError(f"empty assistant content: {json.dumps(body, ensure_ascii=False)[:500]}")
    parsed, parse_error = parse_json_response(text)
    if parsed is None:
        raise ValueError(f"unparseable assistant content: {parse_error}; raw={text[:500]}")
    validation_warnings = validate_controlled_fields(parsed)
    return {
        "object_id": int(obj["object_id"]),
        "input_semantic_label": obj.get("semantic_label", ""),
        "evidence_count": len(evidence_rows),
        "image_mode": image_mode,
        "review_task": task,
        "raw_response": text,
        "parsed": parsed,
        "parse_error": "",
        "validation_warnings": validation_warnings,
        "elapsed_sec": elapsed,
        "usage": body.get("usage", {}),
    }


def review_one(args_tuple: tuple[argparse.Namespace, str, str, str, dict[str, Any], list[dict[str, Any]]]) -> dict[str, Any]:
    args, base_url, api_key, model, obj, ev_rows = args_tuple
    last_error = ""
    for attempt in range(args.retries + 1):
        try:
            return call_chat_completion(
                base_url,
                api_key,
                model,
                obj,
                ev_rows,
                args.timeout,
                args.max_tokens,
                args.image_mode,
                args.task,
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = repr(exc)
            time.sleep(min(2.0 * (attempt + 1), 8.0))
    return {
        "object_id": int(obj["object_id"]),
        "input_semantic_label": obj.get("semantic_label", ""),
        "evidence_count": len(ev_rows),
        "image_mode": args.image_mode,
        "review_task": args.task,
        "raw_response": "",
        "parsed": None,
        "parse_error": last_error,
        "elapsed_sec": 0.0,
        "usage": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-jsonl", type=Path, required=True)
    parser.add_argument("--evidence-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--limit-objects", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--image-mode", choices=("overlay", "crop", "both"), default="overlay")
    parser.add_argument("--task", choices=("object", "structure"), default="object")
    parser.add_argument("--base-url", default=os.environ.get("MIMO_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1"))
    parser.add_argument("--model", default=os.environ.get("MIMO_MODEL", "mimo-v2.5"))
    args = parser.parse_args()

    api_key = os.environ.get("MIMO_API_KEY")
    if not api_key:
        raise SystemExit("MIMO_API_KEY is required in the environment.")

    objects = read_jsonl(args.objects_jsonl)
    if args.limit_objects:
        objects = objects[:args.limit_objects]
    evidence_rows = read_jsonl(args.evidence_jsonl)
    grouped = evidence_by_object(evidence_rows, args.top_k)
    tasks = [
        (args, args.base_url, api_key, args.model, obj, grouped.get(int(obj["object_id"]), []))
        for obj in objects
        if grouped.get(int(obj["object_id"]))
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_dir / "mimo_object_review.jsonl"
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [pool.submit(review_one, task) for task in tasks]
        for fut in as_completed(futures):
            rows.append(fut.result())
            # Keep completed reviews recoverable when a long VLM batch is interrupted.
            write_jsonl(output_jsonl, sorted(rows, key=lambda r: int(r["object_id"])))
            print(json.dumps({
                "done": len(rows),
                "object_id": rows[-1]["object_id"],
                "parsed": bool(rows[-1].get("parsed")),
                "parse_error": rows[-1].get("parse_error", "")[:120],
            }, ensure_ascii=False), flush=True)
    rows.sort(key=lambda r: int(r["object_id"]))
    write_jsonl(output_jsonl, rows)

    label_counts = Counter()
    action_counts = Counter()
    parse_ok = 0
    for row in rows:
        parsed = row.get("parsed") or {}
        if parsed:
            parse_ok += 1
            label_counts[str(parsed.get("controlled_label") or "")] += 1
            action_counts[str(parsed.get("action") or "")] += 1
    report = {
        "objects_jsonl": str(args.objects_jsonl),
        "evidence_jsonl": str(args.evidence_jsonl),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "reviewed_objects": len(rows),
        "parse_ok": parse_ok,
        "parse_ok_ratio": parse_ok / max(len(rows), 1),
        "controlled_label_counts": dict(label_counts),
        "action_counts": dict(action_counts),
        "avg_elapsed_sec": sum(float(r.get("elapsed_sec") or 0.0) for r in rows) / max(len(rows), 1),
    }
    (args.output_dir / "mimo_object_review_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
