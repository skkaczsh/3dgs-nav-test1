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
    "building_part",
    "unknown",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def prompt_for_object(obj: dict[str, Any], evidence_rows: list[dict[str, Any]]) -> str:
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
        ]
        if key in obj
    }
    evidence = [
        {
            "rank": row.get("rank"),
            "frame_id": row.get("frame_id"),
            "cam_id": row.get("cam_id"),
            "projected_points": row.get("projected_points"),
            "bbox_area_ratio": row.get("bbox_area_ratio"),
            "median_depth": row.get("median_depth"),
            "bbox_xyxy": row.get("bbox_xyxy"),
        }
        for row in evidence_rows
    ]
    labels = ", ".join(CONTROLLED_LABELS)
    return (
        "You are reviewing one 3D point-cloud object from an outdoor parking-lot scan. "
        "Images are undistorted camera evidence. Red points / yellow boxes indicate where this 3D object projects. "
        "Use the images plus the 3D geometry summary. Do not classify the whole image; classify only the projected object.\n\n"
        f"Allowed controlled labels: {labels}.\n"
        "Important rules:\n"
        "- Broad flat pavement is floor, not railing/car.\n"
        "- Vertical building facade or wall panels are wall/building_part, not car/railing.\n"
        "- Railing/guardrail should be thin linear metal fence/handrail geometry.\n"
        "- Car must be an actual vehicle body, not a wall reflection, facade, or flat surface.\n"
        "- HVAC outdoor units, utility boxes, exposed machines, or installed mechanical devices are equipment/hvac_outdoor_unit, not wall.\n"
        "- Doors and windows are door_or_window unless the evidence only shows an unresolved building surface fragment.\n"
        "- If evidence mostly shows a surface fragment, say surface_fragment and choose floor/wall/grass/building_part.\n"
        "- If evidence is unclear, choose unknown and action review_manually.\n\n"
        "Return strict JSON only with keys:\n"
        "{"
        "\"controlled_label\": string, "
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
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_for_object(obj, evidence_rows)}]
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
    return {
        "object_id": int(obj["object_id"]),
        "input_semantic_label": obj.get("semantic_label", ""),
        "evidence_count": len(evidence_rows),
        "image_mode": image_mode,
        "raw_response": text,
        "parsed": parsed,
        "parse_error": "",
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
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = repr(exc)
            time.sleep(min(2.0 * (attempt + 1), 8.0))
    return {
        "object_id": int(obj["object_id"]),
        "input_semantic_label": obj.get("semantic_label", ""),
        "evidence_count": len(ev_rows),
        "image_mode": args.image_mode,
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
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [pool.submit(review_one, task) for task in tasks]
        for fut in as_completed(futures):
            rows.append(fut.result())
            print(json.dumps({
                "done": len(rows),
                "object_id": rows[-1]["object_id"],
                "parsed": bool(rows[-1].get("parsed")),
                "parse_error": rows[-1].get("parse_error", "")[:120],
            }, ensure_ascii=False), flush=True)
    rows.sort(key=lambda r: int(r["object_id"]))
    write_jsonl(args.output_dir / "mimo_object_review.jsonl", rows)

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
