#!/usr/bin/env python3
"""Review cross-candidate merge proposals with a VLM."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from vlm_scene_prompt import merge_review_prompt


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def encode_image_data_url(path: Path, long_edge: int = 1280, jpeg_quality: int = 88) -> str:
    if long_edge <= 0:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        suffix = path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        return f"data:{mime};base64,{data}"
    import io

    img = Image.open(path).convert("RGB")
    scale = min(1.0, long_edge / max(img.size))
    if scale < 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg"
    return f"data:{mime};base64,{data}"


def prompt_for_item(item: dict) -> str:
    return merge_review_prompt(item)


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_decision(parsed: dict[str, Any]) -> dict[str, Any]:
    decision = str(parsed.get("decision", "uncertain")).strip().lower()
    if decision not in {"merge", "keep_split", "uncertain"}:
        decision = "uncertain"
    relation = str(parsed.get("physical_relation", "unclear")).strip().lower()
    confidence = parsed.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    evidence = parsed.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        evidence = []
    return {
        "decision": decision,
        "confidence": confidence,
        "physical_relation": relation,
        "reason": str(parsed.get("reason", "")).strip(),
        "evidence": [str(v).strip() for v in evidence if str(v).strip()],
        "risk": str(parsed.get("risk", "")).strip(),
    }


def build_payload(item: dict, sheet_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_for_item(item)},
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image_data_url(sheet_path, args.image_long_edge, args.jpeg_quality)},
                    },
                ],
            }
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if not args.enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def error_result(item: dict, sheet_path: Path, reason: str, raw_content: str = "", usage: dict | None = None) -> dict:
    return {
        "review_id": item["review_id"],
        "object_a": item["proposal"]["object_a"],
        "object_b": item["proposal"]["object_b"],
        "candidate_a": item["proposal"]["candidate_a"],
        "candidate_b": item["proposal"]["candidate_b"],
        "sheet_path": str(sheet_path),
        "vlm": {"decision": "uncertain", "confidence": 0.0, "reason": reason},
        "raw_content": raw_content,
        "usage": usage or {},
        "status": "error",
    }


def call_vlm(item: dict, sheet_path: Path, args: argparse.Namespace) -> dict:
    payload = build_payload(item, sheet_path, args)
    response = requests.post(args.endpoint, json=payload, timeout=args.timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:2000]}") from exc
    body = response.json()
    message = body["choices"][0]["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    usage = body.get("usage", {})
    try:
        parsed = normalize_decision(extract_json(content))
    except Exception as exc:  # noqa: BLE001
        raw = content or reasoning
        if not content and reasoning:
            reason = "empty assistant content; model returned reasoning_content only. disable thinking or increase max_tokens."
        else:
            reason = str(exc)
        return error_result(item, sheet_path, reason, raw_content=raw[:8000], usage=usage)
    return {
        "review_id": item["review_id"],
        "object_a": item["proposal"]["object_a"],
        "object_b": item["proposal"]["object_b"],
        "candidate_a": item["proposal"]["candidate_a"],
        "candidate_b": item["proposal"]["candidate_b"],
        "sheet_path": str(sheet_path),
        "vlm": parsed,
        "raw_content": content,
        "usage": usage,
        "status": "ok",
    }


def existing_results(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {row["review_id"]: row for row in load_jsonl(path) if row.get("status") == "ok"}


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "vlm_merge_review_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    csv_path = output_dir / "vlm_merge_review_results.csv"
    fields = [
        "review_id",
        "object_a",
        "object_b",
        "candidate_a",
        "candidate_b",
        "decision",
        "confidence",
        "physical_relation",
        "reason",
        "risk",
        "status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            vlm = row.get("vlm", {})
            writer.writerow(
                {
                    "review_id": row.get("review_id", ""),
                    "object_a": row.get("object_a", ""),
                    "object_b": row.get("object_b", ""),
                    "candidate_a": row.get("candidate_a", ""),
                    "candidate_b": row.get("candidate_b", ""),
                    "decision": vlm.get("decision", ""),
                    "confidence": vlm.get("confidence", ""),
                    "physical_relation": vlm.get("physical_relation", ""),
                    "reason": vlm.get("reason", ""),
                    "risk": vlm.get("risk", ""),
                    "status": row.get("status", ""),
                }
            )
    counts: dict[str, int] = {}
    for row in rows:
        key = row.get("vlm", {}).get("decision", row.get("status", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    report = {
        "result_jsonl": str(jsonl_path),
        "result_csv": str(csv_path),
        "item_count": len(rows),
        "decision_counts": counts,
        "ok_count": sum(1 for row in rows if row.get("status") == "ok"),
        "error_count": sum(1 for row in rows if row.get("status") != "ok"),
    }
    (output_dir / "vlm_merge_review_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-jsonl", type=Path, required=True)
    parser.add_argument("--contact-sheet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://localhost:8001/v1/chat/completions")
    parser.add_argument("--model", default="Qwen3.6-35B-A3B-Q4_K_M")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--image-long-edge", type=int, default=1280)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    items = load_jsonl(args.review_jsonl)
    result_path = args.output_dir / "vlm_merge_review_results.jsonl"
    done = existing_results(result_path) if args.resume else {}
    rows = [done[item["review_id"]] for item in items if item["review_id"] in done]
    pending = [item for item in items if item["review_id"] not in done]
    errors = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {}
        for item in pending:
            sheet = args.contact_sheet_dir / f"{item['review_id']}_contact_sheet.jpg"
            futures[pool.submit(call_vlm, item, sheet, args)] = item
        for future in as_completed(futures):
            item = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:  # noqa: BLE001
                sheet = args.contact_sheet_dir / f"{item['review_id']}_contact_sheet.jpg"
                errors.append(error_result(item, sheet, str(exc)))
    rows.extend(errors)
    rows.sort(key=lambda row: row["review_id"])
    write_outputs(rows, args.output_dir)
    print(json.dumps({"items": len(rows), "errors": len(errors), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
