#!/usr/bin/env python3
"""Re-label merged SAM2 masks with a stricter outdoor/stuff prompt."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from run_eval import (
    LABEL_TO_ID,
    Mask,
    apply_sky_override,
    draw_overlay_with_ids,
    image_to_base64,
    normalize_label,
    parse_vlm_response,
    write_combo_artifacts,
)


PROMPT = """
/no_think
你正在复审室外扫描/全景图的合并 mask。天空已经由 SkyMask 单独处理；请只分类图中编号区域。

只允许输出这些英文标签之一：
floor, road, wall, building, railing, equipment, tree, grass, car, person, other, ignore

关键规则：
1. 水平或近似水平的大面积可行走/可承载表面，输出 floor。包括地面、屋面、屋顶平台、水泥平台、楼顶地坪、灰色铺装面。
2. 不要因为水平屋面属于建筑物，就把它标成 building；水平面优先 floor。
3. building 只用于独立建筑体、远处楼体、明显垂直的建筑立面或建筑块。
4. wall 用于近处墙面、围墙、女儿墙、立面墙，不用于水平地面/屋面。
5. railing 用于栏杆、护栏、细长网格/扶手。
6. equipment 用于空调外机、管道、机械设备、箱体、天线、太阳能/机电设施。
7. tree/grass/car/person 按可见物体分类。
8. 如果区域是阴影但仍属于地面/屋面，输出 floor。
9. 如果区域横跨多个类别且无法判断，输出 other。无效黑边/镜头边缘输出 ignore。
10. 全景/鱼眼图中，下半部或图像中心的大面积灰色水泥/铺装/屋面，即使透视扭曲，也优先输出 floor。
11. 如果几何信息显示区域面积很大且中心在图像下半部，除非明显是树/车/栏杆/设备/垂直楼体，否则优先 floor。

输出必须是合法 JSON，不要解释，不要 Markdown：
{"items":[{"mask_id":"1","label":"floor","confidence":0.90}]}
""".strip()



def vlm_headers() -> dict[str, str]:
    import os

    headers = {"Content-Type": "application/json"}
    key = os.environ.get("VLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def apply_vlm_payload_options(payload: dict) -> dict:
    import os

    if os.environ.get("VLM_DISABLE_THINKING", "").lower() in {"1", "true", "yes", "on"}:
        payload["thinking"] = {"type": "disabled"}
    return payload


def vlm_post(requests_module, endpoint: str, payload: dict, timeout: int):
    import os
    import sys
    import time

    retries = int(os.environ.get("VLM_RETRIES", "2"))
    sleep_base = float(os.environ.get("VLM_RETRY_SLEEP", "5"))
    retry_statuses = {429, 500, 502, 503, 504}
    extra_statuses = os.environ.get("VLM_RETRY_STATUS_CODES", "")
    for raw_status in extra_statuses.split(","):
        raw_status = raw_status.strip()
        if raw_status:
            retry_statuses.add(int(raw_status))
    last_exc = None
    for attempt in range(retries + 1):
        try:
            response = requests_module.post(
                endpoint,
                json=apply_vlm_payload_options(payload),
                headers=vlm_headers(),
                timeout=timeout,
            )
            if response.status_code >= 400:
                body = response.text[:2000].replace("\\n", " ")
                print(f"VLM HTTP {response.status_code}: {body}", file=sys.stderr, flush=True)
            if response.status_code not in retry_statuses or attempt >= retries:
                return response
            time.sleep(sleep_base * (attempt + 1))
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                raise
            time.sleep(sleep_base * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("VLM request failed without response")

def load_masks(combo_dir: Path) -> list[Mask]:
    inst = np.array(Image.open(combo_dir / "instance.png"))
    masks: list[Mask] = []
    for mask_id in sorted(int(x) for x in np.unique(inst) if int(x) > 0):
        seg = inst == mask_id
        area = int(seg.sum())
        ys, xs = np.where(seg)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else [0, 0, 0, 0]
        masks.append(Mask(seg, area, 1.0, bbox, "sam2_sky_label_merge_prompt_v2"))
    return masks


def classify_once(endpoint: str, model: str, image: np.ndarray, masks: list[Mask],
                  mask_ids: list[int], timeout: int, max_tokens: int) -> tuple[dict[str, str], dict[str, Any]]:
    overlay = draw_overlay_with_ids(image, masks, mask_ids, {str(i): "?" for i in mask_ids})
    ids_text = ", ".join(str(i) for i in mask_ids)
    h, w = image.shape[:2]
    region_lines = []
    for mask, mask_id in zip(masks, mask_ids):
        x0, y0, x1, y1 = mask.bbox
        cx = (x0 + x1) / 2 / max(w, 1)
        cy = (y0 + y1) / 2 / max(h, 1)
        area_frac = mask.area / max(h * w, 1)
        region_lines.append(
            f"- {mask_id}: area={area_frac:.3f}, center=({cx:.2f},{cy:.2f}), "
            f"bbox=({x0/w:.2f},{y0/h:.2f},{x1/w:.2f},{y1/h:.2f})"
        )
    region_text = "\n".join(region_lines)
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT + f"\n\n需要分类的编号：{ids_text}\n\n区域几何信息，坐标已归一化，y 越大越靠近图像下方：\n{region_text}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(overlay)}"}},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.time()
    resp = vlm_post(requests, endpoint, payload, timeout)
    elapsed = time.time() - started
    resp.raise_for_status()
    data = resp.json()
    message = data["choices"][0]["message"]
    text = message.get("content", "") or message.get("reasoning_content", "") or ""
    labels, ok, parse_mode = parse_vlm_response(text)
    wanted = {str(i) for i in mask_ids}
    labels = {k: normalize_label(v) for k, v in labels.items() if k in wanted}
    return labels, {
        "parse_ok": ok and len(labels) == len(mask_ids),
        "parse_mode": parse_mode,
        "elapsed_sec": elapsed,
        "finish_reason": data["choices"][0].get("finish_reason"),
        "raw": text[:2000],
    }


def classify(endpoint: str, model: str, image: np.ndarray, masks: list[Mask],
             chunk_size: int, timeout: int, max_tokens: int) -> tuple[dict[str, str], dict[str, Any]]:
    labels: dict[str, str] = {}
    chunks: list[dict[str, Any]] = []
    all_ok = True
    for start in range(0, len(masks), chunk_size):
        end = min(start + chunk_size, len(masks))
        ids = list(range(start + 1, end + 1))
        chunk_labels, info = classify_once(endpoint, model, image, masks[start:end], ids, timeout, max_tokens)
        labels.update(chunk_labels)
        all_ok = all_ok and bool(info.get("parse_ok"))
        chunks.append({"mask_ids": ids, **info})
    for i in range(1, len(masks) + 1):
        labels.setdefault(str(i), "other")
    return labels, {"parse_ok": all_ok and len(labels) == len(masks), "chunks": chunks}


def process_one(image_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    src = image_dir / args.source_combo
    out = image_dir / args.output_combo
    image = np.array(Image.open(src / "image.png").convert("RGB"))
    sky = np.array(Image.open(src / "sky_mask.png").convert("L")) > 128
    masks = load_masks(src)
    h, w = image.shape[:2]
    if masks:
        labels, vlm_info = classify(
            args.vlm_endpoint,
            args.vlm_model,
            image,
            masks,
            args.chunk_size,
            args.vlm_timeout,
            args.vlm_max_tokens,
        )
    else:
        labels, vlm_info = {}, {"parse_ok": False, "skipped": "no_masks"}
    labels = apply_sky_override(masks, labels, sky, threshold=0.35)
    union = np.logical_or.reduce([m.segmentation for m in masks]) if masks else np.zeros((h, w), dtype=bool)
    non_sky = ~sky
    sem_floor = np.zeros((h, w), dtype=bool)
    for i, mask in enumerate(masks, start=1):
        if labels.get(str(i)) == "floor":
            sem_floor |= mask.segmentation
    summary = {
        "image_id": image_dir.name,
        "combo": args.output_combo,
        "source_combo": args.source_combo,
        "blocked": False,
        "blocker": "",
        "mask_count": len(masks),
        "coverage": float(union.sum() / (h * w)),
        "non_sky_coverage": float((union & non_sky).sum() / max(non_sky.sum(), 1)),
        "ground_non_sky_ratio": float((sem_floor & non_sky).sum() / max(non_sky.sum(), 1)),
        "sky_source": "from_source_combo",
        "sky_mask_ratio": float(sky.sum() / (h * w)),
        "sky_labeled_ratio": float(sky.sum() / (h * w)),
        "vlm": vlm_info,
    }
    write_combo_artifacts(out, image, masks, labels, sky, summary, mark_sky_semantic=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/root/epfs/manifold_3dgs_project/processed/semantic_eval_20260605"))
    parser.add_argument("--source-combo", default="sam2_sky_label_merge_qwen_review")
    parser.add_argument("--output-combo", default="sam2_sky_label_merge_prompt_v2")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Optional manifest; when set, process only image_ids listed in it.")
    parser.add_argument("--vlm-endpoint", default="http://localhost:8001/v1/chat/completions")
    parser.add_argument("--vlm-model", default="Qwen3.6-35B-A3B-Q4_K_M")
    parser.add_argument("--vlm-timeout", type=int, default=180)
    parser.add_argument("--vlm-max-tokens", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    image_dirs = sorted(p for p in (args.output_dir / "images").iterdir() if (p / args.source_combo).exists())
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        wanted = {item["image_id"] for item in manifest.get("items", [])}
        image_dirs = [p for p in image_dirs if p.name in wanted]
    if args.limit:
        image_dirs = image_dirs[:args.limit]
    rows = []
    for image_dir in image_dirs:
        row = process_one(image_dir, args)
        rows.append(row)
        print(f"{row['image_id']} {args.output_combo}: masks={row['mask_count']} non_sky={row['non_sky_coverage']:.3f} ground={row['ground_non_sky_ratio']:.3f} parse={row['vlm'].get('parse_ok')}")
    report = {
        "combo": args.output_combo,
        "source_combo": args.source_combo,
        "images": len(rows),
        "avg_non_sky_coverage": float(np.mean([r["non_sky_coverage"] for r in rows])) if rows else 0.0,
        "avg_ground_non_sky_ratio": float(np.mean([r["ground_non_sky_ratio"] for r in rows])) if rows else 0.0,
        "parse_success_rate": float(np.mean([bool(r["vlm"].get("parse_ok")) for r in rows])) if rows else 0.0,
        "rows": rows,
    }
    path = args.output_dir / f"{args.output_combo}_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
