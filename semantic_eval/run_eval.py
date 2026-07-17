#!/usr/bin/env python3
"""Run small-sample semantic mask evaluation.

This script is designed to run on the GPU server. It produces deterministic
artifacts for a fixed manifest and can be smoke-tested without a live VLM via
--skip-vlm.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
try:
    from scripts.sam_rle import decode_rle
except ModuleNotFoundError:  # Supports direct `python semantic_eval/run_eval.py` execution.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.sam_rle import decode_rle


COMBOS = [
    "sam2_qwen",
    "sam3_qwen",
    "sam3_sky_qwen",
    "sam3_sky_rules_qwen_review",
    "sky_sam3_qwen",
    "sky_sam3_rules_qwen_review",
]

LABEL_TO_ID = {
    "unknown": 0,
    "other": 1,
    "wall": 2,
    "floor": 3,
    "ceiling": 4,
    "grass": 5,
    "tree": 6,
    "person": 7,
    "car": 8,
    "railing": 9,
    "building": 10,
    "sky": 11,
    "road": 12,
    "water": 13,
    "furniture": 14,
    "pipe": 15,
    "equipment": 16,
    "ignore": 255,
}

LABEL_ALIASES = {
    "墙": "wall", "墙壁": "wall", "墙面": "wall", "wall": "wall",
    "地面": "floor", "地板": "floor", "floor": "floor", "ground": "floor",
    "屋面": "floor", "屋顶": "floor", "平台": "floor", "roof": "floor", "rooftop": "floor",
    "天花板": "ceiling", "天花": "ceiling", "ceiling": "ceiling",
    "草地": "grass", "草": "grass", "grass": "grass",
    "树木": "tree", "树": "tree", "tree": "tree",
    "人": "person", "行人": "person", "person": "person",
    "汽车": "car", "车辆": "car", "car": "car",
    "栏杆": "railing", "护栏": "railing", "railing": "railing",
    "建筑": "building", "建筑物": "building", "building": "building",
    "天空": "sky", "云": "sky", "sky": "sky",
    "道路": "road", "路面": "road", "road": "road",
    "水面": "water", "水": "water", "water": "water",
    "家具": "furniture", "桌子": "furniture", "椅子": "furniture",
    "管道": "pipe", "管线": "pipe", "pipe": "pipe",
    "设备": "equipment", "机器": "equipment", "equipment": "equipment",
    "忽略": "ignore", "无效": "ignore", "invalid": "ignore",
    "其他": "other", "物体": "other", "other": "other",
}


@dataclass
class Mask:
    segmentation: np.ndarray
    area: int
    score: float
    bbox: list[int]
    source: str



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

def normalize_label(label: str) -> str:
    raw = str(label or "").strip().lower()
    if raw in LABEL_ALIASES:
        return LABEL_ALIASES[raw]
    for key, value in LABEL_ALIASES.items():
        if key.lower() in raw:
            return value
    return "other"


def image_to_base64(image: np.ndarray, max_size: int = 1024) -> str:
    import os
    try:
        max_size = int(os.environ.get("VLM_IMAGE_MAX_SIZE", max_size))
    except (TypeError, ValueError):
        pass
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = np.array(Image.fromarray(image).resize((int(w * scale), int(h * scale)), Image.LANCZOS))
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_sam_segmentation(segmentation: Any) -> np.ndarray:
    """Decode dense masks plus legacy and COCO-compressed RLE."""
    if isinstance(segmentation, dict) and {"counts", "size"} <= segmentation.keys():
        return decode_rle(segmentation)
    return np.asarray(segmentation, dtype=bool)


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


class Sam3Runner:
    def __init__(self, model: str, device: int, allow_online_download: bool) -> None:
        self.model = model
        self.device = device
        self.allow_online_download = allow_online_download
        self.generator = None
        self.load_error: str | None = None

    def load(self) -> bool:
        if self.generator is not None:
            return True
        if self.load_error:
            return False
        if not self.allow_online_download:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            from transformers import pipeline
            self.generator = pipeline("mask-generation", model=self.model, device=self.device)
            return True
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            return False

    def run(self, image: np.ndarray, min_area: int = 500) -> list[Mask]:
        if not self.load():
            raise RuntimeError(f"SAM3 model unavailable ({self.model}): {self.load_error}")
        outputs = self.generator(Image.fromarray(image), points_per_batch=64)
        masks: list[Mask] = []
        raw_masks = outputs.get("masks", []) if isinstance(outputs, dict) else []
        raw_scores = outputs.get("scores", []) if isinstance(outputs, dict) else []
        for i, seg_raw in enumerate(raw_masks):
            if hasattr(seg_raw, "cpu"):
                seg_raw = seg_raw.cpu().numpy()
            seg = np.array(seg_raw, dtype=bool)
            area = int(seg.sum())
            if area < min_area:
                continue
            ys, xs = np.where(seg)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else [0, 0, 0, 0]
            score = raw_scores[i] if i < len(raw_scores) else 0.5
            if hasattr(score, "cpu"):
                score = score.cpu().item()
            masks.append(Mask(seg, area, float(score), bbox, "sam3"))
        return sorted(masks, key=lambda m: m.area, reverse=True)


def normalize_overlaps(masks: list[Mask], shape: tuple[int, int], min_area: int = 200) -> list[Mask]:
    h, w = shape
    owner = np.full((h, w), -1, dtype=np.int32)
    score_map = np.full((h, w), -1.0, dtype=np.float32)
    for i, mask in enumerate(masks):
        score = float(mask.score)
        better = mask.segmentation & (score > score_map)
        owner[better] = i
        score_map[better] = score
    out: list[Mask] = []
    for i, mask in enumerate(masks):
        seg = owner == i
        area = int(seg.sum())
        if area < min_area:
            continue
        ys, xs = np.where(seg)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else [0, 0, 0, 0]
        out.append(Mask(seg, area, mask.score, bbox, mask.source))
    return sorted(out, key=lambda m: m.area, reverse=True)


def clip_masks_to_region(masks: list[Mask], keep: np.ndarray, min_area: int) -> list[Mask]:
    out: list[Mask] = []
    for mask in masks:
        seg = mask.segmentation & keep
        area = int(seg.sum())
        if area < min_area:
            continue
        ys, xs = np.where(seg)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else [0, 0, 0, 0]
        out.append(Mask(segmentation=seg, area=area, score=mask.score, bbox=bbox, source=f"{mask.source}_sky_first"))
    return sorted(out, key=lambda m: m.area, reverse=True)


def apply_rules(masks: list[Mask], shape: tuple[int, int]) -> list[Mask]:
    # Keep this intentionally conservative for evaluation: remove tiny fragments
    # and cap pathological mask counts so the VLM prompt stays parseable.
    masks = normalize_overlaps(masks, shape, min_area=500)
    return masks[:80]


def load_sky_mask(existing_path: str | None, image: np.ndarray, model_path: Path | None) -> tuple[np.ndarray, str]:
    h, w = image.shape[:2]
    if existing_path and Path(existing_path).exists():
        mask = np.array(Image.open(existing_path).convert("L").resize((w, h), Image.NEAREST)) > 128
        return mask, "existing"
    if model_path and model_path.exists():
        try:
            import cv2
            net = cv2.dnn.readNetFromONNX(str(model_path))
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            blob = cv2.dnn.blobFromImage(bgr, scalefactor=1 / 255.0, size=(320, 320), swapRB=False, crop=False)
            net.setInput(blob)
            outs = net.forward(net.getUnconnectedOutLayersNames())
            pred = np.mean([o[0, 0] for o in outs], axis=0)
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
            return pred > 0.5, "cv2_dnn"
        except Exception as exc:
            return np.zeros((h, w), dtype=bool), f"failed:{exc}"
    return np.zeros((h, w), dtype=bool), "missing"


def build_sky_first_image(image: np.ndarray, sky: np.ndarray) -> np.ndarray:
    out = image.copy()
    non_sky = ~sky
    if non_sky.any():
        fill = np.median(image[non_sky], axis=0).astype(np.uint8)
    else:
        fill = np.array([127, 127, 127], dtype=np.uint8)
    out[sky] = fill
    return out


def draw_overlay_with_ids(image: np.ndarray, masks: list[Mask], mask_ids: list[int], labels: dict[str, str]) -> np.ndarray:
    import cv2
    rng = np.random.default_rng(42)
    colors = rng.integers(80, 230, size=(max(len(masks), 1), 3), dtype=np.uint8)
    overlay = image.copy()
    for i, (mask, mask_id) in enumerate(zip(masks, mask_ids)):
        color = colors[i].astype(np.float32)
        seg = mask.segmentation
        overlay[seg] = (overlay[seg].astype(np.float32) * 0.45 + color * 0.55).astype(np.uint8)
        ys, xs = np.where(seg)
        if len(xs):
            text = f"{mask_id}:{labels.get(str(mask_id), '?')}"
            cv2.putText(overlay, text, (int(xs.mean()) - 8, int(ys.mean()) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(overlay, text, (int(xs.mean()) - 8, int(ys.mean()) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return overlay


def draw_overlay(image: np.ndarray, masks: list[Mask], labels: dict[str, str]) -> np.ndarray:
    return draw_overlay_with_ids(image, masks, list(range(1, len(masks) + 1)), labels)


def build_instance_png(masks: list[Mask], shape: tuple[int, int]) -> np.ndarray:
    inst = np.zeros(shape, dtype=np.uint16)
    for i, mask in enumerate(masks, start=1):
        inst[mask.segmentation] = i
    return inst


def build_semantic_png(masks: list[Mask], labels: dict[str, str], shape: tuple[int, int],
                       sky: np.ndarray | None = None, mark_sky: bool = False) -> np.ndarray:
    sem = np.zeros(shape, dtype=np.uint8)
    if mark_sky and sky is not None:
        sem[sky] = LABEL_TO_ID["sky"]
    for i, mask in enumerate(masks, start=1):
        label = normalize_label(labels.get(str(i), "unknown"))
        sem[mask.segmentation] = LABEL_TO_ID.get(label, 0)
    return sem


FALLBACK_LABEL_RE = (
    "地面|墙壁|墙面|天花板|草地|树木|人|汽车|栏杆|建筑|天空|道路|水面|家具|管道|设备|其他|忽略|"
    "floor|wall|ceiling|grass|tree|person|car|railing|building|sky|road|water|furniture|pipe|equipment|other|ignore"
)


def parse_text_labels(text: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    patterns = [
        rf"编号\s*(\d{{1,4}}).{{0,240}}?(?:->|类别选择[:：]|类别[:：])\s*({FALLBACK_LABEL_RE})",
        rf"(?:^|\n)\s*[-*]?\s*(\d{{1,4}})\s*(?:[:：]|指向|在|是).{{0,180}}?->\s*({FALLBACK_LABEL_RE})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            mask_id = str(int(match.group(1)))
            labels[mask_id] = normalize_label(match.group(2))
    return labels


def parse_vlm_response(text: str) -> tuple[dict[str, str], bool, str]:
    if not text:
        return {}, False, "empty"
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and first < last:
        try:
            data = json.loads(text[first:last + 1])
        except Exception:
            data = None
        labels: dict[str, str] = {}
        if isinstance(data, dict) and "items" in data:
            for item in data.get("items", []):
                raw_id = str(item.get("mask_id", item.get("id", "")))
                match = re.search(r"(\d+)", raw_id)
                if match:
                    labels[str(int(match.group(1)))] = normalize_label(item.get("label", "other"))
        elif isinstance(data, dict):
            for key, value in data.items():
                match = re.search(r"(\d+)", str(key))
                if match:
                    labels[str(int(match.group(1)))] = normalize_label(str(value))
        if labels:
            return labels, True, "json"

    labels = parse_text_labels(text)
    return labels, bool(labels), "text_fallback" if labels else "none"


def filter_labels_to_ids(labels: dict[str, str], mask_ids: list[int]) -> dict[str, str]:
    wanted = {str(i) for i in mask_ids}
    return {k: v for k, v in labels.items() if k in wanted}


def classify_with_vlm_once(endpoint: str, model: str, image: np.ndarray, masks: list[Mask],
                           mask_ids: list[int], timeout: int, max_tokens: int) -> tuple[dict[str, str], dict[str, Any]]:
    import requests
    overlay = draw_overlay_with_ids(image, masks, mask_ids, {str(i): "?" for i in mask_ids})
    labels_text = "地面, 墙壁, 天花板, 草地, 树木, 人, 汽车, 栏杆, 建筑, 天空, 道路, 水面, 家具, 管道, 设备, 其他, 忽略"
    ids_text = ", ".join(str(i) for i in mask_ids)
    prompt = (
        "/no_think\n"
        f"图片中只需要分类这些编号分割区域：{ids_text}。"
        f"请为每个区域选择一个类别：{labels_text}。"
        "不要解释，不要输出 Markdown，不要逐步推理。"
        "只输出合法 JSON，格式为 {\"items\":[{\"mask_id\":\"1\",\"label\":\"类别\",\"confidence\":0.9}]}。"
    )
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
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
    content = message.get("content", "") or ""
    reasoning = message.get("reasoning_content", "") or ""
    response_text = content or reasoning
    labels, ok, parse_mode = parse_vlm_response(response_text)
    labels = filter_labels_to_ids(labels, mask_ids)
    ok = ok and len(labels) == len(mask_ids)
    return labels, {
        "parse_ok": ok,
        "parse_mode": parse_mode,
        "elapsed_sec": elapsed,
        "finish_reason": data["choices"][0].get("finish_reason"),
        "raw": response_text[:4000],
        "content": content[:2000],
        "reasoning_content": reasoning[:4000],
    }


def classify_with_vlm(endpoint: str, model: str, image: np.ndarray, masks: list[Mask],
                      timeout: int, chunk_size: int, max_tokens: int) -> tuple[dict[str, str], dict[str, Any]]:
    if chunk_size > 0 and len(masks) > chunk_size:
        labels: dict[str, str] = {}
        chunks: list[dict[str, Any]] = []
        elapsed = 0.0
        all_ok = True
        for start in range(0, len(masks), chunk_size):
            end = min(start + chunk_size, len(masks))
            ids = list(range(start + 1, end + 1))
            chunk_labels, info = classify_with_vlm_once(endpoint, model, image, masks[start:end], ids, timeout, max_tokens)
            labels.update(chunk_labels)
            elapsed += float(info.get("elapsed_sec", 0.0))
            all_ok = all_ok and bool(info.get("parse_ok"))
            chunks.append({
                "mask_ids": ids,
                "parse_ok": info.get("parse_ok"),
                "parse_mode": info.get("parse_mode"),
                "finish_reason": info.get("finish_reason"),
                "elapsed_sec": info.get("elapsed_sec"),
                "raw": (info.get("raw") or "")[:1000],
            })
        return labels, {
            "parse_ok": all_ok and len(labels) == len(masks),
            "elapsed_sec": elapsed,
            "chunk_size": chunk_size,
            "chunks": chunks,
        }

    return classify_with_vlm_once(
        endpoint,
        model,
        image,
        masks,
        list(range(1, len(masks) + 1)),
        timeout,
        max_tokens,
    )


def apply_sky_override(masks: list[Mask], labels: dict[str, str], sky: np.ndarray, threshold: float = 0.35) -> dict[str, str]:
    out = dict(labels)
    for i, mask in enumerate(masks, start=1):
        inter = (mask.segmentation & sky).sum()
        ratio = float(inter / max(mask.area, 1))
        if ratio >= threshold:
            out[str(i)] = "sky"
    return out


def masks_signature(masks: list[Mask]) -> tuple[tuple[Any, ...], ...]:
    return tuple((m.source, m.area, round(float(m.score), 5), tuple(m.bbox)) for m in masks)


def complete_mask_labels(labels: dict[str, str], mask_count: int, parse_ok: bool) -> dict[str, str]:
    """Never turn a failed VLM request into a confident `other` label."""
    completed = dict(labels)
    fallback = "other" if parse_ok else "unknown"
    for i in range(1, mask_count + 1):
        completed.setdefault(str(i), fallback)
    return completed


def write_combo_artifacts(combo_dir: Path, image: np.ndarray, masks: list[Mask], labels: dict[str, str],
                          sky: np.ndarray, summary: dict[str, Any], mark_sky_semantic: bool = False) -> None:
    combo_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(combo_dir / "image.png")
    Image.fromarray((sky.astype(np.uint8) * 255)).save(combo_dir / "sky_mask.png")
    Image.fromarray(build_instance_png(masks, image.shape[:2])).save(combo_dir / "instance.png")
    Image.fromarray(build_semantic_png(masks, labels, image.shape[:2], sky, mark_sky_semantic)).save(combo_dir / "semantic.png")
    Image.fromarray(draw_overlay(image, masks, labels)).save(combo_dir / "overlay.png")
    (combo_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2))
    (combo_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


def run_item(item: dict[str, Any], args: argparse.Namespace, sam3: Sam3Runner | None) -> list[dict[str, Any]]:
    image_id = item["image_id"]
    image = np.array(Image.open(item["image_path"]).convert("RGB"))
    h, w = image.shape[:2]
    sky, sky_source = load_sky_mask(item.get("sky_mask_path"), image, args.sky_model)

    sam2_masks = decode_sam2_masks(args.sam_masks_dir / f"{image_id}_sam_masks.json")
    sam3_masks_cache: list[Mask] | None = None
    sky_sam3_masks_cache: list[Mask] | None = None
    sam3_blocker = ""
    sky_sam3_blocker = ""
    vlm_cache: dict[tuple[tuple[Any, ...], ...], tuple[dict[str, str], dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []

    for combo in args.combos:
        blocked = False
        blocker = ""
        if combo.startswith("sam2"):
            masks = sam2_masks
        elif combo.startswith("sky_sam3"):
            if sky_sam3_masks_cache is None:
                if sam3 is None:
                    sky_sam3_masks_cache = []
                    sky_sam3_blocker = "sam3_skipped_by_user"
                else:
                    try:
                        sky_first_image = build_sky_first_image(image, sky)
                        sky_sam3_masks_cache = clip_masks_to_region(
                            sam3.run(sky_first_image, min_area=args.min_area),
                            ~sky,
                            args.min_area,
                        )
                    except Exception as exc:
                        sky_sam3_masks_cache = []
                        sky_sam3_blocker = str(exc)
            masks = sky_sam3_masks_cache or []
            blocked = bool(sky_sam3_blocker)
            blocker = sky_sam3_blocker
        else:
            if sam3_masks_cache is None:
                if sam3 is None:
                    sam3_masks_cache = []
                    sam3_blocker = "sam3_skipped_by_user"
                else:
                    try:
                        sam3_masks_cache = sam3.run(image, min_area=args.min_area)
                    except Exception as exc:
                        sam3_masks_cache = []
                        sam3_blocker = str(exc)
            masks = sam3_masks_cache or []
            blocked = bool(sam3_blocker)
            blocker = sam3_blocker

        masks = normalize_overlaps(masks, (h, w), min_area=args.min_area)
        if combo.endswith("_rules_qwen_review"):
            masks = apply_rules(masks, (h, w))

        labels: dict[str, str] = {}
        vlm_info: dict[str, Any] = {"parse_ok": False, "skipped": args.skip_vlm}
        if not args.skip_vlm and masks:
            cache_key = masks_signature(masks)
            if cache_key in vlm_cache:
                cached_labels, cached_info = vlm_cache[cache_key]
                labels = dict(cached_labels)
                vlm_info = dict(cached_info)
                vlm_info["cached"] = True
            else:
                try:
                    labels, vlm_info = classify_with_vlm(
                        args.vlm_endpoint,
                        args.vlm_model,
                        image,
                        masks,
                        args.vlm_timeout,
                        args.vlm_chunk_size,
                        args.vlm_max_tokens,
                    )
                    vlm_cache[cache_key] = (dict(labels), dict(vlm_info))
                except Exception as exc:
                    vlm_info = {"parse_ok": False, "error": str(exc)}
                    labels = {}
        labels = complete_mask_labels(labels, len(masks), bool(vlm_info.get("parse_ok")))

        if "_sky_" in combo:
            labels = apply_sky_override(masks, labels, sky)

        coverage = float(np.logical_or.reduce([m.segmentation for m in masks]).sum() / (h * w)) if masks else 0.0
        mark_sky_semantic = combo.startswith("sky_sam3")
        sky_labeled_pixels = int(sky.sum()) if mark_sky_semantic else 0
        for i, mask in enumerate(masks, start=1):
            if labels.get(str(i)) == "sky":
                sky_labeled_pixels += int(mask.segmentation.sum())
        summary = {
            "image_id": image_id,
            "combo": combo,
            "sky_first": mark_sky_semantic,
            "blocked": blocked,
            "blocker": blocker,
            "mask_count": len(masks),
            "coverage": coverage,
            "coverage_with_sky": float((np.logical_or.reduce([m.segmentation for m in masks]) | sky).sum() / (h * w)) if masks else float(sky.sum() / (h * w)),
            "sky_source": sky_source,
            "sky_mask_ratio": float(sky.sum() / (h * w)),
            "sky_labeled_ratio": float(sky_labeled_pixels / (h * w)),
            "semantic_ready": bool(vlm_info.get("parse_ok")),
            "vlm": vlm_info,
        }
        combo_dir = args.output_dir / "images" / image_id / combo
        write_combo_artifacts(combo_dir, image, masks, labels, sky, summary, mark_sky_semantic)
        rows.append(summary)
        status = f" blocked={blocker[:80]}" if blocked else ""
        print(f"{image_id} {combo}: masks={len(masks)} coverage={coverage:.3f} parse={vlm_info.get('parse_ok')}{status}")
    return rows


def write_report(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    by_combo: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_combo.setdefault(row["combo"], []).append(row)
    report: dict[str, Any] = {"total_rows": len(rows), "combos": {}}
    for combo, items in by_combo.items():
        parse_values = [bool(x.get("vlm", {}).get("parse_ok")) for x in items]
        report["combos"][combo] = {
            "images": len(items),
            "blocked_images": int(sum(1 for x in items if x.get("blocked"))),
            "blockers": sorted({x.get("blocker", "") for x in items if x.get("blocker")}),
            "avg_mask_count": float(np.mean([x["mask_count"] for x in items])) if items else 0,
            "avg_coverage": float(np.mean([x["coverage"] for x in items])) if items else 0,
            "avg_coverage_with_sky": float(np.mean([x.get("coverage_with_sky", x["coverage"]) for x in items])) if items else 0,
            "avg_sky_mask_ratio": float(np.mean([x["sky_mask_ratio"] for x in items])) if items else 0,
            "vlm_parse_success_rate": float(np.mean(parse_values)) if parse_values else 0,
        }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic small-sample evaluation")
    parser.add_argument("--manifest", type=Path,
                        default=Path("/root/epfs/manifold_3dgs_project/processed/semantic_eval_20260605/manifest.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/root/epfs/manifold_3dgs_project/processed/semantic_eval_20260605"))
    parser.add_argument("--sam-masks-dir", type=Path,
                        default=Path("/root/epfs/manifold_3dgs_project/processed/sam_masks"))
    parser.add_argument("--sky-model", type=Path, default=Path("/root/epfs/skyseg_model/skyseg.onnx"))
    parser.add_argument("--vlm-endpoint", default="http://localhost:8001/v1/chat/completions")
    parser.add_argument("--vlm-model", default="Qwen3.6-35B-A3B")
    parser.add_argument("--vlm-timeout", type=int, default=180)
    parser.add_argument("--vlm-chunk-size", type=int, default=20,
                        help="Classify masks in chunks to keep Qwen from spending all tokens on long reasoning.")
    parser.add_argument("--vlm-max-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0,
                        help="Start offset in manifest items, useful for resuming a partial run.")
    parser.add_argument("--end-index", type=int, default=None,
                        help="Exclusive end offset in manifest items.")
    parser.add_argument("--combos", nargs="+", default=COMBOS, choices=COMBOS)
    parser.add_argument("--skip-vlm", action="store_true")
    parser.add_argument("--skip-sam3", action="store_true")
    parser.add_argument("--sam3-model", default="facebook/sam3")
    parser.add_argument("--sam3-device", type=int, default=0)
    parser.add_argument("--allow-online-model-download", action="store_true",
                        help="Allow transformers to download SAM3 weights if they are not cached locally.")
    parser.add_argument("--min-area", type=int, default=500)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    all_items = manifest["items"]
    end_index = args.end_index if args.end_index is not None else len(all_items)
    items = all_items[args.start_index:end_index]
    if args.limit:
        items = items[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.used.json").write_text(json.dumps({
        "start_index": args.start_index,
        "end_index": end_index,
        "limit": args.limit,
        "items": items,
    }, ensure_ascii=False, indent=2))

    needs_sam3 = any(c.startswith("sam3") or c.startswith("sky_sam3") for c in args.combos)
    sam3 = None if args.skip_sam3 or not needs_sam3 else Sam3Runner(
        model=args.sam3_model,
        device=args.sam3_device,
        allow_online_download=args.allow_online_model_download,
    )

    rows: list[dict[str, Any]] = []
    for item in items:
        rows.extend(run_item(item, args, sam3))
        write_report(args.output_dir, rows)
    write_report(args.output_dir, rows)
    print(f"Wrote report to {args.output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
