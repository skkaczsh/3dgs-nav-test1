#!/usr/bin/env python3
"""Patch semantic_eval scripts to support OpenAI-compatible Bearer auth.

The patched scripts read the API key from an environment variable at runtime.
No secret is written to disk.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TARGETS = [
    "run_eval.py",
    "review_merged_labels_prompt_v2.py",
    "complete_unknown_regions.py",
]


HEADERS_HELPER = '''


def vlm_headers() -> dict[str, str]:
    import os

    headers = {"Content-Type": "application/json"}
    key = os.environ.get("VLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers
'''.rstrip()


PAYLOAD_HELPER = '''


def apply_vlm_payload_options(payload: dict) -> dict:
    import os

    if os.environ.get("VLM_DISABLE_THINKING", "").lower() in {"1", "true", "yes", "on"}:
        payload["thinking"] = {"type": "disabled"}
    return payload
'''.rstrip()


POST_HELPER = '''


def vlm_post(requests_module, endpoint: str, payload: dict, timeout: int):
    import os
    import sys
    import time

    retries = int(os.environ.get("VLM_RETRIES", "2"))
    sleep_base = float(os.environ.get("VLM_RETRY_SLEEP", "5"))
    retry_statuses = {429, 500, 502, 503, 504}
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
                body = response.text[:2000].replace("\\\\n", " ")
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
'''.rstrip()


HELPER = HEADERS_HELPER + PAYLOAD_HELPER + POST_HELPER


def patch_text(text: str) -> tuple[str, bool]:
    changed = False
    needs_payload_helper = False
    needs_post_helper = False
    if "def vlm_headers()" not in text:
        marker = "\n\ndef "
        idx = text.find(marker)
        if idx == -1:
            idx = text.find("def ")
        if idx == -1:
            text = text.rstrip() + HELPER + "\n"
        else:
            text = text[:idx] + HELPER + text[idx:]
        changed = True
    else:
        apply_idx = text.find("def apply_vlm_payload_options(")
        post_idx = text.find("def vlm_post(")
        use_idx = text.find("apply_vlm_payload_options(payload)")
        needs_payload_helper = apply_idx == -1 or (use_idx != -1 and apply_idx > use_idx)
        needs_post_helper = post_idx == -1
    if "def vlm_post(" in text and "VLM HTTP" not in text:
        updated, count = re.subn(
            r"\n\ndef vlm_post\(requests_module, endpoint: str, payload: dict, timeout: int\):.*?raise RuntimeError\(\"VLM request failed without response\"\)",
            POST_HELPER,
            text,
            count=1,
            flags=re.S,
        )
        if count == 1:
            text = updated
            changed = True
    if "def vlm_headers()" in text and needs_payload_helper:
        marker = "\n\ndef normalize_label"
        if marker not in text:
            marker = "\n\ndef classify_once"
        if marker not in text:
            marker = "\n\ndef classify_with_vlm_once"
        if marker not in text:
            marker = "\n\ndef main"
        idx = text.find(marker)
        if idx == -1:
            text = text.rstrip() + PAYLOAD_HELPER + "\n"
        else:
            text = text[:idx] + PAYLOAD_HELPER + text[idx:]
        changed = True
    if "def vlm_headers()" in text and needs_post_helper:
        marker = "\n\ndef normalize_label"
        if marker not in text:
            marker = "\n\ndef classify_once"
        if marker not in text:
            marker = "\n\ndef classify_with_vlm_once"
        if marker not in text:
            marker = "\n\ndef main"
        idx = text.find(marker)
        if idx == -1:
            text = text.rstrip() + POST_HELPER + "\n"
        else:
            text = text[:idx] + POST_HELPER + text[idx:]
        changed = True
    old = "requests.post(endpoint, json=payload, timeout=timeout)"
    new = "requests.post(endpoint, json=payload, headers=vlm_headers(), timeout=timeout)"
    if old in text:
        text = text.replace(old, new)
        changed = True
    old = "requests.post(endpoint, json=payload, headers=vlm_headers(), timeout=timeout)"
    new = "requests.post(endpoint, json=apply_vlm_payload_options(payload), headers=vlm_headers(), timeout=timeout)"
    if old in text:
        text = text.replace(old, new)
        changed = True
    old = "requests.post(endpoint, json=apply_vlm_payload_options(payload), headers=vlm_headers(), timeout=timeout)"
    new = "vlm_post(requests, endpoint, payload, timeout)"
    if old in text:
        text = text.replace(old, new)
        changed = True
    old_resize = "def image_to_base64(image: np.ndarray, max_size: int = 1024) -> str:\n    h, w = image.shape[:2]"
    new_resize = (
        "def image_to_base64(image: np.ndarray, max_size: int = 1024) -> str:\n"
        "    import os\n"
        "    try:\n"
        "        max_size = int(os.environ.get(\"VLM_IMAGE_MAX_SIZE\", max_size))\n"
        "    except (TypeError, ValueError):\n"
        "        pass\n"
        "    h, w = image.shape[:2]"
    )
    if old_resize in text:
        text = text.replace(old_resize, new_resize)
        changed = True
    return text, changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for name in TARGETS:
        path = args.semantic_root / name
        if not path.exists():
            rows.append({"path": str(path), "patched": False, "changed": False, "error": "missing"})
            continue
        original = path.read_text(encoding="utf-8")
        updated, changed = patch_text(original)
        if changed and not args.dry_run:
            backup = path.with_suffix(path.suffix + ".vlm_auth_bak")
            if not backup.exists():
                backup.write_text(original, encoding="utf-8")
            path.write_text(updated, encoding="utf-8")
        rows.append({"path": str(path), "patched": True, "changed": changed, "dry_run": args.dry_run})

    report = {
        "semantic_root": str(args.semantic_root),
        "dry_run": args.dry_run,
        "patched_count": sum(1 for row in rows if row.get("patched")),
        "changed_count": sum(1 for row in rows if row.get("changed")),
        "errors": [row for row in rows if row.get("error")],
        "files": rows,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
