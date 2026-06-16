#!/usr/bin/env python3
import json
import os
import subprocess
from pathlib import Path


ROOT = Path("/Users/skkac/Work/SCAN")
EXPERIMENT_DIR = ROOT / "new_route/experiments/fine_object_grounded_small_eval"
ASSET_ROOT = ROOT / "server_conceptseg_fine_object_runlist_v008/assets"
REMOTE = "root@10.0.8.114"
REMOTE_PORT = "31909"
REMOTE_PROJECT = "/root/epfs/vlm_seg_project"
REMOTE_PYTHON = "/root/epfs/conda_envs/vlm_seg/bin/python"
REMOTE_TMP = f"{REMOTE_PROJECT}/tmp_fine_object_grounded_small_eval"
PROMPTS = [
    "railing",
    "guardrail",
    "handrail",
    "metal fence",
    "pipe",
    "cable",
    "HVAC outdoor unit",
]

PROMPT_GROUPS = {
    "railing": ["railing", "guardrail", "handrail", "metal fence"],
    "pipe": ["pipe", "cable"],
    "hvac": ["HVAC outdoor unit", "outdoor unit", "air conditioning unit"],
    "equipment": ["HVAC outdoor unit", "outdoor unit", "air conditioning unit"],
}

SAMPLES = [
    {"id": "review_001__a0", "rel": "review_001/a0", "focus": ["hvac", "equipment"]},
    {"id": "review_002__a0", "rel": "review_002/a0", "focus": ["railing", "equipment"]},
    {"id": "review_002__a1", "rel": "review_002/a1", "focus": ["railing", "equipment"]},
    {"id": "review_002__b0", "rel": "review_002/b0", "focus": ["railing", "equipment"]},
    {"id": "review_003__a0", "rel": "review_003/a0", "focus": ["railing", "equipment"]},
    {"id": "review_003__b0", "rel": "review_003/b0", "focus": ["railing", "equipment"]},
    {"id": "review_003__b1", "rel": "review_003/b1", "focus": ["railing", "equipment"]},
    {"id": "review_004__b0", "rel": "review_004/b0", "focus": ["railing", "equipment"]},
    {"id": "review_005__a1", "rel": "review_005/a1", "focus": ["railing", "equipment"]},
    {"id": "review_005__b1", "rel": "review_005/b1", "focus": ["railing", "equipment"]},
    {"id": "review_006__a0", "rel": "review_006/a0", "focus": ["pipe", "equipment"]},
    {"id": "review_006__a1", "rel": "review_006/a1", "focus": ["pipe", "equipment"]},
    {"id": "review_008__a0", "rel": "review_008/a0", "focus": ["railing", "equipment"]},
    {"id": "review_008__b0", "rel": "review_008/b0", "focus": ["railing", "equipment"]},
]


def load_label_counts(rel: str) -> dict:
    label_path = ASSET_ROOT / rel / "labels.json"
    data = json.loads(label_path.read_text())
    counts = {}
    for value in data.values():
        counts[value] = counts.get(value, 0) + 1
    return counts


def build_manifest() -> dict:
    samples = []
    for sample in SAMPLES:
        rel = sample["rel"]
        local_dir = ASSET_ROOT / rel
        prompt_terms = []
        prompt_groups = []
        for focus in sample["focus"]:
            group_terms = []
            for term in PROMPT_GROUPS.get(focus, []):
                if term not in prompt_terms:
                    prompt_terms.append(term)
                if term not in group_terms:
                    group_terms.append(term)
            if group_terms:
                prompt_groups.append({"focus": focus, "terms": group_terms})
        samples.append(
            {
                **sample,
                "image": str(local_dir / "image.png"),
                "overlay": str(local_dir / "overlay.png"),
                "instance": str(local_dir / "instance.png"),
                "semantic": str(local_dir / "semantic.png"),
                "label_counts": load_label_counts(rel),
                "prompt_terms": prompt_terms,
                "prompt_groups": prompt_groups,
            }
        )
    return {
        "experiment": "fine_object_grounded_small_eval",
        "remote": {
            "host": REMOTE,
            "port": REMOTE_PORT,
            "project_dir": REMOTE_PROJECT,
            "python": REMOTE_PYTHON,
            "tmp_dir": REMOTE_TMP,
        },
        "prompt_terms": PROMPTS,
        "prompt_text": " . ".join(PROMPTS),
        "samples": samples,
    }


def main() -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    manifest_path = EXPERIMENT_DIR / "sample_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(manifest_path)
    print("ssh -F /dev/null -p 31909 root@10.0.8.114")
    print(f"Remote env: {REMOTE_PYTHON}")


if __name__ == "__main__":
    main()
