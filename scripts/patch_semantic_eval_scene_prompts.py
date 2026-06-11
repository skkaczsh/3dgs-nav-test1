#!/usr/bin/env python3
"""Patch server-side semantic_eval prompts with rooftop scene constraints.

The server semantic generation scripts live outside this repository in
`/root/epfs/manifold_3dgs_project/semantic_eval`. This patcher keeps their
existing `{"items":[...]}` response contract while replacing the generic
prompts with scene-aware prompts for the rooftop dense semantic route.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REVIEW_PROMPT = r'''
/no_think
You are reviewing merged SAM2 masks for a rooftop MANIFOLD/Mid360 scan.
Sky has already been handled by SkyMask. Classify only the numbered regions.

Scene constraints:
1. A high ratio of large horizontal roof/floor surface is expected.
2. Large horizontal walkable or load-bearing surfaces should be labeled floor.
   This includes roof, rooftop platform, concrete platform, gray paving, and shadow on roof/floor.
3. Do not label a horizontal roof surface as building just because it belongs to a building.
4. building is for independent building mass, distant buildings, facade-like vertical structures, or architectural blocks.
5. wall is for nearby vertical wall, parapet side, retaining wall, or facade plane. Do not use wall for horizontal roof/floor.
6. railing is for guardrail, fence, handrail, thin grid, or continuous thin metal barrier.
7. equipment is for rooftop boxes, cabinets, HVAC-like units, sensors, antennas, machinery, fixtures, and compact devices.
8. pipe is for pipes, cables, conduits, cable trays, or long utility lines if visible.
9. Thin railings/equipment often touch floor-like light brown/gray roof pixels. Prefer the thin-object label when the highlighted mask follows the thin foreground structure.
10. Keep label coarse and fixed, but add a short description and identity_hint for the physical instance when visible.
    Examples: label equipment + description "white HVAC outdoor unit"; label pipe + description "thin gray conduit"; label railing + description "yellow metal guardrail".
11. If one mask mixes a large surface and a thin object, choose the dominant physical object; use other only when no reliable dominant object exists.
12. Invalid black borders, lens edges, and unusable regions should be ignore.

Point-cloud semantic goal:
1. This is mask-level evidence for dense point-cloud semantics, not a generic image caption.
2. floor, wall, and building are large stable surface layers.
3. railing, pipe, and equipment are fine foreground targets. Do not absorb them into floor only because their mask touches roof pixels.
4. Prefer ignore for sky/background/invalid regions that should not create valid point-cloud objects.

Allowed English labels only:
floor, road, wall, building, railing, equipment, pipe, tree, grass, car, person, other, ignore

Return valid JSON only. No explanation. No Markdown.
{"items":[{"mask_id":"1","label":"equipment","confidence":0.90,"description":"white HVAC outdoor unit","identity_hint":"white rectangular HVAC unit on rooftop","attributes":{"color":"white","material":"metal","shape":"rectangular box","function":"HVAC outdoor unit"}}]}
'''.strip()


COMPLETION_PROMPT = r'''
/no_think
You are completing large non-sky unknown regions for a rooftop MANIFOLD/Mid360 scan.
Sky has already been removed by SkyMask. Numbered regions are non-sky unknown gaps.

Scene constraints:
1. A high ratio of large horizontal roof/floor surface is expected.
2. If a region is a horizontal or near-horizontal walkable/load-bearing surface, label it floor.
   This includes roof, rooftop platform, concrete platform, gray paving, and shadow on roof/floor.
3. Do not label horizontal roof/platform as building; horizontal surfaces have priority as floor.
4. building is for independent building mass, distant buildings, facade-like vertical structures, or architectural blocks.
5. wall is for nearby vertical wall, parapet side, retaining wall, or facade plane. Do not use wall for horizontal roof/floor.
6. railing is for guardrail, fence, handrail, thin grid, or continuous thin metal barrier.
7. equipment is for rooftop boxes, cabinets, HVAC-like units, sensors, antennas, machinery, fixtures, and compact devices.
8. pipe is for pipes, cables, conduits, cable trays, or long utility lines if visible.
9. Panoramic/fisheye perspective can distort geometry. Large gray/brown regions near the lower half or image center are still likely floor unless clearly vertical or a fine object.
10. Keep label coarse and fixed, but add a short description and identity_hint for the physical instance when visible.
    Examples: label equipment + description "white HVAC outdoor unit"; label pipe + description "thin gray conduit"; label railing + description "yellow metal guardrail".
11. Use ignore only for invalid border/lens artifacts. Use other for valid but uncertain scene content.

Point-cloud semantic goal:
1. This completion fills non-sky semantic gaps for dense point-cloud projection.
2. floor, wall, and building are large stable surface layers.
3. railing, pipe, and equipment are fine foreground targets. Preserve them when a numbered region follows a thin/compact physical object.
4. Do not create valid semantics for sky/background/invalid image regions.

Allowed English labels only:
floor, road, wall, building, railing, equipment, pipe, tree, grass, car, person, other, ignore

Return valid JSON only. No explanation. No Markdown.
{"items":[{"mask_id":"1","label":"floor","confidence":0.90,"description":"large gray rooftop floor surface","identity_hint":"broad horizontal roof surface","attributes":{"color":"gray","material":"concrete","shape":"large horizontal plane","function":"walkable roof surface"}}]}
'''.strip()


TARGETS = {
    "review_merged_labels_prompt_v2.py": REVIEW_PROMPT,
    "complete_unknown_regions.py": COMPLETION_PROMPT,
}


PROMPT_RE = re.compile(r'PROMPT = """\n.*?\n"""\.strip\(\)', flags=re.S)


def prompt_assignment(prompt: str) -> str:
    return 'PROMPT = """\n' + prompt + '\n""".strip()'


def patch_text(text: str, prompt: str) -> tuple[str, int]:
    return PROMPT_RE.subn(prompt_assignment(prompt), text, count=1)


def patch_file(path: Path, prompt: str, dry_run: bool) -> dict:
    original = path.read_text(encoding="utf-8")
    updated, count = patch_text(original, prompt)
    changed = updated != original
    if count != 1:
        return {"path": str(path), "patched": False, "changed": False, "error": f"prompt_matches={count}"}
    if changed and not dry_run:
        backup = path.with_suffix(path.suffix + ".scene_prompt_bak")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        path.write_text(updated, encoding="utf-8")
    return {"path": str(path), "patched": True, "changed": changed, "dry_run": dry_run}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for name, prompt in TARGETS.items():
        path = args.semantic_root / name
        if not path.exists():
            rows.append({"path": str(path), "patched": False, "changed": False, "error": "missing"})
            continue
        rows.append(patch_file(path, prompt, args.dry_run))

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
