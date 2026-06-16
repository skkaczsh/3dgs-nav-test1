#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path("/Users/skkac/Work/SCAN")
OUT_DIR = ROOT / "new_route/experiments/conceptseg_r1_small_eval"
VIS_DIR = OUT_DIR / "visuals"

RUNLIST_PATH = ROOT / "server_conceptseg_fine_object_runlist_v008/runlist.json"
REPORT_PATH = ROOT / "server_conceptseg_fine_object_runlist_v008_outputs_all/report.json"
QA_PATH = ROOT / "server_conceptseg_fine_object_runlist_v008_outputs_all/conceptseg_fine_object_all_qa.json"
ALIGN_PATH = ROOT / "server_conceptseg_fine_object_alignment_v008/conceptseg_target_object_alignment_report.json"

SELECTED = [
    {
        "image_id": "review_005_b1_fine_t_000241_cam2_mask0006_sem16_cc01_railing_or_thin_metal_guardrail",
        "assessment": "fail",
        "failure_mode": "background_swallow_mesh",
        "note": "Mesh fence is treated as one broad concept area and most of the fenced background is eaten.",
    },
    {
        "image_id": "review_007_b0_fine_t_000228_cam2_mask0002_sem16_cc00_railing_or_thin_metal_guardrail",
        "assessment": "fail",
        "failure_mode": "coarse_mesh_region",
        "note": "Predicts fence/mesh instead of isolating thin rails or poles; region is too coarse for fine-target routing.",
    },
    {
        "image_id": "review_006_a1_fine_t_000181_cam1_mask0026_sem16_cc00_railing_or_thin_metal_guardrail",
        "assessment": "fail",
        "failure_mode": "thin_structure_miss",
        "note": "Returns nonexistent although a thin railing-like structure is present.",
    },
    {
        "image_id": "review_004_b0_fine_t_000454_cam1_mask0001_sem16_cc02_railing_or_thin_metal_guardrail",
        "assessment": "mixed",
        "failure_mode": "undersegment",
        "note": "Finds the right concept word but the visible mask is tiny and unstable on thin structures.",
    },
    {
        "image_id": "review_006_a0_fine_t_000189_cam1_mask0023_sem16_cc00_pipe_or_thin_utility_conduit",
        "assessment": "mixed",
        "failure_mode": "thickened_pipe_region",
        "note": "Detects the pipe, but expands to a thick blob rather than preserving thin geometry.",
    },
    {
        "image_id": "review_005_b0_fine_t_000991_cam1_mask0010_sem16_cc01_pipe_or_thin_utility_conduit",
        "assessment": "mixed",
        "failure_mode": "concept_drift_pole",
        "note": "Localizes a narrow vertical structure but drifts semantically from pipe/conduit to pole.",
    },
    {
        "image_id": "review_002_b0_fine_t_000507_cam1_mask0013_sem16_cc00_pipe_or_thin_utility_conduit",
        "assessment": "mixed",
        "failure_mode": "concept_drift_cables",
        "note": "Still points to utility-like structure, but answer class drifts to cables and mask support is modest.",
    },
    {
        "image_id": "review_004_a1_fine_t_000430_cam1_mask0008_sem16_cc01_pipe_or_thin_utility_conduit",
        "assessment": "fail",
        "failure_mode": "undersegment",
        "note": "Very thin target is barely covered; too brittle for stable thin-pole extraction.",
    },
    {
        "image_id": "review_005_a0_fine_t_000970_cam1_mask0001_sem16_cc05_rooftop_equipment_box_or_HVAC_unit",
        "assessment": "pass",
        "failure_mode": "none",
        "note": "Boxy rooftop equipment is localized cleanly enough; this is the strongest concept family in the set.",
    },
    {
        "image_id": "review_006_a1_fine_t_000181_cam1_mask0026_sem16_cc00_rooftop_equipment_box_or_HVAC_unit",
        "assessment": "mixed",
        "failure_mode": "lexical_drift",
        "note": "Region is usable, but the textual answer drifts to red box instead of equipment/HVAC.",
    },
    {
        "image_id": "review_006_b0_fine_t_000188_cam1_mask0031_sem16_cc00_rooftop_equipment_box_or_HVAC_unit",
        "assessment": "mixed",
        "failure_mode": "semantic_alias_drift",
        "note": "Finds relevant machinery but prefers duct, reinforcing that concept wording is loose.",
    },
    {
        "image_id": "review_001_b0_fine_t_000735_cam2_mask0008_sem16_cc00_rooftop_equipment_box_or_HVAC_unit",
        "assessment": "mixed",
        "failure_mode": "undersegment",
        "note": "Correct family, but the mask is tiny and not robust when the equipment occupies little area.",
    },
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def copy_visual(src: Path, dst_name: str) -> str:
    dst = VIS_DIR / dst_name
    shutil.copy2(src, dst)
    return str(dst)


def short_concept(concept: str) -> str:
    if concept.startswith("railing"):
        return "railing"
    if concept.startswith("pipe"):
        return "pipe"
    if concept.startswith("rooftop"):
        return "equipment"
    return concept


def make_contact_sheet(entries: list[dict], dst: Path) -> None:
    thumb_w = 480
    thumb_h = 160
    label_h = 60
    cols = 2
    rows = math.ceil(len(entries) / cols)
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    font = None
    for idx, entry in enumerate(entries):
        img = Image.open(entry["local_output_path"]).convert("RGB")
        img = img.resize((thumb_w, thumb_h))
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + label_h)
        canvas.paste(img, (x, y))
        label = f"{short_concept(entry['concept'])} | {entry['answer']} | red={entry['red_overlay_ratio']:.3f}\n{entry['failure_mode']}"
        draw.text((x + 8, y + thumb_h + 8), label, fill=(235, 235, 235), font=font)
    canvas.save(dst)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    runlist = load_json(RUNLIST_PATH)
    report = load_json(REPORT_PATH)
    qa = load_json(QA_PATH)
    align = load_json(ALIGN_PATH)

    runlist_by_id = {item["image_id"]: item for item in runlist["items"]}
    report_by_id = {item["image_id"]: item for item in report["items"]}
    qa_by_id = {item["image_id"]: item for item in qa["items_detail"]}

    selected_entries = []
    for idx, spec in enumerate(SELECTED, start=1):
        image_id = spec["image_id"]
        run_item = runlist_by_id[image_id]
        report_item = report_by_id[image_id]
        qa_item = qa_by_id[image_id]
        output_path = Path(qa_item["output_path"])
        local_output_path = ROOT / "server_conceptseg_fine_object_runlist_v008_outputs_all/outputs" / output_path.name
        visual_name = f"{idx:02d}_{short_concept(run_item['concept'])}_{spec['failure_mode']}.png"
        copied_visual = copy_visual(local_output_path, visual_name)
        selected_entries.append(
            {
                "sample_index": idx,
                "image_id": image_id,
                "concept": run_item["concept"],
                "answer": qa_item["answer"],
                "mode": qa_item["mode"],
                "assessment": spec["assessment"],
                "failure_mode": spec["failure_mode"],
                "note": spec["note"],
                "red_overlay_ratio": qa_item["overlay"]["red_overlay_ratio"],
                "output_path": report_item["output_path"],
                "local_output_path": str(local_output_path),
                "visual_path": copied_visual,
                "local_image_path": run_item["metadata"]["local_assets"]["image"],
                "local_overlay_path": run_item["metadata"]["local_assets"]["overlay"],
                "source_label": run_item["metadata"]["source_label"],
                "review_id": run_item["metadata"]["review_id"],
                "target_id": run_item["metadata"]["representative"]["target_id"],
            }
        )

    selected_entries.sort(key=lambda x: x["sample_index"])

    make_contact_sheet(selected_entries[:4], VIS_DIR / "contact_sheet_railing.png")
    make_contact_sheet(selected_entries[4:8], VIS_DIR / "contact_sheet_pipe.png")
    make_contact_sheet(selected_entries[8:12], VIS_DIR / "contact_sheet_equipment.png")
    make_contact_sheet(
        [selected_entries[0], selected_entries[1], selected_entries[4], selected_entries[8]],
        VIS_DIR / "contact_sheet_mixed_summary.png",
    )

    counts = Counter((entry["concept"], entry["assessment"]) for entry in selected_entries)
    by_concept = defaultdict(lambda: Counter())
    for entry in selected_entries:
        by_concept[entry["concept"]][entry["assessment"]] += 1

    manifest = {
        "task": "ConceptSeg-R1 small-sample eval for railing/mesh/thin-pole style concept regions",
        "source_runlist": str(RUNLIST_PATH),
        "source_report": str(REPORT_PATH),
        "source_qa": str(QA_PATH),
        "source_alignment": str(ALIGN_PATH),
        "remote_check": {
            "attempted_host": "scan-train",
            "status": "blocked",
            "reason": "ssh bind/connect failure before remote command execution",
        },
        "selection_policy": "12 representative items emphasizing railing/mesh/thin-structure failure modes, plus pipe and equipment controls.",
        "samples": selected_entries,
        "summary": {
            "selected_count": len(selected_entries),
            "assessment_counts": dict(Counter(entry["assessment"] for entry in selected_entries)),
            "failure_mode_counts": dict(Counter(entry["failure_mode"] for entry in selected_entries)),
            "by_concept_assessment": {
                concept: dict(counter) for concept, counter in by_concept.items()
            },
            "full_run_metrics": {
                "item_count": qa["items"],
                "concept_match_count": align["concept_match_count"],
                "overlarge_count": align["overlarge_count"],
                "non_discriminative_target_count": align["non_discriminative_target_count"],
                "concept_summary": qa["concept_summary"],
            },
        },
    }
    (OUT_DIR / "sample_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report_md = f"""# ConceptSeg-R1 small eval

## Scope

- Input package: `server_conceptseg_fine_object_runlist_v008`
- Evidence base: existing `v008` run (`90` prompt-image pairs), structured QA, object/target alignment, plus a failed live remote reconnect attempt to `scan-train`
- Sampled review set here: `12` representative items with emphasis on railing / mesh / thin-pole failure modes

## What was reused

- Local scripts: `run_server_conceptseg_r1_smoke.sh`, `run_server_conceptseg_smoke.sh`, `build_conceptseg_fine_object_runlist.py`, `validate_conceptseg_fine_object_runlist.py`
- Existing outputs: `server_conceptseg_fine_object_runlist_v008_outputs_all`, `server_conceptseg_problem40`, `server_conceptseg_fine_object_alignment_v008`
- Existing v008 package already validated locally and contains `90` items / `30` targets

## Full-run signal

- All `90/90` runs returned code `0`
- Alignment report marks `89/90` prompt-answer pairs as concept-matched, but **`30/30` targets are non-discriminative across prompts**
- `railing or thin metal guardrail` is the unstable family:
  - median red overlay ratio `0.0416`
  - p90 `0.1047`
  - max `0.4032`
  - common answers: `guardrail`, `rail`, `fence`
- `pipe or thin utility conduit` is more stable semantically, but often thickens the region instead of preserving thin geometry
- `rooftop equipment box or HVAC unit` is the cleanest family and behaves like a coarse object concept detector

## Judgment

### Can it stably segment railing / mesh / thin pole as concept regions?

Short answer: **not stably enough for fine-target mainline use**.

- On railing / mesh scenes, ConceptSeg-R1 often locks onto the **entire fence/mesh field** instead of isolating the thin structural elements.
- On very thin structures, it can also **undersegment badly** or even answer `nonexistent`.
- For pipe/pole-like scenes it is directionally useful, but the mask usually becomes a **fatter concept blob** rather than a precise thin structure.

### Relative to current SAM2+VLM

Best fit: **second-stage review / proposal signal**, not a replacement for the current mainline.

- Good at: asking "is there something fence-like / pipe-like / equipment-like here?"
- Weak at: providing the **tight, stable, topology-aware masks** needed for thin targets in production routing
- The existing alignment report's own interpretation is consistent with this: usable as constrained candidate review, not as dense semantic production output

### Most visible failure modes

1. **Over-coarse concept regions**: mesh/fence gets swallowed as one broad region.
2. **Concept drift**: `railing -> fence`, `pipe -> pole/cables`, `equipment -> duct/red box`.
3. **Background inclusion**: large chunks of rooftop/background get pulled in with the concept.
4. **Thin-structure instability**: extremely narrow poles/rails either disappear or get only a tiny sliver mask.

## Sample review verdict

| concept | pass | mixed | fail | read |
| --- | ---: | ---: | ---: | --- |
| railing / guardrail | 0 | 1 | 3 | worst family; mesh and thin rails are not stable |
| pipe / conduit | 0 | 3 | 1 | useful hint, but geometry is too thick/coarse |
| equipment / HVAC | 1 | 3 | 0 | strongest concept family, but not the target problem |

## Remote blocker

- Tried to reconnect to `scan-train` for a live `12`-image rerun using the existing remote assets.
- SSH failed **before** remote execution with local bind/connect errors:
  - `bind 192.168.100.115: Can't assign requested address`
  - `ssh: connect to host 10.0.8.114 port 31909: failure`
- Because this failed before entering the remote shell, this is a connectivity/config blocker, not a ConceptSeg-R1 environment blocker.

## Recommendation

- Keep ConceptSeg-R1 as a **side-track reviewer / candidate proposer** for ambiguous fine targets.
- Do **not** move it into the fine-target mainline for railing / mesh / thin pole extraction in its current form.
- If revisited, the next worthwhile experiment is not a broader rollout, but a **post-filtered second stage**:
  1. run ConceptSeg-R1 only on SAM2/VLM suspicious regions
  2. intersect with existing instance/support masks
  3. reject high-area mesh/background expansions
  4. score whether any residual thin structure signal survives
"""
    (OUT_DIR / "report.md").write_text(report_md, encoding="utf-8")


if __name__ == "__main__":
    main()
