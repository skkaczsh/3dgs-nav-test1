# Side-Track Execution Queue

Context:

- Main 0-999 dataset delivery manifest passed.
- Dataset package exists at `/Users/skkac/Work/SCAN/dataset_delivery_0000_0999.tgz`.
- Dataset package was refreshed on 2026-06-11 with route decision,
  ConceptSeg fine-object intersection QA, and old-route reference validation.
- Main route should remain the authoritative semantic route:
  `sam2_prompt_v3_sky_label_merge_completion` plus scanner-native projection.

## Server State Checked 2026-06-11

`scan-train`:

- GPUs: two 4090D cards.
- observed memory/utilization:
  - GPU0: `7676 / 49140 MiB`, utilization `17%`
  - GPU1: `23602 / 49140 MiB`, utilization `0%`
- root filesystem: `35G / 50G` used.
- EPFS: `100T / 108T` used.
- Suitable for controlled side-track GPU jobs.

`scan-vlm`:

- GPU: L20.
- observed memory/utilization: `9805 / 46068 MiB`, utilization `0%`
- root filesystem: `45G / 50G` used.
- EPFS: `100T / 108T` used.
- Avoid downloads/cache growth on root. Use only EPFS-backed jobs.

## Queue

1. ConceptSeg-R1 constrained fine-object test on `scan-train`.
   - Scope: selected reviewed fine-object crops/masks, not full 1000-frame dense run.
   - Goal: determine whether ConceptSeg helps equipment/railing second-stage review.
   - Do not promote to main path unless it beats SAM2+Qwen on the same artifacts.
   - Use EPFS cache only.
   - Status 2026-06-11:
     - v008 constrained runlist package is built and validated with `90` items.
     - 12-item smoke succeeded with stable constrained labels and no overlarge
       broad-surface masks.
     - Remaining `78` items completed successfully; combined result is
       `90 / 90` successful ConceptSeg runs.
     - Server output:
       `/root/epfs/new_route_stage1_skymask/conceptseg_fine_object_runlist_v008_outputs_full`.
     - Local merged QA:
       `/Users/skkac/Work/SCAN/server_conceptseg_fine_object_runlist_v008_outputs_all/conceptseg_fine_object_all_qa.json`.
     - Target/object alignment:
       `/Users/skkac/Work/SCAN/server_conceptseg_fine_object_alignment_v008/conceptseg_target_object_alignment_report.json`.
     - Alignment result: `30 / 30` targets have usable candidates, but
       `0 / 30` targets are semantically discriminative because multiple prompts
       can match different local structures in the same image.
     - Next action: use ConceptSeg only to split/refine fine residual masks
       after intersection with existing SAM2/Qwen masks and 3D connected
       components. Do not promote it to dense semantic image generation or
       target-level classification.
     - Instance-mask intersection QA:
       `/Users/skkac/Work/SCAN/server_conceptseg_instance_intersection_v008/conceptseg_instance_intersection_report.json`.
     - Intersection result: `10 / 90` candidates accepted, covering `7 / 30`
       targets. This is useful as a conservative refinement signal but too low
       coverage for dataset-wide mask production.

2. Old-route visual-reference expansion on `scan-train` only if GPU/CPU is idle.
   - Scope: color/reference comparison, not semantic source.
   - Use the fixed scanner-native color route.
   - Do not reintroduce deprecated `transforms.json + project_world_points()`.
   - Status 2026-06-11:
     - Old-route smoke PLY and debug images were pulled locally.
     - Reference validation passed:
       `/Users/skkac/Work/SCAN/server_old_route_smoke/old_route_reference_validation.json`.
     - Validated metrics: `31,323` RGB vertices, colored ratio `0.8816`,
       `12` color frames, `8` sections.
     - No reusable server runner was found under EPFS; keep this as a fixed
       visual/color reference until a reproducible runner is rebuilt from the
       validated scanner-native route.

3. Main-route next dataset increment.
   - Candidate: extend from 0-999 to the next contiguous range only after visual QA accepts the current package.
   - Reuse the same delivery manifest/package validators.

## Current No-Go Conditions

- Do not run large Hugging Face downloads on `scan-vlm` root.
- Do not rerun full Qwen semantic completion unless the current manifest fails or the prompt/schema changes.
- Do not use ConceptSeg broad `floor`/`wall` prompts for production; problem40 QA showed broad/unstable masks.
- Do not treat old-route color smoke as semantic correctness evidence.
