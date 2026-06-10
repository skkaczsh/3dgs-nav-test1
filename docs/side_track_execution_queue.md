# Side-Track Execution Queue

Context:

- Main 0-999 dataset delivery manifest passed.
- Dataset package exists at `/Users/skkac/Work/SCAN/dataset_delivery_0000_0999.tgz`.
- Main route should remain the authoritative semantic route:
  `sam2_prompt_v3_sky_label_merge_completion` plus scanner-native projection.

## Server State Checked 2026-06-10

`scan-train`:

- GPUs: two 4090D cards.
- observed memory/utilization:
  - GPU0: `7676 / 49140 MiB`, utilization `15%`
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

2. Old-route visual-reference expansion on `scan-train` only if GPU/CPU is idle.
   - Scope: color/reference comparison, not semantic source.
   - Use the fixed scanner-native color route.
   - Do not reintroduce deprecated `transforms.json + project_world_points()`.

3. Main-route next dataset increment.
   - Candidate: extend from 0-999 to the next contiguous range only after visual QA accepts the current package.
   - Reuse the same delivery manifest/package validators.

## Current No-Go Conditions

- Do not run large Hugging Face downloads on `scan-vlm` root.
- Do not rerun full Qwen semantic completion unless the current manifest fails or the prompt/schema changes.
- Do not use ConceptSeg broad `floor`/`wall` prompts for production; problem40 QA showed broad/unstable masks.
- Do not treat old-route color smoke as semantic correctness evidence.
