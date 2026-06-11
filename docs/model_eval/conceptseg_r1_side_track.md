# ConceptSeg-R1 Side-Track Evaluation Notes

Purpose:

- Keep ConceptSeg-R1 as a side-track model for second-stage review of ambiguous/fine-object masks.
- Do not replace the current main 2D semantic source: `sam2_prompt_v3_sky_label_merge_completion`.
- Do not spend main route GPU resources unless GPU1 is idle.

Security:

- Hugging Face tokens must not be committed, copied into scripts, or written into docs.
- Use an ephemeral environment variable only when a gated model download requires authentication.

Evaluation scope:

- Start with the existing problem sample set, not 1000 frames.
- Inputs should be original image crops/overlays and candidate masks from fine residual traces.
- Outputs should be binary/semantic masks plus a JSON response that can be compared against SAM2+Qwen labels.

Promotion criteria:

- Stable binary masks on fine objects.
- Better separation for railing/equipment/building edge cases than SAM2+Qwen.
- No regression on sky/background exclusion.
- Runtime acceptable on scan-train GPU1 without blocking Qwen/SAM2 production.

Current status:

- Previous 40-image smoke produced non-empty outputs but category stability was insufficient for main route replacement.
- Source repository is prepared on `scan-train` at `/root/epfs/model_side_tracks/ConceptSeg-R1`.
- Preparation runner: `/root/epfs/model_side_tracks/run_server_conceptseg_r1_smoke.sh`.
- Local tracked runner: `/Users/skkac/Work/SCAN/new_route/scripts/run_server_conceptseg_r1_smoke.sh`.
- Preparation check passed with `RUN_INFERENCE=0` on GPU1.
- `ConceptSeg-R1-7B` weights are downloaded to `/root/epfs/model_side_tracks/ConceptSeg-R1/ConceptSeg-R1-7B`.
- Release assets are downloaded:
  - `/root/epfs/model_side_tracks/ConceptSeg-R1/sam3-main.zip`
  - `/root/epfs/model_side_tracks/ConceptSeg-R1/all_meta.json.zip`
- The existing environment `/root/epfs/conda_envs/conceptseg-r1` can load the model on GPU1 with `ATTENTION_IMPLEMENTATION=sdpa`; `flash-attn` is not required for smoke.
- Smoke output:
  - server: `/root/epfs/model_side_tracks/ConceptSeg-R1/example_images/outputs_scan_smoke/scan_smoke_railing_or_thin_metal_structure.png`
  - local: `/Users/skkac/Work/SCAN/server_conceptseg_r1_smoke/scan_smoke_railing_or_thin_metal_structure.png`
- Problem40 structured QA:
  - script: `/Users/skkac/Work/SCAN/new_route/scripts/analyze_conceptseg_problem_outputs.py`
  - local report: `/Users/skkac/Work/SCAN/server_conceptseg_problem40/conceptseg_problem40_structured_qa.json`
  - outputs copied locally under: `/Users/skkac/Work/SCAN/server_conceptseg_problem40/outputs`
  - items: `40 / 40` succeeded
  - inference modes: `sam3=26`, `mllm=14`
  - average red-overlay ratio:
    - floor: `0.2813`, with `5 / 10` over-large masks
    - wall: `0.0447`
    - railing: `0.0869`, with `1 / 10` over-large masks
    - equipment: `0.0456`
  - answer stability:
    - equipment answers include `equipment`, `barrel`, `pipe`, `crane`,
      `aircon`, and `debris`
    - railing answers include both `rail` and `railing`
    - floor/wall often return empty answer text when using direct SAM3 mode

Smoke interpretation:

- The full inference chain runs on GPU1.
- The smoke response to `railing or thin metal structure` produced a non-empty mask, but the model response labeled the target as `wall` and visually selected the blue bird-house region rather than the intended thin metal support.
- This reinforces the current decision: ConceptSeg-R1 is a side-track candidate for constrained second-stage review, not a replacement for `sam2_prompt_v3_sky_label_merge_completion`.
- Next useful test should use our own problem crops with reference boxes/masks, not only the public example image.
- Problem40 confirms the same direction: ConceptSeg-R1 can produce useful
  constrained masks for some equipment/railing prompts, but broad surface
  prompts and free-text answers are not stable enough for automatic dense
  semantic production. Keep it as a second-stage candidate generator for
  reviewed fine-object crops.

Fine-object v008 constrained run:

- Builder: `/Users/skkac/Work/SCAN/new_route/scripts/build_conceptseg_fine_object_runlist.py`
- Validator: `/Users/skkac/Work/SCAN/new_route/scripts/validate_conceptseg_fine_object_runlist.py`
- Runner: `/Users/skkac/Work/SCAN/new_route/scripts/run_server_conceptseg_fine_object_runlist.sh`
- Local runlist package:
  `/Users/skkac/Work/SCAN/server_conceptseg_fine_object_runlist_v008`
- Server runlist package:
  `/root/epfs/new_route_stage1_skymask/conceptseg_fine_object_runlist_v008`
- Runlist items: `90`
- Prompts:
  - `railing or thin metal guardrail`
  - `rooftop equipment box or HVAC unit`
  - `pipe or thin utility conduit`
- Broad prompts intentionally excluded: `floor`, `wall`, `building facade`
- 12-item smoke output:
  - server:
    `/root/epfs/new_route_stage1_skymask/conceptseg_fine_object_runlist_v008_outputs`
  - local:
    `/Users/skkac/Work/SCAN/server_conceptseg_fine_object_runlist_v008_outputs`
  - structured QA:
    `/Users/skkac/Work/SCAN/server_conceptseg_fine_object_runlist_v008_outputs/conceptseg_fine_object_smoke_qa.json`
  - contact sheet:
    `/Users/skkac/Work/SCAN/server_conceptseg_fine_object_runlist_v008_outputs/conceptseg_fine_object_smoke_contact_sheet.jpg`
- 12-item smoke result:
  - `12 / 12` succeeded.
  - all runs used MLLM mode.
  - average red-overlay ratios:
    - railing/guardrail: `0.0082`
    - rooftop equipment/HVAC: `0.0039`
    - pipe/conduit: `0.0066`
  - answer stability:
    - railing prompt returned `rail` or `guardrail`
    - equipment prompt returned `HVAC` or `unit`
    - pipe prompt returned `pipe`, `pipes`, `conduit`, or `cables`
- Interpretation:
  - The v008 fine-object smoke is better scoped than problem40 because it
    avoids floor/wall prompts and uses reviewed fine-object representative
    images.
  - ConceptSeg-R1 is promising as a constrained fine-object candidate generator
    with low broad-surface contamination on the reviewed samples.
  - It should still be compared inside the existing target/object review
    workflow, not promoted to a new dense semantic image source.
- Full constrained run:
  - started on `scan-train` in tmux session `conceptseg_fine_v008_full`
  - command range: `START_INDEX=12 LIMIT=78`
  - output:
    `/root/epfs/new_route_stage1_skymask/conceptseg_fine_object_runlist_v008_outputs_full`

References:

- Project page: `https://ntu-ai4x.github.io/ConceptSeg-R1/`
- Repository: `https://github.com/NTU-AI4X/ConceptSeg-R1`
- Weights: `https://huggingface.co/zhaoyuan666/ConceptSeg-R1-7B`
