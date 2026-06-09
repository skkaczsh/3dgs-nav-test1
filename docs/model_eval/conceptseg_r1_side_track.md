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
- The next blocker is external assets/weights:
  - `sam3-main.zip` and `all_meta.json.zip` from ConceptSeg-R1 GitHub releases.
  - `ConceptSeg-R1-7B` weights from Hugging Face.

References:

- Project page: `https://ntu-ai4x.github.io/ConceptSeg-R1/`
- Repository: `https://github.com/NTU-AI4X/ConceptSeg-R1`
- Weights: `https://huggingface.co/zhaoyuan666/ConceptSeg-R1-7B`
