# Surface Baseline Small Eval

Purpose:

- test whether large-surface confusion should be handled by a semantic
  segmentation baseline instead of the current `SAM2 + Mimo` mask-label route
- compare only the coarse classes that matter for global structure:
  `floor/ground`, `wall`, `ceiling`, `building`, `sky`

Current status:

- entry script:
  `/Users/skkac/Work/SCAN/new_route/experiments/surface_baseline_small_eval/run_surface_baseline_small_eval.py`
- sample manifest:
  `/Users/skkac/Work/SCAN/new_route/experiments/surface_baseline_small_eval/samples/sample_manifest.json`
- server run output:
  `/root/epfs/vlm_seg_project/tmp_surface_baseline_small_eval/outputs_compare_gpu0`
- local synced report:
  `/Users/skkac/Work/SCAN/new_route/experiments/surface_baseline_small_eval/outputs_compare_gpu0/report.md`
- additional server run output:
  `/root/epfs/vlm_seg_project/tmp_surface_baseline_small_eval/outputs_compare_city_map`
- additional local synced report:
  `/Users/skkac/Work/SCAN/new_route/experiments/surface_baseline_small_eval/outputs_compare_city_map/report.md`
- final missing large-checkpoint run output:
  `/root/epfs/vlm_seg_project/tmp_surface_baseline_small_eval/outputs_compare_oneformer_ade20k_large`
- final missing large-checkpoint local report:
  `/Users/skkac/Work/SCAN/new_route/experiments/surface_baseline_small_eval/outputs_compare_oneformer_ade20k_large/report.md`

Server result on 12 representative tail samples:

- device: `cuda`
- models:
  - `shi-labs/oneformer_ade20k_swin_tiny`
  - `facebook/mask2former-swin-tiny-ade-semantic`
- baseline reference: `sam2_prompt_v3_sky_label_merge_completion`

Additional outdoor-prior run:

- device: `cuda`
- models:
  - `shi-labs/oneformer_cityscapes_swin_large`
  - `facebook/mask2former-swin-large-cityscapes-semantic`
  - `facebook/mask2former-swin-large-mapillary-vistas-semantic`

Final missing checkpoint run:

- device: `cuda`
- model:
  - `shi-labs/oneformer_ade20k_swin_large`
- note:
  - the first attempt failed only because `/root/.cache/huggingface` on the
    container overlay was full
  - rerunning with `HF_HOME=/root/epfs/hf_home` and
    `HUGGINGFACE_HUB_CACHE=/root/epfs/hf_cache` completed successfully

Decision:

- neither `OneFormer(ADE20K)` nor `Mask2Former(ADE20K)` is good enough to
  replace the main path directly
- both models are smoother than the current noisy point-level semantic output,
  but they still systematically confuse rooftop `floor/wall/ceiling`
- `OneFormer` is slightly better on some wall-vs-ceiling cases
- `Mask2Former` is slightly more stable overall, but still over-predicts
  `wall/ceiling` on rooftop scenes
- `Cityscapes/Mapillary` do not solve the problem either; they mostly shift the
  failure mode toward over-predicting `building` on rooftop surfaces
- `OneFormer(ADE20K large)` also does not solve the problem; it still heavily
  over-predicts `wall` on rooftop planes and only modestly changes the
  floor/ceiling balance

Implication:

- the current failure is not only a VLM label problem
- a generic ADE20K semantic model also lacks rooftop-domain prior
- outdoor street-scene priors (`Cityscapes/Mapillary`) are also insufficient
  because they collapse many rooftop planes into `building` and do not provide
  a real `ceiling` prior
- if we want a "surface-first" branch, it likely needs either:
  - geometry-aware post rules on top of image semantics, or
  - a stronger outdoor/urban semantic model with labels better matched to the
    scan domain

Current route decision:

- within the current `Mask2Former/OneFormer` family and the already staged
  12-sample rooftop benchmark, the branch is now effectively exhausted
- no additional pure `Mask2Former/OneFormer` reruns are justified unless we
  introduce a genuinely new dataset prior or a geometry-aware post layer
