# Route1 Small Eval Pack - 2026-06-16

Scope:

- large-surface semantic baseline:
  `/Users/skkac/Work/SCAN/new_route/experiments/surface_baseline_small_eval`
- grounded fine-object pipeline:
  `/Users/skkac/Work/SCAN/new_route/experiments/fine_object_grounded_small_eval`
- ConceptSeg-R1 side-track:
  `/Users/skkac/Work/SCAN/new_route/experiments/conceptseg_r1_small_eval`

This file is a scoped comparison pack for Route1 only. It does not modify the
main recommendation document.

## 1. Large-Surface Baseline

### Evidence checked

- ADE20K tiny:
  - `outputs_compare_gpu0/report.json`
  - `outputs_compare_gpu0/report.md`
- Cityscapes / Mapillary:
  - `outputs_compare_city_map/report.json`
  - `outputs_compare_city_map/report.md`
- ADE20K large:
  - `outputs_compare_oneformer_ade20k_large/report.json`
  - `outputs_compare_oneformer_ade20k_large/report.md`
- Remote output directories were re-checked on `scan-train` and confirmed to
  contain `predictions/`, `visualizations/`, `report.json`, `report.md`.

### Result

Small-sample benchmark size: `12` rooftop-tail images.

Relevant aggregate behavior:

- `OneFormer ADE20K tiny`
  - smoother than current point-level semantics
  - still heavily shifts rooftop planes into `wall`
- `Mask2Former ADE20K tiny`
  - slightly lower transition rate than `OneFormer tiny`
  - still confuses `floor/wall/ceiling`
- `OneFormer ADE20K large`
  - completed successfully after moving HF cache to `/root/epfs`
  - still over-predicts `wall` on rooftop planes
- `Cityscapes / Mapillary`
  - mostly collapse rooftop surfaces into `building`
  - do not provide a usable `ceiling` prior

### Verdict against current `Mimo/SAM2`

Answer required by this track:

- Are these models **more stable visually** than current `Mimo/SAM2` point
  labels? Yes, they are smoother.
- Are these models **more stable in the sense needed by the project**, namely
  reliable `floor/ground`, `wall`, `ceiling`, `building`, `sky` separation on
  rooftop imagery? No.

Operational conclusion:

- `Mask2Former/OneFormer` can be kept only as a coarse surface prior or
  reviewer.
- They are not good enough to replace the current mainline by themselves.
- If this branch is revisited, it must be combined with geometry-aware post
  rules. Pure checkpoint swapping inside the current
  `Mask2Former/OneFormer/ADE20K/Cityscapes/Mapillary` family is effectively
  exhausted on this small eval.

## 2. Grounded Fine Objects

### Evidence checked

- small grouped-detector reports:
  - `report_20260616.md`
  - `report_florence2_large_ft_small_eval.md`
- focus-rich tail reports:
  - `report_focus_rich_2000_2999.md`
  - `report_railing_rich_2000_2999.md`
- synced server evidence:
  - `railing_rich_2000_2999_strict_v2/accepted_report.json`
  - `railing_rich_2000_2999_strict_v2/fine_object_report.json`
- local evidence already present:
  - `pipe_rich_2000_2999_cpu/*.json`
  - `equipment_rich_2000_2999_cpu/*.json`
  - `multi_focus_v4_std_mapped/*.json`

### Phrase-level conclusion

Useful proposal families:

- `railing / guardrail / handrail`
  - usable only after strict prompting and geometry filtering
  - can produce compact projected subsets
  - still fragments badly across views
- `pipe`
  - cleanest current fine-object branch
  - survives 2D filter -> 3D projection -> target/object promotion best
- `cable`
  - usable only when treated as the same linear-utility family as `pipe`
  - should not be promoted as an independent semantic class
- `air conditioning unit`
  - strongest equipment phrase
  - can survive filtering without swallowing entire surfaces
- `hvac outdoor unit`
  - usable, but only under strict phrase and geometry guards

Risky or broad phrases:

- `metal fence`
  - worst railing phrase
  - repeatedly returns broad fence/mesh/wall-like regions
  - top large-mask examples in grouped summaries are dominated by this phrase
- `outdoor unit`
  - worst equipment phrase
  - strongly associated with oversized rooftop-band masks
- `air`, `unit`, `unit unit`
  - weak equipment phrases
  - should remain rejected by strict filtering

### Category-level ranking

1. `pipe`
   - strongest small-sample route
   - best end-to-end survival into compact 3D subsets
2. `railing`
   - usable as a proposal route after strict prompt + geometry filter
   - still suffers from thin-structure fragmentation
3. `equipment/HVAC`
   - can produce valid subsets
   - still the most likely branch to reintroduce broad surface pollution if
     phrase guards are relaxed
4. `Florence-2`
   - not a replacement for `GroundingDINO + SAM2`
   - only a broad proposal side-branch

### Did this round add a new sample rerun?

No new rerun was required for this scoped pack.

Reason:

- the current small-sample evidence is already sufficient to separate
  `pipe`, `railing`, and `equipment/HVAC` by utility and failure mode
- the missing problem was evidence completeness, not lack of another batch
- this round therefore focused on syncing missing server-side JSON evidence back
  into the local experiment directories

## 3. ConceptSeg-R1

### Final positioning

`ConceptSeg-R1` is only suitable as a **reviewer / proposal signal**, not as a
mainline fine-mask generator.

Why:

- concept answer can be semantically right while the highlighted region remains
  coarse, mesh-like, or topologically wrong
- railing / mesh / thin-pole scenes are still the least stable family
- the rich-tail rerun confirms it can recognize concepts, but not produce the
  precise masks required by the production route

## 4. Recommended Route1 Combination

Recommended combination after the current small-sample pack:

- large surfaces:
  - keep current `Mimo/SAM2 + geometry-aware post reasoning`
  - do not switch to pure `Mask2Former/OneFormer`
- fine objects:
  - `GroundingDINO + SAM2`
  - with grouped prompts, strict phrase gating, and geometry guards
- category emphasis:
  - promote `pipe` first
  - keep `railing` as a guarded proposal path
  - keep `equipment/HVAC` behind strict phrase gating and review
- ConceptSeg-R1:
  - reviewer / candidate proposer only

## 5. Submit Readiness

This scoped pack is submit-ready because:

- surface baseline has complete local and remote evidence, including the
  previously missing `OneFormer ADE20K large` JSON report
- grounded fine-object has enough synced JSON evidence to support a phrase-level
  recommendation without another rerun
- ConceptSeg-R1 has both the original small eval and the rich-tail rerun
  already documented

Remaining non-blocking gap:

- the next grounded improvement should be a targeted `railing` mask-quality
  refinement experiment, not a broader small-sample benchmark reset
