# Scene-Aware VLM Prompting

This route should describe the global scene and the segmentation goal to the VLM, but only as structured constraints. The model output is used as point-cloud evidence, so free-form labels and caption-like answers are harmful.

## Current Rule

Use a scene-aware prompt for every mask classification and merge review:

- The scene is a rooftop MANIFOLD/Mid360 scan.
- A high ratio of large roof/floor surface is expected.
- Sky and distant background should be ignored.
- The task is dense point-level semantic evidence extraction, not image captioning.
- `floor`, `wall`, and `building` form the large stable surface layer.
- `railing`, `pipe`, and `equipment` form the fine foreground target layer.
- Thin railings, pipes, cables, and equipment frequently touch floor-like roof pixels in 2D masks.
- Labels must come from the fixed taxonomy used by `scripts/project_semantic.py`.
- The model must return strict JSON, including confidence and ambiguity fields.

## Why

The previous bottleneck is not just label wording. Multi-view conflicts remain high because coarse masks, mask boundaries, viewpoint changes, and context-dependent 2D mistakes create inconsistent observations. Scene context can reduce systematic roof/floor/railing/equipment confusion, but it cannot repair mixed masks by itself.

## Expected 2D Mask Output

The VLM should classify only the highlighted mask and return:

```json
{
  "label": "railing",
  "parent_class": "structure",
  "confidence": 0.82,
  "mixed": false,
  "is_large_surface": false,
  "can_merge_to_surface": false,
  "ambiguous_with": ["pipe"],
  "reason": "thin continuous foreground metal structure along roof edge"
}
```

The fields `mixed`, `is_large_surface`, and `can_merge_to_surface` should be preserved on `Target` records when available. Object fusion should use them as quality signals rather than directly trusting every single mask label.

Current object fusion behavior:

- `confidence` is used as label vote weight: `cluster_size * confidence`.
- Targets below `MIN_MERGE_CONFIDENCE` are preserved but do not actively merge by geometry/color alone.
- `mixed=true` blocks merging unless `can_merge_to_surface=true`.
- QA reports summarize low-confidence and mixed-object counts through `quality_stats`.

## Implementation Hook

The shared prompt source is `scripts/vlm_scene_prompt.py`.

When the server-side 2D semantic generator is updated, it should use `mask_label_prompt()` for `sam2_prompt_v3_sky_label_merge_completion` or for the next prompt variant. The existing cross-candidate Qwen review now uses `merge_review_prompt()` from the same module.

The compatibility patcher `scripts/patch_semantic_eval_scene_prompts.py` cannot
add new JSON fields to older server scripts without changing their parser
contract. It therefore keeps the existing `{"items":[...]}` schema but embeds
the point-cloud semantic goal, large-surface layer, and fine-target layer in the
prompt text.
