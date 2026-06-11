# Dense Semantic Route Decision

## Decision

- Main route status: `continue_as_authoritative_route`.
- ConceptSeg-R1 status: `keep_as_conservative_fine_object_refinement_only`.
- Old route status: `keep_as_fixed_visual_color_reference_only`.

## Main Route Evidence

- Dataset manifest passed: `True`
- Output validation passed: `True`
- Frame range: `0-999`
- Semantic combo: `sam2_prompt_v3_sky_label_merge_completion`
- Projection route: `img_pos.txt + cam_in_ex.txt + Tcl + Til`
- Target count: `34252`
- Object count: `2978`
- Object ambiguous ratio: `0.0695`
- Surface-first changed ratio: `0.0714`

## ConceptSeg-R1 Evidence

- Candidate runs: `90`
- Aligned targets: `30`
- Concept matches: `89`
- Semantically discriminative targets: `0`
- Instance-intersection accepted candidates: `10`
- Instance-intersection target coverage: `7 / 30`
- Conclusion: Useful for a small subset of local fine-object mask refinements after strict instance-mask intersection; not suitable for dense semantic generation or target-level classification.

## Old Route Evidence

- Reference validation passed: `True`
- Colored ratio: `0.8816`
- PLY vertices: `31323`
- RGB fields present: `True`
- Conclusion: Validated as an RGB visual sanity reference; no reusable production runner found.

## Next Steps

- Do not expand ConceptSeg to all frames; first integrate only accepted intersection candidates into fine-object split/refine QA.
- Do not revive deprecated transforms.json/project_world_points semantic projection.
- For main route, continue from object/residual refinement: stable surface layer first, then fine-object 3D connected components.
- Before extending beyond 0-999 frames, validate the current reviewed package visually in the PLY viewer/CloudCompare.

## Delivery Package

- Refreshed package: `/Users/skkac/Work/SCAN/dataset_delivery_0000_0999.tgz`
- Package validation: `/Users/skkac/Work/SCAN/dataset_delivery_0000_0999_validation.json`
- Manifest validation: `/Users/skkac/Work/SCAN/route_status_20260610/dataset_delivery_manifest_0000_0999_validation.json`
- Delivery acceptance: `/Users/skkac/Work/SCAN/route_status_20260610/delivery_acceptance_20260611.json`
- Packaged files: `24`
- Large referenced files: `3`
- Included side-track evidence:
  - route decision JSON/Markdown
  - ConceptSeg fine-object alignment report
  - ConceptSeg instance-intersection report and accepted sheet
  - old-route reference validation
