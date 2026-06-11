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
- Residual surface assignment ratio: `0.4618`
- Residual surface unassigned points: `1609501`
- Residual absorption sweep best ratio: `0.5576`
- Residual miss reasons: `no_candidate_cell=767620`, `label_incompatible=396654`, `plane_distance_failed=225349`, `color_distance_failed=147082`, `bbox_distance_failed=72796`
- Residual candidate coverage best ratio: `0.6253`
- Surface seed augmented best ratio: `0.6349`
- Surface target/fusion bottleneck: base fusion absorbs wall targets into floor/ambiguous objects; only `38,248` wall object points remain from `391,427` wall target points.
- Strict surface-label fusion test: wall object points recover to `391,427`, ambiguous object points drop from `704,054` to `174,874`; wall object count rises from `51` to `310`, so same-label wall/plane consolidation is still needed.

## 1000-1999 Partial Increment

- Remote semantic production is still running; as of `2026-06-11T12:08Z`,
  `sam2_prompt_v3_sky_label_merge_completion` and `label_records` are
  `1375 / 3000`.
- A partial target/object fusion was run on the currently available
  `1000-1999` semantic directory using `WORK_MODE=semantic-dir`.
- Output:
  `/root/epfs/new_route_stage1_skymask/target_object_fusion_1000_1999_partial`
  and local QA copy:
  `/Users/skkac/Work/SCAN/server_target_object_fusion_1000_1999_partial`.
- Target/object QA:
  - frames processed: `1000`
  - targets: `11951`
  - target points: `2394103`
  - small target residual points: `1551092`
  - small residual ratio: `0.3932`
  - objects: `2080`
  - merge ratio: `0.8260`
  - ambiguous ratio: `0.0327`
  - object identity enrichment ratio: `0.9966`
- Identity-description relabel QA:
  - script: `scripts/relabel_objects_from_identity.py`
  - changed objects: `94 / 2080` (`0.0452`)
  - ambiguous objects reduced from `68` to `4`
  - railing objects increased from `124` to `153`
  - pipe objects increased from `48` to `58`
  - local preview PLY:
    `/Users/skkac/Work/SCAN/server_target_object_fusion_1000_1999_partial/objects/object_points_identity_relabel_stride10.ply`

Current interpretation: the two-level label model is useful. Coarse labels
remain noisy, but `description`/`identity_hint` can safely correct a narrow set
of obvious object-level mistakes such as `equipment -> railing` for guardrails
and `ambiguous -> building/floor/pipe` when the identity text is clear. This
should remain a post-fusion QA variant until visual acceptance confirms it.

### Label-Specific Connectivity Variant

The initial target generation used one 3D connected-component voxel size
(`0.08m`) for every label. On high-residual frames, a parameter sweep showed
that the residual was dominated by large surface/structure labels being split
into many components below `min_target_points=20`, not only by the minimum
target threshold:

| variant | residual ratio | targets |
| --- | ---: | ---: |
| `voxel=0.08,min=20` | `0.4273` | `425` |
| `voxel=0.08,min=10` | `0.3844` | `804` |
| `voxel=0.08,min=5` | `0.3345` | `1751` |
| `voxel=0.24,min=20` | `0.1946` | `491` |
| `surface=0.24,fine=0.12,min=20` | `0.2114` | `491` |

Based on this, `scripts/build_targets_from_masks.py` now supports optional
label-specific connectivity:

- `--surface-voxel-size` for `floor`, `road`, `wall`, `building`, `ceiling`
- `--fine-voxel-size` for `equipment`, `railing`, `pipe`, `furniture`
- default behavior remains unchanged unless these arguments are passed

Full `1000-1999` partial rerun with `VOXEL_SIZE=0.08`,
`SURFACE_VOXEL_SIZE=0.24`, `FINE_VOXEL_SIZE=0.12`:

- output:
  `/root/epfs/new_route_stage1_skymask/target_object_fusion_1000_1999_surface024_fine012`
- local QA copy:
  `/Users/skkac/Work/SCAN/server_target_object_fusion_1000_1999_surface024_fine012`
- frames processed: `1000`
- targets: `13010`
- target points: `3722904`
- small target residual points: `617915`
- small residual ratio: `0.1423`
- objects: `2196`
- merge ratio: `0.8312`
- ambiguous ratio: `0.0305`
- identity relabel changed objects: `101 / 2196` (`0.0460`)
- identity relabel reduced ambiguous objects from `67` to `7`
- local preview PLY:
  `/Users/skkac/Work/SCAN/server_target_object_fusion_1000_1999_surface024_fine012/objects/object_points_identity_relabel_stride10.ply`

Current recommendation: for this rooftop dataset, use the label-specific
connectivity variant as the next QA candidate. It preserves fine-target
connectivity separately while substantially reducing residual loss on large
surfaces. Visual QA is still required because a larger surface voxel can
over-connect adjacent planar regions.

Operational note: `scripts/start_remote_scan_train_target_refresh_loop.sh`
starts a scan-train watcher that refreshes the `surface024_fine012` target/object
variant when `label_records` grows by `MIN_COMPLETION_DELTA` records. It uses
the artifact-aware resume logic in `build_targets_from_masks.py`, then rebuilds
the identity-relabel JSONL and stride preview PLY.

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
- Replace broad cross-label surface-parent merges with strict surface labels for floor/wall/building, then run same-label plane/structure consolidation.
- Before extending beyond 0-999 frames, validate the current reviewed package visually in the PLY viewer/CloudCompare.

## Delivery Package

- Refreshed package: `/Users/skkac/Work/SCAN/dataset_delivery_0000_0999.tgz`
- Package validation: `/Users/skkac/Work/SCAN/dataset_delivery_0000_0999_validation.json`
- Manifest validation: `/Users/skkac/Work/SCAN/route_status_20260610/dataset_delivery_manifest_0000_0999_validation.json`
- Delivery acceptance: `/Users/skkac/Work/SCAN/route_status_20260610/delivery_acceptance_20260611.json`
- Packaged files: `34`
- Large referenced files: `3`
- Included side-track evidence:
  - route decision JSON/Markdown
  - residual surface-assignment report and preview
  - residual absorption parameter sweep
  - residual surface miss-reason report
  - residual candidate surface coverage sweep
  - surface seed candidate/promotion diagnostics
  - surface target/fusion bottleneck base and strict-surface A/B diagnostics
  - ConceptSeg fine-object alignment report
  - ConceptSeg instance-intersection report and accepted sheet
  - old-route reference validation
