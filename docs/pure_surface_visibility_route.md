# Pure Surface-Visibility Semantic Route

## Principle

The clean route separates three kinds of evidence:

- `drivability_cpp` is a structural region prior, not semantic truth.
- first-touch depth is a visibility prior, not dense ground-truth depth.
- 2D segmentation is candidate evidence, not final object identity.

The fusion target is therefore not `mask label -> point label`.  It is:

```text
view-valid target points
+ structural-region compatibility
+ local 3D geometry
+ color/texture/mask evidence
+ multi-view consistency
=> surface merge / attached object / ambiguous target
```

## Frozen Baseline

- Full raw 0.01m voxel PLY with frame metadata.
- Source-frame guard: `frame_mean +/- 20`.
- Visibility: z-buffer + first-touch.
- Disabled by default: image-space splat, hole fill, pixel-height guard, blue-sky heuristics.

## Structural Region Field

`scripts/build_structural_region_field.py` converts the color-coded
`drivability_cpp` PCD into a non-semantic voxel field:

- red -> `ground_like_region`
- white -> `vertical_surface_region`
- green -> `upper_horizontal_region`
- blue/other -> `other_structure_region`

Downstream code must not directly relabel these as floor/wall/ceiling.  They
are compatibility evidence for surface attachment decisions.

## Surface Attachment

`scripts/classify_surface_attachment.py` annotates each frame-local target with:

- `merge_to_structural_region`: target behaves like part of a large structure.
- `attached_object_candidate`: target lies in/on a structural region but has
  independent geometry such as linearity, thickness, or scattering.
- `independent_object_candidate`: target has weak surface-region support or is
  dominated by `other_structure_region`.
- `ambiguous_surface_attachment`: insufficient evidence.

Examples:

- Wall texture misdetected as railing: vertical region + planar + no independent
  line geometry -> merge into structural region.
- Railing along a wall: vertical region + line-like independent geometry ->
  attached object candidate.
- Cabinet top near upper-horizontal prior: small/isolated/high-scattering target
  remains attached/ambiguous instead of becoming ceiling.

## Object Fusion Contract

`scripts/fuse_targets_to_objects.py` consumes the attachment metadata as a merge
gate:

- `merge_to_structural_region` targets may merge with compatible large-surface
  objects, but must not pollute fine-object objects.
- `attached_object_candidate` and `independent_object_candidate` targets must
  not be absorbed by broad surface-parent rules unless their own surface label
  evidence says they are a surface.
- Object JSONL keeps weighted `surface_attachment_votes` and
  `structural_region_votes`, plus dominant summaries for viewer QA.

This is the practical boundary between structure priors and semantics.  A
region prior can explain why a target is near a wall-like/ground-like region,
but it cannot by itself confirm `wall`, `floor`, or `ceiling`.

## 5070Ti Smoke

`scripts/run_rtx5070_pure_surface_visibility_smoke.sh` runs the closed loop on
`scan-rtx5070`:

```text
drivability PCD -> structural field -> surface attachment targets
-> object fusion -> viewer PLY/JSONL export
```

The current smoke window `3400..3500` produced:

- `83` targets
- `42` objects
- merge ratio `0.494`
- viewer points `378,817`
- missing target points `0`

## 5070Ti Production Entry

`scripts/run_rtx5070_pure_surface_visibility_route.sh` is the fixed runner for
the clean mainline.  It executes on `scan-rtx5070` and keeps the heavy work
remote:

```text
run_parking_safe_semantic_prior_route.sh to Target stage
-> build_structural_region_field.py
-> classify_surface_attachment.py
-> fuse_targets_to_objects.py with attachment gates
-> export_frame_target_objects_for_viewer.py
-> qa_viewer_candidate.py
```

Default mode is dry-run.  Use `RUN=1`; use `PULL_RESULTS=1` only for review
artifacts.  Current verified window:

```bash
RUN=1 OVERWRITE=1 PULL_RESULTS=1 START=3400 END=3500 STRIDE=10 \
  OUT_SUFFIX=pure_surface_visibility_window_3400_3500 \
  ./scripts/run_rtx5070_pure_surface_visibility_route.sh
```

Result summary:

- geometry guidance images: `33/33 ok`
- frame targets: `93`
- attachment targets: `93`, missing target points `0`
- object fusion: `43` objects, merge ratio `0.538`
- viewer points: `386,721`, missing target points `0`
- base viewer QA: `ok`, with warning for one large railing object
- local-geometry viewer QA: `ok`, warnings `[]`

Interpretation: the route is now reproducible end-to-end.  The remaining
quality issue is not structural-prior fusion; it is source mask / target split
quality for large fine objects such as railing.  The runner now applies a
post-viewer local geometry split for large fine objects:

- candidate: `object_id=28`, source `obj_000028`, `railing`, `11,537` points
- split result: `wall=4,389`, `railing=1,616`, `ground=636`,
  `unknown=4,896`
- final semantic point counts: `wall=369,329`, `ground/floor=7,551`,
  `railing=3,247`, `unknown=4,896`

This confirms the principle: broad image masks can propose a fine-object
region, but local point geometry must decide which parts are actually railing
versus surface or unresolved evidence.

## Next Integration

The next production orchestrator should run:

```text
structural field -> first-touch visibility -> mask gated targets
-> surface attachment -> object fusion -> viewer export
```

Mimo/VLM should operate only after this stage, using target/object evidence
summaries rather than raw masks alone.
