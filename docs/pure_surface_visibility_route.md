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

## Next Integration

The next production orchestrator should run:

```text
structural field -> first-touch visibility -> mask gated targets
-> surface attachment -> object fusion -> viewer export
```

Mimo/VLM should operate only after this stage, using target/object evidence
summaries rather than raw masks alone.
