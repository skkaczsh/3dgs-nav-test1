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

Expanded validation window:

```bash
RUN=1 OVERWRITE=1 PULL_RESULTS=1 START=3000 END=3600 STRIDE=10 \
  OUT_SUFFIX=pure_surface_visibility_window_3000_3600 \
  ./scripts/run_rtx5070_pure_surface_visibility_route.sh
```

Result summary:

- geometry guidance images: `183/183 ok`
- frame targets: `760`
- attachment targets: `760`, missing target points `0`
- object fusion: `281` base objects, merge ratio `0.630`
- base viewer points: `1,130,157`, missing target points `0`
- base viewer QA: `ok`, with one large railing warning
- local-geometry viewer: `289` objects, QA `ok`, warnings `[]`
- final semantic point counts: `wall=1,044,360`, `ground/floor=59,242`,
  `grass=11,655`, `car=4,885`, `railing=5,011`, `unknown=5,004`

The same large-railing pattern appeared as `obj_000224` (`11,669` points) and
was split into `wall=4,413`, `railing=1,616`, `ground=636`,
`unknown=5,004`.  This suggests the local-geometry fine-object split is a
stable post-target correction, not a one-off patch.

Full parking run:

```bash
RUN=1 OVERWRITE=1 PULL_RESULTS=0 START=0 END=6180 STRIDE=10 \
  OUT_SUFFIX=pure_surface_visibility_full_0000_6180 \
  ./scripts/run_rtx5070_pure_surface_visibility_route.sh
```

Remote viewer entry:

```text
http://scan-rtx5070:8765/tools/semantic_viewer_index.html
```

Focused object review entry:

```text
http://scan-rtx5070:8765/work_MT20260616-175807/review_pure_surface_visibility_full_0000_6180/semantic_object_review_index.html
```

The review page is intentionally a manual-QA entry, not an implicit relabel
stage.  It also writes `manual_object_review_decisions.csv` with one row per
selected object:

```text
object_id, source_object_id, current_label, decision, new_label, confidence, reviewer, notes
```

Valid decisions are `keep`, `relabel`, `demote_unknown`, `split_review`, and
`reject_artifact`.  Normalize filled decisions before any downstream
application:

```bash
python3 scripts/normalize_manual_object_review_decisions.py \
  --decisions-csv <review_dir>/manual_object_review_decisions.csv \
  --review-index-json <review_dir>/semantic_object_review_index.json \
  --output-jsonl <review_dir>/manual_object_review_decisions.normalized.jsonl \
  --report-json <review_dir>/manual_object_review_decisions.report.json
```

This keeps human QA explicit and auditable.  A later apply stage may consume the
normalized JSONL, but object labels must not be silently rewritten from the HTML
view alone.

Apply normalized manual decisions to object metadata:

```bash
python3 scripts/apply_manual_object_review_decisions.py \
  --objects-jsonl <viewer_dir>/frame_objects_viewer.jsonl \
  --decisions-jsonl <review_dir>/manual_object_review_decisions.normalized.jsonl \
  --output-objects-jsonl <review_dir>/frame_objects_viewer.manual_reviewed.jsonl \
  --report-json <review_dir>/manual_object_review_apply_report.json
```

This updates JSONL metadata only.  Because semantic color in the viewer PLY is
stored in the PLY `semantic` field, point colors change only after re-exporting
the viewer PLY from the reviewed object JSONL with
`scripts/export_frame_target_objects_for_viewer.py`.

The preferred wrapper for the whole reviewed-artifact export is:

```bash
python3 scripts/run_manual_object_review_export.py \
  --decisions-csv <review_dir>/manual_object_review_decisions.csv \
  --review-index-json <review_dir>/semantic_object_review_index.json \
  --objects-jsonl <work_dir>/frame_objects_attachment_pure_surface_visibility_full_0000_6180/objects.jsonl \
  --targets-jsonl <work_dir>/frame_targets_pure_surface_visibility_full_0000_6180/frame_targets.jsonl \
  --target-ply <work_dir>/frame_targets_pure_surface_visibility_full_0000_6180/frame_targets.ply \
  --output-dir <work_dir>/frame_object_viewer_manual_reviewed_pure_surface_visibility_full_0000_6180 \
  --stride 10 \
  --copy-review-inputs
```

The wrapper performs normalization, apply, PLY re-export, viewer QA, and writes
`manual_object_review_export_report.json`.  Without reviewed CSV rows it exits
at the normalization gate unless `--allow-normalize-errors` is explicitly used.

Latest full artifact in the index:

```text
frame_object_viewer_attachment_localgeom_pure_surface_visibility_full_0000_6180
```

Result summary:

- geometry guidance images: `1857/1857 ok`
- geometry guidance elapsed: `1926.7s`
- refined priority images: `1857/1857 ok`
- priority projection frames: `619`
- projection visible non-sky points: `9,341,265`
- projection priority points: `9,023,936`
- projection residual points: `317,329`
- frame targets: `11,928`
- target points: `8,291,613`
- target label counts: `ground=1,000`, `wall=4,073`, `grass=4,805`,
  `car=1,310`, `railing=740`
- surface attachment targets: `11,928`, missing target points `0`
- attachment status counts: `merge_to_structural_region=1,653`,
  `ambiguous_surface_attachment=4,260`,
  `independent_object_candidate=5,470`,
  `attached_object_candidate=457`, `unstructured_target=88`
- object fusion: `3,839` base objects, merge ratio `0.678`
- base viewer points: `8,291,613`, QA `ok`
- base semantic point counts: `ground=1,676,883`, `wall=5,301,875`,
  `grass=929,448`, `car=218,854`, `railing=164,553`
- local-geometry split candidates: `18` selected (`15` railing, `3` car)
- local-geometry output: `3,900` objects, QA `ok`
- local-geometry QA after class-aware large-fine thresholds: warnings `[]`
- local-geometry final semantic point counts: `ground=1,682,648`,
  `wall=5,348,985`, `grass=929,448`, `car=218,854`,
  `railing=93,090`, `unknown=18,588`

Interpretation:

- The full clean route is reproducible end-to-end on `scan-rtx5070`.
- The first-touch/full-pointcloud visibility stage is healthy: no failed images,
  no missing target points, and no sky/back-wall projection regression was
  reported by the cheap QA gate.
- The local-geometry split substantially reduced surface-swallowing risk for
  broad railing masks: railing points dropped from `164,553` to `93,090`, while
  `47,110` points became wall, `5,765` became ground, and `18,588` became
  unknown rather than forced fine-object evidence.
- QA now uses class-aware large-fine thresholds: `railing >= 10,000` points is
  still suspicious, while `car >= 25,000` points is the warning threshold.
  The previously reported `10,698`-point car (`object_id=1742`) is therefore
  retained as a normal large object candidate instead of a system-level warning.
- This means the next QA step should be visual review of the full remote viewer,
  not another blind parameter sweep.  If a visually invalid car remains, add a
  targeted car split rule using local geometry; do not lower the generic
  threshold back to a class-agnostic value.
- `tools/semantic_ply_viewer.html` now supports `object=<id>` URL filtering.
  The focused object review page contains direct semantic/object/RGB links for
  top car, railing, wall, ground, grass, unknown, and local-geometry child
  objects.  Use this page for manual QA before changing fusion parameters.

## Next Integration

The next production orchestrator should run:

```text
structural field -> first-touch visibility -> mask gated targets
-> surface attachment -> object fusion -> viewer export
```

Mimo/VLM should operate only after this stage, using target/object evidence
summaries rather than raw masks alone.
