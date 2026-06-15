# Increment 2000-2999 Status

Checked on 2026-06-15.

## Main Route Readiness

Remote dataset readiness report:
`/root/epfs/new_route_stage1_skymask/reports/dataset_readiness_2000_2999.json`

- frames: `1000`
- images: `3000`
- complete camera frames: `1000/1000`
- color PLY: `1000/1000`
- sky masks: `3000/3000`
- Python SAM2 masks: `3000/3000`
- completion semantic images: `3000/3000`

All readiness ratios are `1.0`.

## Target/Object Fusion

Remote output:
`/root/epfs/new_route_stage1_skymask/target_object_fusion_2000_2999_surface024_fine012`

QA:
`/root/epfs/new_route_stage1_skymask/target_object_fusion_2000_2999_surface024_fine012/reports/target_object_qa.json`

- frames ok: `1000/1000`
- targets: `5957`
- target points: `13,910,027`
- small residual ratio: `0.01095`
- objects: `1101`
- merge ratio: `0.8152`
- ambiguous objects: `35`
- ambiguous ratio: `0.0318`
- object status counts:
  - stable: `630`
  - single_target: `436`
  - ambiguous_object: `35`
- object semantic label counts:
  - floor: `588`
  - equipment: `141`
  - building: `107`
  - railing: `101`
  - person: `61`
  - other: `45`
  - ambiguous: `35`
  - pipe: `17`
  - wall: `6`

The current object QA passes the intended merge/ambiguity sanity threshold for
this increment. Remaining QA should be visual: inspect the stride PLY and the
known ambiguous wall/floor and building/railing conflicts.

## C++ TensorRT Side Candidate

Remote C++ semantic smoke summary:
`/root/epfs/sam2_tensorrt/reports/trt50_semantic_smoke_summary.json`

For `cam0_002000` to `cam0_002049`:

- C++ RLE masks linked: `50/50`
- downstream semantic stages: all `50/50`
- completion label records: `1352`
- top labels:
  - equipment: `668`
  - floor: `281`
  - ignore: `197`
  - wall: `121`
  - pipe: `44`
  - building: `40`
  - railing: `1`
- parse failures from logs:
  - initial `sam2_qwen`: `5/50`
  - prompt-v3 review: `0/50`
  - completion: `0/50`

Conclusion: C++/TensorRT is operationally compatible with the semantic pipeline
but should not replace the Python SAM2 main route yet. The high-coverage masks
inflate region count and over-bias labels toward `equipment`, while thin-object
recall remains weak. Keep it as a high-coverage side candidate for point
projection and large-surface-first experiments.

