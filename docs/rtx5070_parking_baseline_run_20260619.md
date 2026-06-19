# RTX 5070Ti Parking Baseline Run - 2026-06-19

## Summary

- Host: `scan-rtx5070`
- Remote repo: `/home/zsh/Work/SCAN/new_route`
- Remote dataset: `/home/zsh/Work/SCAN/datasets/MT20260616-175807`
- Remote workdir: `/home/zsh/Work/SCAN/work_MT20260616-175807`
- Local review copy: `server_parking_priority_s10/frame_object_viewer_priority_full_s10_v5_geometry_ceiling_rtx5070`

This run restores the current validated frame-local route on the RTX 5070Ti host.
The old global object back-projection route is still invalid and was not used.

## Environment

- Released Gemma/llama-server before running: PID `1723`, about `8708MiB` VRAM.
- Post-release GPU baseline: about `1.3GB` VRAM used by desktop processes.
- Python env: `/home/zsh/Work/SCAN/.venvs/scan-semantic`
- CUDA stack: `torch 2.11.0+cu130`, CUDA available.
- Added dependencies: `transformers==4.47.1`, `scipy`, `scikit-learn`, `accelerate`, `safetensors`, `tqdm`.
- Clash/Mihomo proxy used for downloads: `127.0.0.1:7897`.

## TensorRT Readiness

Probe command:

```bash
cd /Users/skkac/Work/SCAN/new_route
python3 scripts/check_rtx5070_tensorrt_readiness.py \
  --output server_parking_priority_s10/parking_candidate_manifest_rtx5070/rtx5070_tensorrt_readiness.json
```

Current 5070Ti result:

- passed: `true` as a probe, but TensorRT is **not ready**
- PyTorch/CUDA is ready:
  - `torch_version=2.11.0+cu130`
  - `torch_cuda_available=1`
  - `torch_cuda_version=13.0`
- GPU driver: `580.126.20`
- CUDA home exists: `true`
- missing:
  - `trtexec`
  - TensorRT C++ headers
  - TensorRT C++ libraries
  - Python `tensorrt`
  - Python `onnx`
  - Python `polygraphy`
  - Python `onnxruntime`
  - ONNX Runtime TensorRT provider

Interpretation: the 5070Ti host is ready for PyTorch GPU inference but not yet
ready for TensorRT acceleration. The next acceleration setup should first install
TensorRT runtime/dev tooling plus Python ONNX helpers, then run a tiny ONNX
TensorRT smoke before attempting SAM2/DINO engine work. Do not treat this as a
semantic-quality fix; it is only a throughput optimization path.

5070Ti-specific setup assets now exist:

```bash
scripts/setup_rtx5070_tensorrt_env.sh
scripts/verify_rtx5070_tensorrt_env.sh
```

Dry-run result:

- OS: `Ubuntu 24.04.4 LTS`
- CUDA: `/usr/local/cuda-13.2`
- matching TensorRT pin: `11.0.0.114-1+cuda13.2`
- Python package pin: `tensorrt-cu13==11.0.0.114`
- apt simulation is solvable after explicitly pinning
  `libnvinfer-safe-headers-dev`
- planned apt install:
  - `16` new packages
  - no CUDA 13.3 packages pulled
  - no removals
- dry-run intentionally does not invoke `pip install --dry-run`, because NVIDIA
  TensorRT Python wheels can still download multi-GB payloads during dry-run.
  It uses `pip index versions` instead.

Initial blocker for automatic install:

- `sudo -n true` failed on `scan-rtx5070`; apt installation required an
  interactive password.
- Disk was sufficient: root/home filesystem had more than `100G` free.

The intended install command remains:

```bash
cd /home/zsh/Work/SCAN/new_route
APPLY=1 scripts/setup_rtx5070_tensorrt_env.sh
```

The setup script will install Python helpers, install pinned TensorRT C++
runtime and dev packages, then run `verify_rtx5070_tensorrt_env.sh`.
Verification builds a C++ TensorRT smoke binary and a tiny ONNX TensorRT engine
before this environment is treated as ready for SAM2/DINO engine work.

Final setup status on `scan-rtx5070`:

- TensorRT C++ runtime/dev packages installed through apt:
  `11.0.0.114-1+cuda13.2`.
- Installed packages include `libnvinfer11`, `libnvinfer-dev`,
  `libnvinfer-plugin11`, `libnvonnxparsers11`, and `libnvinfer-bin`.
- Python TensorRT binding installed through apt:
  `python3-libnvinfer=11.0.0.114-1+cuda13.2`.
- ONNX helper packages installed in
  `/home/zsh/Work/SCAN/.venvs/scan-semantic`: `onnx`, `onnxsim`,
  `polygraphy`.
- Python `tensorrt` import reports `11.0.0.114`.
- `scripts/verify_rtx5070_tensorrt_env.sh` passed:
  - C++ TensorRT builder smoke: `builder_ok=1`
  - tiny ONNX engine build with `/usr/bin/trtexec`: `PASSED TensorRT.trtexec`
  - output engine:
    `/home/zsh/Work/SCAN/work_MT20260616-175807/tensorrt_smoke/engines/tiny_conv.plan`
- Latest readiness report:
  `/home/zsh/Work/SCAN/work_MT20260616-175807/logs/rtx5070_tensorrt_readiness_after_python_apt.json`
  - `torch_cuda_ready=true`
  - `onnx_export_ready=true`
  - `python_tensorrt_ready=true`
  - `cpp_tensorrt_ready=true`
  - `onnxruntime_tensorrt_ready=false`

Remaining optional gap: ONNX Runtime TensorRT provider is not installed. This is
not required for a C++ TensorRT runner, but should be treated separately if a
Python ONNX Runtime acceleration path is needed later.

## Data Migration

Synced only the files required by the frame-local route:

- `MANIFOLD_MT20260616-175807.lx`
- `image/video_cam0.mkv`, `image/video_cam1.mkv`, `image/video_cam2.mkv`
- `image/img_pos.txt`, `image/cam_in_ex.txt`
- `calib_online.yaml`, `calib_online_final.yaml`, `project.log`

The LAS and zip outputs were intentionally not migrated.

## Smoke Run

- Range: `0..190`, `stride=10`, `20` frames x `3` cameras.
- Extracted frames: `60/60`.
- Priority segmentation: `60/60`.
- Projection:
  - raw points: `244,700`
  - visible non-sky: `233,269`
  - priority points: `226,701`
  - residual points: `6,568`
- Frame targets:
  - input targets: `219`
  - refined targets: `221`
  - missing target points: `0`
- Object fusion:
  - objects: `77`
  - merge ratio: `0.652`
- Viewer export:
  - output vertices: `216,039`
  - missing target points: `0`

## Full Stride10 Run

- Range: `0..6180`, `stride=10`, `619` frames x `3` cameras.
- Extracted frames: `1,857/1,857`, elapsed `54.1s`.
- Priority segmentation: `1,857/1,857`.
- Projection:
  - raw points: `9,905,881`
  - visible non-sky: `9,341,265`
  - priority points: `8,758,542`
  - residual points: `582,723`
  - class counts: `ground=1,589,600`, `wall=5,642,609`, `grass=1,095,682`, `car=237,629`, `railing=193,022`
- Frame targets:
  - source targets: `9,493`
  - source target points: `8,654,647`
  - source labels: `ground=729`, `wall=2,987`, `car=1,124`, `grass=4,099`, `railing=554`
- Geometry refinement:
  - refined targets: `9,593`
  - split source targets: `54`
  - relabelled targets: `149`
  - output labels: `ground=799`, `wall=2,964`, `car=1,109`, `grass=4,099`, `railing=561`, `ceiling=61`
  - missing target points: `0`
- Object fusion:
  - objects: `3,193`
  - zones: `62`
  - merge ratio: `0.667`
  - statuses: `stable=1,567`, `single_target=1,568`, `ambiguous_object=58`
- Viewer export:
  - input vertices: `8,631,602`
  - stride10 output vertices: `863,161`
  - missing target points: `0`
  - label counts: `wall=475,505`, `ground=126,662`, `ambiguous=119,105`, `grass=96,762`, `car=22,874`, `railing=14,084`, `ceiling=8,169`

## Review Entrypoints

Local viewer URL:

```text
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/frame_object_viewer_priority_full_s10_v5_geometry_ceiling_rtx5070/frame_object_points_stride10.ply&objects=/server_parking_priority_s10/frame_object_viewer_priority_full_s10_v5_geometry_ceiling_rtx5070/frame_objects_viewer.jsonl&mode=semantic&stride=1&pointSize=1.5
```

QA contact sheet:

```text
server_parking_priority_s10/frame_local_object_qa_full_s10_v5_geometry_ceiling_rtx5070/frame_local_object_qa_contact.jpg
```

Risk trace for the current guarded candidate:

```text
server_parking_priority_s10/frame_object_trace_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070/risky_object_target_trace.jsonl
server_parking_priority_s10/frame_object_trace_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070/risky_object_target_trace.csv
```

## Interpretation

- The 5070Ti migration is operational: data, environment, model cache, GPU inference, frame-local projection, target fusion, and local review export all work.
- This run reproduces the current validated route with no missing target-point mapping.
- The bottleneck remains source mask quality and surface/fine-object confusion, not calibration or global point projection.
- Next optimization should compare source priority mask refinements on selected bad windows before any new full run.

Current guarded candidate risk trace:

- traced high-risk objects: `240`
- label counts in trace:
  - `wall=88`
  - `car=53`
  - `ground=44`
  - `railing=29`
  - `ceiling=15`
  - `grass=8`
  - `other=3`
- point counts in trace:
  - `ground=1,367,192`
  - `wall=436,686`
  - `ceiling=53,420`
  - `car=35,255`
  - `railing=28,038`
- dominant risk reasons:
  - `large_single_target_object=106`
  - `fine_object_low_points=41`
  - `ground_has_large_height_span=36`
  - `car_extent_suspicious=33`
  - `wall_normal_too_up=28`

Interpretation: remaining high-impact errors are mostly born before object
fusion. Large bad objects are usually single frame-local targets from the
priority mask stage. Next work should inspect those source masks and either
split/demote them during target construction or add a stricter geometry guard at
priority-mask refinement time. Do not spend the next iteration on global object
relabeling.

## Batch Size Benchmark

Measured on `60` undistorted frames with Mask2Former Mapillary priority segmentation:

| batch size | elapsed | max RSS | result |
| --- | ---: | ---: | --- |
| 1 | `12.37s` | `2,106,072KB` | ok |
| 2 | `12.04s` | `1,982,436KB` | ok |
| 4 | `11.69s` | `2,109,460KB` | ok |
| 8 | `11.62s` | `2,125,896KB` | ok |

Batch `8` is safe on the 5070Ti but only marginally faster than batch `4`.
Use batch `8` for long full-scene priority segmentation when VRAM is otherwise idle;
use batch `4` when sharing the GPU with another lightweight task.

## Bad-Window Geometry-Guided Mask Refinement

Tested six high-risk windows: `2200_2300`, `3400_3500`, `4000_4100`,
`5380_5420`, `5680_5800`, `5980_6040`.

Modes:

- `safe`: fill only residual holes from trusted projected surface prior.
- `diag045`: diagnostic upper bound; allow trusted surfaces to overwrite `residual/car/railing`.
- `guarded_v2`: production candidate; allow fine-label overwrite only inside projected fine components strongly supported by nearby trusted surface priors.

Aggregate projected point counts across the six windows:

| mode | ground | wall | grass | car | railing | residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| safe | `266,471` | `913,190` | `112,820` | `55,112` | `58,147` | `23,054` |
| guarded_v2 | `266,471` | `934,083` | `112,821` | `48,952` | `43,413` | `23,054` |
| diag045 | `267,586` | `942,439` | `113,690` | `46,668` | `35,358` | `23,053` |

`guarded_v2` changed `20,893` points back to wall, primarily by reducing false
`railing` (`-14,734`) and false `car` (`-6,160`). It is intentionally less
aggressive than `diag045`, which is useful as a diagnostic ceiling but too broad
for default production.

Important implementation detail: the fine/surface support ratio must be computed
over projected LiDAR pixels inside the 2D component, not over the full dense 2D
mask area. Otherwise sparse 3D priors are diluted by image pixels and the guard
never triggers.

Local QA artifacts:

```text
server_parking_priority_s10/mask_refine_guarded_v2_bad_windows_rtx5070/mask_refine_guarded_v2_bad_windows_summary_rtx5070.json
server_parking_priority_s10/mask_refine_guarded_v2_bad_windows_rtx5070/<range>/guarded_v2_contact.jpg
server_parking_priority_s10/mask_refine_guarded_v2_bad_windows_rtx5070/<range>/safe_contact.jpg
```

Next step: run `guarded_v2` on a medium slice and rebuild target/object outputs,
then verify in the viewer that wall/ground recovery does not erase real railing
or cars.

## Full Guarded V2 Run

After the bad-window and medium-window checks, `guarded_v2` was run on the full
stride10 parking dataset.

- Range: `0..6180`, `stride=10`, `619` frames x `3` cameras.
- Geometry guidance maps: `1,857/1,857`, elapsed `224.5s`.
- Guarded priority refinement:
  - images: `1,857/1,857`
  - residual surface fill pixels: `410,123`
  - guarded fine-surface override pixels: `118,023`
  - main recoveries: `railing->wall=93,165`, `car->wall=24,019`
- Projection:
  - raw points: `9,905,881`
  - visible non-sky: `9,341,265`
  - priority points: `9,140,355`
  - residual points: `200,910`
  - class counts: `ground=1,687,223`, `wall=6,019,917`, `grass=1,111,815`, `car=214,043`, `railing=107,357`
- Frame targets:
  - source targets: `9,532`
  - source target points: `9,056,679`
  - source labels: `ground=763`, `wall=3,081`, `car=1,103`, `grass=4,178`, `railing=407`
- Target geometry refinement:
  - refined targets: `9,605`
  - split source targets: `45`
  - relabelled targets: `154`
  - output labels: `ground=836`, `wall=3,051`, `car=1,089`, `grass=4,178`, `railing=387`, `ceiling=64`
  - missing target points: `0`
- Object fusion:
  - objects: `3,100`
  - zones: `62`
  - merge ratio: `0.677`
  - statuses: `stable=1,537`, `single_target=1,501`, `ambiguous_object=62`
- Viewer export:
  - input vertices: `9,033,868`
  - stride10 output vertices: `903,387`
  - missing target points: `0`
  - label counts: `wall=498,786`, `ground=127,949`, `ambiguous=143,754`, `grass=98,460`, `car=20,491`, `railing=8,228`, `ceiling=5,719`

Compared with the previous full v5 baseline viewer export:

| label | v5 baseline | guarded_v2 full | change |
| --- | ---: | ---: | ---: |
| wall | `475,505` | `498,786` | `+23,281` |
| railing | `14,084` | `8,228` | `-5,856` |
| car | `22,874` | `20,491` | `-2,383` |
| ambiguous | `119,105` | `143,754` | `+24,649` |
| ground | `126,662` | `127,949` | `+1,287` |
| grass | `96,762` | `98,460` | `+1,698` |

Interpretation:

- The guarded mask stage does reduce false fine-object leakage into large
  surfaces, especially `railing` and broad planar `car` regions.
- The increase in `ambiguous` is expected because recovered large surfaces expose
  more ground/wall/ceiling vote conflicts during object fusion rather than
  hiding them as fine-object labels.
- Remaining bottleneck has shifted from raw fine-label pollution to large
  surface object consolidation, especially ground/wall/ceiling separation in
  zones with mixed normals or scan geometry.

Local viewer URL:

```text
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/frame_object_viewer_guarded_v2_full_s10_geometry_ceiling_rtx5070/frame_object_points_stride10.ply&objects=/server_parking_priority_s10/frame_object_viewer_guarded_v2_full_s10_geometry_ceiling_rtx5070/frame_objects_viewer.jsonl&mode=semantic&stride=1&pointSize=1.5
```

Local QA contact sheet:

```text
server_parking_priority_s10/frame_local_object_qa_guarded_v2_full_s10_geometry_ceiling_rtx5070/frame_local_object_qa_contact.jpg
```

Local report bundle:

```text
server_parking_priority_s10/guarded_v2_full_reports/guarded_v2_full_summary.local.json
```

Recommended next step: keep `guarded_v2` as the default priority-mask correction
candidate, then improve object fusion for large surfaces with a stricter
plane/normal/height-aware split before label voting.

## Strict Surface Fusion Check

The full guarded_v2 target set was re-fused with `--strict-surface-labels`,
without rerunning mask refinement, projection, or target generation.

Command effect:

- Blocks `ground/wall/ceiling/building` targets from merging across surface
  labels through the relaxed parent-class rule.
- Keeps fine-object labels unchanged.
- Turns surface vote conflicts into separate objects instead of `ambiguous`
  objects.

Results:

| metric | guarded_v2 default fusion | guarded_v2 strict surface |
| --- | ---: | ---: |
| objects | `3,100` | `3,198` |
| merge ratio | `0.677` | `0.667` |
| ambiguous objects | `62` | `0` |
| stable objects | `1,537` | `1,663` |
| single-target objects | `1,501` | `1,535` |

Viewer label counts:

| label | default fusion | strict surface | change |
| --- | ---: | ---: | ---: |
| ambiguous | `143,754` | `0` | `-143,754` |
| wall | `498,786` | `569,838` | `+71,052` |
| ground | `127,949` | `183,693` | `+55,744` |
| ceiling | `5,719` | `22,677` | `+16,958` |
| car | `20,491` | `20,491` | `0` |
| railing | `8,228` | `8,228` | `0` |
| grass | `98,460` | `98,460` | `0` |

Interpretation:

- `strict_surface` fixes the main remaining fusion artifact: large
  ground/wall/ceiling targets were being merged into one object and then marked
  `ambiguous`.
- It does not change the fine-object mask result, so it preserves the guarded_v2
  reduction of false `railing` and false `car`.
- The next remaining risk is not ambiguity but surface over-splitting and
  geometry-label errors, visible in QA as `ground_has_large_height_span`,
  `wall_too_flat_low_height`, and `wall_normal_too_up`.

Current recommended review URL:

```text
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/frame_object_viewer_guarded_v2_full_s10_strict_surface_rtx5070/frame_object_points_stride10.ply&objects=/server_parking_priority_s10/frame_object_viewer_guarded_v2_full_s10_strict_surface_rtx5070/frame_objects_viewer.jsonl&mode=semantic&stride=1&pointSize=1.5
```

Strict-surface QA contact sheet:

```text
server_parking_priority_s10/frame_local_object_qa_guarded_v2_full_s10_strict_surface_rtx5070/frame_local_object_qa_contact.jpg
```

Local comparison report:

```text
server_parking_priority_s10/guarded_v2_strict_surface_reports/guarded_v2_strict_surface_compare.local.json
```

Recommended default for the parking route is now:

```text
guarded_v2 priority masks -> target geometry refinement -> fuse_targets_to_objects --strict-surface-labels
```

Next optimization should focus on surface target refinement, especially splitting
large horizontal/vertical surfaces before fusion so that `wall_too_flat_low_height`
and `ground_has_large_height_span` are handled earlier.

## Surface Refinement Follow-Up

Two low-cost post-baseline checks were run from the full guarded_v2 target set.

### Target Height Split

`refine_frame_targets_by_geometry.py` was rerun with
`--split-horizontal-wall-by-height`, then fused with `--strict-surface-labels`.

Result:

- targets: `9,605 -> 9,820`
- objects: `3,198 -> 3,231`
- `ground_has_large_height_span`: `20 -> 18`
- `wall_too_flat_low_height`: unchanged at `7`
- `wall_normal_too_up`: unchanged at `7`
- viewer `ceiling`: `22,677 -> 11,746`

Interpretation: this is not a good default. It splits more surfaces but does not
fix the high-risk flat wall cases, and it reduces ceiling coverage.

### Object-Level Surface Relabel

`refine_target_fusion_objects.py` was extended to understand the parking
pipeline's `ground` label and to optionally relabel flat horizontal `wall`
objects from geometry:

```text
refine_target_fusion_objects.py --geometry-relabel-flat-wall --horizontal-surface-label ground
```

Result:

- changed objects: `46 / 3,198`
- object label counts:
  - `wall: 787 -> 741`
  - `ground: 84 -> 100`
  - `ceiling: 31 -> 61`
- viewer label counts:
  - `wall: 569,838 -> 566,089`
  - `ground: 183,693 -> 186,245`
  - `ceiling: 22,677 -> 23,874`
- Top160 QA candidate risks:
  - `wall_too_flat_low_height: 7 -> 0`
  - `wall_normal_too_up: 7 -> 0`
  - `ground_has_large_height_span: 20 -> 25`

Interpretation: object-level relabel is useful as an audit/local correction tool
for obvious flat wall mistakes. The apparent `ground_has_large_height_span`
increase is a Top160 candidate-selection effect, not a global-risk increase;
see the full-object QA comparison below.

Object-relabel viewer:

```text
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/frame_object_viewer_guarded_v2_full_s10_strict_surface_object_relabel_rtx5070/frame_object_points_stride10.ply&objects=/server_parking_priority_s10/frame_object_viewer_guarded_v2_full_s10_strict_surface_object_relabel_rtx5070/frame_objects_viewer.jsonl&mode=semantic&stride=1&pointSize=1.5
```

Comparison report:

```text
server_parking_priority_s10/guarded_v2_object_relabel_reports/surface_refinement_compare.local.json
```

Current default remains:

```text
guarded_v2 priority masks -> target geometry refinement -> fuse_targets_to_objects --strict-surface-labels
```

Next real improvement should target `ground_has_large_height_span` at the target
level, likely by splitting large ground targets using local PCA/height layers and
only relabelling subclusters whose normals are clearly vertical or overhead.

## Ground Artifact Guard Follow-Up

`refine_frame_targets_by_geometry.py` now has an opt-in guard for linear,
high-span `ground` targets:

```text
refine_frame_targets_by_geometry.py --guard-linear-ground-artifacts
```

The rule is intentionally narrow. It only fires when a target labelled `ground`
has high Z span, high linearity, and low planarity. Up-facing line artifacts are
demoted to `other`; non-up line artifacts are relabelled to `wall`.

### Target-Level Guard Only

Result from full guarded_v2 targets, followed by strict-surface fusion:

- target relabels added by this guard:
  - `linear_ground_artifact_to_wall: 3`
  - `linear_ground_artifact_to_other: 4`
- object count: `3,198 -> 3,197`
- viewer label counts:
  - `ground: 183,693 -> 182,951`
  - `wall: 569,838 -> 570,236`
  - `other: 0 -> 344`
- QA risks:
  - `ground_has_large_height_span: 20 -> 17`
  - `wall_too_flat_low_height`: unchanged at `7`
  - `wall_normal_too_up`: unchanged at `7`

Interpretation: this is a low-risk target-level patch for a small class of line
artifacts, but it does not address flat wall mistakes by itself. Keep it opt-in
until a viewer pass confirms the new `other` points are acceptable.

Viewer:

```text
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/frame_object_viewer_guarded_v2_full_s10_ground_artifact_guard_strict_rtx5070/frame_object_points_stride10.ply&objects=/server_parking_priority_s10/frame_object_viewer_guarded_v2_full_s10_ground_artifact_guard_strict_rtx5070/frame_objects_viewer.jsonl&mode=semantic&stride=1&pointSize=1.5
```

### Target Guard Plus Object Relabel

The target-level guard was also combined with object-level surface relabel:

```text
refine_frame_targets_by_geometry.py --guard-linear-ground-artifacts
fuse_targets_to_objects.py --strict-surface-labels
refine_target_fusion_objects.py --geometry-relabel-flat-wall --horizontal-surface-label ground
```

Result:

- changed objects in object relabel: `46 / 3,197`
- object label counts:
  - `ground: 79 -> 95`
  - `wall: 787 -> 741`
  - `ceiling: 31 -> 61`
  - `other: 4 -> 4`
- viewer label counts:
  - `ground: 182,951 -> 185,503`
  - `wall: 570,236 -> 566,487`
  - `ceiling: 22,677 -> 23,874`
  - `other: 344 -> 344`
- QA risks:
  - `wall_too_flat_low_height: 7 -> 0`
  - `wall_normal_too_up: 7 -> 0`
  - `ground_has_large_height_span: 17 -> 22`

Initial interpretation was based on `risk_reason_counts`, which only described
the selected Top160 evidence candidates. `build_frame_local_object_qa_pack.py`
now also reports `all_risk_reason_counts` over every risky object. The full
counts show that object-level relabel does not increase the global
`ground_has_large_height_span`; it only moves more ground-risk objects into the
Top160 review set.

Full-object QA comparison:

| version | all risky objects | ground high span | wall flat low height | wall normal up |
| --- | ---: | ---: | ---: | ---: |
| strict surface default | `965` | `41` | `31` | `36` |
| strict surface + object relabel | `944` | `41` | `9` | `28` |
| ground artifact guard + strict surface | `963` | `36` | `31` | `36` |
| ground artifact guard + object relabel | `942` | `36` | `9` | `28` |

Interpretation: the combined version is the best full-QA candidate so far: it
keeps the target-level reduction in high-span ground artifacts and also removes
most flat/up-normal wall risks. It should be treated as the next candidate for
viewer review, not blindly as the committed default, because it introduces
`other=344` viewer points and changes surface label balance.

Combined viewer:

```text
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/frame_object_viewer_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070/frame_object_points_stride10.ply&objects=/server_parking_priority_s10/frame_object_viewer_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070/frame_objects_viewer.jsonl&mode=semantic&stride=1&pointSize=1.5
```

Full-risk comparison report:

```text
server_parking_priority_s10/guarded_v2_surface_refinement_all_risk_compare/qa_compare.md
server_parking_priority_s10/guarded_v2_surface_refinement_all_risk_compare/qa_compare.json
```

The report is generated by `scripts/compare_frame_local_object_qa.py` from the
QA pack outputs, so future route variants can be compared without hand-rolled
JSON snippets.

Current default remains unchanged until visual review accepts the combined
candidate:

```text
guarded_v2 priority masks -> target geometry refinement -> fuse_targets_to_objects --strict-surface-labels
```

Candidate default under review:

```text
guarded_v2 priority masks -> target geometry refinement --guard-linear-ground-artifacts -> fuse_targets_to_objects --strict-surface-labels -> refine_target_fusion_objects --geometry-relabel-flat-wall --horizontal-surface-label ground
```

## Reproducible Candidate Rebuild

Before starting or resuming remote jobs, run the runtime healthcheck from the
local repo:

```bash
cd /Users/skkac/Work/SCAN/new_route
python3 scripts/check_rtx5070_parking_runtime.py \
  --output server_parking_priority_s10/parking_candidate_manifest_rtx5070/rtx5070_runtime_check.json
```

Current healthcheck result:

- passed: `true`
- host: `scan-rtx5070` / `zsh-AORUS`
- GPU: `NVIDIA GeForce RTX 5070 Ti`
- VRAM: `1,288MiB / 16,303MiB`, free `15,015MiB`
- GPU utilization: `3%`
- tmux session: `scan_migrate`
- remote workdir size: `8.8G`
- remote venv CUDA: `torch_cuda_available=1`, `torch_cuda_version=13.0`
- proxy: port `7897` listening
- required remote candidate artifacts: all present

The healthcheck also verifies that the candidate source targets, viewer PLY,
viewer objects JSONL, viewer export report, QA report, and full-risk comparison
JSON exist on the 5070Ti workdir. Treat a failed healthcheck as a pre-run blocker
unless the failing artifact is intentionally being regenerated.

The preferred local launcher is:

```bash
cd /Users/skkac/Work/SCAN/new_route
scripts/start_rtx5070_parking_candidate_surface_route.sh
```

It runs the runtime healthcheck, runs the remote candidate script in
`CHECK_ONLY=1` mode, then starts the rebuild inside tmux session
`rtx5070_parking_candidate`.

Useful local launcher options:

```bash
DRY_RUN=1 scripts/start_rtx5070_parking_candidate_surface_route.sh
RESTART=1 scripts/start_rtx5070_parking_candidate_surface_route.sh
FORCE=1 scripts/start_rtx5070_parking_candidate_surface_route.sh
```

Current launcher verification:

- healthcheck passed
- remote `CHECK_ONLY=1` passed
- tmux session `rtx5070_parking_candidate` started successfully
- the run completed immediately because all candidate artifacts already existed
- remote log:
  `/home/zsh/Work/SCAN/work_MT20260616-175807/logs/rtx5070_parking_candidate.log`

For direct remote debugging, the candidate surface route can still be rebuilt on
`scan-rtx5070` with:

```bash
cd /home/zsh/Work/SCAN/new_route
scripts/run_rtx5070_parking_candidate_surface_route.sh
```

The script starts from the validated full guarded_v2 target set:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/frame_targets_guarded_v2_full_s10_geometry_ceiling_rtx5070
```

It intentionally does not rerun expensive priority segmentation/projection.
Instead, it rebuilds or reuses these cheap surface-object branches:

- `strict_surface`
- `strict_surface_object_relabel`
- `ground_artifact_guard_strict`
- `ground_guard_object_relabel`

Useful options:

```bash
CHECK_ONLY=1 scripts/run_rtx5070_parking_candidate_surface_route.sh
FORCE=1 scripts/run_rtx5070_parking_candidate_surface_route.sh
```

Primary candidate output:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/frame_object_viewer_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070
```

Primary comparison report:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/guarded_v2_surface_refinement_all_risk_compare/qa_compare.md
```

To pull the review-sized candidate artifacts back to the local viewer workspace:

```bash
cd /Users/skkac/Work/SCAN/new_route
scripts/pull_rtx5070_parking_candidate_surface_route.sh
```

Preferred local refresh command after any remote run:

```bash
cd /Users/skkac/Work/SCAN/new_route
scripts/refresh_rtx5070_parking_candidate_review.sh
```

This runs the 5070Ti healthcheck, pulls review-sized artifacts, rebuilds
`manifest.json` / `manifest.md`, validates the manifest, and prints the viewer
URL. Use `DRY_RUN=1` to verify the remote pull plan while still validating the
current local package:

```bash
DRY_RUN=1 scripts/refresh_rtx5070_parking_candidate_review.sh
```

Current refresh dry-run result:

- remote runtime/artifact healthcheck passed
- rsync dry-run completed
- local manifest rebuilt
- local manifest validation passed
- viewer URL printed for the current candidate

The output handoff files are:

```text
server_parking_priority_s10/parking_candidate_manifest_rtx5070/rtx5070_runtime_check.json
server_parking_priority_s10/parking_candidate_manifest_rtx5070/manifest.json
server_parking_priority_s10/parking_candidate_manifest_rtx5070/manifest.md
server_parking_priority_s10/parking_candidate_manifest_rtx5070/validation.json
```

The pull script syncs the candidate viewer PLY/JSONL, compact QA files, and the
full-risk comparison report into `server_parking_priority_s10/`. It skips QA
crop images by default to avoid local disk growth. To pull crops for offline
image review:

```bash
PULL_QA_CROPS=1 scripts/pull_rtx5070_parking_candidate_surface_route.sh
```

Use `DRY_RUN=1` before large syncs or when changing network routes:

```bash
DRY_RUN=1 scripts/pull_rtx5070_parking_candidate_surface_route.sh
```

After pulling, build the local review manifest:

```bash
python3 scripts/build_rtx5070_parking_candidate_manifest.py
```

Current manifest outputs:

```text
server_parking_priority_s10/parking_candidate_manifest_rtx5070/manifest.md
server_parking_priority_s10/parking_candidate_manifest_rtx5070/manifest.json
```

The manifest records the candidate viewer paths, compact QA artifacts, full-risk
comparison metrics, and the fixed rebuild/pull commands. It is the current
handoff point for this candidate route before visual acceptance.

Validate the handoff manifest before switching machines or treating the candidate
as reusable input:

```bash
python3 scripts/validate_rtx5070_parking_candidate_manifest.py \
  --manifest server_parking_priority_s10/parking_candidate_manifest_rtx5070/manifest.json \
  --output server_parking_priority_s10/parking_candidate_manifest_rtx5070/validation.json
```

Current validation result:

- passed: `true`
- viewer output vertices: `903,387`
- objects with points: `3,197`
- missing target points: `0`
- strict-surface baseline risky objects: `965`
- candidate risky objects: `942`
- key full-risk deltas:
  - `ground_has_large_height_span: -5`
  - `wall_normal_too_up: -8`
  - `wall_too_flat_low_height: -22`

The validator is intentionally stricter than the builder: it rechecks required
artifact files on disk, confirms the current candidate route name/stages, checks
the embedded builder checks, and fails if the candidate no longer improves the
surface-risk metrics over the strict-surface baseline.
