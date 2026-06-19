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

Fine-surface guard candidate:

```text
server_parking_priority_s10/frame_object_viewer_guarded_v3_full_s10_fine_surface_guard_object_relabel_rtx5070/frame_object_points_stride10.ply
server_parking_priority_s10/frame_object_viewer_guarded_v3_full_s10_fine_surface_guard_object_relabel_rtx5070/frame_objects_viewer.jsonl
server_parking_priority_s10/frame_object_trace_guarded_v3_full_s10_fine_surface_guard_object_relabel_rtx5070/risky_object_target_trace.csv
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

## Fine-Surface Guard Candidate

Candidate name:
`frame_object_viewer_guarded_v3_full_s10_fine_surface_guard_object_relabel_rtx5070`.

Change under test:

- Added optional target-level flag `--guard-fine-surface-artifacts`.
- Rule is deliberately narrow: only `railing` targets that are thin, horizontal,
  and planar are demoted to the corresponding surface label.
- Existing default route is unchanged unless the flag is explicitly enabled.

Target refine summary:

- input targets: `9605`
- output targets: `9640`
- split source targets: `17`
- relabelled targets: `16`
- new rule hits: `flat_horizontal_railing_to_surface=9`
- output label counts:
  - `ground=845`
  - `wall=3054`
  - `car=1089`
  - `grass=4178`
  - `railing=378`
  - `ceiling=92`
  - `other=4`

Viewer stride10 delta versus guarded v2 object-relabel candidate:

| label | v2 | v3 | delta |
| --- | ---: | ---: | ---: |
| wall | `566487` | `566496` | `+9` |
| ground | `185503` | `185480` | `-23` |
| grass | `98460` | `98453` | `-7` |
| ceiling | `23874` | `23721` | `-153` |
| car | `20491` | `20489` | `-2` |
| railing | `8228` | `8145` | `-83` |
| other | `344` | `344` | `0` |

Interpretation:

- This guard safely removes a small number of obvious horizontal railing
  artifacts, but it does not materially change the scene.
- The remaining large errors are still upstream: priority mask / source target
  generation is producing broad surface/fine-object confusion before object
  fusion.
- The next useful iteration should operate at source mask or target construction
  level, with bad-window overlays and point-depth/geometry-aware mask splitting.
  More object-level relabeling is unlikely to solve the core issue.

## Target Geometry Conflict QA

New reusable diagnostic:

```bash
python scripts/diagnose_frame_target_geometry_conflicts.py \
  --targets-jsonl <frame_targets_refined.jsonl> \
  --output-jsonl <conflicts.jsonl> \
  --report <report.json>
```

Purpose: inspect frame-local targets before object fusion and report target
labels that already contradict 3D geometry. This keeps the next iteration
focused on source mask / target construction instead of downstream object
relabeling.

Full guarded v3 target diagnosis:

- targets: `9640`
- findings: `1277`
- finding labels:
  - `car=646`
  - `wall=321`
  - `railing=259`
  - `ground=50`
  - `ceiling=1`
- top bad windows:
  - `2700-2800 cam0`: `42` findings, score `1335`
  - `2200-2300 cam2`: `30` findings, score `1055`
  - `5800-5900 cam1`: `32` findings, score `860`

Interpretation:

- The dominant target-level issues are many small `car` / `railing` fragments,
  plus horizontal/flat `wall` targets.
- These are present before object fusion, so object-level voting cannot remove
  them reliably.
- The diagnostic report is now the preferred input for choosing bad windows and
  for testing mask/target construction changes.

## Top-Window Neighbor Surface Prior Test

Tested windows:

- `2700-2800`
- `2200-2300`
- `5800-5900`

Command shape:

```bash
python scripts/refine_priority_masks_with_geometry.py \
  --priority-suffix _priority_refined \
  --guarded-fine-surface-override \
  --fine-surface-neighbor-radius 2 \
  --fine-surface-neighbor-min-support 4 \
  --fine-surface-min-ratio 0.30 \
  --fine-surface-dominant-ratio 0.65
```

Results after rebuilding targets and applying target geometry refine:

| window | mask override | refined target findings | conclusion |
| --- | ---: | ---: | --- |
| `2700-2800` | `0` pixels | `64` | no added signal; same top cam0 score as full v3 |
| `2200-2300` | `0` pixels | `82` | no added signal; same top cam scores as full v3 |
| `5800-5900` | `591` railing->wall pixels | `41` | small improvement only |

Conclusion:

- Loosening neighbor surface overwrite is not the right main fix. For the worst
  windows, the projected surface prior often does not overlap the wrong fine
  mask enough to trigger a safe correction.
- The next route should modify target construction itself: use 2D mask
  connected components plus same-frame depth discontinuities and 3D connected
  components to split broad or fragmented targets before object fusion.
- Do not run a full-scene neighbor-prior sweep from this result; it would mostly
  add compute without addressing the dominant conflict modes.

## 2D Mask Component Split Probe

New optional target-construction flag:

```bash
python scripts/build_frame_targets_from_priority.py \
  --split-by-image-components \
  --image-component-min-pixels 32
```

This first splits sampled points by connected components in the 2D priority
mask, then applies the existing 3D voxel connectivity. The flag defaults to off
so the validated baseline remains unchanged.

Probe on the worst window `2700-2800` using the neighbor-r2 refined masks:

- without 2D component split, after target geometry refine:
  - findings: `64`
  - top `2700-2800 cam0`: `42` findings, score `1335`
- with 2D component split, after target geometry refine:
  - findings: `63`
  - top `2700-2800 cam0`: `42` findings, score `1335`

Interpretation:

- 2D connected-component pre-splitting is useful infrastructure, but it does not
  solve the current worst window. The dominant issue there is not disconnected
  same-label islands being merged; it is source priority masks producing many
  low-point `car` / `railing` fragments plus flat/horizontal wall fragments.
- The next meaningful target-construction change should use depth
  discontinuities and projected point support directly, not only 2D binary mask
  connectivity.

## Depth-Support Split Probe

New optional target-construction flags:

```bash
python scripts/build_frame_targets_from_priority.py \
  --split-by-image-components \
  --split-by-depth-support \
  --depth-support-pixel-radius 8 \
  --depth-support-max-gap 0.6
```

The split groups projected LiDAR samples by local image proximity and camera
depth continuity before the existing 3D voxel connectivity. This is designed to
work with sparse projected points, where generated depth-edge PNGs are often too
sparse to constrain masks directly.

Probe on `2700-2800` after target geometry refine:

| mode | findings | top cam0 findings | top cam0 score | result |
| --- | ---: | ---: | ---: | --- |
| neighbor-r2 only | `64` | `42` | `1335` | baseline for this probe |
| neighbor-r2 + 2D components | `63` | `42` | `1335` | effectively unchanged |
| depth support, radius `8`, gap `0.6` | `91` | `70` | `2065` | worse; more low-point fragments |
| depth support, radius `16`, gap `1.2` | `148` | `118` | `3725` | much worse |

Conclusion:

- Sparse projected-support splitting is not suitable as a production target
  split rule for this dataset. It fragments targets and increases low-point
  `car` / `railing` / flat-wall conflicts.
- Keep the flag as an experimental diagnostic tool, but do not enable it for the
  full-scene route.
- The next practical direction is fragment absorption / confirmation: low-point
  fine targets should be merged into neighboring trusted surfaces or sent to a
  stronger detector, instead of being preserved as independent objects.

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

## Target Fragment Absorption Probe

Date: 2026-06-19

New script:

```text
scripts/absorb_fine_fragments_into_surfaces.py
```

Purpose:

- pre-fusion JSONL-only target cleanup
- keep provenance via `raw_label`, absorption metadata, and optional demotion
- avoid rewriting target PLY or committing generated artifacts

Validation window:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/frame_targets_targetdiag_2700_2800_neighbor_r2_geometry_refined_rtx5070
```

This is the previous top bad window from target conflict QA.

Results:

| Variant | Absorbed/Demoted | Conflict findings | Top cam0 findings | Notes |
| --- | ---: | ---: | ---: | --- |
| baseline neighbor_r2 geometry refined | n/a | 64 | 42 | original target conflict baseline |
| loose surface absorb | 25 absorbed | 48 | 33 | improves metrics but can absorb vertical/real fine fragments too aggressively |
| strict surface absorb | 11 absorbed | 59 | 39 | safer but weak improvement |
| geometry-compatible absorb only | 2 absorbed | 62 | 40 | too conservative |
| geometry-compatible absorb + weak fine demote | 2 absorbed, 42 demoted | 20 | 9 | best current target-level cleanup |

Best probe output:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/frame_targets_targetdiag_2700_2800_neighbor_r2_absorbed_geomcompat_demote_rtx5070
```

Local compact reports:

```text
server_parking_priority_s10/frame_targets_targetdiag_2700_2800_neighbor_r2_absorbed_geomcompat_demote_rtx5070/absorb_report.json
server_parking_priority_s10/frame_targets_targetdiag_2700_2800_neighbor_r2_absorbed_geomcompat_demote_rtx5070/conflict_report.json
```

Interpretation:

- The main failure mode is not object fusion; it is low-point fine targets from
  source masks entering fusion as hard `car` / `railing` labels.
- Safe absorption into trusted surfaces helps only a little when made
  geometry-compatible.
- The larger win is to mark unconfirmed weak fine targets as `unknown` before
  object fusion. This prevents noisy fragments from polluting global object
  labels while preserving `raw_label` for later VLM/manual review.
- This should be tested next on a wider stride10 slice before promoting it into
  the full candidate route.

Recommended next command shape:

```bash
python scripts/absorb_fine_fragments_into_surfaces.py \
  --targets-jsonl frame_targets_refined.jsonl \
  --output-jsonl frame_targets_absorbed.jsonl \
  --report absorb_report.json \
  --demote-unabsorbed-weak-label unknown
```

Full stride10 target-level run:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/frame_targets_guarded_v3_full_s10_fine_surface_guard_absorbed_demote_rtx5070
```

Summary:

- input targets: `9,640`
- absorbed/demoted count in report: `907`
- weak fine targets demoted to `unknown`: `884`
- target conflict findings: `381`
- previous full v3 target conflict findings: `1,277`
- remaining conflict labels: `wall=321`, `ground=50`, `railing=5`,
  `car=4`, `ceiling=1`

Full stride10 viewer candidate:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/frame_object_viewer_guarded_v3_full_s10_absorbed_demote_rtx5070
```

Local viewer package:

```text
server_parking_priority_s10/frame_object_viewer_guarded_v3_full_s10_absorbed_demote_rtx5070/frame_object_points_stride10.ply
server_parking_priority_s10/frame_object_viewer_guarded_v3_full_s10_absorbed_demote_rtx5070/frame_objects_viewer.jsonl
server_parking_priority_s10/frame_object_viewer_guarded_v3_full_s10_absorbed_demote_rtx5070/frame_object_viewer_export_report.json
```

Viewer URL:

```text
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/frame_object_viewer_guarded_v3_full_s10_absorbed_demote_rtx5070/frame_object_points_stride10.ply&objects=/server_parking_priority_s10/frame_object_viewer_guarded_v3_full_s10_absorbed_demote_rtx5070/frame_objects_viewer.jsonl&mode=semantic&stride=1&pointSize=1.5
```

Viewer export label counts:

```text
ground=133041
ambiguous=139889
wall=500823
unknown=3666
grass=98466
car=17668
railing=7085
other=344
ceiling=2146
```

Current read:

- The target cleanup successfully suppresses most low-point `car` / `railing`
  hard-label pollution.
- The remaining dominant issue is now surface geometry splitting/relabeling,
  especially `wall_too_flat` and `ground_large_z_span`.
- Next optimization should target surface-plane partition and wall/ground/ceiling
  geometry rules, not VLM relabeling.

## Surface Target Repair Probe

Date: 2026-06-19

New script:

```text
scripts/repair_surface_target_labels.py
```

Purpose:

- JSONL-only repair after fine fragment cleanup and before object fusion
- fix only geometry-obvious surface contradictions
- keep target provenance with `raw_label`, `surface_repaired_from_label`, and
  `surface_repair_reason`

Input:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/frame_targets_guarded_v3_full_s10_fine_surface_guard_absorbed_demote_rtx5070/frame_targets_absorbed.jsonl
```

Output:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/frame_targets_guarded_v3_full_s10_absorbed_demote_surface_repair_rtx5070
```

Surface repair summary:

- repaired targets: `83`
- label flows: `wall->ground=47`, `wall->ceiling=28`, `ceiling->wall=8`
- target conflict findings after repair: `306`
- previous cleanup-only target conflict findings: `381`

Remaining conflict labels:

```text
wall=246
ground=50
railing=5
car=4
ceiling=1
```

Viewer candidate:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/frame_object_viewer_guarded_v3_full_s10_absorbed_demote_surface_repair_rtx5070
```

Local viewer URL:

```text
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/frame_object_viewer_guarded_v3_full_s10_absorbed_demote_surface_repair_rtx5070/frame_object_points_stride10.ply&objects=/server_parking_priority_s10/frame_object_viewer_guarded_v3_full_s10_absorbed_demote_surface_repair_rtx5070/frame_objects_viewer.jsonl&mode=semantic&stride=1&pointSize=1.5
```

Viewer export label counts:

```text
ground=136956
ambiguous=135974
wall=500737
unknown=3666
grass=98466
car=17668
ceiling=2232
railing=7085
other=344
```

Current read:

- This is a measurable but smaller win than fine fragment demotion.
- The easy horizontal-wall errors are partly fixed.
- The remaining surface issue likely needs true plane/component splitting for
  mixed large targets, not only whole-target relabeling.

## Surface Repair Threshold And Split Sweep

Date: 2026-06-19

Tested variants after fine fragment cleanup:

| Variant | Target conflict findings | Notes |
| --- | ---: | --- |
| cleanup only | 381 | after weak fine demotion |
| surface repair default | 306 | `wall->ground=47`, `wall->ceiling=28` |
| repair default + split | 305 | split alone did not help with conservative planarity |
| split with `surface_planarity=0.30` | 252 | better, but still leaves many horizontal wall fragments |
| repair `horizontal_surface_min_planarity=0.08` | 219 | cheaper and stronger than split alone |
| repair p008 + low-planarity split | 165 | best current geometry QA result |

Best current target output:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/frame_targets_guarded_v3_full_s10_absorbed_demote_surface_repair_p008_split_lowplanar_rtx5070
```

Commands:

```bash
python scripts/repair_surface_target_labels.py \
  --targets-jsonl frame_targets_absorbed.jsonl \
  --output-jsonl frame_targets_repaired.jsonl \
  --report surface_repair_report.json \
  --horizontal-surface-min-planarity 0.08

python scripts/refine_frame_targets_by_geometry.py \
  --targets-jsonl frame_targets_repaired.jsonl \
  --target-ply frame_targets_refined.ply \
  --output-dir frame_targets_surface_repair_p008_split_lowplanar \
  --split-horizontal-wall-by-height \
  --guard-linear-ground-artifacts \
  --guard-fine-surface-artifacts \
  --surface-planarity 0.30 \
  --wall-max-normal-z 0.75 \
  --ceiling-min-z 2.2 \
  --surface-height-split-threshold 0.8 \
  --surface-height-bin 0.45 \
  --surface-min-split-points 800 \
  --surface-split-min-points 100 \
  --surface-split-voxel 0.16 \
  --keep-residual
```

Best current metrics:

- surface repair p008 repaired targets: `177`
- split source targets: `247`
- output targets after split: `10,359`
- relabelled targets during split: `183`
- target conflict findings: `165`
- remaining conflicts: `wall=102`, `ground=53`, `railing=5`, `car=4`,
  `ceiling=1`

Best current viewer candidate:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/frame_object_viewer_guarded_v3_full_s10_absorbed_demote_surface_repair_p008_split_lowplanar_rtx5070
```

Local viewer URL:

```text
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/frame_object_viewer_guarded_v3_full_s10_absorbed_demote_surface_repair_p008_split_lowplanar_rtx5070/frame_object_points_stride10.ply&objects=/server_parking_priority_s10/frame_object_viewer_guarded_v3_full_s10_absorbed_demote_surface_repair_p008_split_lowplanar_rtx5070/frame_objects_viewer.jsonl&mode=semantic&stride=1&pointSize=1.5
```

Viewer export label counts:

```text
ground=145472
wall=456025
ceiling=7978
unknown=3669
grass=98473
ambiguous=166353
car=17667
railing=7050
other=428
```

Tradeoff:

- Geometry conflict findings are much lower than the previous candidate.
- Object fusion ambiguity increased (`ambiguous` points rose to `166,353`),
  likely because target splitting creates more fragmented object evidence.
- The next step should compare visual quality in the viewer before making this
  the default route. If visual quality is better, object fusion needs a
  same-surface post-consolidation pass to reduce ambiguity after splitting.

## Same-Surface Object Consolidation

Date: 2026-06-19

New helper:

```text
scripts/prepare_consolidated_viewer_objects.py
```

Purpose:

- `remap_ply_object_ids.py` rewrites PLY object ids to compact integers.
- This helper applies the same sidecar mapping to consolidated object JSONL so
  `tools/semantic_ply_viewer.html` can join PLY points with object metadata.

Input object route:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/frame_objects_guarded_v3_full_s10_absorbed_demote_surface_repair_p008_split_lowplanar_rtx5070/objects.jsonl
```

Balanced same-label consolidation:

```bash
python scripts/consolidate_same_label_surface_objects.py \
  --objects-jsonl objects.jsonl \
  --output-jsonl objects_consolidated.jsonl \
  --output-report consolidation_report.json \
  --output-mapping object_mapping.jsonl \
  --labels ground wall ceiling \
  --min-points 60 \
  --max-bbox-gap 0.35 \
  --max-centroid-distance 1.0 \
  --max-normal-angle 15 \
  --max-plane-distance 0.22 \
  --max-color-distance 70
```

Consolidation result:

- input objects: `3,279`
- output objects: `3,217`
- merged object reduction: `62`
- `surface_consolidated` objects: `31`
- remapped PLY vertices: `903,115`
- unmapped PLY vertices: `0`
- viewer object metadata missing mappings: `0`

Viewer candidate:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/frame_object_viewer_guarded_v3_full_s10_p008_split_lowplanar_surface_consolidated_balanced_rtx5070
```

Local viewer URL:

```text
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/frame_object_viewer_guarded_v3_full_s10_p008_split_lowplanar_surface_consolidated_balanced_rtx5070/frame_object_points_stride10.ply&objects=/server_parking_priority_s10/frame_object_viewer_guarded_v3_full_s10_p008_split_lowplanar_surface_consolidated_balanced_rtx5070/frame_objects_viewer.jsonl&mode=semantic&stride=1&pointSize=1.5
```

Current read:

- Balanced consolidation is conservative: it reduces object fragmentation without
  changing semantic labels.
- It does not reduce `ambiguous_object` count because cross-label ambiguous
  objects are intentionally left untouched.
- If visual QA confirms the p008 split candidate is better, the next useful pass
  is a targeted ambiguous-surface resolver using geometry and dominant label
  thresholds, not broad same-label merging.

## Ambiguous Surface Resolver

Date: 2026-06-19

New script:

```text
scripts/resolve_ambiguous_surface_objects.py
```

Purpose:

- resolve only surface-only `ambiguous_object` records
- accepted vote labels must be within `ground/floor/wall/ceiling`
- use dominant vote ratio plus object normal, z span, and height
- update PLY semantic/RGB by object id so viewer color matches JSON metadata

Input:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/frame_object_viewer_guarded_v3_full_s10_p008_split_lowplanar_surface_consolidated_balanced_rtx5070
```

Output:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/frame_object_viewer_guarded_v3_full_s10_p008_split_lowplanar_surface_consolidated_balanced_ambresolved_rtx5070
```

Result:

- ambiguous objects: `87 -> 29`
- resolved objects: `58`
- changed vertices: `126,131`
- unmapped vertices: `0`

Reason counts:

```text
strong_horizontal_high_to_ceiling=28
dominant_wall_geometry_ok=20
dominant_ground_high_horizontal_to_ceiling=3
dominant_ground=3
strong_horizontal_low_to_ground=3
dominant_ceiling_geometry_ok=1
kept_ambiguous=29
```

Object label counts after resolve:

```text
grass=1602
wall=691
unknown=532
car=196
railing=65
ceiling=54
ground=43
ambiguous=29
other=5
```

Viewer URL:

```text
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/frame_object_viewer_guarded_v3_full_s10_p008_split_lowplanar_surface_consolidated_balanced_ambresolved_rtx5070/frame_object_points_stride10.ply&objects=/server_parking_priority_s10/frame_object_viewer_guarded_v3_full_s10_p008_split_lowplanar_surface_consolidated_balanced_ambresolved_rtx5070/frame_objects_viewer.jsonl&mode=semantic&stride=1&pointSize=1.5
```

Current read:

- This is the best current object-level candidate by ambiguity count.
- It is intentionally conservative: 29 ambiguous objects remain unresolved
  because their votes/geometries are mixed.
- Next QA should visually compare this against the pre-resolve consolidated
  candidate, especially high-z horizontal areas now promoted to ceiling.

## Viewer Candidate QA Gate

Date: 2026-06-19

New script:

```text
scripts/qa_viewer_candidate.py
```

Purpose:

- validate only the final viewer artifacts: ASCII PLY + object JSONL
- check PLY header/data count, PLY object ids vs JSONL object ids, and object
  label vs PLY semantic consistency
- treat unresolved `ambiguous` objects as conservative `unknown` point semantics;
  this is a warning, not a hard failure
- emit JSON and Markdown summaries with Chinese label names for human review

Command used for the current best local copy:

```bash
python3 scripts/qa_viewer_candidate.py \
  --ply server_parking_priority_s10/frame_object_viewer_guarded_v3_full_s10_p008_split_lowplanar_surface_consolidated_balanced_ambresolved_rtx5070/frame_object_points_stride10.ply \
  --objects-jsonl server_parking_priority_s10/frame_object_viewer_guarded_v3_full_s10_p008_split_lowplanar_surface_consolidated_balanced_ambresolved_rtx5070/frame_objects_viewer.jsonl \
  --ambiguous-report server_parking_priority_s10/frame_object_viewer_guarded_v3_full_s10_p008_split_lowplanar_surface_consolidated_balanced_ambresolved_rtx5070/ambiguous_surface_resolve_report.json \
  --consolidation-report server_parking_priority_s10/frame_object_viewer_guarded_v3_full_s10_p008_split_lowplanar_surface_consolidated_balanced_ambresolved_rtx5070/consolidation_report.json \
  --output-json server_parking_priority_s10/frame_object_viewer_guarded_v3_full_s10_p008_split_lowplanar_surface_consolidated_balanced_ambresolved_rtx5070/viewer_candidate_qa.json \
  --output-md server_parking_priority_s10/frame_object_viewer_guarded_v3_full_s10_p008_split_lowplanar_surface_consolidated_balanced_ambresolved_rtx5070/viewer_candidate_qa.md \
  --top-n 20
```

Result:

- status: `ok`
- PLY vertices: `903,115`; header and data rows match
- PLY objects: `3,217`; JSON objects: `3,217`
- object/PLY semantic mismatch: `0`
- warnings:
  - remaining ambiguous objects: `29`
  - large fine objects exist and need visual inspection:
    - railing object `2828`: `13,071` points
    - car object `2997`: `10,416` points

Largest unresolved ambiguous objects:

```text
object 2998: 46,826 pts, frames 4870/4930/4940...
object 2839: 36,110 pts, frames 3550/3570/3580...
object 2850: 34,872 pts, frames 3720/3730/3770...
object 2754: 31,205 pts, frames 3020/3030/3040...
object 2925: 29,901 pts, frames 4500/4590/4600...
```

Interpretation:

- The current best viewer package is internally consistent.
- Remaining issues are semantic/geometry quality warnings, not export
  corruption.
- The next targeted QA should inspect the 29 ambiguous objects and the two
  large fine objects before changing source-mask or object-fusion thresholds.

## Reproducible Best-Route Runner

Date: 2026-06-19

New script:

```text
scripts/run_parking_frame_local_best_route.sh
```

Purpose:

- reproduce the current best frame-local post-processing route from the
  validated `fine_surface_guard` target artifacts
- keep the route out of older failed global-projection/VLM-relabel scripts
- default to dry-run; execute only with `RUN=1`
- refuse to overwrite existing output directories unless `OVERWRITE=1`

Default remote usage:

```bash
cd /home/zsh/Work/SCAN/new_route
RUN=0 OUT_SUFFIX=dryrun_check scripts/run_parking_frame_local_best_route.sh

RUN=1 OUT_SUFFIX=rtx5070_repro_$(date +%Y%m%d_%H%M%S) \
  scripts/run_parking_frame_local_best_route.sh
```

Pipeline stages encoded in the runner:

1. demote/absorb weak fine fragments
2. repair geometry-obvious surface labels with p008 planarity threshold
3. split low-planarity mixed surface targets
4. fuse frame-local targets to objects
5. export viewer PLY/JSONL
6. consolidate same-label surface objects
7. remap PLY object ids and JSONL metadata together
8. resolve surface-only ambiguous objects
9. run `qa_viewer_candidate.py` as the final consistency gate

## Parking Car Height Guard

Date: 2026-06-19

Evidence from the focused QA pack showed one large false `car` object:

```text
object 2997:
  label=car
  points=10,416
  centroid_z=10.3966
  bbox_z=9.525..11.621
  source crop: large high wall/panel surface, not a parking-lot car
```

The actual `car` object z distribution had a clear gap:

```text
car centroid z p90 ~= 0.99
next outliers: 2.71, then 9.17+
```

Change:

- `scripts/refine_frame_targets_by_geometry.py` now supports opt-in
  `--car-max-centroid-z`.
- Default is `None`, so non-parking/general datasets are unaffected.
- `scripts/run_parking_frame_local_best_route.sh` enables
  `--car-max-centroid-z 2.5` for this parking dataset.

Full rerun on `scan-rtx5070`:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/frame_object_viewer_best_p008_split_lowplanar_surface_consolidated_ambresolved_rtx5070_carz25_20260619_125444
```

Result:

- refine stage: `high_car_to_unknown=18`
- car objects: `196 -> 180`
- car points: `176,654 -> 149,376`
- unknown points: `36,681 -> 63,959`
- PLY/object QA: `ok`
- PLY vertices: `903,115`
- object records: `3,214`
- semantic mismatch: `0`
- remaining ambiguous objects: `29`
- remaining large fine object: only the stair/handrail object `2828`, visually
  valid from the QA crop

Local viewer URL:

```text
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/frame_object_viewer_best_p008_split_lowplanar_surface_consolidated_ambresolved_rtx5070_carz25_20260619_125444/frame_object_points_stride10.ply&objects=/server_parking_priority_s10/frame_object_viewer_best_p008_split_lowplanar_surface_consolidated_ambresolved_rtx5070_carz25_20260619_125444/frame_objects_viewer.jsonl&mode=semantic&stride=1&pointSize=1.5
```

Focused QA pack:

```text
server_parking_priority_s10/frame_object_qa_best_carz25_focus_rtx5070/frame_local_object_qa_contact.jpg
server_parking_priority_s10/frame_object_qa_best_carz25_focus_rtx5070/frame_local_object_qa_report.json
```

Current read:

- The car height guard fixes a concrete false-positive class without changing
  wall/ground/ceiling totals.
- It should replace the previous ambresolved viewer as the current best
  candidate.
- Remaining work is still the 29 ambiguous wall/ground/ceiling objects; evidence
  shows they are mostly multi-plane boundary/height-layer targets, not global
  calibration failures.

## Surface-Ambiguous Split Pass

Date: 2026-06-19

The remaining `29` ambiguous objects in the car-height-guard candidate were all
surface-only conflicts (`ground`/`wall`/`ceiling`). They were not useful as a
single semantic object in the viewer, but the source target labels were still
traceable. A narrow post-viewer pass was added:

```text
scripts/split_ambiguous_surface_viewer_objects.py
```

Scope:

- only touches objects with `semantic_label=ambiguous` or
  `status=ambiguous_object`
- only splits when all source target labels are surface labels:
  `ground`, `floor`, `wall`, `ceiling`
- preserves non-surface ambiguity for manual review
- rewrites both viewer JSONL and ASCII PLY `object`/`semantic` fields together
- records `parent_object_id`, split source labels, target ids, and a report

Run on `scan-rtx5070`:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/frame_object_viewer_best_p008_split_lowplanar_surface_consolidated_ambsplit_rtx5070_carz25_20260619_125444
```

Result:

- input objects: `3,214`
- output objects: `3,253`
- split objects: `29`
- split children: `68`
- kept ambiguous objects: `0`
- changed PLY vertices: `40,222`
- changed label point counts:
  - `wall`: `21,069`
  - `ground`: `11,782`
  - `ceiling`: `7,371`
- QA status: `ok`
- PLY vertices: `903,115`
- semantic mismatch: `0`
- warnings: only the large stair/handrail `railing` object remains for visual
  inspection

Viewer URL:

```text
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/frame_object_viewer_best_p008_split_lowplanar_surface_consolidated_ambsplit_rtx5070_carz25_20260619_125444/frame_object_points_stride10.ply&objects=/server_parking_priority_s10/frame_object_viewer_best_p008_split_lowplanar_surface_consolidated_ambsplit_rtx5070_carz25_20260619_125444/frame_objects_viewer.jsonl&mode=semantic&stride=1&pointSize=1.5
```

Current read:

- This is now the best viewer candidate for parking stride10.
- The split pass fixes a concrete post-fusion representation problem; it does
  not claim that the underlying mask/target labels are perfect.
- Remaining quality work should focus on source target quality and fine-object
  masks, not on global point reprojection or VLM relabel from already-corrupted
  evidence.

## Railing Local-Geometry Probe

Date: 2026-06-19

Problem:

- The best ambsplit candidate had no ambiguous objects, but QA still warned that
  large `car`/`railing` objects should be inspected.
- The remaining large `railing` cases were source-mask artifacts: the visual
  crop often contained real handrail/guardrail, but the mask swallowed adjacent
  stairs, panels, walls, or ground.
- Therefore a pure height guard would be too blunt; the useful operation is
  local 3D geometry splitting inside selected `railing` objects.

Tooling fix:

- `scripts/split_priority_objects_by_local_geometry.py` previously assumed PLY
  rows were contiguous by object id.
- Viewer PLY rows are not guaranteed to be object-contiguous, so the script
  duplicated object metadata and produced invalid object counts.
- The script now collects only selected split-candidate object rows and streams
  all non-candidates through once. A regression test covers non-contiguous PLY
  object runs.

Probe input:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/frame_object_viewer_best_p008_split_lowplanar_surface_consolidated_ambsplit_rtx5070_carz25_20260619_125444
```

Forced split candidates:

```text
2828, 2916, 2934, 3001, 3046, 3103, 3130, 3178, 3179, 3193
```

Output:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/frame_object_viewer_ambsplit_railing_localgeom_probe_v4_ground_rtx5070
```

QA result:

- status: `ok`
- PLY vertices: `903,115`
- input objects: `3,253`
- output objects: `3,266`
- split source objects: `10`
- large fine object warning: `none`
- semantic mismatch: `0`

Label movement:

| label | before points | after points | delta |
| --- | ---: | ---: | ---: |
| railing | `70,549` | `21,609` | `-48,940` |
| unknown | `63,959` | `67,972` | `+4,013` |
| wall | `5,206,131` | `5,206,425` | `+294` |
| ground | `1,637,998` | `1,638,491` | `+493` |

Viewer URL:

```text
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/frame_object_viewer_ambsplit_railing_localgeom_probe_v4_ground_rtx5070/frame_object_points_railing_localgeom.ply&objects=/server_parking_priority_s10/frame_object_viewer_ambsplit_railing_localgeom_probe_v4_ground_rtx5070/frame_object_points_railing_localgeom.jsonl&mode=semantic&stride=1&pointSize=1.5
```

Current read:

- This is a valid candidate result, but not yet promoted to the default best
  route because it aggressively reduces `railing`.
- The result should be visually checked against the previous ambsplit best
  candidate. If real railings are over-suppressed, tune local voxel thresholds
  or apply the split only to stricter high-risk objects.
- This probe confirms that local point geometry inside a bad fine mask is a
  useful next optimization axis.

## Automated Local-Geometry Candidate Gate

Date: 2026-06-19

New script:

```text
scripts/build_local_geometry_split_candidates.py
```

Purpose:

- convert viewer-object QA risk into a splitter-compatible conflicts JSONL
- avoid manual lists like the first railing probe
- remain pure JSON only; no image/OpenCV dependency

Default selection:

```text
labels=railing
min_points=2000
require_reasons=large_single_target_object,railing_not_linear,railing_extent_too_large
```

Running this gate on the current ambsplit best candidate selects exactly `10`
railing objects, matching the successful local-geometry probe:

```text
2828, 2916, 2934, 3001, 3046, 3103, 3130, 3178, 3179, 3193
```

The best-route runner now supports the probe as an opt-in final stage:

```bash
RUN=1 \
OUT_SUFFIX=rtx5070_localgeom_$(date +%Y%m%d_%H%M%S) \
SPLIT_RAILING_LOCAL_GEOMETRY=1 \
scripts/run_parking_frame_local_best_route.sh
```

Default remains:

```text
SPLIT_RAILING_LOCAL_GEOMETRY=0
```

Reason:

- the local-geometry result is internally valid and removes the large-fine QA
  warning, but it reduces `railing` points from `70,549` to `21,609`
- this is likely correct for source-mask spillover, but it still needs visual
  acceptance before becoming the default best route

## Local-Geometry Runner Reproduction

Date: 2026-06-19

The opt-in local-geometry stage was run through the full best-route runner on
`scan-rtx5070`:

```bash
RUN=1 \
OVERWRITE=0 \
OUT_SUFFIX=localgeom_full_20260619_131820 \
SPLIT_RAILING_LOCAL_GEOMETRY=1 \
scripts/run_parking_frame_local_best_route.sh
```

Final output:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/frame_object_viewer_best_p008_split_lowplanar_surface_consolidated_localgeom_localgeom_full_20260619_131820
```

Local mirror:

```text
server_parking_priority_s10/frame_object_viewer_best_p008_split_lowplanar_surface_consolidated_localgeom_localgeom_full_20260619_131820
```

QA:

- status: `ok`
- warnings: `none`
- errors: `none`
- PLY vertices: `903,115`
- objects: `3,266`
- candidate objects selected for local geometry split: `10`
- split source objects: `10`

Comparison to the previous ambsplit candidate:

| metric | ambsplit | local-geometry runner |
| --- | ---: | ---: |
| objects | `3,253` | `3,266` |
| warnings | `large car/railing objects exist` | `none` |
| railing points | `70,549` | `21,609` |
| unknown points | `63,959` | `67,972` |
| wall points | `5,206,131` | `5,206,425` |
| ground points | `1,637,998` | `1,638,491` |

Viewer URL:

```text
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/frame_object_viewer_best_p008_split_lowplanar_surface_consolidated_localgeom_localgeom_full_20260619_131820/frame_object_points_stride10.ply&objects=/server_parking_priority_s10/frame_object_viewer_best_p008_split_lowplanar_surface_consolidated_localgeom_localgeom_full_20260619_131820/frame_objects_viewer.jsonl&mode=semantic&stride=1&pointSize=1.5
```

Current read:

- This validates the opt-in runner path end-to-end.
- The local-geometry stage remains opt-in until visual review confirms that the
  reduced `railing` coverage is preferable to the source-mask spillover.
- If accepted, the next code change should simply flip
  `SPLIT_RAILING_LOCAL_GEOMETRY` to default `1` and rerun once with a clean
  suffix that does not repeat `localgeom`.

## Viewer Candidate QA Comparison Tool

Date: 2026-06-19

New script:

```text
scripts/compare_viewer_candidate_qa.py
```

Purpose:

- compare two or more `qa_viewer_candidate.py` JSON reports
- report label point/object deltas, warning/error changes, status deltas, and
  large fine-object risk changes
- make route comparison reproducible instead of relying on manual arithmetic

Command used locally and on `scan-rtx5070`:

```bash
python scripts/compare_viewer_candidate_qa.py \
  --report ambsplit=.../frame_object_viewer_best_p008_split_lowplanar_surface_consolidated_ambsplit_rtx5070_carz25_20260619_125444/viewer_candidate_qa.json \
  --report localgeom=.../frame_object_viewer_best_p008_split_lowplanar_surface_consolidated_localgeom_localgeom_full_20260619_131820/viewer_candidate_qa.json \
  --output-json .../viewer_candidate_comparisons/ambsplit_vs_localgeom_20260619.json \
  --output-md .../viewer_candidate_comparisons/ambsplit_vs_localgeom_20260619.md
```

Observed comparison:

| metric | ambsplit | local-geometry | delta |
| --- | ---: | ---: | ---: |
| vertices | `903,115` | `903,115` | `0` |
| objects | `3,253` | `3,266` | `+13` |
| warnings | `1` | `0` | `-1` |
| large fine objects | `1` | `0` | `-1` |
| large fine points | `13,071` | `0` | `-13,071` |
| railing points | `70,549` | `21,609` | `-48,940` |
| unknown points | `63,959` | `67,972` | `+4,013` |

Interpretation:

- local-geometry removes the measurable large railing/car swallowing risk
- it is conservative: a large share of previously labeled `railing` is no
  longer accepted as railing
- this strengthens the case for local-geometry as a guard stage, but visual QA
  must decide whether the `railing` recall loss is acceptable

## Local-Geometry Candidate Evidence Pack

Date: 2026-06-19

Evidence pack generated on `scan-rtx5070` and mirrored locally:

```text
server_parking_priority_s10/frame_object_qa_localgeom_candidate_ids_20260619/
```

Forced object IDs:

```text
2828, 2916, 2934, 3001, 3046, 3103, 3130, 3178, 3179, 3193
```

Summary:

- candidates: `10`
- evidence crop images: `23`
- candidate labels: all `railing`
- risk reasons:
  - `railing_not_linear`: `10`
  - `large_single_target_object`: `3`
  - `railing_extent_too_large`: `1`

Visual read from the contact sheet:

- multiple source masks swallow stair treads, wall/door panels, corrugated
  panels, floor patches, or loose background material together with true rail
  structure
- this explains why local-geometry reduces `railing` aggressively: it is
  removing mask spillover rather than only removing random true rail points
- the upstream priority/fine mask is currently the limiting stage for railing
  recall and precision

Next engineering direction:

- keep local-geometry as a guard against fine-mask spillover
- improve the source fine-target mask stage for railing/handrail before
  promoting more railing points
- avoid object-level relabel from only image crops when the source crop already
  contains mixed wall/floor/stair evidence

## Default Route Update

Date: 2026-06-19

`scripts/run_parking_frame_local_best_route.sh` now enables local-geometry
railing split by default:

```text
SPLIT_RAILING_LOCAL_GEOMETRY=1
```

Reason:

- the dominant current failure mode is wall/ground/stair pixels being swallowed
  by coarse `railing` priority masks
- local-geometry is the only currently validated guard that removes this
  spillover before final viewer export
- if a future source mask improves railing recall/precision, this guard can
  remain as a safety gate or be made less strict

Remote dry-run on `scan-rtx5070` confirmed the default runner now includes:

```text
build_local_geometry_split_candidates.py
split_priority_objects_by_local_geometry.py
qa_viewer_candidate.py
```

Final default viewer directory pattern:

```text
frame_object_viewer_best_p008_split_lowplanar_surface_consolidated_localgeom_${OUT_SUFFIX}
```
