# RTX 5070Ti Parking Baseline Run - 2026-06-19

## Summary

- Host: `scan-rtx5070`
- Remote repo: `/home/zsh/Work/SCAN/new_route`
- Remote dataset: `/home/zsh/Work/SCAN/datasets/MT20260616-175807`
- Remote workdir: `/home/zsh/Work/SCAN/work_MT20260616-175807`
- Local review copy: `server_parking_priority_s10/frame_object_viewer_priority_full_s10_v5_geometry_ceiling_rtx5070`

This run restores the current validated frame-local route on the RTX 5070Ti host.
The old global object back-projection route is still invalid and was not used.

## Current Gate: Video/LiDAR Sync

The parking dataset must not proceed to semantic production until the
image/LiDAR sync map is accepted.  The reason is now explicit: `img_pos.txt` is
not a uniform frame sequence.

Measured `img_pos.timestamp` statistics:

- rows: `6181`
- time span: `835.3239300251007s`
- delta min / p50 / mean / p90 / max:
  `0.09968 / 0.10002 / 0.13517 / 0.20003 / 2.00031`
- deltas `>0.15s`: `1507`
- deltas `>0.3s`: `231`
- deltas `<0.05s`: `0`

Interpretation:

- The images used for derived datasets should be extracted by calibrated video
  time/index, not by assuming `section_id == video_idx`.
- The sync model needs to estimate at least:
  - effective video fps / time scale from `img_pos.timestamp` to video frames
  - global video offset
  - possible per-camera offset
  - exposure phase relative to a LiDAR section, i.e. start/middle/end of the
    local nonuniform scan interval.  This must be modeled as
    `timestamp + phase_fraction * local_dt`; a pure constant offset is absorbed
    by the absolute intercept and cannot distinguish start/middle/end.
- Uniform frame-id stepping can still be used as a diagnostic baseline, but it is
  not a valid production assumption for this dataset.

Implemented sync tooling:

- `scripts/solve_sync_path_from_candidates.py`
  - supports `--time-mode frame-id|timestamp`
  - supports `--img-pos-file` and `--video-fps`
  - supports `--timestamp-phase-fraction` where `0=start`, `0.5=middle`, and
    `1=end` of the local `img_pos` interval
  - supports an optional absolute timestamp prior:
    `video_idx ~= (timestamp - t0) * fps + intercept`
  - accepted manual anchors remain hard constraints
- `scripts/sweep_sync_timestamp_fps.py`
  - sweeps effective fps values in timestamp mode
  - can sweep timestamp phase fractions with `--phase-values 0,0.5,1`
  - can also sweep per-camera intercepts when the absolute prior is enabled
  - writes the best path plus a sweep report
- `scripts/summarize_sync_option_sources.py`
  - compares direct, independent-best, and smooth-path candidate sources

Current automatic results:

- frame-id smooth + sky penalty:
  - status: `rejected`
  - temporally stable but score loss remains too high
- timestamp smooth at `10fps` + sky penalty:
  - status: `rejected`
  - max step deviation by camera reaches about `0.45`
- timestamp phase sweep on 5070Ti with sky-penalty fullprobe candidates:
  - output: `server_parking_priority_s10/sync_phase_sweep_sky_penalty_20260619/`
  - swept `fps=5.5..7.0` and `phase=0,0.5,1`
  - best automatic setting: `fps=6.0`, `phase=1.0`, cam intercepts
    `cam0=700`, `cam1=600`, `cam2=800`
  - status remains `rejected`; use this as a diagnostic prior only, not as a
    production sync map without accepted manual anchors.
  - `scripts/run_rtx5070_sync_anchor_solver.sh` now defaults
    `SOLVER_TIMESTAMP_PHASE_FRACTION=1.0` so the constrained anchor run starts
    from this best current phase prior. Override the variable if manual anchors
    contradict it.
- timestamp fps sweep:
  - best effective fps: `10.25`
  - status: `rejected`
  - mean score loss: about `0.156`
  - max step deviation: about `0.461`
- timestamp fps/intercept sweep with absolute prior:
  - best effective fps: `6.0`
  - per-camera intercepts: `cam0=700`, `cam1=600`, `cam2=800`
  - status: `rejected`
  - mean score loss: about `0.172`
  - max step deviation: about `0.139`
  - mean absolute prior error by camera: about `18-25` video frames

Visual review of the timestamp/fps sweep contact sheet shows it captures the
non-uniform sampling hypothesis, but still has suspicious early mappings such
as `video_idx=0/100` around `frame_id=1000`.  Therefore it is useful as an
anchor review candidate, not as an automatic truth source.

The absolute-prior sweep fixes that specific failure mode by keeping early
mappings near plausible video indices, but it still has enough low-score
frame/camera pairs that it must remain a manual-anchor candidate rather than a
production sync map.

Current review pages:

```text
http://127.0.0.1:8765/server_parking_priority_s10/sync_anchor_review_priority_sky_penalty_smooth_preselect_20260619/anchor_review_priority.html
http://127.0.0.1:8765/server_parking_priority_s10/sync_anchor_review_priority_sky_penalty_timestamp_fps_sweep_20260619/anchor_review_priority.html
http://127.0.0.1:8765/server_parking_priority_s10/sync_anchor_review_priority_sky_penalty_timestamp_absprior_dot3_20260619/anchor_review_priority.html
```

The priority review page now shows a live export-readiness bar:

- accepted anchor count per camera, default target `2` per camera
- total accepted anchors
- accepted rows that are missing a selected option
- a confirmation prompt if exporting before coverage is ready

The current dot3 priority page has been regenerated with this coverage UI.

Required next action: manually accept enough anchor rows from a review page,
then stage and solve with:

```bash
python3 scripts/stage_accepted_sync_anchors.py --force --run-solver
```

This discovers the newest `/Users/skkac/Downloads/accepted_sync_anchors*.jsonl`,
validates it, stages it to the current review directory, and runs
`scripts/run_rtx5070_sync_anchor_solver.sh`.  The readiness gate must pass
before extraction, segmentation, target building, or semantic fusion uses the
new sync map.

`scripts/run_rtx5070_sync_anchor_solver.sh` now defaults to the latest
abs-prior review pack:

- `REVIEW_NAME=sync_anchor_review_priority_sky_penalty_timestamp_absprior_dot3_20260619`
- `RUN_NAME=sync_anchor_constrained_timestamp_absprior_dot3_20260619`
- solver mode: `timestamp`
- solver fps: `6.0`
- absolute prior: enabled with tolerance `200`
- intercept source: `anchors`, estimated independently per camera from accepted
  manual anchors
- review dot size: `3`

Use `DRY_RUN=1 scripts/run_rtx5070_sync_anchor_solver.sh` to inspect the exact
remote command even before anchors have been exported.  Real execution still
requires `accepted_sync_anchors.jsonl`.

Before rsyncing anchors or starting the remote solver, the launcher now runs:

```bash
python3 scripts/validate_sync_anchors.py \
  --anchors-jsonl accepted_sync_anchors.jsonl \
  --img-pos-file ../MT20260616-175807/image/img_pos.txt \
  --timestamp-phase-fraction 1.0 \
  --expected-fps 6.0
```

The validator catches insufficient per-camera accepted anchors, accepted rows
without a selected video frame, and non-monotonic video indices.  It writes
`accepted_sync_anchor_validation.json` next to the constrained sync output so
manual-anchor quality is traceable before the expensive remote solve.

After the constrained sync run passes, use the gated production launcher:

```bash
scripts/run_rtx5070_sync_gated_parking_dataset.sh
RUN=1 scripts/run_rtx5070_sync_gated_parking_dataset.sh
```

For a single read-only status snapshot of the whole sync-gated route:

```bash
python3 scripts/summarize_sync_gated_parking_status.py
```

It writes:

- `server_parking_priority_s10/sync_gated_parking_status.json`
- `server_parking_priority_s10/sync_gated_parking_status.md`

The current status is `waiting_for_manual_anchors`, so the next command remains
`python3 scripts/stage_accepted_sync_anchors.py --force --run-solver` after
exporting accepted anchors from the review page.

This launcher refuses to run unless:

- `sync_frame_map_readiness.exit_code` is `0`
- `sync_frame_map_readiness.json` exists
- `expanded_frame_map.jsonl` exists

When `RUN=1`, it first runs
`scripts/check_rtx5070_parking_runtime.py` with
`--no-default-required-files` against the exact sync-gated inputs:

- `.lx`
- `video_cam0/1/2.mkv`
- `img_pos.txt`
- `cam_in_ex.txt`
- `expanded_frame_map.jsonl`
- `sync_frame_map_readiness.json`
- `sync_frame_map_readiness.exit_code`

The preflight also checks RTX 5070Ti free VRAM, torch CUDA availability, and
the Python modules used by extraction and priority segmentation.  It writes a
local JSON report such as
`server_parking_priority_s10/sync_absprior_s10_preflight.json` before starting
the remote tmux job.  Set `RUN_PREFLIGHT=0` only for diagnostics.

It then uses `expanded_frame_map.jsonl` with `--require-frame-map` for
sync-correct frame extraction and optional colorization.  By default it prepares
the reusable synchronized frames and Mapillary priority masks only:

- `frames_jpeg_sync_absprior_s10`
- `priority_surface_mapillary_sync_absprior_s10`

It does **not** default to the older safe semantic-prior object route, because
that route can depend on prior PLY evidence generated before this sync gate.
Set `DO_SAFE_ROUTE=1` only after the semantic-prior input is known to be
compatible with the accepted sync map.

## Environment

- Released Gemma/llama-server before running: PID `1723`, about `8708MiB` VRAM.
- Post-release GPU baseline: about `1.3GB` VRAM used by desktop processes.
- Python env: `/home/zsh/Work/SCAN/.venvs/scan-semantic`
- CUDA stack: `torch 2.11.0+cu130`, CUDA available.
- Added dependencies: `transformers==4.47.1`, `scipy`, `scikit-learn`, `accelerate`, `safetensors`, `tqdm`.
- Clash/Mihomo proxy used for downloads: `127.0.0.1:7897`.
- Basic sync-gated input preflight on `2026-06-19` passed:
  - GPU: RTX 5070 Ti, `15015MiB` free of `16303MiB`
  - torch CUDA available, CUDA `13.0`
  - dataset files present: `.lx`, three videos, `img_pos.txt`, `cam_in_ex.txt`
  - warning only: remote git dirty count was `48` from working-copy/generated
    files; this is not a runtime blocker but should be cleaned before treating
    the remote checkout as a release artifact.

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


## LiDAR/Video Sync Recheck

The latest visual evidence shows the parking dataset still has image/point-cloud
misalignment. Treat this as a synchronization gate failure, not as a semantic
model failure. The validated geometry chain can project same-frame `.lx` points,
but direct `frame_id -> video_idx` is not reliable enough for semantic
production on this dataset.

Timing audit facts:

- `.lx` sections: `6181`; `img_pos` rows: `6181`; each camera video: `6181` frames at `10 fps`.
- `.lx` section pose and `img_pos` pose match almost exactly, so the LiDAR/pose
  stream itself is consistent.
- `img_pos.timestamp` spans about `835.32s`, while video frame count at `10 fps`
  spans `618.1s`; there are many irregular timestamp gaps.
- Candidate projection scoring shows direct video index is often not the best
  visual match. Smooth-path solving without manual anchors is still too lossy.

Current gate:

```text
Do not continue semantic production until manual sync anchors are selected and
the constrained sync solver produces a visually accepted path.
```

New review artifact format:

- `manual_anchor_manifest.jsonl`: source manifest with candidate options.
- `manual_anchor_review_sheet.jpg`: quick contact sheet.
- `panels/*.jpg`: one rendered panel per candidate option.
- `manual_anchor_review.html`: static reviewer that selects candidate frames and
  exports `accepted_sync_anchors.jsonl` for
  `solve_sync_path_from_candidates.py --anchors-jsonl`.

Generated v2 review pack:

```text
server_parking_priority_s10/sync_anchor_review_small_20260619_v2/manual_anchor_review.html
server_parking_priority_s10/sync_anchor_review_small_20260619_v2/manual_anchor_manifest.jsonl
server_parking_priority_s10/sync_anchor_review_small_20260619_v2/manual_anchor_review_sheet.jpg
server_parking_priority_s10/sync_anchor_review_small_20260619_v2/panels/
```

Generation report: `27` probes and `152` option panels.

After exporting accepted anchors from the HTML page, run the constrained solver
from the local repo:

```bash
cd /Users/skkac/Work/SCAN/new_route
scripts/run_rtx5070_sync_anchor_solver.sh
```

Default expected anchor file:

```text
server_parking_priority_s10/sync_anchor_review_small_20260619_v2/accepted_sync_anchors.jsonl
```

The launcher syncs anchors to `scan-rtx5070`, runs
`solve_sync_path_from_candidates.py --anchors-jsonl`, rebuilds a constrained
review pack, and pulls outputs to:

```text
server_parking_priority_s10/sync_anchor_constrained_from_review_v2/
```

Only after that constrained path is visually accepted should priority masks,
frame-local targets, or object fusion be rerun.

Automatic high-confidence anchor diagnostic from the v2 manifest:

- probes: `27`
- candidates with `score >= 0.55` and top-vs-second margin `>= 0.08`: `3`
- selected offsets among those strong candidates ranged from `-1300` to `+800`
- per-frame top offsets are not temporally smooth

Interpretation: score-only automatic anchors are not reliable enough for this
dataset. They can help prioritize human review, but must not be used as
production synchronization truth.

Video/time-model audit:

- `ffprobe` shows all three MKV files are constant `10fps`, PTS starts at `0`,
  and container duration is `618.1s`; there is no hidden variable-frame-rate PTS
  stream to recover the missing timing.
- `analyze_sync_time_models.py` compared simple models against visual sync
  candidates. None is reliable:
  - direct `frame_id`: exact candidate always present, but median rank `12`,
    mean score loss `0.246`, max score loss `0.550`
  - `img_pos.timestamp * 10fps`: exact candidate ratio `0.111`, median nearest
    distance `19` frames
  - timestamp compressed to video span: exact candidate ratio `0.111`, median
    nearest distance `15` frames
  - `cam_info` values: exact candidate ratio `0`; median nearest distance about
    `1990` frames, so these fields are not direct video frame indices
  - affine fit to independent best: no exact candidates and median nearest
    distance `30` frames

Report artifacts:

```text
server_parking_priority_s10/sync_time_model_analysis_20260619/sync_time_model_report.json
server_parking_priority_s10/sync_time_model_analysis_20260619/sync_time_model_details.jsonl
```

Conclusion: current evidence rules out a cheap deterministic time model. The
next sync step still needs human anchors or a stronger visual sequence matching
method.

LX coordinate-frame audit:

- `audit_lx_coordinate_frame.py` confirms the parking `.lx` sections behave like
  world-coordinate points, not local LiDAR-coordinate points.
- raw section centroid vs pose position correlations: `x=0.9973`, `y=0.9977`,
  `z=0.9955`
- after world-to-lidar transform, median centroid span across sampled frames is
  only `[2.11, 2.32, 1.35]m`

Report artifact:

```text
server_parking_priority_s10/lx_coordinate_frame_audit_20260619/lx_coordinate_frame_report.json
```

Conclusion: the projection chain's world-coordinate assumption is valid for
this dataset. The remaining image/point-cloud mismatch should be treated as
sync/path selection or visual scoring, not as `.lx` coordinate-frame misuse.

Full-range edge-score probe:

- Ran absolute `video_idx=0..6180 step=100` search on 9 sampled sections x 3
  cameras.
- Raw edge scoring is dominated by frame-level image-edge bias: cam0/cam1 select
  `video_idx=300` for almost every probe, which is physically impossible as a
  synchronization path.
- Smooth solver can force monotonic paths, but only with high score loss: mean
  score loss `0.304` for cam0, `0.249` for cam1, `0.228` for cam2 under the
  relaxed full-range run.
- Projected sparse depth-edge scoring was tested and found ineffective: depth
  discontinuity samples are often `0` because Mid360 ring projections are too
  sparse for pixel-adjacent depth gradients.
- Projected silhouette scoring also degenerates to all projected points on this
  sparse data, so it does not remove the frame-prior bias.

Report artifacts:

```text
server_parking_priority_s10/sync_calibration_abs_fullrange_step100_20260619/sync_calibration_report.json
server_parking_priority_s10/sync_smooth_abs_fullrange_step100_20260619/sync_smooth_path_report.json
server_parking_priority_s10/sync_silhouette_probe_fields_20260619/sync_candidates.jsonl
```

Conclusion: the current edge-distance visual score is suitable for producing
human review candidates, but not sufficient as an automatic sequence matcher.
The next automatic route should use stronger image descriptors or manual anchors,
not more edge-score parameter sweeps.

Video frame access audit:

- All three MKV files have `6181` readable frames according to `ffprobe
  -count_frames`.
- `audit_video_frame_access.py` compared OpenCV random frame seek against ffmpeg
  exact `select=eq(n,idx)` on 27 sampled camera/frame pairs.
- OpenCV vs exact ffmpeg: gray correlation min `0.9977`, mean `0.9992`; mean
  absolute difference max `3.19`, mean `1.80`. This is decoder/encoding-level
  difference, not a frame-index mismatch.
- ffmpeg timestamp seek is also close to exact ffmpeg: gray correlation mean
  `0.9984`.

Report artifact:

```text
server_parking_priority_s10/video_frame_access_audit_20260619/video_frame_access_report.json
```

Conclusion: OpenCV random seek is not the root cause of the current sync
misalignment. It remains acceptable for candidate generation speed; ffmpeg exact
frame extraction is too slow for full candidate scoring and should remain an
audit/reference path.

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

## Clean Default Local-Geometry Run

Date: 2026-06-19

Command:

```bash
RUN=1 \
OVERWRITE=0 \
OUT_SUFFIX=rtx5070_default_localgeom_20260619_132858 \
scripts/run_parking_frame_local_best_route.sh
```

Remote output:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/frame_object_viewer_best_p008_split_lowplanar_surface_consolidated_localgeom_rtx5070_default_localgeom_20260619_132858
```

Local mirror:

```text
server_parking_priority_s10/frame_object_viewer_best_p008_split_lowplanar_surface_consolidated_localgeom_rtx5070_default_localgeom_20260619_132858
```

QA:

- status: `ok`
- warnings: `none`
- errors: `none`
- PLY vertices: `903,115`
- objects: `3,266`
- ambiguous objects: `0`
- large fine objects: `0`

Point labels:

| label | points |
| --- | ---: |
| wall | `5,206,425` |
| ground | `1,638,491` |
| grass | `984,548` |
| ceiling | `914,287` |
| car | `149,376` |
| unknown | `67,972` |
| railing | `21,609` |
| other | `4,295` |

Viewer URL:

```text
http://127.0.0.1:8765/tools/semantic_ply_viewer.html?file=/server_parking_priority_s10/frame_object_viewer_best_p008_split_lowplanar_surface_consolidated_localgeom_rtx5070_default_localgeom_20260619_132858/frame_object_points_stride10.ply&objects=/server_parking_priority_s10/frame_object_viewer_best_p008_split_lowplanar_surface_consolidated_localgeom_rtx5070_default_localgeom_20260619_132858/frame_objects_viewer.jsonl&mode=semantic&stride=1&pointSize=1.5
```

Current status:

- this is now the cleanest reproducible parking-lot viewer candidate
- it prioritizes precision over railing recall
- next optimization target is source fine-mask quality for `railing/handrail`
  and `car`, not global object relabeling

## Current-Best Geometry-Prior Source-Mask Probe

Date: 2026-06-19

Test window:

```text
3400..3500 stride=10, cams=0/1/2
```

Reason:

- local-geometry evidence showed typical `railing` source-mask spillover around
  this window, especially object `2828`
- this is an appropriate small probe before any full-scene source-mask change

Inputs:

- priority masks:
  `/home/zsh/Work/SCAN/work_MT20260616-175807/priority_surface_mapillary_s10_rtx5070`
- current best semantic prior PLY:
  `/home/zsh/Work/SCAN/work_MT20260616-175807/frame_object_viewer_best_p008_split_lowplanar_surface_consolidated_localgeom_rtx5070_default_localgeom_20260619_132858/frame_object_points_stride10.ply`
- geometry guidance:
  `/home/zsh/Work/SCAN/work_MT20260616-175807/geometry_guidance_currentbest_3400_3500`
- refined priority masks:
  `/home/zsh/Work/SCAN/work_MT20260616-175807/geometry_refine_currentbest_3400_3500_guarded`

Mask-refine result:

- images: `33/33 ok`
- semantic-prior voxels: `198,800`
- `residual->wall`: `42,754` pixels
- `railing->wall`: `9,921` pixels
- depth-edge cut: `0`

Target-level gate:

| metric | baseline priority | geometry-refined priority | delta |
| --- | ---: | ---: | ---: |
| targets | `83` | `98` | `+15` |
| findings | `8` | `13` | `+5` |
| finding points | `31,994` | `33,040` | `+1,046` |
| top-window score | `260` | `405` | `+145` |
| wall findings | `2` | `4` | `+2` |
| railing findings | `6` | `9` | `+3` |

Local comparison report:

```text
server_parking_priority_s10/frame_target_geometry_conflict_comparisons/baseline_vs_geomrefined_3400_3500.md
```

Decision:

- reject this geometry-prior source-mask refine for the current mainline
- it fixes some 2D pixels but worsens target-level geometry consistency
- the projected prior is too sparse/aliased in this window to safely guide
  fine-mask overwrite

New gate:

```text
scripts/compare_frame_target_geometry_conflicts.py
```

Use this gate for future source-mask experiments. A candidate should not be
promoted if target findings or top-window score increase on the probe windows,
even if image overlays look cleaner.

## GroundingDINO Frame-Level Fine-Object Probe

Date: 2026-06-19

New script:

```text
scripts/run_groundingdino_frame_probe.py
```

Purpose:

- test whether a text-conditioned detector can provide narrower source
  candidates for `railing/handrail` and `car`
- produce annotated frame contact sheets and box-area statistics
- avoid modifying point-cloud artifacts until the detector passes a small
  visual/quantitative gate

Environment:

- host: `scan-rtx5070`
- model: `IDEA-Research/grounding-dino-tiny`
- implementation: Transformers `AutoModelForZeroShotObjectDetection`
- proxy: remote Clash/Mihomo listens on `127.0.0.1:7897`
- note: `127.0.0.1:7890` is not valid on this host

Window:

```text
3400..3500 stride=10, cams=0/1/2
```

Loose prompt/threshold:

```text
box_threshold=0.22
text_threshold=0.18
```

Result:

- images: `33`
- detections: `railing=143`, `car=37`, `unknown=8`
- visual read: many large false-positive boxes over walls, doors, stairs, and
  floors

Strict prompt/threshold:

```text
box_threshold=0.35
text_threshold=0.25
railing prompts=handrail, stair railing, metal handrail, guardrail
car prompts=car, vehicle
```

Result:

- images: `33`
- detections: `railing=23`, `unknown=1`
- `railing` large boxes (`>=12%` image area): `16/23`
- mean `railing` box area ratio: `0.298`
- visual read: still contains broad boxes on wall/stair/floor regions

Local outputs:

```text
server_parking_priority_s10/groundingdino_frame_probe_3400_3500_tiny_fullwindow/
server_parking_priority_s10/groundingdino_frame_probe_3400_3500_tiny_strict_v3/
```

Decision:

- do not promote GroundingDINO-tiny frame boxes into the source mask pipeline
  yet
- it is useful as a diagnostic/evidence generator, but not precise enough for
  `railing` source candidates on this dataset
- next detector attempt should use a stronger open-vocabulary detector or a
  segmentation-specific model, and must pass the same box-area/contact-sheet
  gate before SAM/mask projection

## GroundingDINO Base Probe And Fine-Mask Decision

Date: 2026-06-19

Follow-up model:

```text
IDEA-Research/grounding-dino-base
```

Same window and strict prompts as the tiny probe:

```text
3400..3500 stride=10, cams=0/1/2
box_threshold=0.35
text_threshold=0.25
railing prompts=handrail, stair railing, metal handrail, guardrail
car prompts=car, vehicle
```

Comparison tool:

```text
scripts/compare_groundingdino_frame_probes.py
```

Comparison output:

```text
server_parking_priority_s10/groundingdino_frame_probe_comparisons/tiny_vs_base_strict_3400_3500.md
```

Quantitative result:

| metric | tiny strict | base strict |
| --- | ---: | ---: |
| images | `33` | `33` |
| railing detections | `23` | `19` |
| unknown detections | `1` | `0` |
| railing large boxes | `16` | `9` |
| railing large-box rate | `69.6%` | `47.4%` |
| mean railing box area ratio | `29.8%` | `16.3%` |
| max railing box area ratio | `131.0%` | `60.4%` |

Visual read:

- `grounding-dino-base` is clearly better than `tiny`; boxes are fewer and less
  often scene-scale.
- It still frequently includes stairs, wall panels, floors, or door regions
  together with true handrails.
- It returns boxes, not masks. Sending these broad boxes directly into SAM would
  likely reproduce the same mixed-mask failure already seen in priority masks.

Decision:

- Do not promote GroundingDINO-base into the main mask path.
- Keep GroundingDINO as a candidate/evidence generator only.
- The highest-quality fine-mask method already validated remains:

```text
undistorted image -> skymask -> SAM2 loop / coverage completion -> mask overlay QA -> same-frame point projection
```

Historical best combo name on the previous dataset:

```text
sam2_prompt_v3_sky_label_merge_completion
```

Current route implication:

- SAM2 remains the fine-mask core.
- DINO/GroundingDINO should only propose candidate regions when they pass a
  small-window detector gate.
- Point-cloud local geometry remains the required guard for mixed fine masks,
  especially `railing/handrail` masks that swallow stairs, walls, or floors.

## Fine-Mask Evaluation Manifest

Date: 2026-06-19

New script:

```text
scripts/build_fine_mask_eval_manifest.py
```

Purpose:

- convert object-QA evidence into a stable small sample set for the next
  SAM2-loop / coverage-completion experiment
- preserve object id, target id, frame/camera, current source mask path, crop
  path, 2D bbox, and risk reasons
- avoid ad hoc hand-picked frames when testing fine-mask improvements

Input evidence:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/frame_object_qa_localgeom_candidate_ids_20260619/frame_local_object_qa_evidence.jsonl
```

Remote manifest:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/fine_mask_eval_manifest_localgeom_railing_20260619/manifest.json
/home/zsh/Work/SCAN/work_MT20260616-175807/fine_mask_eval_manifest_localgeom_railing_20260619/manifest.md
```

Local mirror:

```text
server_parking_priority_s10/fine_mask_eval_manifest_localgeom_railing_20260619/manifest.json
server_parking_priority_s10/fine_mask_eval_manifest_localgeom_railing_20260619/manifest.md
```

Command:

```bash
python scripts/build_fine_mask_eval_manifest.py \
  --evidence-jsonl frame_object_qa_localgeom_candidate_ids_20260619/frame_local_object_qa_evidence.jsonl \
  --output-json fine_mask_eval_manifest_localgeom_railing_20260619/manifest.json \
  --output-md fine_mask_eval_manifest_localgeom_railing_20260619/manifest.md \
  --labels railing car \
  --limit 40 \
  --per-object-limit 3
```

Result:

- samples: `21`
- objects: `10`
- labels: `railing=21`, `car=0`

Interpretation:

- The current high-risk fine-mask sample set is railing/handrail dominated.
- The next SAM2-loop test should run on these 21 samples first, not on a broad
  full-scene slice.
- Promotion gate: the new masks must reduce source-mask spillover into
  stairs/walls/floors while preserving obvious handrail coverage. If the mask
  still swallows adjacent surfaces, keep local-geometry suppression as the
  precision guard and do not promote the 2D mask change.

## Fine-Mask Input Package And Color/Shape Probe

Date: 2026-06-19

New scripts:

```text
scripts/prepare_fine_mask_eval_inputs.py
scripts/probe_fine_mask_color_geometry.py
```

Prepared 5070Ti input package:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/fine_mask_eval_inputs_localgeom_railing_20260619
```

Preparation result:

- samples: `21`
- ready images: `21`
- ready current masks: `21`
- ready crops: `21`
- missing: `0`

Color/shape probe output:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/fine_mask_color_geometry_probe_localgeom_railing_20260619
server_parking_priority_s10/fine_mask_color_geometry_probe_localgeom_railing_20260619
```

Probe result:

| flag | count |
| --- | ---: |
| `high_fill_ratio` | `21 / 21` |
| `not_thin` | `21 / 21` |
| `large_bbox` | `17 / 21` |
| `weak_color_boundary` | `12 / 21` |

Visual read from the contact sheet:

- current `railing` masks are usually large filled regions, not thin
  railing/handrail structures
- many masks swallow stair treads, wall panels, floors, corrugated surfaces, or
  adjacent background
- this matches the point-cloud local-geometry finding that most removed
  `railing` points were mask spillover, not random label noise

Decision:

- The proposed visual-depth refinement should be attached after
  `SAM2/skymask` as a mask split/guard layer.
- Do not use VLM relabeling to fix these samples; the 2D evidence is already
  mixed before labeling.
- Next step is to add projected sparse depth continuity to this probe, then use
  depth/color/3D connectedness to split or reject broad fine masks.

5070Ti Python SAM2 setup status:

- installed official `facebookresearch/sam2` code in editable mode:
  `/home/zsh/Work/SCAN/deps/sam2`
- installed with `--no-deps` after first aborting a pip attempt that tried to
  upgrade torch
- verified torch remained unchanged:
  `torch 2.11.0+cu130`, CUDA `13.0`, CUDA available
- checkpoint download target:
  `/home/zsh/Work/SCAN/models/sam2/sam2.1_hiera_large.pt`

## Dense Raw Point Cloud Reverse-Depth Check

Date: 2026-06-19

Problem found during the reverse-depth mask refinement probe:

- Using only colorized point clouds is too sparse because uncolored LiDAR points
  are dropped before depth reconstruction.
- Using a short `3400..3500` window does not cover every camera view.
- Blindly projecting the full raw scene into one camera frame is also invalid:
  historical points can project through current-frame occluders, so the depth
  map can look like a different pose even when the calibration chain is correct.

Code changes:

```text
scripts/build_raw_lx_voxel_cloud.py
scripts/build_geometry_guidance_maps.py
```

`build_raw_lx_voxel_cloud.py` now streams `.lx` sections into a binary
little-endian 1cm voxel PLY without requiring image color.  It can write source
frame metadata per voxel:

```text
frame_min
frame_max
frame_mean
frame_count
```

`build_geometry_guidance_maps.py` can now read that metadata and filter global
PLY points per image frame:

```text
--global-source-filter-mode mean --global-source-frame-window 20
```

Smoke artifacts:

```text
/home/zsh/Work/SCAN/work_MT20260616-175807/raw_lx_voxel_v001_meta_3400_3500/raw_points_voxel001_meta.ply
/home/zsh/Work/SCAN/work_MT20260616-175807/raw_lx_voxel_v001_meta_guidance_3400_3500_s10_nofilter
/home/zsh/Work/SCAN/work_MT20260616-175807/raw_lx_voxel_v001_meta_guidance_3400_3500_s10_source20
server_parking_priority_s10/raw_depth_source_filter_check/
```

Smoke result:

- `3400..3500` raw points: `2,513,722`
- `0.01m` voxel points: `1,411,871`
- no-filter guidance: `33/33` images OK
- source-window guidance: `33/33` images OK
- cam0/frame3400 candidate points:
  - no filter: `1,411,871`, visible pixels `84,349`
  - source20: `203,088`, visible pixels `61,855`

Interpretation:

- Source-frame filtering reduces obvious historical projection leakage.
- It does not make unavailable same-view surfaces appear.  For example, the
  large near wall in cam0/frame3400 still has little or no LiDAR support in the
  current source window.
- Therefore dense raw reverse-depth should be used as a confidence/edge guard
  only where depth support exists.  No-depth pixels must stay low confidence;
  they must not be filled by blind full-scene projection.

Full reusable raw dataset build started on `scan-rtx5070`:

```text
tmux session: raw_voxel_meta_full
output: /home/zsh/Work/SCAN/work_MT20260616-175807/raw_lx_voxel_full_v001_meta/raw_points_full_voxel001_meta.ply
report: /home/zsh/Work/SCAN/work_MT20260616-175807/raw_lx_voxel_full_v001_meta/raw_voxel_report.json
```

It later completed:

- raw points: `97,855,095`
- voxel points: `68,870,431`
- voxel size: `0.01m`
- output size: about `1.8GB`
- embedded metadata: `frame_min`, `frame_max`, `frame_mean`, `frame_count`

Promotion rule:

- Keep frame-local `.lx` visibility as the hard correctness reference.
- Use full raw voxel depth only with source-frame metadata and visibility
  confidence.
- Do not use blind global reverse projection for mask labels or VLM evidence.

Follow-up sync clarification:

- The previously validated color route reads images from `FRAME_OUTPUT_DIRS`;
  its frame/video binding is created by `extract_frames.py`.
- On the parking dataset, the legacy `ffmpeg-time` extraction
  (`target_rel_ts = frame_id * 0.1`) and OpenCV direct-index extraction both
  map sampled frames to `video_idx == frame_id` with `delta=0`.
- Therefore the observed mismatch is not explained by ffmpeg vs OpenCV seeking.
- The important issue is evidence provenance: same-frame `.lx` depth can be
  very sparse in the camera view, while full-global depth fills the image with
  geometry collected from other moments/viewpoints.  That makes the image and
  full-global depth look like different poses in places where the current frame
  has no LiDAR support.
- Production semantic evidence must therefore use same-frame or near-frame
  source voxels first, and use full-global depth only as boundary guidance.

Code guard added:

- `scripts/build_geometry_guidance_maps.py` now defaults global PLY guidance to
  `--global-source-filter-mode mean --global-source-frame-window 20`.
- Unguarded full-global projection requires explicit
  `--allow-unguarded-global`; otherwise the script exits before producing
  misleading guidance maps.
- If a global PLY has no frame metadata, source filtering is treated as an
  error unless `--allow-unguarded-global` is explicitly set.
- Default guarded smoke on `frame=3400, cam0`:
  - source points kept: `366,214`
  - visible pixels: `85,290`
  - previous full-global no-filter visible pixels: `1,300,081`

Safe semantic-prior probe:

- Window: `3400..3500`, `stride=10`, cams `0/1/2`.
- Geometry source:
  `/home/zsh/Work/SCAN/work_MT20260616-175807/raw_lx_voxel_full_v001_meta/raw_points_full_voxel001_meta.ply`
- Semantic prior:
  `/home/zsh/Work/SCAN/work_MT20260616-175807/frame_object_viewer_best_p008_split_lowplanar_surface_consolidated_localgeom_rtx5070_default_localgeom_20260619_132858/frame_object_points_stride10.ply`
- Output:
  `/home/zsh/Work/SCAN/work_MT20260616-175807/geometry_refine_guarded_default_semprior_safe_3400_3500`
- Result:
  - source filter: `mean +/-20`
  - semantic-prior voxels: `198,800`
  - surface fill: `residual->wall 126,988`, `residual->ground 15`
  - fine overwrite: `0`
  - projection residual points: `41,794 -> 31,726`
  - railing points preserved: `13,328`
- Decision: keep semantic prior as residual surface fill by default. Do not
  enable `--guarded-fine-surface-override` in production yet; the same window
  produced `railing->wall 26,273` pixels when that option was enabled.

## Parking Dataset Frame Sync Blocker

Date: 2026-06-19

User review caught that the `original` image and projected full raw point cloud
looked like different poses.  Follow-up checks confirmed a real synchronization
risk in the parking dataset cache.

Current extraction/cache behavior:

- `scripts/extract_undistorted_frames_jpeg.py` reads video frames by direct
  frame index: `CAP_PROP_POS_FRAMES = frame_id`.
- `scripts/colorize_lx_stream.py`, the current full-scene colorization runner,
  uses the same direct `read_video_frame(cap, frame_id)` assumption.
- Therefore the existing colored point cloud is not an independent proof of
  image/LiDAR synchronization; it shares the same frame-index assumption.

Reusable diagnostic added:

```text
scripts/probe_lx_video_alignment.py
```

5070Ti probe command:

```bash
python scripts/probe_lx_video_alignment.py \
  --lx-file /home/zsh/Work/SCAN/datasets/MT20260616-175807/MANIFOLD_MT20260616-175807.lx \
  --output-dir /home/zsh/Work/SCAN/work_MT20260616-175807/alignment_probe_multi_20260619 \
  --frames 1000 2000 3400 5000 6000 \
  --cams 0 1 2 \
  --offsets -1200 -1000 -800 -600 -400 -200 0 200 400
```

Mirrored local QA:

```text
server_parking_priority_s10/alignment_probe_multi_20260619/alignment_probe_sheet.jpg
server_parking_priority_s10/alignment_probe_multi_20260619/alignment_probe_report.json
```

Findings:

- For `section=3400, cam0`, direct `video_idx=3400` scored poorly; a candidate
  near `video_idx=2600` visually/edge-wise fit better.
- Best offsets are not stable across sampled frames/cameras:
  - frame `1000`: often `offset=-200`
  - frame `2000`: often `offset=-1200`
  - frame `3400`: varies by camera from `-1200` to `-400`
  - frame `5000`: often `offset=+400`
  - frame `6000`: varies from `-600` to `0`
- `img_pos` fields `cam0_frame_info/cam1_frame_info/cam2_frame_info` are not
  direct video frame numbers; they only take small values such as `0..10` and
  are identical across cameras in this dataset.
- `.lx` section headers include the same pose/timestamp values as
  `img_pos.txt`; those timestamps do not map directly to MKV PTS.  The MKV is
  `10fps` and about `618s`, while `img_pos` timestamps span about `835s`.

Decision:

- Treat the current parking image cache and all mask/target results derived
  from it as useful experiments, but not final semantic evidence.
- Do not run more production semantic fusion until image/LiDAR frame mapping is
  calibrated.
- Next required step is a dedicated synchronization calibration stage.  It
  should produce an explicit `section_id -> cam_id -> video_frame_idx` mapping
  or reject this dataset for image-based semantic projection.

Production-code fix:

- Added `scripts/sync_frame_map.py` as the shared loader for explicit
  `frame_id/cam_id -> video_idx` JSONL mappings.
- `scripts/colorize_lx_stream.py` now accepts:
  - `--frame-map-jsonl`: read calibrated mappings from solver or manual-review
    JSONL output.
  - `--require-frame-map`: fail image reads for missing frame/camera mappings
    instead of silently falling back to `video_idx == frame_id`.
- Default behavior remains direct `frame_id -> video_idx`, so previous runs are
  reproducible.  Calibrated runs must pass an explicit frame map.
- This closes the structural gap where a correct sync path could be generated
  but not used by the validated colorization route.

Follow-up extraction fix:

- `scripts/extract_undistorted_frames_jpeg.py` now supports
  `--sync-mode frame-map --frame-map-jsonl <path>`.
- In `frame-map` mode it saves files using section ids
  (`frame_003400.jpg`) while reading the mapped video frame
  (`video_idx=2600/2700/...`), so downstream segmentation artifacts keep the
  existing section-id naming convention.
- `--require-frame-map` prevents silent fallback to direct frame ids when a
  frame/camera mapping is missing.
- Performance issue fixed: `frame-map` mode no longer loads full ffprobe
  timestamp tables.  Remote smoke for one frame/cam triplet dropped from about
  `50.5s` to `0.25s`.

Production safety gate:

- `scripts/sync_frame_map.py` now rejects unsafe mapping rows by default,
  including solver output marked as `cam_path_status=rejected_unstable_temporal_path`.
- `scripts/colorize_lx_stream.py` and
  `scripts/extract_undistorted_frames_jpeg.py` expose
  `--allow-rejected-frame-map` only for diagnostics.
- Current full-range smooth path is correctly blocked by default:

```text
ValueError: unsafe sync row status='rejected_unstable_temporal_path'
```

- Diagnostic mode still works when explicitly enabled:
  `--allow-rejected-frame-map`.

Readiness gate added:

- `scripts/check_sync_frame_map_readiness.py` checks accepted anchors, solver
  reports, and frame-map JSONL before any production extraction/colorization.
- It reports structured failures instead of requiring manual inspection of
  several JSON files.
- Current 5070Ti readiness result:
  - `passed=false`
  - accepted anchors: `0`
  - anchor rows: `27` unreviewed
  - solver status: `rejected`
  - rejected cameras: `0,1,2`
  - frame-map rows: `27`, all
    `cam_path_status=rejected_unstable_temporal_path`
- Local mirrored report:

```text
server_parking_priority_s10/sync_readiness_current_20260619.remote.json
```

Runner integration:

- `scripts/run_rtx5070_sync_anchor_solver.sh` now runs
  `check_sync_frame_map_readiness.py` immediately after constrained solver
  output is produced.
- Readiness failure exits the local launcher with code `3`.
- The runner still rebuilds and pulls the constrained review pack before
  returning failure, so the next manual review iteration is not lost.
- Pulled artifacts now include:
  - `sync_frame_map_readiness.json`
  - `sync_frame_map_readiness.exit_code`
  - `solver/`
  - `review/`
- Configurable environment:
  - `READINESS_FRAMES`
  - `READINESS_CAMS`
  - `MIN_ACCEPTED_PER_CAM`

Manual review prioritizer:

- Added `scripts/prioritize_sync_anchor_review.py` to rank existing manual
  anchor review rows without auto-accepting any anchor.
- It reuses existing panel images and outputs a small first-pass batch by
  camera, so manual work can focus on the most informative probes.
- `anchor_review_priority.html` is now interactive: select an option, mark the
  row accepted/rejected/unreviewed, then export `accepted_sync_anchors.jsonl`
  directly from the page.
- Panel image paths are relative to the priority output directory, so the page
  can be served from the existing local static server.
- Current local output:

```text
server_parking_priority_s10/sync_anchor_review_priority_20260619/
```

- Key files:
  - `anchor_review_priority.html`
  - `anchor_review_priority_batch.jsonl`
  - `anchor_review_priority_all.jsonl`
  - `anchor_review_priority_report.json`
- Current batch: `12` rows, `4` per camera.
- This is a review accelerator only. It does not auto-accept anchors; accepted
  anchors must still be explicitly selected before constrained solving.

Accepted-anchor staging:

- Added `scripts/stage_accepted_sync_anchors.py`.
- Default source discovery:
  latest `/Users/skkac/Downloads/accepted_sync_anchors*.jsonl`, so repeated
  browser downloads such as `accepted_sync_anchors (1).jsonl` are accepted
  automatically.  Pass `--source` to force an explicit file.
- Default target:
  `server_parking_priority_s10/sync_anchor_review_priority_sky_penalty_timestamp_absprior_dot3_20260619/accepted_sync_anchors.jsonl`.
- The staging script validates accepted anchors with
  `scripts/validate_sync_anchors.py` before copying.  By default it requires at
  least `2` accepted anchors per camera, uses `timestamp_phase_fraction=1.0`,
  and checks monotonic video indices against `expected_fps=6.0`.
- Current dry-run correctly fails because no exported anchors exist yet:

```text
source_missing=/Users/skkac/Downloads/accepted_sync_anchors.jsonl
```

After exporting anchors from the priority page, run either:

```bash
python scripts/stage_accepted_sync_anchors.py --force
scripts/run_rtx5070_sync_anchor_solver.sh
```

or one-shot:

```bash
python scripts/stage_accepted_sync_anchors.py --force --run-solver
```

Command:

```bash
python scripts/check_sync_frame_map_readiness.py \
  --anchors-jsonl /home/zsh/Work/SCAN/work_MT20260616-175807/sync_anchor_review_small_20260619_v2/manual_anchor_manifest.jsonl \
  --frame-map-jsonl /home/zsh/Work/SCAN/work_MT20260616-175807/sync_smooth_abs_fullrange_step100_20260619/sync_smooth_paths.jsonl \
  --solver-report /home/zsh/Work/SCAN/work_MT20260616-175807/sync_smooth_abs_fullrange_step100_20260619/sync_smooth_path_report.json \
  --frames 1000 1600 2200 2800 3400 4000 4600 5200 5800 \
  --cams 0 1 2 \
  --min-accepted-per-cam 2
```

5070Ti smoke:

```bash
python scripts/extract_undistorted_frames_jpeg.py \
  --output-dir /home/zsh/Work/SCAN/work_MT20260616-175807/extract_frame_map_smoke_fast_3400_20260619 \
  --start 3400 --end 3400 --stride 1 --cams 0 1 2 --workers 3 \
  --sync-mode frame-map \
  --frame-map-jsonl /home/zsh/Work/SCAN/work_MT20260616-175807/sync_smooth_abs_fullrange_step100_20260619/sync_smooth_paths.jsonl \
  --require-frame-map
```

Smoke result:

- cam0: section `3400` read `video_idx=2600`
- cam1: section `3400` read `video_idx=2700`
- cam2: section `3400` read `video_idx=3400`
- failed reads: `0`
- elapsed: `0.25s`

Validation:

```bash
python3 -m py_compile scripts/sync_frame_map.py scripts/colorize_lx_stream.py
pytest -q tests/test_sync_frame_map.py
```

Updated validation:

```bash
python3 -m py_compile \
  scripts/sync_frame_map.py \
  scripts/colorize_lx_stream.py \
  scripts/extract_undistorted_frames_jpeg.py
pytest -q tests/test_sync_frame_map.py tests/test_extract_undistorted_frames_jpeg_sync.py
```

Result: sync frame-map tests `9 passed` locally and on `scan-rtx5070`.

## Safe Semantic-Prior Runner

Date: 2026-06-19

Reusable runner added:

```text
scripts/run_parking_safe_semantic_prior_route.sh
```

Default behavior:

- runs dry by default; set `RUN=1` to execute;
- exports `SCAN_IMAGE_DIR` and `SCAN_VIDEO_DIR` to the 5070Ti parking dataset
  path so scripts do not fall back to old `/root/epfs` calibration files;
- uses the raw `0.01m` voxel PLY with frame metadata as geometry guidance;
- applies `--global-source-filter-mode mean --global-source-frame-window 20`;
- uses the current best local-geometry viewer PLY only as a semantic prior;
- fills residual surface holes only: `--surface-override-from 0`;
- does not enable `--guarded-fine-surface-override` unless the caller sets
  `ALLOW_FINE_SURFACE_OVERRIDE=1`.

Smoke command:

```bash
RUN=1 OVERWRITE=1 OUT_SUFFIX=guarded_semprior_safe_smoke \
START=3400 END=3500 STRIDE=10 \
bash scripts/run_parking_safe_semantic_prior_route.sh
```

Smoke result:

- geometry guidance images: `33/33 ok`;
- semantic-prior voxels: `198,800`;
- residual surface fill: `residual->wall 126,988`, `residual->ground 15`;
- fine overwrite: `0`;
- projection:
  - frame count: `11`;
  - raw points: `450,175`;
  - visible non-sky points: `401,126`;
  - priority points: `369,400`;
  - residual points: `31,726`;
  - priority counts:
    `ground 6,577`, `wall 347,470`, `grass 2,025`, `railing 13,328`.

Interpretation:

- The runner reproduces the conservative safe-prior behavior and avoids the
  previous `railing->wall` fine-label overwrite failure.
- It does not solve the image/LiDAR synchronization blocker.  It only prevents
  unguarded full-global reverse projection from adding additional cross-time
  evidence pollution.

## Sync Calibration Gate

Date: 2026-06-19

Reusable calibration script added:

```text
scripts/calibrate_lx_video_frame_mapping.py
```

Purpose:

- search candidate video frames for selected `.lx` sections;
- score each candidate by projecting same-section LiDAR rings onto the
  undistorted image and measuring distance to image edges;
- write explicit candidates, best matches, affine fit diagnostics, and a QA
  contact sheet;
- reject the dataset for production image projection when best matches cannot
  be explained by a stable per-camera mapping.

5070Ti command:

```bash
export SCAN_IMAGE_DIR=/home/zsh/Work/SCAN/datasets/MT20260616-175807/image
export SCAN_VIDEO_DIR=/home/zsh/Work/SCAN/datasets/MT20260616-175807/image
export PYTHONPATH=$PWD/scripts

/home/zsh/Work/SCAN/.venvs/scan-semantic/bin/python \
  scripts/calibrate_lx_video_frame_mapping.py \
  --lx-file /home/zsh/Work/SCAN/datasets/MT20260616-175807/MANIFOLD_MT20260616-175807.lx \
  --output-dir /home/zsh/Work/SCAN/work_MT20260616-175807/sync_calibration_small_20260619 \
  --frames 1000 1600 2200 2800 3400 4000 4600 5200 5800 6200 \
  --cams 0 1 2 \
  --offsets=-1400:800:100 \
  --panels-per-probe 3 \
  --sheet-cols 4
```

Mirrored local QA:

```text
server_parking_priority_s10/sync_calibration_small_20260619/sync_calibration_report.json
server_parking_priority_s10/sync_calibration_small_20260619/sync_fit.json
server_parking_priority_s10/sync_calibration_small_20260619/sync_best.jsonl
server_parking_priority_s10/sync_calibration_small_20260619/sync_candidates.jsonl
server_parking_priority_s10/sync_calibration_small_20260619/sync_probe_sheet.jpg
```

Result:

- status: `rejected`
- candidates: `594`
- best rows: `27`
- direct-index rank among candidates:
  - count: `27`
  - median rank: `12`
  - mean rank: `11.63`
  - max rank: `23`
- per-camera affine fit:
  - cam0: `rmse=674.64`, `max_abs_residual=1212.22`,
    `slope=1.2056`, `intercept=-1010.0`
  - cam1: `rmse=716.44`, `max_abs_residual=1142.22`,
    `slope=1.1556`, `intercept=-984.44`
  - cam2: `rmse=595.87`, `max_abs_residual=1022.22`,
    `slope=1.1111`, `intercept=-722.22`

Interpretation:

- The current parking image cache remains unsuitable as production semantic
  evidence.
- A constant offset or single affine mapping is not enough.
- Edge-score matching is a useful automatic gate, but it can still choose
  visually plausible local mismatches.  Use the contact sheet as required QA
  before accepting a mapping.
- Next technical step is either:
  - recover authoritative timestamp/frame mapping from MANIFOLD export or MKV
    metadata, or
  - build a stronger visual-LiDAR synchronization optimizer with temporal
    smoothness and manual anchor support.

## Raw Timing Source Audit

Date: 2026-06-19

Reusable audit script added:

```text
scripts/audit_dataset_timing_sources.py
```

5070Ti command:

```bash
export SCAN_IMAGE_DIR=/home/zsh/Work/SCAN/datasets/MT20260616-175807/image
export SCAN_VIDEO_DIR=/home/zsh/Work/SCAN/datasets/MT20260616-175807/image
export PYTHONPATH=$PWD/scripts

/home/zsh/Work/SCAN/.venvs/scan-semantic/bin/python \
  scripts/audit_dataset_timing_sources.py \
  --lx-file /home/zsh/Work/SCAN/datasets/MT20260616-175807/MANIFOLD_MT20260616-175807.lx \
  --output-dir /home/zsh/Work/SCAN/work_MT20260616-175807/timing_sources_audit_20260619
```

Mirrored local QA:

```text
server_parking_priority_s10/timing_sources_audit_20260619/timing_sources_report.json
server_parking_priority_s10/timing_sources_audit_20260619/lx_headers_sample.jsonl
```

Findings:

- `.lx` sections: `6,181`
- `img_pos` rows: `6,181`
- each camera MKV frames: `6,181`, `fps=10`, duration by frame count `618.1s`
- `.lx` header pose matches `img_pos`:
  - position error max: `2.63e-6m`
  - quaternion error max: `4.32e-8`
  - header `uint8` matches `frame_id`: `100%`
- `.lx` header `uint9` correlates with `img_pos.timestamp` and appears to be
  a device tick counter around `2^22 ticks/sec`.
- `img_pos.timestamp` span: `835.32s`
- `img_pos` adjacent interval:
  - median: `0.100016s`
  - mean: `0.135166s`
  - max: `2.000313s`
  - intervals `>0.15s`: `1,507`
  - intervals `>0.5s`: `63`

Interpretation:

- `.lx` and `img_pos` are internally consistent; the pose source is not the
  cause of the mismatch.
- The camera videos are fixed 10fps streams with continuous PTS, while
  `img_pos/.lx` device time contains many long gaps.  The current route has no
  authoritative mapping from device-time sections to compressed video frames.
- Continue to block production semantic projection until this mapping is
  recovered or calibrated with stronger temporal constraints.

## Temporal Smoothness Sync Solver

Date: 2026-06-19

Reusable solver added:

```text
scripts/solve_sync_path_from_candidates.py
```

Purpose:

- consume `sync_candidates.jsonl` from the calibration gate;
- solve a per-camera monotonic/smooth section-to-video path with dynamic
  programming;
- reject paths that are temporally smooth but require too much score loss from
  the independently best visual candidates.

Local sample command:

```bash
python3 scripts/solve_sync_path_from_candidates.py \
  --candidates-jsonl server_parking_priority_s10/sync_calibration_small_20260619/sync_candidates.jsonl \
  --output-dir server_parking_priority_s10/sync_smooth_path_small_20260619 \
  --target-ratio 1.0 \
  --max-ratio-deviation 0.6 \
  --velocity-weight 2.0
```

Result:

- status: `rejected`
- the solver can produce smooth monotonic paths:
  - cam0 step ratio mean: `0.979`, max deviation: `0.167`
  - cam1 step ratio mean: `1.000`, max deviation: `0.000`
  - cam2 step ratio mean: `1.000`, max deviation: `0.000`
- but those paths lose too much image-edge evidence:
  - cam0 score-loss mean/max: `0.158 / 0.441`
  - cam1 score-loss mean/max: `0.227 / 0.426`
  - cam2 score-loss mean/max: `0.171 / 0.386`
  - default acceptance threshold: mean `<=0.10`, max `<=0.25`

Interpretation:

- A smooth timeline exists, but it is not visually well-supported by the
  current edge candidate scores.
- This prevents the route from silently replacing one bad mapping with another
  plausible but unsupported mapping.
- Next step remains: recover authoritative camera timing or add manual anchors
  plus visual QA to constrain the optimizer.

## Manual Sync Anchor Review Pack

Date: 2026-06-19

Reusable review-pack script added:

```text
scripts/build_sync_anchor_review_pack.py
```

Purpose:

- consume sync candidates and optional smooth path;
- for every `(section_id, cam_id)` probe, render deduplicated choices:
  `direct`, `independent_best`, `smooth_path`, and top score candidates;
- output a human-fillable `manual_anchor_manifest.jsonl`;
- output one contact sheet for fast visual review.

5070Ti command:

```bash
export SCAN_IMAGE_DIR=/home/zsh/Work/SCAN/datasets/MT20260616-175807/image
export SCAN_VIDEO_DIR=/home/zsh/Work/SCAN/datasets/MT20260616-175807/image
export PYTHONPATH=$PWD/scripts

/home/zsh/Work/SCAN/.venvs/scan-semantic/bin/python \
  scripts/build_sync_anchor_review_pack.py \
  --lx-file /home/zsh/Work/SCAN/datasets/MT20260616-175807/MANIFOLD_MT20260616-175807.lx \
  --candidates-jsonl /home/zsh/Work/SCAN/work_MT20260616-175807/tmp_sync_review_inputs/sync_candidates.jsonl \
  --smooth-path-jsonl /home/zsh/Work/SCAN/work_MT20260616-175807/tmp_sync_review_inputs/sync_smooth_paths.jsonl \
  --output-dir /home/zsh/Work/SCAN/work_MT20260616-175807/sync_anchor_review_small_20260619 \
  --top-n 4 \
  --sheet-cols 4
```

Mirrored local QA:

```text
server_parking_priority_s10/sync_anchor_review_small_20260619/manual_anchor_review_sheet.jpg
server_parking_priority_s10/sync_anchor_review_small_20260619/manual_anchor_manifest.jsonl
server_parking_priority_s10/sync_anchor_review_small_20260619/manual_anchor_review_report.json
```

Result:

- probe count: `27`
- rendered candidate panels: `152`
- manifest rows: `27`

Next use:

- Review the sheet and fill reliable rows in `manual_anchor_manifest.jsonl`:
  set `anchor_status` to `accepted`, set `selected_option_idx` and
  `selected_video_idx`, and leave uncertain rows as `unreviewed`.
- The next optimizer should use accepted anchors as hard or high-weight
  constraints, not as another soft score.

Anchor consumption support:

- `scripts/solve_sync_path_from_candidates.py` now accepts
  `--anchors-jsonl manual_anchor_manifest.jsonl`.
- Rows with `anchor_status == "accepted"` become hard constraints for their
  `(frame_id, cam_id)` probe.
- `selected_video_idx` is used directly; if omitted, the solver resolves
  `selected_option_idx` through the row's `options`.
- If an accepted anchor points to a video frame not present in the candidate
  set, the solver exits with an error instead of silently ignoring the anchor.

Example:

```bash
python3 scripts/solve_sync_path_from_candidates.py \
  --candidates-jsonl server_parking_priority_s10/sync_calibration_small_20260619/sync_candidates.jsonl \
  --anchors-jsonl server_parking_priority_s10/sync_anchor_review_small_20260619/manual_anchor_manifest.jsonl \
  --output-dir server_parking_priority_s10/sync_smooth_path_anchor_constrained \
  --target-ratio 1.0 \
  --max-ratio-deviation 0.6 \
  --velocity-weight 2.0
```

## Sky-Penalty Sync Calibration

Date: 2026-06-19

Problem observed:

- The previous image/LiDAR sync score was dominated by image-edge-rich frames.
- Many independent probes repeatedly selected early video frames, which is not a physically plausible temporal path.
- This explains why downstream reverse-depth and semantic overlays looked like they came from different poses.

Reusable change:

- `scripts/calibrate_lx_video_frame_mapping.py` now supports optional negative sky evidence:
  - `--sky-filter heuristic`
  - `--sky-penalty-weight <float>`
- The score is reduced when projected LiDAR samples fall into conservative sky-like pixels.  This encodes the physical rule that sky should not contain LiDAR returns.
- Default behavior remains unchanged unless `--sky-filter heuristic` is passed.

5070Ti command used:

```bash
cd /home/zsh/Work/SCAN/new_route
export SCAN_IMAGE_DIR=/home/zsh/Work/SCAN/datasets/MT20260616-175807/image
export SCAN_VIDEO_DIR=/home/zsh/Work/SCAN/datasets/MT20260616-175807/image
export PYTHONPATH=$PWD/scripts

/home/zsh/Work/SCAN/.venvs/scan-semantic/bin/python \
  scripts/calibrate_lx_video_frame_mapping.py \
  --lx-file /home/zsh/Work/SCAN/datasets/MT20260616-175807/MANIFOLD_MT20260616-175807.lx \
  --output-dir /home/zsh/Work/SCAN/work_MT20260616-175807/sync_calibration_sky_penalty_fullprobe_20260619 \
  --frames 1000 1600 2200 2800 3400 4000 4600 5200 5800 \
  --cams 0 1 2 \
  --offsets=-1600:1000:100 \
  --projected-depth-edges \
  --projected-edge-kind silhouette \
  --sky-filter heuristic \
  --sky-penalty-weight 0.45 \
  --panels-per-probe 5 \
  --sheet-cols 6

/home/zsh/Work/SCAN/.venvs/scan-semantic/bin/python \
  scripts/solve_sync_path_from_candidates.py \
  --candidates-jsonl /home/zsh/Work/SCAN/work_MT20260616-175807/sync_calibration_sky_penalty_fullprobe_20260619/sync_candidates.jsonl \
  --output-dir /home/zsh/Work/SCAN/work_MT20260616-175807/sync_smooth_sky_penalty_fullprobe_20260619 \
  --target-ratio 1.0 \
  --max-ratio-deviation 0.45 \
  --velocity-weight 2.0 \
  --max-score-loss-mean 0.12 \
  --max-score-loss-max 0.30
```

Result:

- Full 9-probe calibration still rejects independent best fits.
- The smooth solver finds a physically stable monotonic path with `step_ratio=1.0` for all cameras, but the path is still rejected by production gate because score loss from independent best remains high:
  - cam0 mean loss `0.179`, max `0.342`
  - cam1 mean loss `0.227`, max `0.422`
  - cam2 mean loss `0.202`, max `0.423`
- Interpretation: sky penalty improves the candidate set, but automatic scoring is still not strong enough to be trusted without manual anchors.

Mirrored local QA:

```text
server_parking_priority_s10/sync_calibration_sky_penalty_fullprobe_20260619/
server_parking_priority_s10/sync_smooth_sky_penalty_fullprobe_20260619/
server_parking_priority_s10/sync_anchor_review_sky_penalty_fullprobe_20260619/
server_parking_priority_s10/sync_anchor_review_priority_sky_penalty_20260619/
```

Priority review page:

```text
http://127.0.0.1:8765/server_parking_priority_s10/sync_anchor_review_priority_sky_penalty_20260619/anchor_review_priority.html
```

Next step:

- Use the sky-penalty priority review page to export accepted anchors.
- Stage anchors with `python scripts/stage_accepted_sync_anchors.py`.
- Run `scripts/run_rtx5070_sync_anchor_solver.sh` and require readiness pass before any further semantic production.

## Sync Option Contact Sheets

Date: 2026-06-19

Why:

- RGB colorization can look acceptable even when frame sync is too weak for hard mask projection.
- Semantic and depth-guided masks require stricter frame alignment, so the sync path must be reviewed visually before production.
- The sky-penalty smooth path is temporally plausible, but still rejected automatically because it loses too much local score against independent best candidates.

Reusable tools added:

- `scripts/make_sync_option_sheet.py`
  - extracts one option source from `manual_anchor_manifest.jsonl`;
  - supports sources such as `smooth_path`, `direct`, and `independent_best`;
  - writes a contact sheet and JSON report without modifying anchors.
- `scripts/prioritize_sync_anchor_review.py --preselect-source smooth_path`
  - preselects smooth-path choices in the review UI;
  - keeps rows `unreviewed`, so no anchor is accepted without explicit confirmation.

Generated QA assets:

```text
server_parking_priority_s10/sync_anchor_review_sky_penalty_fullprobe_20260619/smooth_path_contact_sheet.jpg
server_parking_priority_s10/sync_anchor_review_sky_penalty_fullprobe_20260619/direct_contact_sheet.jpg
server_parking_priority_s10/sync_anchor_review_sky_penalty_fullprobe_20260619/independent_best_contact_sheet.jpg
server_parking_priority_s10/sync_anchor_review_priority_sky_penalty_smooth_preselect_20260619/
```

Observed comparison:

- `independent_best` is visually/temporally unstable and often jumps across scenes.
- `direct` is sometimes plausible outdoors, but not reliable enough across the indoor/stair segments.
- `smooth_path` has the best temporal continuity and generally better physical structure, but still has local questionable frames; it should be treated as a review aid, not automatic truth.

Preferred review URL:

```text
http://127.0.0.1:8765/server_parking_priority_s10/sync_anchor_review_priority_sky_penalty_smooth_preselect_20260619/anchor_review_priority.html
```

Next operational sequence:

```bash
cd /Users/skkac/Work/SCAN/new_route
python3 scripts/stage_accepted_sync_anchors.py
scripts/run_rtx5070_sync_anchor_solver.sh
```

Do not resume semantic production until the constrained solver and readiness gate pass.
