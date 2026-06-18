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

## Interpretation

- The 5070Ti migration is operational: data, environment, model cache, GPU inference, frame-local projection, target fusion, and local review export all work.
- This run reproduces the current validated route with no missing target-point mapping.
- The bottleneck remains source mask quality and surface/fine-object confusion, not calibration or global point projection.
- Next optimization should compare source priority mask refinements on selected bad windows before any new full run.

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
