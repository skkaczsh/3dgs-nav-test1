# SAM2 TensorRT Environment

This documents the scan-train SAM2 TensorRT environment and the current
subgraph-level C++ runner foundation.

## Server

- host: `scan-train`
- SSH: `root@10.0.8.114 -p 31909`
- required bind address from this workstation: `192.168.100.119`
- work dir: `/root/epfs/sam2_tensorrt`

## Installed Components

- Python env: `/root/epfs/conda_envs/vlm_seg`
- Python packages:
  - `tensorrt-cu12==11.0.0.114`
  - `onnx`
  - `onnxsim`
  - `polygraphy`
  - `cuda-python`
- C++ TensorRT packages:
  - `libnvinfer-dev==10.9.0.34-1+cuda11.8`
  - `libnvinfer-bin==10.9.0.34-1+cuda11.8`
  - matching `libnvinfer*`, `libnvonnxparsers*` runtime/dev packages
- C++ include path:
  - `/usr/include/x86_64-linux-gnu`
  - `/usr/local/cuda/include`
- C++ libraries:
  - `/usr/lib/x86_64-linux-gnu`
  - `/usr/local/cuda/lib64`
- `trtexec`:
  - `/usr/src/tensorrt/bin/trtexec`

## Verify

Run on scan-train:

```bash
cd /root/epfs/new_route_scripts
./verify_sam2_tensorrt_env.sh
```

Expected evidence:

- Python imports `torch`, `tensorrt`, `onnx`, `onnxsim`, `polygraphy`.
- C++ smoke binary compiles and prints `builder_ok=1`.
- Tiny ONNX model builds a TensorRT FP16 engine at
  `/root/epfs/sam2_tensorrt/engines/tiny_conv_fp16.plan`.
- `trtexec_tiny_conv.log` contains `PASSED TensorRT.trtexec`.

## Build SAM2 Engines

Run on scan-train after syncing scripts to `/root/epfs/new_route_scripts`:

```bash
cd /root/epfs/new_route_scripts
GPU_ID=0 BATCH_SIZE=64 ./build_sam2_tensorrt_engines.sh
```

Current verified artifacts:

- encoder ONNX:
  `/root/epfs/sam2_tensorrt/onnx/sam2_hiera_l_image_encoder.onnx`
- decoder ONNX:
  `/root/epfs/sam2_tensorrt/onnx/sam2_hiera_l_point_decoder_b64.onnx`
- encoder engine:
  `/root/epfs/sam2_tensorrt/engines/sam2_hiera_l_image_encoder_fp16.plan`
- decoder engine:
  `/root/epfs/sam2_tensorrt/engines/sam2_hiera_l_point_decoder_b64_fp16.plan`

The decoder ONNX export is intentionally done on CPU. CUDA export currently
hits a PyTorch ONNX tracer device-mismatch path in SAM2's `repeat_image`
decoder branch. The CPU-exported decoder graph builds and runs under TensorRT.

## C++ Runner Smoke

Build and run the C++ smoke runner on scan-train:

```bash
cd /root/epfs/new_route_scripts
./build_sam2_tensorrt_runner.sh
CUDA_VISIBLE_DEVICES=0 /root/epfs/sam2_tensorrt/bin/sam2_trt_runner \
  --engine /root/epfs/sam2_tensorrt/engines/sam2_hiera_l_image_encoder_fp16.plan \
  --engine /root/epfs/sam2_tensorrt/engines/sam2_hiera_l_point_decoder_b64_fp16.plan \
  --run
```

Verified output includes:

- encoder input/output tensors:
  `image`, `high_res_0`, `high_res_1`, `image_embed`
- decoder input/output tensors:
  `image_embed`, `high_res_0`, `high_res_1`, `point_coords`, `point_labels`,
  `low_res_masks`, `iou_predictions`
- both engines report `run ok` from C++ TensorRT runtime.

Current trtexec smoke performance on scan-train GPU0:

- encoder: about `53 qps`, mean GPU compute about `18.8 ms`
- decoder batch 64: about `84 qps`, mean GPU compute about `11.8 ms`

## C++ AMG Runner

The current production-shaped runner is:

- source: `tools/sam2_trt_amg_runner.cpp`
- remote binary:
  `/root/epfs/sam2_tensorrt/bin/sam2_trt_amg_runner`

Build it on scan-train with:

```bash
cd /root/epfs/new_route_scripts
SRC=/root/epfs/new_route_tools/sam2_trt_amg_runner.cpp \
OUT=/root/epfs/sam2_tensorrt/bin/sam2_trt_amg_runner \
./build_sam2_tensorrt_runner.sh
```

Run an isolated candidate sample:

```bash
CUDA_VISIBLE_DEVICES=0 /root/epfs/sam2_tensorrt/bin/sam2_trt_amg_runner \
  --images "/root/epfs/new_route_stage1_skymask/sam2_input_2000_2999/cam0_00200[0-4].png" \
  --output-dir /root/epfs/sam2_tensorrt/sam_masks_candidate_smoke5 \
  --crop-n-layers 1 \
  --crop-nms-thresh 0.65 \
  --overwrite
```

The runner writes Python-compatible artifact names and schema:

- `{image_id}_sam_masks.json`
- `{image_id}_sam_masks.png`
- `{image_id}_numbered.png`
- `{image_id}_sam_done.flag`

For candidate benchmarks, use `--output-mode uncompressed_rle`. The comparison
tool can read both the original bool-list `binary_mask` schema and the
uncompressed RLE schema, while RLE avoids hundreds of MB of JSON per image.
Use `--output-mode binary_mask` only when testing exact compatibility with
legacy consumers that directly expect a 2D bool list in `segmentation`.

Current implementation coverage:

- SAM2 image preprocessing: resize to `1024x1024`, RGB, ImageNet mean/std.
- full-image and `crop_n_layers=1` crop boxes.
- `points_per_side=32`, `points_per_batch=64`.
- TensorRT encoder and point decoder execution.
- bilinear mask upsample to original image size.
- predicted IoU, stability score, area filtering.
- crop-edge filtering, box NMS, crop NMS, overlap resolution.
- JSON/overlay/numbered PNG/flag output compatible with downstream consumers.
- `binary_mask` and `uncompressed_rle` JSON output modes.

Current gaps:

- It has not yet implemented small-region hole/island cleanup equivalent to
  SAM2's CUDA connected-components postprocess.
- Some downstream scripts still assume bool-list `segmentation`; those should
  be updated before promoting RLE artifacts into the main semantic pipeline.
- It is not promoted to the main mask directory until a 20-50 image benchmark
  passes the promotion gates below.

Verified smoke comparison against Python SAM2 on `cam0_002000` to
`cam0_002004`:

- images: `5`
- mean baseline masks: `30.4`
- mean candidate masks: `32.8`
- mean baseline coverage: `0.8062`
- mean candidate coverage: `0.8120`
- mean coverage delta: `+0.0058`
- mean matched IoU: `0.9663`
- mean unmatched baseline masks: `3.6`
- elapsed wall time: `95.9 s` for 5 images with compatible bool-list JSON.

Verified RLE benchmark on `cam0_002000` to `cam0_002009`:

- images: `10`
- mean baseline masks: `22.9`
- mean candidate masks: `25.5`
- mean baseline coverage: `0.7005`
- mean candidate coverage: `0.7290`
- mean coverage delta: `+0.0285`
- mean matched IoU: `0.9534`
- mean unmatched baseline masks: `3.2`
- mean unmatched candidate masks: `5.8`
- candidate output directory size including PNG previews: `34 MB`

Verified RLE benchmark on `cam0_002000` to `cam0_002019`:

- images: `20`
- mean baseline masks: `27.6`
- mean candidate masks: `31.9`
- mean baseline coverage: `0.6659`
- mean candidate coverage: `0.7158`
- mean coverage delta: `+0.0499`
- mean matched IoU: `0.9469`
- mean unmatched baseline masks: `3.2`
- mean unmatched candidate masks: `7.5`
- candidate output directory size including PNG previews: `65 MB`

The 20-image benchmark confirms model-level alignment, but it also shows that
the TensorRT AMG candidate is more permissive than the Python baseline. The
worst observed frame was `cam0_002005`, where candidate coverage was `+0.2083`
above baseline. Next tuning should focus on crop/candidate NMS and overlap
resolution rather than encoder/decoder export.

NMS sweep result on `cam0_002000` to `cam0_002009`:

- default `box=0.7,crop=0.7,pred=0.7`: coverage delta `+0.0285`,
  matched IoU `0.9534`, unmatched candidate `5.8`.
- `box=0.7,crop=0.65,pred=0.7`: coverage delta `+0.0283`,
  matched IoU `0.9539`, unmatched candidate `5.2`.
- raising `pred_iou_thresh` to `0.75` or `0.8` further reduces coverage delta,
  but increases unmatched baseline masks, so it is not the default candidate.

Benchmark launcher default is therefore `box=0.7,crop=0.65,pred=0.7`. The C++
runner itself still defaults to SAM2-like `crop_nms_thresh=0.7`; pass explicit
parameters for benchmarks and promotion tests.

The high matched IoU means the TensorRT encoder/decoder path is numerically
close enough for a larger side benchmark. With bool-list output, runtime is
dominated by CPU mask upsample/postprocess and very large JSON writes. RLE
removes the JSON-size bottleneck for candidate benchmarks.

## Python AMG Trace Parity

Use the official Python SAM2 trace runner when C++ output drifts from the
Python baseline:

```bash
CUDA_VISIBLE_DEVICES=1 /root/epfs/conda_envs/vlm_seg/bin/python \
  /root/epfs/new_route_scripts/trace_python_sam2_amg.py \
  --images "/root/epfs/sam2_tensorrt/trace_worst3_input/*.png" \
  --output-dir /root/epfs/sam2_tensorrt/python_trace_worst3 \
  --crop-n-layers 1 \
  --crop-nms-thresh 0.7
```

The `cam0_002005`, `cam0_002039`, `cam0_002045` worst-frame trace showed that
the final mask count is already close between Python and C++:

- Python official + project overlap resolution: mean final masks `33.3`.
- C++ trace before any edge-order experiment: mean final masks `33.3`.

Therefore the current promotion blocker is not a count-level failure. The
remaining production issue is mask geometry/coverage parity: the C++ candidate
keeps high matched-mask IoU, but systematically over-covers some frames.

An experiment moving crop-edge filtering before within-crop NMS, matching the
official Python order, was tested on the 50-image gate. It did not improve the
promotion metrics:

- status: `fail`
- mean matched IoU: `0.9474`
- mean coverage delta: `+0.0883`
- mean unmatched baseline masks: `5.56`
- mean unmatched candidate masks: `9.22`
- worst frame: `cam0_002039`, coverage delta `+0.3646`

Do not promote the C++ runner over Python SAM2 until the coverage drift is
fixed. Next parity work should compare per-mask geometry on worst frames, with
focus on low-resolution mask upsampling, bbox edge tests, and crop boundary
handling rather than further threshold sweeps.

For worst-frame visual inspection, generate union-level extra/missing overlays:

```bash
/root/epfs/conda_envs/vlm_seg/bin/python \
  /root/epfs/new_route_scripts/visualize_sam_mask_diff.py \
  --baseline-dir /root/epfs/new_route_stage1_skymask/sam_masks_2000_2999_combined \
  --candidate-dir /root/epfs/new_route_stage1_skymask/sam_masks_2000_2999_trt_candidate_rle50_edgefirst \
  --image-dir /root/epfs/new_route_stage1_skymask/sam2_input_2000_2999 \
  --output-dir /root/epfs/sam2_tensorrt/reports/mask_diff_worst_edgefirst \
  --image-id cam0_002039 \
  --image-id cam0_002045 \
  --image-id cam0_002005
```

Color convention: gray means both baseline and candidate cover the pixel,
magenta means candidate-only extra, and yellow means baseline-only missing.
The worst frames show very low missing ratios but large candidate-only regions:

- `cam0_002039`: extra `36.55%`, missing `0.09%`.
- `cam0_002045`: extra `27.26%`, missing `0.37%`.
- `cam0_002005`: extra `20.91%`, missing `0.12%`.

Visual inspection shows a mixed picture: many extra pixels are plausible large
surface coverage, while some lie on scene boundaries or background structures.
Treat Python parity as a regression guard, not as ground truth. A C++ production
promotion still needs downstream validation with skymask, VLM labels, and point
projection quality.

## Downstream Semantic Smoke

Use the downstream smoke runner to validate that a C++/TensorRT mask directory
can be consumed by the existing semantic route:

```bash
cd /root/epfs/new_route_scripts
MANIFEST=/root/epfs/new_route_stage1_skymask/semantic_manifest_2000_2999.json \
SAM_MASK_SOURCE=/root/epfs/new_route_stage1_skymask/sam_masks_2000_2999_trt_candidate_rle50 \
OUTPUT_DIR=/root/epfs/sam2_tensorrt/semantic_eval_rle50_default_downstream10_cam0 \
FILTER_PREFIX=cam0_ \
LIMIT=10 \
SHARDS=4 \
CHUNK_SIZE=4 \
VLM_ENDPOINT=http://localhost:8001/v1/chat/completions \
bash ./run_server_sam2_trt_semantic_smoke.sh
```

Verified on scan-vlm with Qwen `-np 4`:

- linked C++ RLE masks: `10/10`
- `sam2_qwen`: `10/10`
- `sam2_sky_label_merge_qwen_review`: `10/10`
- `sam2_prompt_v3_sky_label_merge`: `10/10`
- `sam2_prompt_v3_sky_label_merge_completion`: `10/10`
- completion label records: `145`
- top completion labels: `equipment=61`, `floor=42`, `building=20`,
  `ignore=20`, `pipe=1`, `railing=1`

This proves the C++ RLE artifacts are operationally compatible with the current
semantic pipeline. It does not yet prove semantic quality is better than the
Python SAM2 route; the high `equipment` count and large-surface coverage need
visual QA and point-projection QA before promotion.

## Notes

- C++ TensorRT is pinned to CUDA 11.8 because `/usr/local/cuda` points to
  `/usr/local/cuda-11.8` on scan-train.
- Python TensorRT 11 is installed for export/prototyping experiments, but the
  C++ ABI baseline is TensorRT 10.9 CUDA 11.8.
- `pycuda` is intentionally not required. It failed to build against the
  current container CUDA header layout and is not needed for the C++ path.

## Accuracy Comparison Gate

There are now SAM2 encoder/point-decoder TensorRT engines, a C++ smoke runner,
and a production-shaped C++ AMG runner that writes compatible
`*_sam_masks.json` artifacts. This is still a side-track candidate until the
larger benchmark proves that thin-object recall and downstream target/object
quality do not regress.

Once the C++/TensorRT AMG runner writes candidate masks, compare it against the
current Python baseline with:

```bash
python3 compare_sam_mask_dirs.py \
  --baseline-dir /root/epfs/new_route_stage1_skymask/sam_masks_2000_2999_combined \
  --candidate-dir /root/epfs/sam2_tensorrt/sam_masks_candidate \
  --manifest /root/epfs/new_route_stage1_skymask/semantic_manifest_2000_2999.json \
  --limit 50 \
  --json-output /root/epfs/sam2_tensorrt/reports/python_vs_trt_masks.json \
  --csv-output /root/epfs/sam2_tensorrt/reports/python_vs_trt_masks.csv
```

Promotion criteria for replacing the Python SAM2 generator:

- mean matched-mask IoU should be high on the validation sample.
- coverage delta should be small, especially on ground/wall/railing frames.
- unmatched baseline masks should not concentrate on thin objects such as
  railings, pipes, edges, or equipment handles.
- downstream semantic label records and target/object fusion should not regress.

## Production Candidate Runner

Use `run_server_sam2_trt_production.sh` for production-shaped candidate output.
It writes `uncompressed_rle` JSON by default and patches
`semantic_eval/run_eval.py` with `patch_semantic_eval_rle_masks.py` so the
existing VLM, merge, completion, and artifact-writing path can decode compact
RLE masks. This avoids duplicating the current hundreds-of-GB bool-list SAM2
mask cache.

Example 20-image validation run:

```bash
START=2000 END=2999 \
IMAGE_GLOB='/root/epfs/new_route_stage1_skymask/sam2_input_2000_2999/cam0_0020[0-1][0-9].png' \
OUTPUT_DIR=/root/epfs/new_route_stage1_skymask/sam_masks_2000_2999_trt_candidate_20 \
REPORT_DIR=/root/epfs/sam2_tensorrt/reports/production_2000_2999_20 \
BASELINE_DIR=/root/epfs/new_route_stage1_skymask/sam_masks_2000_2999_combined \
bash run_server_sam2_trt_production.sh
```

The script runs:

- Optional semantic-eval RLE loader patch when `OUTPUT_MODE=uncompressed_rle`.
- C++ TensorRT AMG candidate generation.
- Candidate manifest generation.
- Optional baseline comparison when `BASELINE_DIR` is set.
- `gate_sam2_trt_promotion.py`, which fails nonzero if the candidate exceeds
  configured drift thresholds.

Default promotion gate thresholds:

- `mean_matched_iou >= 0.93`
- `abs(mean_coverage_delta) <= 0.06`
- `abs(row coverage_delta) <= 0.25`
- `mean_unmatched_baseline_masks <= 4.0`
- `mean_unmatched_candidate_masks <= 8.0`

If the gate passes, the candidate directory is eligible for downstream
semantic-eval smoke testing. It still should not replace
`sam_masks_${START}_${END}_combined` until target/object QA on the same frame
range is at least neutral against the Python baseline.

Set `OUTPUT_MODE=binary_mask` only for strict legacy compatibility checks. That
mode is disk-expensive: observed Python baseline bool-list masks are already
hundreds of MB per image, and the full `2000-2999` combined cache is hundreds of
GB.

Verified production-shaped RLE smoke on `cam0_002000` to `cam0_002009`:

- candidate dir:
  `/root/epfs/new_route_stage1_skymask/sam_masks_2000_2999_trt_candidate_rle10`
- output size: `34 MB` for 10 images.
- gate status: `pass`
- mean matched IoU: `0.9539`
- mean coverage delta: `+0.0283`
- mean unmatched baseline masks: `3.1`
- mean unmatched candidate masks: `5.2`
- downstream RLE smoke:
  `/root/epfs/sam2_tensorrt/semantic_eval_rle_smoke10`, two images reached
  `sam2_qwen` artifact generation through `semantic_eval/run_eval.py`.

Verified wider production-shaped RLE test on `cam0_002000` to `cam0_002049`:

- candidate dir:
  `/root/epfs/new_route_stage1_skymask/sam_masks_2000_2999_trt_candidate_rle50`
- output size: `168 MB` for 50 images.
- gate status: `fail`
- mean matched IoU: `0.9474`
- mean coverage delta: `+0.0841`
- mean unmatched baseline masks: `5.54`
- mean unmatched candidate masks: `9.08`
- worst coverage delta: `+0.3646` on `cam0_002039`.

Additional 50-image tuning results:

- `PRED_IOU_THRESH=0.75`:
  - gate status: `fail`
  - mean matched IoU: `0.9479`
  - mean coverage delta: `+0.0764`
  - mean unmatched baseline masks: `6.38`
  - mean unmatched candidate masks: `8.48`
  - worst coverage delta: `+0.3536` on `cam0_002039`
- `CROP_N_LAYERS=0`:
  - gate status: `fail`
  - mean matched IoU: `0.9352`
  - mean coverage delta: `-0.0104`
  - mean unmatched baseline masks: `12.6`
  - mean unmatched candidate masks: `2.74`

Interpretation: TensorRT encoder/decoder alignment is acceptable, but the C++
AMG postprocessing is not production-equivalent yet. Raising `pred_iou_thresh`
reduces over-coverage only slightly and increases missed baseline masks. Turning
off crops removes most mean over-coverage but misses too many masks. Therefore
the next production task is crop/uncrop/NMS parity with Python SAM2 AMG, not
more blind threshold sweeps. Keep Python SAM2 as the production mask source
until C++ postprocessing parity passes the 50-image promotion gate.

## Parity Diagnosis

`analyze_sam_mask_parity.py` adds a failure diagnosis layer on top of the
promotion gate. It reports union extra/missing pixels and unmatched-mask area
distributions.

Default 50-image RLE candidate:

- mean coverage delta: `+0.0841`
- mean extra pixel ratio: `0.0948`
- mean missing pixel ratio: `0.0108`
- mean union IoU: `0.8553`
- mean unmatched baseline area ratio: `0.0624`
- mean unmatched candidate area ratio: `0.1847`
- worst extra-pixel rows include `cam0_002039`, `cam0_002045`,
  `cam0_002038`, `cam0_002027`, and `cam0_002035`.

`CROP_N_LAYERS=0` 50-image candidate:

- mean coverage delta: `-0.0104`
- mean extra pixel ratio: `0.0217`
- mean missing pixel ratio: `0.0321`
- mean union IoU: `0.9165`
- mean unmatched baseline area ratio: `0.0513`
- mean unmatched candidate area ratio: `0.0189`

This isolates the primary failure:

- Full crop mode creates too much extra candidate area.
- No-crop mode removes most extra area but misses too many baseline masks.
- The target is not "disable crops"; it is Python SAM2 crop parity.

Relevant Python SAM2 ordering in
`sam2/automatic_mask_generator.py`:

1. For each crop, run point-grid batches through `SAM2ImagePredictor`.
2. Filter by predicted IoU and stability.
3. Threshold logits, compute boxes, and drop boxes near crop edges.
4. Compress crop masks to RLE and return to original image frame.
5. Run within-crop box NMS scored by `iou_preds`.
6. Run cross-crop box NMS scored by `1 / crop_box_area`.
7. In this project wrapper, run project-level overlap resolution by
   `predicted_iou * stability_score`.

The C++ runner should be changed against that ordered checklist before another
large sweep. The next concrete implementation target is to add parity traces
for per-crop candidate counts, crop-edge drops, within-crop NMS keeps, and
cross-crop NMS keeps, then adjust the C++ crop path until the 50-image
diagnosis no longer shows large unmatched candidate area.

## C++ Trace Runner

The C++ runner supports `--write-trace`. It writes
`{image_id}_trace.json` next to mask artifacts with:

- per-crop `raw_candidates`
- `after_within_crop_nms`
- `dropped_near_crop_edge`
- `after_crop_edge_filter`
- image-level `before_cross_crop_nms`
- `after_cross_crop_nms`
- `after_overlap_resolution`

`summarize_sam2_trt_traces.py` aggregates these trace files.

Trace smoke on `cam0_002000` and `cam0_002001`:

- mean raw candidates: `3590.5`
- mean after within-crop NMS: `110.5`
- mean after crop-edge filter: `81.5`
- mean after cross-crop NMS: `49.5`
- mean after overlap resolution: `34.0`
- within-crop NMS keep ratio: `0.0308`
- crop-edge drop ratio: `0.2624`
- cross-crop NMS keep ratio: `0.6074`
- overlap-resolution keep ratio: `0.6869`

Worst over-coverage trace on `cam0_002005`, `cam0_002039`, and
`cam0_002045`:

- mean raw candidates: `2588.0`
- mean after within-crop NMS: `105.0`
- mean after crop-edge filter: `83.7`
- mean after cross-crop NMS: `48.3`
- mean after overlap resolution: `33.3`
- within-crop NMS keep ratio: `0.0406`
- crop-edge drop ratio: `0.2032`
- cross-crop NMS keep ratio: `0.5777`
- overlap-resolution keep ratio: `0.6897`

This trace does not show a single count-level explosion in one crop or one NMS
stage. The worst frames look similar to ordinary frames by stage counts. The
next parity task should therefore compare crop-level boxes and mask shapes
against Python SAM2, especially:

- whether C++ coordinates use the same transformed point grid as
  `SAM2ImagePredictor._transforms.transform_coords`
- whether C++ bilinear upsample and thresholding match SAM2 logits handling
- whether C++ box IoU/NMS uses the same `xyxy` coordinate convention as
  `torchvision.ops.batched_nms`
- whether C++ cross-crop NMS should run before or after full uncrop exactly as
  Python `MaskData` does
